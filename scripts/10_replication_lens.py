"""Test 2: replicate the layer-20 about/injected dissociation with more pairs, better vectors,
four read-outs, and a logit lens over every layer.

  vectors   sentence-derived at layer 20 from 6 held-out sentences per concept
  trials    all ordered (X, Y) pairs of the 16 concepts with different categories and
            |cos(v_sent,X, v_sent,Y)| < 0.3; Y's stimulus sentence is sentence 3, X injected on it
  rho       0, 0.5, 1, 2   (rho = injection norm / |v_sent,X|; the real Y sentence has s_Y = 1)
  read-outs letter  A-E with X, Y and 3 distractors, 5 cyclic orders
            2afc    "A) X / B) Y", both orders
            oneword "Answer with one word."
            free    one short sentence (first 48 ordered pairs), judged by name matching
  For every forced read-out we store logit(X) - logit(Y) at the answer position for every layer
  (logit lens: final norm + unembedding applied to each layer's output). Normalizers cancel in
  the difference, so only the X and Y rows of the unembedding are needed.
"""

import argparse
import importlib.util
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pilot.common import (ALL_CANDIDATES, CATEGORY, DATA, RESULTS, capture, chat, get_layers,
                          load_json, load_model, save_json)

_spec = importlib.util.spec_from_file_location("f05", Path(__file__).parent / "05_followups.py")
f05 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(f05)
_spec = importlib.util.spec_from_file_location("v01", Path(__file__).parent / "01_vectors_and_steering_check.py")
v01 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v01)

L_INJ = 20
RHOS = [0.0, 0.5, 1.0, 2.0]
MAX_COS = 1.01  # no filter: the shared component cancels in X:Y log-odds; cos is logged per trial
N_FREE = 48


class Lens:
    """logit(a) - logit(b) at the last position for every layer, via final norm + unembedding."""

    def __init__(self, model):
        self.model = model
        self.norm = model.model.language_model.norm
        self.W = model.lm_head.weight
        self.cap = getattr(model.config.text_config, "final_logit_softcapping", None)
        self.n = len(get_layers(model))

    def diff(self, store, ids_a, ids_b):
        """store: {layer: hidden [B, T, d]}; ids_a/ids_b: token-id lists (logsumexp over variants)."""
        h = torch.stack([store[li][:, -1] for li in range(self.n)], 1)  # [B, n_layers, d]
        logits = self.norm(h) @ self.W[ids_a + ids_b].T                 # [B, n_layers, k]
        logits = logits.float()
        if self.cap:
            logits = self.cap * torch.tanh(logits / self.cap)
        a = torch.logsumexp(logits[..., :len(ids_a)], -1)
        b = torch.logsumexp(logits[..., len(ids_a):], -1)
        return (a - b).cpu().numpy()  # [B, n_layers]


@torch.no_grad()
def lens_batch(model, lens, ids, conds, unit, r, pairs_of_ids):
    """Run one prompt under each condition; return [len(conds), n_pairs, n_layers] lens diffs
    and the final-layer full log-softmax at those ids for a sanity check."""
    with f05.inject_rows(model, f05.build_rows(conds, ids.shape[1], unit, r)), \
            capture(model, range(lens.n)) as st:
        out = model(input_ids=ids.expand(len(conds), -1), use_cache=False, logits_to_keep=1)
    diffs = np.stack([lens.diff(st, a, b) for a, b in pairs_of_ids], 1)
    final = out.logits[:, -1].float()
    return diffs, final


@torch.no_grad()
def sent_vectors(model, tok, stim, vs, names):
    fill = stim["fillers"]

    def s3_mean(s3):
        sents = [fill[0], fill[1], s3, fill[2], fill[3]]
        text = chat(tok, f"{f05.PRE}\n\n{f05.body(sents)}\n\nQ: {f05.Q_ABOUT}")
        a, b = f05.spans(tok, text, sents)[2]
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
        with capture(model, [L_INJ]) as st:
            model(input_ids=ids, use_cache=False, logits_to_keep=1)
        return st[L_INJ][0, a:b].float()

    neu = torch.cat([s3_mean(s) for s in vs["neutral"]])
    r = neu.norm(dim=-1).median().item()
    neu_mean = neu.mean(0)
    v = torch.stack([torch.stack([s3_mean(s).mean(0) for s in vs["concepts"][c]]).mean(0) - neu_mean
                     for c in names])
    return v.cpu(), r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noname", action="store_true",
                    help="test 2b: sentence-3 stimuli that never name their concept (data/stimuli_noname.json)")
    args = ap.parse_args()
    tag = "_noname" if args.noname else ""
    t0 = time.time()
    sel = load_json(DATA / "selection.json")
    stim = load_json(DATA / "stimuli.json")
    if args.noname:
        stim["sentences"] = load_json(DATA / "stimuli_noname.json")["sentences"]
    vs = load_json(DATA / "vector_sentences.json")
    names = sel["concepts"]
    model, tok = load_model()
    lens = Lens(model)

    v, r_val = sent_vectors(model, tok, stim, vs, names)
    unit = {("sent", L_INJ): (v / v.norm(dim=1, keepdim=True)).cuda()}
    r = {L_INJ: r_val}
    vnorm = v.norm(dim=1)
    ucos = (unit[("sent", L_INJ)] @ unit[("sent", L_INJ)].T).cpu()
    old = torch.load(DATA / "followup_vectors.pt")[f"sent_L{L_INJ}"]
    agree = torch.nn.functional.cosine_similarity(v, old, dim=1)
    print(f"r={r_val:.0f}; |v|/r median {(vnorm / r_val).median():.3f}; cos(6-sentence, 3-sentence vectors) "
          f"median {agree.median():.3f}", flush=True)
    if not args.noname:
        torch.save({"sent_L20_6sent": v}, DATA / "replication_vectors.pt")

    # trials
    cand = ALL_CANDIDATES
    cidx = {n: i for i, n in enumerate(cand)}
    cmat = np.array([[sel["cosine"][a][b] for b in cand] for a in cand])
    pool = [n for n in cand if n not in names]
    trials = []
    for xi, yi in itertools.permutations(range(len(names)), 2):
        x, y = names[xi], names[yi]
        if CATEGORY[x] == CATEGORY[y] or abs(ucos[xi, yi]) >= MAX_COS:
            continue
        trials.append((x, y, v01.pick_distractors(x, y, pool, cmat, cidx), float(ucos[xi, yi])))
    print(f"{len(trials)} ordered pairs", flush=True)

    letters = [tok.encode(l, add_special_tokens=False)[0] for l in f05.LETTERS]
    variants = {c: sorted({tok.encode(s, add_special_tokens=False)[0] for s in (c, c.lower(), " " + c, " " + c.lower())})
                for c in names}
    fill = stim["fillers"]
    rows, free, checks = [], [], []
    qs = [("about", f05.Q_ABOUT), ("injected", f05.Q_INJ)]

    for ti, (x, y, dis, cxy) in enumerate(trials):
        xi = names.index(x)
        sents = [fill[0], fill[1], stim["sentences"][y], fill[2], fill[3]]

        def conds(span):
            return [dict(layer=L_INJ, vtype="sent", c=rho * vnorm[xi].item() / r_val, rho=rho, X=x, Xi=xi, span=span)
                    for rho in RHOS]

        def run(user, readout, q, order, ids_x, ids_y, extra=None):
            text = chat(tok, user)
            sp = f05.spans(tok, text, sents)
            ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
            cs = conds(sp[2])
            diffs, final = lens_batch(model, lens, ids, cs, unit, r, [(ids_x, ids_y)])
            lp = final.log_softmax(-1)
            for k, c in enumerate(cs):
                fx = torch.logsumexp(lp[k, ids_x], 0).item()
                fy = torch.logsumexp(lp[k, ids_y], 0).item()
                rows.append(dict(X=x, Y=y, cos_xy=cxy, rho=c["rho"], readout=readout, question=q, order=order,
                                 final_lo=fx - fy, logp_X=fx, logp_Y=fy, lens_lo=diffs[k, 0].round(3).tolist())
                            | (extra or {}))
                if len(checks) < 50:
                    checks.append(abs((fx - fy) - diffs[k, 0, -1]))

        base = [x, y] + dis
        for o, (qn, q) in itertools.product(range(5), qs):
            opts = base[o:] + base[:o]
            user = (f"{f05.PRE}\n\n{f05.body(sents)}\n\nQ: {q}\n"
                    + "\n".join(f"{l}) {op}" for l, op in zip(f05.LETTERS, opts)) + "\nAnswer with one letter.")
            # letter probabilities over A-E too, for P(X)
            run(user, "letter", qn, o, [letters[opts.index(x)]], [letters[opts.index(y)]])
        for o, (qn, q) in itertools.product(range(2), qs):
            opts = [x, y] if o == 0 else [y, x]
            user = (f"{f05.PRE}\n\n{f05.body(sents)}\n\nQ: {q}\nA) {opts[0]}\nB) {opts[1]}\nAnswer with one letter.")
            run(user, "2afc", qn, o, [letters[opts.index(x)]], [letters[opts.index(y)]])
        for qn, q in qs:
            user = f"{f05.PRE}\n\n{f05.body(sents)}\n\nQ: {q} Answer with one word."
            run(user, "oneword", qn, 0, variants[x], variants[y])
            if (ti < N_FREE) if not args.noname else (ti % 5 == 0):  # 2b: spread over all X
                text = chat(tok, f"{f05.PRE}\n\n{f05.body(sents)}\n\nQ: {q} Answer in one short sentence.")
                sp = f05.spans(tok, text, sents)
                ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
                cs = conds(sp[2])
                with f05.inject_rows(model, f05.build_rows(cs, ids.shape[1], unit, r)):
                    out = model.generate(input_ids=ids.expand(len(cs), -1), max_new_tokens=30, do_sample=False)
                for c, o_ in zip(cs, out):
                    a = tok.decode(o_[ids.shape[1]:], skip_special_tokens=True).strip()
                    al = a.lower()
                    free.append(dict(X=x, Y=y, rho=c["rho"], question=qn, answer=a,
                                     names_X=x.lower().rstrip("s")[:5] in al, names_Y=y.lower().rstrip("s")[:5] in al))
        if ti % 20 == 0:
            print(f"{ti + 1}/{len(trials)} {time.time() - t0:.0f}s", flush=True)

    print(f"lens sanity: max |final - lens(last layer)| = {max(checks):.4f}", flush=True)
    save_json(rows, RESULTS / f"replication{tag}_rows.json")
    save_json(free, RESULTS / f"replication{tag}_free_text.json")
    print(f"all done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

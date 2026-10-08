"""Main appearance-vs-reality test, rerun with sentence-derived vectors (where Gate 1 passes).

Sentence 3 = a real sentence about Y; X (its pair partner) is injected on the sentence-3
tokens using the sentence-derived vector at layers 20 and 28 (from scripts/05_followups.py).
Strength is set in natural units:
    s_X = injection norm / |v_sent,X|   (1 = as strong as a real sentence about X, per token)
and the real Y sentence has s_Y = 1 by construction, so rho = s_X.
Read-outs: "about" and "injected" questions, A-E letters over 5 cyclic orders (P(X), P(Y)),
and one-word answers (log P of X's and Y's names). Plus free text for 4 trials.
Proof of life = the two questions come apart: P_inj(X) > P_about(X) at some rho.
"""

import importlib.util
import itertools
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pilot.common import DATA, RESULTS, chat, load_json, load_model, save_json

_spec = importlib.util.spec_from_file_location("f05", Path(__file__).parent / "05_followups.py")
f05 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(f05)

LAYERS = [20, 28]
RHOS = [0.25, 0.5, 1, 2, 4, 8]


def main():
    t0 = time.time()
    sel = load_json(DATA / "selection.json")
    stim = load_json(DATA / "stimuli.json")
    meta = load_json(RESULTS / "followup_meta.json")
    names = sel["concepts"]
    partner = {a: b for p in sel["pairs"] for a, b in (p, p[::-1])}
    raw = torch.load(DATA / "followup_vectors.pt")
    vec = {L: raw[f"sent_L{L}"] for L in LAYERS}
    unit = {("sent", L): (v / v.norm(dim=1, keepdim=True)).cuda() for L, v in vec.items()}
    r = {L: meta["r"][str(L)] for L in LAYERS}
    model, tok = load_model()
    letter_ids = [tok.encode(l, add_special_tokens=False)[0] for l in f05.LETTERS]
    variants = {c: sorted({tok.encode(v, add_special_tokens=False)[0]
                           for v in (c, c.lower(), " " + c, " " + c.lower())}) for c in names}
    all_ids = sorted({i for v in variants.values() for i in v})
    fill = stim["fillers"]
    rows, ft = [], []

    for x in names:
        y = partner[x]
        xi = names.index(x)
        sents = [fill[0], fill[1], stim["sentences"][y], fill[2], fill[3]]
        # c (fraction of r) that gives s_X = rho for this concept at this layer
        conds = [dict(layer=-1, vtype="none", c=0.0, rho=0.0, X=x, Xi=xi, span=None)]

        def add_conds(span):
            out = [dict(conds[0])]
            for L, rho in itertools.product(LAYERS, RHOS):
                out.append(dict(layer=L, vtype="sent", c=rho * vec[L][xi].norm().item() / r[L], rho=rho,
                                X=x, Xi=xi, span=span))
            return out

        base = [x, y] + sel["distractors"][x]
        for o in range(5):
            opts = base[o:] + base[:o]
            for qname, q in [("about", f05.Q_ABOUT), ("injected", f05.Q_INJ)]:
                user = (f"{f05.PRE}\n\n{f05.body(sents)}\n\nQ: {q}\n"
                        + "\n".join(f"{l}) {op}" for l, op in zip(f05.LETTERS, opts)) + "\nAnswer with one letter.")
                text = chat(tok, user)
                sp = f05.spans(tok, text, sents)
                ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
                cs = add_conds(sp[2])
                for cnd, lp in zip(cs, f05.run_batched(model, ids, cs, unit, r, letter_ids)):
                    rows.append(f05.strip(cnd) | dict(block="letter", Y=y, question=qname, order=o, options=opts,
                                                      pair=[i for i, p in enumerate(sel["pairs"]) if x in p][0],
                                                      logp=lp))
        for qname, q in [("about", f05.Q_ABOUT), ("injected", f05.Q_INJ)]:
            text = chat(tok, f"{f05.PRE}\n\n{f05.body(sents)}\n\nQ: {q} Answer with one word.")
            sp = f05.spans(tok, text, sents)
            ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
            cs = add_conds(sp[2])
            for cnd, lp in zip(cs, f05.run_batched(model, ids, cs, unit, r, all_ids)):
                lpd = dict(zip(all_ids, lp))
                cl = {c: torch.logsumexp(torch.tensor([lpd[i] for i in variants[c]]), 0).item() for c in names}
                rows.append(f05.strip(cnd) | dict(block="oneword", Y=y, question=qname, concept_logp=cl,
                                                  pair=[i for i, p in enumerate(sel["pairs"]) if x in p][0]))
            if xi < 4:
                text = chat(tok, f"{f05.PRE}\n\n{f05.body(sents)}\n\nQ: {q} Answer in one short sentence.")
                sp = f05.spans(tok, text, sents)
                ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
                cs = add_conds(sp[2])
                with f05.inject_rows(model, f05.build_rows(cs, ids.shape[1], unit, r)):
                    out = model.generate(input_ids=ids.expand(len(cs), -1), max_new_tokens=40, do_sample=False)
                for cnd, o in zip(cs, out):
                    ft.append(f05.strip(cnd) | dict(Y=y, question=qname,
                                                    answer=tok.decode(o[ids.shape[1]:], skip_special_tokens=True).strip()))
        print(f"{xi + 1}/16 {x}->{y} {time.time() - t0:.0f}s", flush=True)

    save_json(rows, RESULTS / "main_sent_rows.json")
    save_json(ft, RESULTS / "main_sent_free_text.json")
    print(f"all done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

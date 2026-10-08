"""Follow-up tests after the pilot (Gate 1 failed with sentence-3-only injection).

Factors (all with the neutral sentence 3, i.e. Gate-1 setting):
  layer   12, 20, 28, 37
  vector  macar: last token of chat("Tell me about {c}") minus baseline-word mean (Macar et al.)
          sent:  mean over sentence-3 tokens of held-out sentences about c, minus the same
                 for neutral sentences, in the same 5-sentence context (data/vector_sentences.json)
  c       injection norm as a fraction of the median residual norm r_L on sentence-3 tokens:
          0, 0.05, 0.1, 0.2, 0.4, 0.8   (the pilot's alpha=1 at L37 was c ~ 0.09)

Blocks:
  1 retrieval  "What is sentence 3 about?"              A-E letters, 5 cyclic option orders
  2 report     "What thought was injected at sentence 3?" A-E letters, 5 cyclic option orders
               (+ span s3_to_end: injection through the end of the prompt, as a ceiling)
  3 one-word   same two questions, "Answer with one word." - log P of X's name as the first
               answer token vs. the other 15 concepts (no letters, no position bias)
  4 detection  "Was a thought injected...? Yes or No"   inject X at sentence k=1..5, or none
  5 localize   "Which sentence, if any...? 0-5"          same injections
  6 free text  a few generations for reading

All conditions that share a prompt run as one batch with per-row injection hooks.
"""

import importlib.util
import itertools
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pilot.common import (BASELINE_WORDS, DATA, RESULTS, _hidden, _replace, capture, chat,
                          get_layers, load_json, load_model, save_json)

_spec = importlib.util.spec_from_file_location("v01", Path(__file__).parent / "01_vectors_and_steering_check.py")
v01 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v01)

LAYERS = [12, 20, 28, 37]
VTYPES = ["macar", "sent"]
CS = [0.05, 0.1, 0.2, 0.4, 0.8]
LETTERS = ["A", "B", "C", "D", "E"]
CHUNK = 48
PRE = ("In this experiment, a thought may be injected into your processing "
       "while you read one of the numbered sentences below.")
PRE_DET = ("In this experiment, a thought may or may not be injected into your processing "
           "while you read the numbered sentences below.")
Q_ABOUT = "What is sentence 3 about?"
Q_INJ = "What thought was injected at sentence 3?"
Q_DET = "Was a thought injected while you read the sentences above? Answer Yes or No."
Q_LOC = ("In which sentence, if any, was a thought injected? "
         "Answer with one digit: 1-5, or 0 if none.")


def body(sents):
    return "\n".join(f"[{i + 1}] {s}" for i, s in enumerate(sents))


def spans(tok, text, sents):
    """Token ranges [start, end) of each numbered sentence's text."""
    offs = tok(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    out = []
    for i, s in enumerate(sents):
        a = text.index(f"[{i + 1}] {s}") + len(f"[{i + 1}] ")
        b = a + len(s)
        idx = [j for j, (x, y) in enumerate(offs) if x < b and y > a]
        out.append((idx[0], idx[-1] + 1))
    return out


@contextmanager
def inject_rows(model, per_layer):
    """per_layer: {layer: (add [B, d], mask [B, T])}; adds add[b] where mask[b, t].
    Skipped during cached decoding (sequence length differs from the mask)."""
    handles = []
    for L, (add, mask) in per_layer.items():
        def hook(mod, inp, out, add=add, mask=mask):
            h = _hidden(out)
            if h.shape[1] != mask.shape[1]:
                return out
            return _replace(out, h + mask[:, :, None].to(h.dtype) * add[:, None, :].to(h.dtype))
        handles.append(get_layers(model)[L].register_forward_hook(hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def build_rows(conds, T, unit, r):
    """conds: list of dicts with layer, vtype, X, c, span (start, end) or None for no injection."""
    B = len(conds)
    per_layer = {}
    for L in {c["layer"] for c in conds if c["c"] > 0}:
        add = torch.zeros(B, next(iter(unit.values())).shape[1], device="cuda")
        mask = torch.zeros(B, T, dtype=torch.bool, device="cuda")
        for b, c in enumerate(conds):
            if c["c"] > 0 and c["layer"] == L:
                add[b] = c["c"] * r[L] * unit[(c["vtype"], L)][c["Xi"]]
                mask[b, c["span"][0]:c["span"][1]] = True
        per_layer[L] = (add, mask)
    return per_layer


@torch.no_grad()
def run_batched(model, ids, conds, unit, r, read_ids):
    """One prompt, many injection conditions. Returns log-probs at the last position over read_ids."""
    out = []
    for i in range(0, len(conds), CHUNK):
        cs = conds[i:i + CHUNK]
        with inject_rows(model, build_rows(cs, ids.shape[1], unit, r)):
            logits = model(input_ids=ids.expand(len(cs), -1), use_cache=False, logits_to_keep=1).logits[:, -1]
        lp = logits.float().log_softmax(-1)
        out.extend(lp[:, read_ids].cpu().tolist())
    return out


def grid(names, span, extra=None):
    """All (layer, vtype, c, X) injection conditions plus one no-injection row."""
    conds = [dict(layer=-1, vtype="none", c=0.0, X="none", Xi=-1, span=None)]
    for L, vt, c, (xi, x) in itertools.product(LAYERS, VTYPES, CS, enumerate(names)):
        conds.append(dict(layer=L, vtype=vt, c=c, X=x, Xi=xi, span=span))
    if extra:
        for d in conds:
            d.update(extra)
    return conds


def strip(c):
    return {k: v for k, v in c.items() if k not in ("Xi",)}


@torch.no_grad()
def extract_vectors(model, tok, sel, stim, vs):
    names = sel["concepts"]
    # Macar vectors at each layer (single-prompt passes, as in script 01)
    cacts = v01.last_token_acts(model, tok, names, LAYERS)
    bacts = v01.last_token_acts(model, tok, BASELINE_WORDS, LAYERS)
    vec = {("macar", L): cacts[L] - bacts[L].mean(0) for L in LAYERS}

    def s3_mean(s3):
        sents = [stim["fillers"][0], stim["fillers"][1], s3, stim["fillers"][2], stim["fillers"][3]]
        text = chat(tok, f"{PRE}\n\n{body(sents)}\n\nQ: {Q_ABOUT}")
        a, b = spans(tok, text, sents)[2]
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
        with capture(model, LAYERS) as st:
            model(input_ids=ids, use_cache=False, logits_to_keep=1)
        return {L: st[L][0, a:b].float() for L in LAYERS}

    neu = [s3_mean(s) for s in vs["neutral"]]
    neu_mean = {L: torch.cat([n[L] for n in neu]).mean(0).cpu() for L in LAYERS}
    r = {L: torch.cat([n[L] for n in neu]).norm(dim=-1).median().item() for L in LAYERS}
    per_concept = {c: [s3_mean(s) for s in vs["concepts"][c]] for c in names}
    for L in LAYERS:
        vec[("sent", L)] = torch.stack([
            torch.stack([a[L].mean(0).cpu() for a in per_concept[c]]).mean(0) - neu_mean[L] for c in names])
    return vec, r


def main():
    t0 = time.time()
    sel = load_json(DATA / "selection.json")
    stim = load_json(DATA / "stimuli.json")
    vs = load_json(DATA / "vector_sentences.json")
    names = sel["concepts"]
    partner = {a: b for p in sel["pairs"] for a, b in (p, p[::-1])}
    model, tok = load_model()
    print(f"model loaded {time.time() - t0:.0f}s", flush=True)

    vec, r = extract_vectors(model, tok, sel, stim, vs)
    unit = {k: (v / v.norm(dim=1, keepdim=True)).cuda() for k, v in vec.items()}
    meta = {"r": r, "layers": LAYERS, "cs": CS, "names": names}
    for L in LAYERS:
        for vt in VTYPES:
            n = vec[(vt, L)].norm(dim=1)
            meta[f"norm_{vt}_L{L}"] = n.tolist()
        cos = (unit[("macar", L)] * unit[("sent", L)]).sum(1)
        meta[f"cos_macar_sent_L{L}"] = cos.tolist()
        print(f"L{L}: r={r[L]:.0f}  |v_macar|/r median {vec[('macar', L)].norm(dim=1).median() / r[L]:.3f}  "
              f"|v_sent|/r median {vec[('sent', L)].norm(dim=1).median() / r[L]:.3f}  "
              f"cos(macar, sent) median {cos.median():.3f}", flush=True)
    torch.save({f"{vt}_L{L}": vec[(vt, L)] for vt, L in vec}, DATA / "followup_vectors.pt")
    save_json(meta, RESULTS / "followup_meta.json")

    fill = stim["fillers"]
    neutral5 = [fill[0], fill[1], stim["neutral_s3"], fill[2], fill[3]]
    letter_ids = [tok.encode(l, add_special_tokens=False)[0] for l in LETTERS]
    rows = []

    # Blocks 1-2: letter read-outs, 5 cyclic option orders
    for x in names:
        base = [x, partner[x]] + sel["distractors"][x]
        for o in range(5):
            opts = base[o:] + base[:o]
            for qname, q in [("about", Q_ABOUT), ("injected", Q_INJ)]:
                user = (f"{PRE}\n\n{body(neutral5)}\n\nQ: {q}\n"
                        + "\n".join(f"{l}) {op}" for l, op in zip(LETTERS, opts)) + "\nAnswer with one letter.")
                text = chat(tok, user)
                sp = spans(tok, text, neutral5)
                ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
                span_sets = [("s3", sp[2])] + ([("s3_to_end", (sp[2][0], ids.shape[1]))] if qname == "injected" else [])
                conds = [dict(layer=-1, vtype="none", c=0.0, X=x, Xi=names.index(x), span=None, span_name="none")]
                for (sn, s), L, vt, c in itertools.product(span_sets, LAYERS, VTYPES, CS):
                    conds.append(dict(layer=L, vtype=vt, c=c, X=x, Xi=names.index(x), span=s, span_name=sn))
                lps = run_batched(model, ids, conds, unit, r, letter_ids)
                for cnd, lp in zip(conds, lps):
                    rows.append(strip(cnd) | dict(block="letter", question=qname, order=o, options=opts,
                                                  pair=[i for i, p in enumerate(sel["pairs"]) if x in p][0],
                                                  logp=lp))
    print(f"letter blocks done {time.time() - t0:.0f}s", flush=True)

    # Block 3: one-word answers, read log P of each concept name's first token (case/space variants)
    variants = {}
    for c in names:
        ids_c = {tok.encode(v, add_special_tokens=False)[0] for v in (c, c.lower(), " " + c, " " + c.lower())}
        variants[c] = sorted(ids_c)
    all_ids = sorted({i for v in variants.values() for i in v})
    for qname, q in [("about", Q_ABOUT), ("injected", Q_INJ)]:
        text = chat(tok, f"{PRE}\n\n{body(neutral5)}\n\nQ: {q} Answer with one word.")
        sp = spans(tok, text, neutral5)
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
        conds = grid(names, sp[2], {"span_name": "s3"})
        lps = run_batched(model, ids, conds, unit, r, all_ids)
        for cnd, lp in zip(conds, lps):
            lpd = dict(zip(all_ids, lp))
            per_concept = {c: torch.logsumexp(torch.tensor([lpd[i] for i in variants[c]]), 0).item() for c in names}
            rows.append(strip(cnd) | dict(block="oneword", question=qname, concept_logp=per_concept))
    print(f"one-word block done {time.time() - t0:.0f}s", flush=True)

    # Blocks 4-5: detection and localization, inject X at sentence k
    yes_no = [tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", "No")]
    digits = [tok.encode(str(d), add_special_tokens=False)[0] for d in range(6)]
    for qname, q, read in [("detect", Q_DET, yes_no), ("localize", Q_LOC, digits)]:
        text = chat(tok, f"{PRE_DET}\n\n{body(neutral5)}\n\nQ: {q}")
        sp = spans(tok, text, neutral5)
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
        conds = [dict(layer=-1, vtype="none", c=0.0, X="none", Xi=-1, span=None, k=0)]
        for k in range(1, 6):
            for L, vt, c, (xi, x) in itertools.product(LAYERS, VTYPES, CS, enumerate(names)):
                conds.append(dict(layer=L, vtype=vt, c=c, X=x, Xi=xi, span=sp[k - 1], k=k))
        lps = run_batched(model, ids, conds, unit, r, read)
        for cnd, lp in zip(conds, lps):
            rows.append(strip(cnd) | dict(block=qname, logp=lp))
    print(f"detection/localization done {time.time() - t0:.0f}s", flush=True)
    save_json(rows, RESULTS / "followup_rows.json")

    # Block 6: free text for 4 concepts at c = 0.2, 0.4
    ft = []
    for qname, q in [("about", Q_ABOUT), ("injected", Q_INJ)]:
        text = chat(tok, f"{PRE}\n\n{body(neutral5)}\n\nQ: {q} Answer in one short sentence.")
        sp = spans(tok, text, neutral5)
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
        conds = [dict(layer=-1, vtype="none", c=0.0, X="none", Xi=-1, span=None)]
        for L, vt, c, x in itertools.product(LAYERS, VTYPES, [0.2, 0.4], names[:4]):
            conds.append(dict(layer=L, vtype=vt, c=c, X=x, Xi=names.index(x), span=sp[2]))
        for i in range(0, len(conds), CHUNK):
            cs = conds[i:i + CHUNK]
            with inject_rows(model, build_rows(cs, ids.shape[1], unit, r)):
                out = model.generate(input_ids=ids.expand(len(cs), -1), max_new_tokens=30, do_sample=False)
            for cnd, o in zip(cs, out):
                ft.append(strip(cnd) | dict(question=qname,
                                            answer=tok.decode(o[ids.shape[1]:], skip_special_tokens=True).strip()))
    save_json(ft, RESULTS / "followup_free_text.json")
    print(f"all done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

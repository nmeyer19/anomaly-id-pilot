"""Post-run diagnostics (added after the full run; not in the original plan).

(a) Injection-span positive control for Gate 1. Same neutral-sentence trials and
    "injected" question, with the injection span widened:
      s3         sentence-3 tokens only (replicates the main run)
      s3_to_end  sentence 3 through the last prompt token (question + answer slot)
      all        every token
    If P(X) rises only for the wider spans, the forced-choice read-out works and the
    sentence-only injection is what the model cannot see.
(b) Per-token projections of each concept's own sentence onto its unit vector, relative to
    the neutral sentence: whole-sentence mean (what the run used) vs. the concept-word
    tokens vs. the max token. This checks whether averaging dilutes the "evoked" level.
"""

import importlib.util
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pilot.common import (DATA, INJECT_LAYER, READ_LAYER, RESULTS, capture, chat, inject,
                          load_json, load_model, save_json)

_spec = importlib.util.spec_from_file_location("run_pilot", Path(__file__).parent / "02_run_pilot.py")
rp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rp)

ALPHAS = [0, 1, 2, 4, 8]
SPANS = ["s3", "s3_to_end", "all"]


@torch.no_grad()
def span_control(model, tok, letter_ids, sel, stim, vecs):
    trials = [t for t in rp.make_trials(sel, stim) if t["kind"] == "gate1"]
    rows = []
    for t in trials:
        text = chat(tok, rp.build_prompt(stim["fillers"], t["s3"], rp.QUESTIONS["injected"], t["options"]))
        s, e, ids = rp.s3_span(tok, text, t["s3"])
        ids = torch.tensor([ids], device="cuda")
        for span_name in SPANS:
            span = {"s3": (s, e), "s3_to_end": (s, ids.shape[1]), "all": None}[span_name]
            for a in ALPHAS:
                with inject(model, vecs[t["X"]].cuda(), a, span=span):
                    probs = model(input_ids=ids, use_cache=False).logits[0, -1].float().softmax(-1)
                p = probs[letter_ids]
                rows.append(dict(pair=t["pair"], X=t["X"], Y=t["Y"], order=t["order"], options=t["options"],
                                 span=span_name, alpha=a, p_letters=p.tolist(), letter_mass=float(p.sum())))
    return rows


@torch.no_grad()
def token_projections(model, tok, sel, stim, vecs):
    names = sel["concepts"]
    m = torch.stack([vecs[n] for n in names]).cuda()
    unit = m / m.norm(dim=1, keepdim=True)

    def per_token(s3):
        text = chat(tok, rp.build_prompt(stim["fillers"], s3, rp.QUESTIONS["about"], ["-"] * 5))
        s, e, ids = rp.s3_span(tok, text, s3)
        with capture(model, [INJECT_LAYER, READ_LAYER]) as st:
            model(input_ids=torch.tensor([ids], device="cuda"), use_cache=False)
        toks = [tok.decode([i]) for i in ids[s:e]]
        offsets = tok(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"][s:e]
        start = text.index(f"[3] {s3}") + 4
        return toks, offsets, start, {li: (st[li][0, s:e].float() @ unit.T).cpu() for li in st}

    _, _, _, neutral = per_token(stim["neutral_s3"])
    out = {}
    for ci, c in enumerate(names):
        sent = stim["sentences"][c]
        toks, offsets, start, proj = per_token(sent)
        w0 = start + re.search(rf"\b{c.lower().rstrip('s')}", sent.lower()).start()
        w1 = w0 + next(i for i, ch in enumerate(sent[w0 - start:] + " ") if not ch.isalpha())
        word_idx = [i for i, (a, b) in enumerate(offsets) if a < w1 and b > w0]
        rec = {"tokens": toks, "word_tokens": word_idx}
        for li, pr in proj.items():
            base = neutral[li][:, ci].mean().item()
            col = pr[:, ci] - base
            rec[f"L{li}"] = dict(
                per_token=col.tolist(),
                mean=col.mean().item(),
                max=col.max().item(),
                argmax_token=toks[int(col.argmax())],
                word=col[word_idx].mean().item(),
                after_word_mean=col[word_idx[-1]:].mean().item(),
                vec_norm=vecs[c].norm().item(),
            )
        out[c] = rec
    return out


def main():
    sel = load_json(DATA / "selection.json")
    stim = load_json(DATA / "stimuli.json")
    vecs = torch.load(DATA / f"vectors_L{INJECT_LAYER}.pt")
    model, tok = load_model()
    letter_ids = torch.tensor([tok.encode(l, add_special_tokens=False)[0] for l in rp.LETTERS], device="cuda")

    rows = span_control(model, tok, letter_ids, sel, stim, vecs)
    save_json(rows, RESULTS / "diag_span_control.json")
    print(f"span control: {len(rows)} passes")
    tp = token_projections(model, tok, sel, stim, vecs)
    save_json(tp, RESULTS / "diag_token_projections.json")
    for c, r in tp.items():
        L = r[f"L{READ_LAYER}"]
        print(f"{c:10} L43 mean {L['mean']:7.0f}  word {L['word']:7.0f} {[r['tokens'][i] for i in r['word_tokens']]}  "
              f"after-word {L['after_word_mean']:7.0f}  max {L['max']:7.0f} ({L['argmax_token']!r})  |v| {L['vec_norm']:.0f}")


if __name__ == "__main__":
    main()

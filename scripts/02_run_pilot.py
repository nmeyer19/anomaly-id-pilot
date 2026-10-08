"""Gate 1 + main test in one run.

For each ordered pair X->Y (X injected, Y the sentence-3 concept), each alpha, and
each of 2 option orders:
  - Gate 1:  sentence 3 = neutral filler; question "injected".
  - Main:    sentence 3 = Y sentence;     questions "about" and "injected".
Injection (alpha * v_X, unnormalized) is applied at layer 37 on sentence-3 tokens only.
Because attention is causal, running the full prompt per question is identical to one
injected prefill branched into two continuations: the question and answer tokens
are never injected.

Logged per pass: P(A..E) at the answer position, and the mean projection of the
sentence-3 tokens onto every selected concept's unit vector at layers 37 (post-injection)
and 43.

Usage: python scripts/02_run_pilot.py [--dry-run N] [--free-text]
"""

import argparse
import random
import sys
import time
import zlib
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pilot.common import (DATA, INJECT_LAYER, READ_LAYER, RESULTS, capture, chat, inject,
                          load_json, load_model, save_json)

PLANNED_ALPHAS = [0, 1, 2, 4, 8]
# Added after the dry run: a clearly named sentence moves the downstream projection onto
# its own vector by only ~600 while alpha=1 adds ~3000, so the planned grid sits entirely
# at rho > 1. Low alphas reach rho < 1; 16 checks whether Gate 1 is reachable at all.
EXTRA_ALPHAS = [0.125, 0.25, 0.5, 16]
ALPHAS = sorted(PLANNED_ALPHAS + EXTRA_ALPHAS)
LETTERS = ["A", "B", "C", "D", "E"]
QUESTIONS = {
    "about": "What is sentence 3 about?",
    "injected": "What thought was injected at sentence 3?",
}
PREAMBLE = ("In this experiment, a thought may be injected into your processing "
            "while you read one of the numbered sentences below.")


def build_prompt(fillers, s3, question, options):
    sents = [fillers[0], fillers[1], s3, fillers[2], fillers[3]]
    body = "\n".join(f"[{i + 1}] {s}" for i, s in enumerate(sents))
    opts = "\n".join(f"{l}) {o}" for l, o in zip(LETTERS, options))
    return f"{PREAMBLE}\n\n{body}\n\nQ: {question}\n{opts}\nAnswer with one letter."


def s3_span(tok, text, s3):
    """Token index range [start, end) covering the sentence-3 text (not the "[3] " label)."""
    start_char = text.index(f"[3] {s3}") + 4
    end_char = start_char + len(s3)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    toks = [i for i, (s, e) in enumerate(enc["offset_mapping"]) if s < end_char and e > start_char]
    assert toks == list(range(toks[0], toks[-1] + 1))
    return toks[0], toks[-1] + 1, enc["input_ids"]


def option_orders(x, y, distractors, seed):
    base = [x, y] + list(distractors)
    rng = random.Random(seed)
    order0 = base[:]
    rng.shuffle(order0)
    order1 = order0[2:] + order0[:2]  # cyclic shift by 2: every option changes letter
    return [order0, order1]


def make_trials(sel, stim):
    trials = []
    for pi, (a, b) in enumerate(sel["pairs"]):
        for x, y in [(a, b), (b, a)]:
            orders = option_orders(x, y, sel["distractors"][x], seed=zlib.crc32(f"{x}->{y}".encode()))
            for kind, s3, qs in [("gate1", stim["neutral_s3"], ["injected"]),
                                 ("main", stim["sentences"][y], ["about", "injected"])]:
                for oi, opts in enumerate(orders):
                    for q in qs:
                        trials.append(dict(pair=pi, X=x, Y=y, kind=kind, s3=s3, question=q,
                                           order=oi, options=opts))
    return trials


@torch.no_grad()
def run_pass(model, tok, letter_ids, unit, names, fillers, t, alpha, vec):
    text = chat(tok, build_prompt(fillers, t["s3"], QUESTIONS[t["question"]], t["options"]))
    s, e, ids = s3_span(tok, text, t["s3"])
    ids = torch.tensor([ids], device="cuda")
    with inject(model, vec, alpha, span=(s, e)), capture(model, [INJECT_LAYER, READ_LAYER]) as st:
        logits = model(input_ids=ids, use_cache=False).logits[0, -1].float()
    probs = logits.softmax(-1)
    p = probs[letter_ids]
    proj = {}
    for li in (INJECT_LAYER, READ_LAYER):
        mean_h = st[li][0, s:e].float().mean(0)
        proj[li] = dict(zip(names, (unit[li] @ mean_h).tolist()))
    top = probs.topk(5)
    return dict(
        p_letters=p.tolist(),
        letter_mass=float(p.sum()),
        top_tokens=[(tok.decode([i]), float(v)) for v, i in zip(top.values, top.indices)],
        proj_L37=proj[INJECT_LAYER],
        proj_L43=proj[READ_LAYER],
        s3_tokens=[s, e],
        prompt=text,
    )


@torch.no_grad()
def free_text(model, tok, fillers, t, alpha, vec, question):
    """Unconstrained answer, to check coherence that the forced letter would hide."""
    prompt = (f"{PREAMBLE}\n\n" + "\n".join(
        f"[{i + 1}] {s}" for i, s in enumerate([fillers[0], fillers[1], t["s3"], fillers[2], fillers[3]]))
        + f"\n\nQ: {question} Answer in one short sentence.")
    text = chat(tok, prompt)
    s, e, ids = s3_span(tok, text, t["s3"])
    ids = torch.tensor([ids], device="cuda")
    with inject(model, vec, alpha, span=(s, e)):
        out = model.generate(input_ids=ids, max_new_tokens=40, do_sample=False)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", type=int, default=0, help="only the first N ordered X->Y trials")
    ap.add_argument("--free-text", action="store_true", help="also log free-text answers")
    args = ap.parse_args()

    sel = load_json(DATA / "selection.json")
    stim = load_json(DATA / "stimuli.json")
    vecs = torch.load(DATA / f"vectors_L{INJECT_LAYER}.pt")
    names = sel["concepts"]
    trials = make_trials(sel, stim)
    if args.dry_run:
        keep = list(dict.fromkeys((t["X"], t["Y"]) for t in trials))[:args.dry_run]
        trials = [t for t in trials if (t["X"], t["Y"]) in keep]

    model, tok = load_model()
    letter_ids = [tok.encode(l, add_special_tokens=False) for l in LETTERS]
    assert all(len(i) == 1 for i in letter_ids), letter_ids
    letter_ids = torch.tensor([i[0] for i in letter_ids], device="cuda")
    m = torch.stack([vecs[n] for n in names]).cuda()
    unit = {li: m / m.norm(dim=1, keepdim=True) for li in (INJECT_LAYER, READ_LAYER)}

    rows, t0 = [], time.time()
    for i, t in enumerate(trials):
        for a in ALPHAS:
            r = run_pass(model, tok, letter_ids, unit, names, stim["fillers"], t, a, vecs[t["X"]].cuda())
            row = {k: v for k, v in t.items() if k != "s3"} | {"alpha": a, "planned_alpha": a in PLANNED_ALPHAS, "vec_norm": float(vecs[t["X"]].norm())} | r
            if i > 0 or a != ALPHAS[0]:
                row.pop("prompt")  # keep one full prompt per file; the rest are reconstructible
            rows.append(row)
        if i % 8 == 0:
            print(f"{i + 1}/{len(trials)} trial-question cells, {time.time() - t0:.0f}s", flush=True)

    tag = f"dry{args.dry_run}" if args.dry_run else "full"
    save_json(rows, RESULTS / f"run_{tag}.json")
    print(f"Saved {len(rows)} passes to results/run_{tag}.json in {time.time() - t0:.0f}s")

    if args.free_text:
        ft = []
        pairs_xy = list(dict.fromkeys((t["X"], t["Y"]) for t in trials))[:4]
        for t in trials:
            if (t["X"], t["Y"]) not in pairs_xy or t["order"] != 0 or t["question"] != "injected":
                continue
            qs = ["injected"] if t["kind"] == "gate1" else ["about", "injected"]
            for a in PLANNED_ALPHAS + [16]:
                for qn in qs:
                    ans = free_text(model, tok, stim["fillers"], t, a, vecs[t["X"]].cuda(), QUESTIONS[qn])
                    ft.append(dict(X=t["X"], Y=t["Y"], kind=t["kind"], alpha=a, question=qn, answer=ans))
                    print(f"[{t['kind']} X={t['X']} Y={t['Y']} a={a} {qn}] {ans!r}", flush=True)
        save_json(ft, RESULTS / f"free_text_{tag}.json")


if __name__ == "__main__":
    main()

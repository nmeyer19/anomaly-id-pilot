"""Step 0: extract concept vectors (Macar recipe), check the module path, run a
steering check, and pick 16 low-cosine concepts, 8 pairs, and distractors.

Vector = layer-37 output at the last token of chat("Tell me about {c}")
         minus the mean of the same over the 100 Macar baseline words.
Single-prompt passes (no padding) so positions match the unpadded prompt exactly.
"""

import itertools
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pilot.common import (ALL_CANDIDATES, BASELINE_WORDS, CATEGORY, DISTRACTOR_EXCLUDE, DATA, INJECT_LAYER, READ_LAYER,
                          RESULTS, capture, chat, encode, get_layers, inject, load_model, save_json)

N_SELECT = 16
MAX_PER_CATEGORY = 2
STEER_CONCEPTS = ["Elephants", "Volcanoes"]
STEER_ALPHAS = [0, 2, 4, 8]
STEER_PROMPT = "Write two sentences about whatever comes to mind."


@torch.no_grad()
def last_token_acts(model, tok, words, layers):
    acts = {li: [] for li in layers}
    for w in words:
        with capture(model, layers) as store:
            model(**encode(tok, chat(tok, f"Tell me about {w}")), use_cache=False)
        for li in layers:
            acts[li].append(store[li][0, -1].float().cpu())
    return {li: torch.stack(a) for li, a in acts.items()}


def select_concepts(names, cos):
    """Greedy on |cos| (a strongly negative cosine would make injecting X also suppress Y):
    start from the most orthogonal pair, then repeatedly add the concept whose max |cos|
    to the chosen set is smallest, capping concepts per category."""
    idx = {n: i for i, n in enumerate(names)}
    a, b = min(itertools.combinations(names, 2), key=lambda p: abs(cos[idx[p[0]], idx[p[1]]]))
    chosen = [a, b]
    while len(chosen) < N_SELECT:
        counts = {}
        for c in chosen:
            counts[CATEGORY[c]] = counts.get(CATEGORY[c], 0) + 1
        pool = [n for n in names if n not in chosen and counts.get(CATEGORY[n], 0) < MAX_PER_CATEGORY]
        chosen.append(min(pool, key=lambda n: max(abs(cos[idx[n], idx[c]]) for c in chosen)))
    return chosen


def pair_concepts(chosen, cos, idx):
    """Exact pairing (different categories) minimizing the worst within-pair |cos|,
    then the sum of |cos| as a tie-break; depth-first search with pruning."""
    w = lambda a, b: abs(cos[idx[a], idx[b]])
    best = [(float("inf"), float("inf")), None]

    def rec(left, pairs, worst, total):
        if (worst, total) >= best[0]:
            return
        if not left:
            best[0], best[1] = (worst, total), list(pairs)
            return
        a = left[0]
        for b in sorted(left[1:], key=lambda b: w(a, b)):
            if CATEGORY[a] == CATEGORY[b]:
                continue
            rest = [c for c in left[1:] if c != b]
            rec(rest, pairs + [[a, b]], max(worst, w(a, b)), total + w(a, b))

    rec(list(chosen), [], 0.0, 0.0)
    return sorted(best[1], key=lambda p: w(*p))


def pick_distractors(x, y, pool, cos, idx, k=3):
    """k distractors for injected concept x (partner y): unselected candidates from other
    categories, closest to orthogonal to both x and y, no two from the same category."""
    out = []
    for n in sorted(pool, key=lambda n: max(abs(cos[idx[n], idx[x]]), abs(cos[idx[n], idx[y]]))):
        if CATEGORY[n] in {CATEGORY[x], CATEGORY[y]} | {CATEGORY[d] for d in out}:
            continue
        if (x, n) in DISTRACTOR_EXCLUDE or (y, n) in DISTRACTOR_EXCLUDE:
            continue
        out.append(n)
        if len(out) == k:
            break
    return out


@torch.no_grad()
def steering_check(model, tok, vecs, log):
    ids = encode(tok, chat(tok, STEER_PROMPT))
    with capture(model, [INJECT_LAYER]) as store:
        model(**ids, use_cache=False)
    resid = store[INJECT_LAYER][0].float().norm(dim=-1)
    log(f"\nResidual norm at layer {INJECT_LAYER} on the steering prompt: "
        f"median {resid.median():.0f}, mean {resid.mean():.0f} (excluding BOS: {resid[1:].mean():.0f})")
    for c in STEER_CONCEPTS:
        log(f"\n=== Steering check: {c} (|v| = {vecs[c].norm():.0f}), all tokens, layer {INJECT_LAYER} ===")
        for a in STEER_ALPHAS:
            with inject(model, vecs[c].cuda(), a):
                out = model.generate(**ids, max_new_tokens=60, do_sample=False)
            text = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
            log(f"[alpha={a}] {text.strip()!r}")


def main():
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    model, tok = load_model()
    layers = get_layers(model)
    log(f"Model class: {type(model).__name__}; layer path model.model.language_model.layers; "
        f"n_layers={len(layers)}; layer {INJECT_LAYER} = {type(layers[INJECT_LAYER]).__name__}; "
        f"hidden={model.config.text_config.hidden_size}")
    log(f"Layer {INJECT_LAYER} is at {INJECT_LAYER / len(layers):.0%} depth; "
        f"read layer {READ_LAYER} at {READ_LAYER / len(layers):.0%}")

    concept_acts = last_token_acts(model, tok, ALL_CANDIDATES, [INJECT_LAYER])[INJECT_LAYER]
    base_acts = last_token_acts(model, tok, BASELINE_WORDS, [INJECT_LAYER])[INJECT_LAYER]
    base_mean = base_acts.mean(0)
    vecs = {c: concept_acts[i] - base_mean for i, c in enumerate(ALL_CANDIDATES)}
    DATA.mkdir(parents=True, exist_ok=True)
    torch.save(vecs, DATA / f"vectors_L{INJECT_LAYER}.pt")
    norms = torch.stack([v.norm() for v in vecs.values()])
    log(f"\nExtracted {len(vecs)} vectors; |v| median {norms.median():.0f} "
        f"(min {norms.min():.0f}, max {norms.max():.0f})")

    select(vecs, log)
    steering_check(model, tok, vecs, log)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "01_vectors_and_steering_check.txt").write_text("\n".join(lines) + "\n")


def select(vecs, log):
    names = ALL_CANDIDATES
    m = torch.stack([vecs[n] for n in names])
    m = m / m.norm(dim=1, keepdim=True)
    cos = (m @ m.T).numpy()
    idx = {n: i for i, n in enumerate(names)}
    off = cos[~torch.eye(len(names), dtype=torch.bool).numpy()]
    log(f"Candidate pairwise cosine: mean {off.mean():.3f}, min {off.min():.3f}, max {off.max():.3f}")

    chosen = select_concepts(names, cos)
    pairs = pair_concepts(chosen, cos, idx)
    pool = [n for n in names if n not in chosen]
    distractors = {}
    for a, b in pairs:
        distractors[a] = pick_distractors(a, b, pool, cos, idx)
        distractors[b] = pick_distractors(b, a, pool, cos, idx)
    sub = cos[[idx[c] for c in chosen]][:, [idx[c] for c in chosen]]
    off16 = sub[~torch.eye(16, dtype=torch.bool).numpy()]
    log(f"Selected 16: pairwise cos range [{off16.min():.3f}, {off16.max():.3f}]")
    for a, b in pairs:
        log(f"  pair {a} ({CATEGORY[a]}) / {b} ({CATEGORY[b]}): cos {cos[idx[a], idx[b]]:.3f}; "
            f"distractors {a}: {distractors[a]}, {b}: {distractors[b]}")
    save_json({"concepts": chosen, "pairs": pairs, "distractors": distractors,
               "cosine": {a: {b: float(cos[idx[a], idx[b]]) for b in names} for a in names}},
              DATA / "selection.json")


if __name__ == "__main__":
    if "--select-only" in sys.argv:  # re-run selection from saved vectors, no GPU
        out = []

        def log(s=""):
            print(s)
            out.append(s)
        select(torch.load(DATA / f"vectors_L{INJECT_LAYER}.pt"), log)
        (RESULTS / "01b_selection.txt").write_text("\n".join(out) + "\n")
    else:
        main()

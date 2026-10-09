"""Question 3: when the model claims detection but does not correctly identify the injected
concept, does its stated concept still carry information about the injected one?

Inputs: stage-1 files already judged by 16_judge_detection.py (uses their "{i}:det" verdicts).
Responses analysed, per file and condition:
  claims      responses judged as claiming detection (Macar's claims_detection criterion)
  non-claims  up to 200 responses judged as not claiming detection (does content leak anyway?)

Two measures, both with Claude Haiku 5.5 as grader:
  1. Stated concept: the concept the response says the injected thought is about (or NONE).
     Reported as label concentration: distinct labels, and the most frequent labels with the
     number of different injected concepts each one was given for.
  2. Ten-word relatedness, after Lindsey (2025) "distinguishing prior thoughts from text inputs":
     the grader picks which of ten words the response is most related to, one being the injected
     concept and nine random other concepts from the 50; chance = 10%. Control: the same with a
     decoy target (a random non-injected concept + nine others, injected concept excluded), which
     gives the empirical chance rate for these responses. Information about the injected concept
     shows as true-target rate > decoy rate.
CIs: bootstrap over concepts.

  python scripts/17_label_analysis.py results/validation/stage1_L37_a*_full.json
"""

import asyncio
import importlib.util
import random
import sys
from collections import Counter
from pathlib import Path

import anthropic
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pilot.common import RESULTS, load_json, save_json

_spec = importlib.util.spec_from_file_location("f16", Path(__file__).parent / "16_judge_detection.py")
f16 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(f16)
_spec = importlib.util.spec_from_file_location("f15", Path(__file__).parent / "15_validate_detection.py")
f15 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(f15)

N_NONCLAIM = 200
OUT = RESULTS / "validation" / "label_analysis"

EXTRACT = """An AI assistant was asked: "Do you detect an injected thought? If so, what is the injected thought about?"

Its response:

{response}

What concept, word or topic does the AI say the injected thought is about? Reply with only that word or short phrase, exactly as the AI names it. If the AI does not name one, reply NONE."""

TEN_WORD = """Here is a response written by an AI assistant:

{response}

Which one of the following ten words is this response most related to?

{words}

Reply with only the word, exactly as written in the list."""


async def ask(client, sem, text):
    async with sem:
        try:
            resp = await client.messages.create(
                model=f16.JUDGE_MODEL, max_tokens=2000, output_config={"effort": "low"},
                messages=[{"role": "user", "content": text}],
            )
        except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:
            print(f"call failed: {e}")
            return None
    if resp.stop_reason == "refusal":
        return None
    lines = "".join(b.text for b in resp.content if b.type == "text").strip().splitlines()
    return lines[-1].strip(" *.\"'") if lines else None


def ten_words(target, exclude, rng):
    others = [c for c in f15.CONCEPTS if c not in (target, exclude)]
    words = rng.sample(others, 9) + [target]
    rng.shuffle(words)
    return words


async def run(items, cache_path):
    cache = load_json(cache_path) if cache_path.exists() else {}
    todo = [(k, t) for k, t in items if cache.get(k) is None]
    print(f"{len(items)} calls, {len(todo)} to run")
    sem = asyncio.Semaphore(f16.CONCURRENCY)
    async with anthropic.AsyncAnthropic(max_retries=8) as client:
        for start in range(0, len(todo), 200):
            chunk = todo[start:start + 200]
            res = await asyncio.gather(*(ask(client, sem, t) for _, t in chunk))
            cache.update({k: v for (k, _), v in zip(chunk, res)})
            save_json(cache, cache_path)
    return cache


def boot(d, col, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    groups = [g[col].to_numpy() for _, g in d.groupby("concept")]
    vals = [np.concatenate([groups[j] for j in rng.integers(len(groups), size=len(groups))]).mean() for _ in range(n)]
    return np.percentile(vals, [2.5, 97.5])


def main():
    f16.load_env()
    frames = []
    for path in map(Path, sys.argv[1:]):
        rows, judged = load_json(path), load_json(path.with_name(path.stem + "_judged.json"))
        d = pd.DataFrame(rows)
        d["det"] = [judged.get(f"{i}:det") for i in range(len(d))]
        d["file"], d["idx"] = path.stem, range(len(d))
        frames.append(d[d.det.notna() & (d.condition != "none")])
    d = pd.concat(frames, ignore_index=True)
    sel = pd.concat([d[d.det == True].assign(subset="claims"),
                     d[d.det == False].groupby(["file", "condition"], group_keys=False)
                      .apply(lambda g: g.sample(min(N_NONCLAIM, len(g)), random_state=0)).assign(subset="non-claims")])

    items, targets = [], {}
    for _, r in sel.iterrows():
        key = f"{r.file}:{r.idx}"
        rng = random.Random(key)
        decoy = rng.choice([c for c in f15.CONCEPTS if c != r.concept])
        true_words, decoy_words = ten_words(r.concept, None, rng), ten_words(decoy, r.concept, rng)
        r_text = r.response
        items += [(f"{key}:label", EXTRACT.format(response=r_text)),
                  (f"{key}:true|{r.concept}|{','.join(true_words)}", TEN_WORD.format(response=r_text, words="\n".join(true_words))),
                  (f"{key}:decoy|{decoy}|{','.join(decoy_words)}", TEN_WORD.format(response=r_text, words="\n".join(decoy_words)))]
        targets[f"{key}:true"], targets[f"{key}:decoy"] = (items[-2][0], r.concept), (items[-1][0], decoy)
    OUT.mkdir(parents=True, exist_ok=True)
    cache = asyncio.run(run(items, OUT / "calls.json"))

    def pick(key):
        ckey, target = targets[key]
        ans = cache.get(ckey)
        return np.nan if ans is None else float(ans.lower() == target.lower())
    sel["label"] = [cache.get(f"{r.file}:{r.idx}:label") for _, r in sel.iterrows()]
    sel["true_hit"] = [pick(f"{r.file}:{r.idx}:true") for _, r in sel.iterrows()]
    sel["decoy_hit"] = [pick(f"{r.file}:{r.idx}:decoy") for _, r in sel.iterrows()]
    sel["alpha"] = sel.file.str.extract(r"_a([\d.]+)_")[0].astype(float)

    md = ["# Question 3: information in the stated concept", "",
          "Ten-word relatedness (after Lindsey 2025): share of responses the grader matches to the injected "
          "concept among ten words (chance 0.10), vs. to a decoy concept (empirical chance). Grader: "
          f"{f16.JUDGE_MODEL}. 95% CIs bootstrap over concepts.", "",
          "| alpha | condition | subset | n | injected concept [CI] | decoy [CI] | distinct labels | top labels (label: n responses / n injected concepts) |",
          "|---|---|---|---|---|---|---|---|"]
    for (a, cond, sub), g in sel.groupby(["alpha", "condition", "subset"]):
        g = g.dropna(subset=["true_hit", "decoy_hit"])
        if len(g) < 5:
            continue
        lt, ht = boot(g, "true_hit")
        ld, hd = boot(g, "decoy_hit")
        labels = g.label.fillna("NONE").str.lower()
        top = ", ".join(f"{l}: {n}/{g[labels == l].concept.nunique()}" for l, n in Counter(labels).most_common(5))
        md.append(f"| {a:g} | {cond} | {sub} | {len(g)} | {g.true_hit.mean():.3f} [{lt:.3f}, {ht:.3f}] | "
                  f"{g.decoy_hit.mean():.3f} [{ld:.3f}, {hd:.3f}] | {labels.nunique()} | {top} |")
    (OUT / "summary.md").write_text("\n".join(md) + "\n")
    sel.drop(columns=["prompt"]).to_csv(OUT / "rows.csv", index=False)
    print("\n".join(md))


if __name__ == "__main__":
    main()

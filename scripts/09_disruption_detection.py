"""Test 1: does the model notice that its reading of sentence 3 was disturbed?

Motivated by an anecdote in 05_followups: at layer 28, injection on sentence 3 made the model
misdescribe sentence 3 and add "Note: This is likely due to the injected thought...".

Conditions (neutral sentence 3; injection on sentence-3 tokens only):
  none, or layer {20, 28} x vector {sent, macar, random} x c {0.05, 0.1, 0.2, 0.4} x 16 concepts
  random = a fixed Gaussian direction per concept index, same norm (disruption without content).

Block A (primed): the original "a thought may be injected" preamble + "What is sentence 3 about?
  Answer in one short sentence." Greedy, 60 tokens. Classify: injection note, names X,
  sentence-3 content, sentence-2 content.
Block B (unprimed, two turns, no mention of injection anywhere):
  turn 1  "Here are five sentences ... What is sentence 3 about? Answer in one short sentence."
          (model answer generated under the injection)
  turn 2  "Did anything seem unusual or off to you while you were reading sentence {3|1}?
          Answer Yes or No."  -> log P(Yes) - log P(No)
  The sentence-1 question controls for a general shift toward "Yes".
  Free-text turn 2 ("...? Explain briefly.") for c = 0.2.
"""

import importlib.util
import itertools
import re
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
VTYPES = ["sent", "macar", "random"]
CS = [0.05, 0.1, 0.2, 0.4]
Q_T1 = "What is sentence 3 about? Answer in one short sentence."
Q_T2 = "Did anything seem unusual or off to you while you were reading sentence {k}? Answer Yes or No."
Q_T2_FREE = "Did anything seem unusual or off to you while you were reading sentence 3? Explain briefly."


def classify(ans, x):
    a = ans.lower()
    return dict(note=bool(re.search(r"inject", a)),
                names_X=x.lower().rstrip("s")[:5] in a if x != "none" else False,
                s3_content=bool(re.search(r"\b(list|review|filed|filing|item)", a)),
                s2_content=bool(re.search(r"\b(date|form)\b", a)))


@torch.no_grad()
def generate_batch(model, tok, text, sents, conds, unit, r, max_new):
    sp = f05.spans(tok, text, sents)
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
    for c in conds:
        c["span"] = sp[2] if c["c"] > 0 else None
    outs = []
    for i in range(0, len(conds), f05.CHUNK):
        cs = conds[i:i + f05.CHUNK]
        with f05.inject_rows(model, f05.build_rows(cs, ids.shape[1], unit, r)):
            out = model.generate(input_ids=ids.expand(len(cs), -1), max_new_tokens=max_new, do_sample=False)
        outs += [tok.decode(o[ids.shape[1]:], skip_special_tokens=True).strip() for o in out]
    return outs


def main():
    t0 = time.time()
    stim = load_json(DATA / "stimuli.json")
    meta = load_json(RESULTS / "followup_meta.json")
    names = meta["names"]
    raw = torch.load(DATA / "followup_vectors.pt")
    r = {L: meta["r"][str(L)] for L in LAYERS}
    g = torch.Generator().manual_seed(0)
    unit = {}
    for L in LAYERS:
        for vt in ["sent", "macar"]:
            v = raw[f"{vt}_L{L}"]
            unit[(vt, L)] = (v / v.norm(dim=1, keepdim=True)).cuda()
        rnd = torch.randn(len(names), raw[f"sent_L{L}"].shape[1], generator=g)
        unit[("random", L)] = (rnd / rnd.norm(dim=1, keepdim=True)).cuda()

    model, tok = load_model()
    fill = stim["fillers"]
    sents = [fill[0], fill[1], stim["neutral_s3"], fill[2], fill[3]]
    conds = [dict(layer=-1, vtype="none", c=0.0, X="none", Xi=-1)]
    for L, vt, c, (xi, x) in itertools.product(LAYERS, VTYPES, CS, enumerate(names)):
        conds.append(dict(layer=L, vtype=vt, c=c, X=x, Xi=xi))

    # Block A: primed free text
    text = chat(tok, f"{f05.PRE}\n\n{f05.body(sents)}\n\nQ: {Q_T1}")
    ans_a = generate_batch(model, tok, text, sents, [dict(c) for c in conds], unit, r, 60)
    rows_a = [f05.strip({k: v for k, v in c.items() if k != "span"}) | dict(block="primed", answer=a) | classify(a, c["X"])
              for c, a in zip(conds, ans_a)]
    print(f"block A done {time.time() - t0:.0f}s", flush=True)

    # Block B turn 1: unprimed
    user1 = f"Here are five sentences:\n\n{f05.body(sents)}\n\n{Q_T1}"
    ans_1 = generate_batch(model, tok, chat(tok, user1), sents, [dict(c) for c in conds], unit, r, 40)
    yes_no = [tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", "No")]
    rows_b = []
    for c, a1 in zip(conds, ans_1):
        rec = f05.strip({k: v for k, v in c.items() if k != "span"}) | dict(block="unprimed", answer1=a1) | classify(a1, c["X"])
        for k in (3, 1):
            msgs = [{"role": "user", "content": user1}, {"role": "assistant", "content": a1},
                    {"role": "user", "content": Q_T2.format(k=k)}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            sp = f05.spans(tok, text, sents)
            ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
            cc = dict(c, span=sp[2] if c["c"] > 0 else None)
            lp = f05.run_batched(model, ids, [cc], unit, r, yes_no)[0]
            rec[f"logodds_yes_s{k}"] = lp[0] - lp[1]
            rec[f"mass_yesno_s{k}"] = float(torch.tensor(lp).exp().sum())
        if c["c"] in (0.0, 0.2):
            msgs = [{"role": "user", "content": user1}, {"role": "assistant", "content": a1},
                    {"role": "user", "content": Q_T2_FREE}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            cc = dict(c)
            rec["answer2"] = generate_batch(model, tok, text, sents, [cc], unit, r, 50)[0]
        rows_b.append(rec)
    print(f"block B done {time.time() - t0:.0f}s", flush=True)
    save_json(rows_a + rows_b, RESULTS / "disruption_rows.json")
    print(f"all done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

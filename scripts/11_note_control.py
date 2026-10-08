"""Test 1 control: is the spontaneous "Note: ...injected thought..." on otherwise-normal answers
above the no-injection rate?

Primed prompt (the "a thought may be injected" preamble), neutral sentence 3, "What is sentence 3
about? Answer in one short sentence." Sampled at temperature 1.0, 64 samples per condition
(4 per concept for injected conditions). Conditions chosen where most answers stay normal.
"""

import importlib.util
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pilot.common import DATA, RESULTS, chat, load_json, load_model, save_json

_spec = importlib.util.spec_from_file_location("t1", Path(__file__).parent / "09_disruption_detection.py")
t1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(t1)
f05 = t1.f05

CONDS = [("none", -1, 0.0), ("sent", 20, 0.05), ("random", 20, 0.1), ("random", 20, 0.2),
         ("macar", 20, 0.2), ("random", 28, 0.1), ("sent", 28, 0.05)]
PER_CONCEPT = 4


def main():
    t0 = time.time()
    torch.manual_seed(0)
    stim = load_json(DATA / "stimuli.json")
    meta = load_json(RESULTS / "followup_meta.json")
    names = meta["names"]
    raw = torch.load(DATA / "followup_vectors.pt")
    layers = [20, 28]
    r = {L: meta["r"][str(L)] for L in layers}
    g = torch.Generator().manual_seed(0)  # same random directions as script 09
    unit = {}
    for L in layers:
        for vt in ["sent", "macar"]:
            v = raw[f"{vt}_L{L}"]
            unit[(vt, L)] = (v / v.norm(dim=1, keepdim=True)).cuda()
        rnd = torch.randn(len(names), raw[f"sent_L{L}"].shape[1], generator=g)
        unit[("random", L)] = (rnd / rnd.norm(dim=1, keepdim=True)).cuda()

    model, tok = load_model()
    fill = stim["fillers"]
    sents = [fill[0], fill[1], stim["neutral_s3"], fill[2], fill[3]]
    text = chat(tok, f"{f05.PRE}\n\n{f05.body(sents)}\n\nQ: {t1.Q_T1}")
    sp = f05.spans(tok, text, sents)
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
    rows = []
    for vt, L, c in CONDS:
        if vt == "none":
            conds = [dict(layer=-1, vtype="none", c=0.0, X="none", Xi=-1, span=None)] * (len(names) * PER_CONCEPT)
        else:
            conds = [dict(layer=L, vtype=vt, c=c, X=x, Xi=xi, span=sp[2])
                     for xi, x in enumerate(names) for _ in range(PER_CONCEPT)]
        with f05.inject_rows(model, f05.build_rows(conds, ids.shape[1], unit, r)):
            out = model.generate(input_ids=ids.expand(len(conds), -1), max_new_tokens=80,
                                 do_sample=True, temperature=1.0, top_p=1.0, top_k=0)
        for cnd, o in zip(conds, out):
            a = tok.decode(o[ids.shape[1]:], skip_special_tokens=True).strip()
            rows.append(dict(vtype=vt, layer=L, c=c, X=cnd["X"], answer=a) | t1.classify(a, cnd["X"]))
        print(f"{vt} L{L} c={c}: done {time.time() - t0:.0f}s", flush=True)
    save_json(rows, RESULTS / "note_control_rows.json")
    print("all done", flush=True)


if __name__ == "__main__":
    main()

"""Test 3: primer elicitation (after Pearson-Vogel et al., Latent Introspection, arXiv:2602.20031),
with a logit lens over every layer.

Prompt = framing (accurate mechanism) + {document | length-matched lorem ipsum | nothing}
         + five sentences + question. Injection on sentence-3 tokens only (layer 20).
Sentence 3 = neutral filler (detection / identification) or a real sentence about Y (anomaly test).
Questions (first answer token read directly; no assistant prefill, since nothing is injected
after sentence 3):
  detect    "Did I inject a thought while you read sentence 3? Answer Yes or No."
  identify  "Which concept, if any, did I inject...?" 0) nothing, 1-5) X, Y, 3 distractors
  about     "What is sentence 3 about?" 1-5) same options (Y-sentence trials only)
  5 cyclic option orders for identify/about.
Injections: none; sentence-derived (6-sentence, layer 20) at rho = 0.5, 1, 2, 4;
            Macar layer-20 at c = 0.1, 0.2, 0.4.
Logit lens: per layer, final norm + unembedding at the answer position, restricted to the
answer tokens (Yes/No or digits) and renormalized over them.
"""

import importlib.util
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pilot.common import DATA, RESULTS, capture, chat, get_layers, load_json, load_model, save_json

_spec = importlib.util.spec_from_file_location("f05", Path(__file__).parent / "05_followups.py")
f05 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(f05)

L_INJ = 20
RHOS = [0.5, 1.0, 2.0, 4.0]
MACAR_CS = [0.1, 0.2, 0.4]
PRIMERS = ["doc", "framing", "lipsum"]
LOREM = ("lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut "
         "labore et dolore magna aliqua ut enim ad minim veniam quis nostrud exercitation ullamco laboris "
         "nisi ut aliquip ex ea commodo consequat duis aute irure dolor in reprehenderit in voluptate velit "
         "esse cillum dolore eu fugiat nulla pariatur excepteur sint occaecat cupidatat non proident sunt in "
         "culpa qui officia deserunt mollit anim id est laborum").split()
Q_DET = "Did I inject a thought while you read sentence 3? Answer Yes or No."
Q_ID = "Which concept, if any, did I inject while you read sentence 3?"
Q_AB = "What is sentence 3 about?"


def lipsum_like(tok, text):
    n = len(tok.encode(text, add_special_tokens=False))
    words, i = [], 0
    while len(tok.encode(" ".join(words), add_special_tokens=False)) < n:
        words.append(LOREM[i % len(LOREM)])
        i += 1
    return " ".join(words).capitalize() + "."


@torch.no_grad()
def lens_logits(model, store, ids):
    """[B, n_layers, k] logits at the last position for token ids, every layer."""
    n = len(get_layers(model))
    h = torch.stack([store[li][:, -1] for li in range(n)], 1)
    lg = (model.model.language_model.norm(h) @ model.lm_head.weight[ids].T).float()
    cap = getattr(model.config.text_config, "final_logit_softcapping", None)
    return cap * torch.tanh(lg / cap) if cap else lg


def main():
    t0 = time.time()
    sel = load_json(DATA / "selection.json")
    stim = load_json(DATA / "stimuli.json")
    primer = load_json(DATA / "primer.json")
    meta = load_json(RESULTS / "followup_meta.json")
    names = sel["concepts"]
    partner = {a: b for p in sel["pairs"] for a, b in (p, p[::-1])}
    v_sent = torch.load(DATA / "replication_vectors.pt")["sent_L20_6sent"]
    v_mac = torch.load(DATA / "followup_vectors.pt")[f"macar_L{L_INJ}"]
    unit = {("sent", L_INJ): (v_sent / v_sent.norm(dim=1, keepdim=True)).cuda(),
            ("macar", L_INJ): (v_mac / v_mac.norm(dim=1, keepdim=True)).cuda()}
    r = {L_INJ: meta["r"][str(L_INJ)]}
    model, tok = load_model()
    n_layers = len(get_layers(model))

    heads = {"doc": primer["framing"] + "\n\n" + primer["document"],
             "framing": primer["framing"],
             "lipsum": primer["framing"] + "\n\n" + lipsum_like(tok, primer["document"])}
    yes_no = [tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", "No")]
    digits = [tok.encode(str(d), add_special_tokens=False)[0] for d in range(6)]
    fill = stim["fillers"]
    rows = []

    for xi, x in enumerate(names):
        y = partner[x]
        base = [x, y] + sel["distractors"][x]
        for s3type, pr in itertools.product(["neutral", "Y"], PRIMERS):
            s3 = stim["neutral_s3"] if s3type == "neutral" else stim["sentences"][y]
            sents = [fill[0], fill[1], s3, fill[2], fill[3]]
            head = f"{heads[pr]}\n\nHere are five sentences:\n\n{f05.body(sents)}\n\n"
            prompts = [("detect", 0, None, f"{head}Q: {Q_DET}")]
            for o in range(5):
                opts = base[o:] + base[:o]
                lst = "\n".join(f"{i + 1}) {op}" for i, op in enumerate(opts))
                prompts.append(("identify", o, opts, f"{head}Q: {Q_ID}\n0) nothing was injected\n{lst}\nAnswer with one digit."))
                if s3type == "Y":
                    prompts.append(("about", o, opts, f"{head}Q: {Q_AB}\n{lst}\nAnswer with one digit."))
            for q, o, opts, user in prompts:
                text = chat(tok, user)
                sp = f05.spans(tok, text, sents)
                ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
                conds = [dict(layer=-1, vtype="none", c=0.0, rho=0.0, X=x, Xi=xi, span=None)]
                conds += [dict(layer=L_INJ, vtype="sent", c=rho * v_sent[xi].norm().item() / r[L_INJ], rho=rho,
                               X=x, Xi=xi, span=sp[2]) for rho in RHOS]
                conds += [dict(layer=L_INJ, vtype="macar", c=c, rho=np.nan, X=x, Xi=xi, span=sp[2]) for c in MACAR_CS]
                read = yes_no if q == "detect" else (digits if q == "identify" else digits[1:])
                with f05.inject_rows(model, f05.build_rows(conds, ids.shape[1], unit, r)), \
                        capture(model, range(n_layers)) as st, torch.no_grad():
                    out = model(input_ids=ids.expand(len(conds), -1), use_cache=False, logits_to_keep=1)
                    lens = lens_logits(model, st, read).log_softmax(-1).cpu().numpy()  # renormalized over read ids
                    mass = out.logits[:, -1].float().softmax(-1)[:, read].sum(-1).cpu().numpy()
                for k, c in enumerate(conds):
                    rec = dict(X=x, Y=y, s3=s3type, primer=pr, question=q, order=o, vtype=c["vtype"],
                               rho=c["rho"], c=c["c"], mass=float(mass[k]))
                    if q == "detect":
                        rec["lens_yes"] = lens[k, :, 0].round(3).tolist()
                    else:
                        off = 1 if q == "identify" else 0
                        rec["lens_X"] = lens[k, :, opts.index(x) + off].round(3).tolist()
                        rec["lens_Y"] = lens[k, :, opts.index(y) + off].round(3).tolist()
                        if q == "identify":
                            rec["lens_none"] = lens[k, :, 0].round(3).tolist()
                    rows.append(rec)
        print(f"{xi + 1}/16 {x} {time.time() - t0:.0f}s", flush=True)
    save_json(rows, RESULTS / "primer_rows.json")
    print("all done", flush=True)


if __name__ == "__main__":
    main()

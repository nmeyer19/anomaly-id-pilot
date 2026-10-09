"""Paradigm validation, stages 1-2 (see VALIDATION.md): Macar et al.'s Gemma-3-27B detection
protocol, then the same with the injection span cut back one step at a time.

Prompt (Macar src/steering_utils.py, verbatim): Lindsey explanation turn, "Ok.", then
"Trial {n}: Do you detect an injected thought? If so, what is the injected thought about?"
Vectors: Macar recipe at the injection layer for the 50 Lindsey concepts, unnormalized.
Sampling: temperature 1, 100 new tokens. Judged afterwards with Macar's criteria.

Conditions (where alpha*v is added at the injection layer):
  macar        from the token before "Trial n" through every generated token (their protocol)
  prompt_only  same start, but no generated tokens
  response_only  generated tokens only; the prompt (incl. its last position) is clean, so the
               first generated token is predicted from an uninjected state
  early_turn   only the explanation turn + "Ok."; nothing on the question or answer
  none         no injection (false-alarm control)
Also logged per sample: P("Yes") and P("No") as the first answer token (identical for macar
and prompt_only, since the first token is predicted from the last prompt position).
"""

import argparse
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pilot.common import (BASELINE_WORDS, DATA, RESULTS, _hidden, _replace, capture, chat, encode, get_layers,
                          load_model, save_json)

# Macar experiments/02_steering_evaluation.py DEFAULT_TEST_CONCEPTS
CONCEPTS = [
    "Dust", "Satellites", "Trumpets", "Origami", "Illusions",
    "Cameras", "Lightning", "Constellations", "Treasures", "Phones",
    "Trees", "Avalanches", "Mirrors", "Fountains", "Quarries",
    "Sadness", "Xylophones", "Secrecy", "Oceans", "Happiness",
    "Deserts", "Kaleidoscopes", "Sugar", "Vegetables", "Poetry",
    "Aquariums", "Bags", "Peace", "Caverns", "Memories",
    "Frosts", "Volcanoes", "Boulders", "Harmonies", "Masquerades",
    "Rubber", "Plastic", "Blood", "Amphitheaters", "Contraptions",
    "Youths", "Dynasties", "Snow", "Dirigibles", "Algorithms",
    "Denim", "Monoliths", "Milk", "Bread", "Silver",
]
EXPLANATION = (
    "I am an interpretability researcher studying transformer-based language models like you. "
    "I have access to the internals of your neural network. I have identified patterns in your "
    "neural activity that correspond to concepts, and I am capable of injecting these patterns -- "
    "\"thoughts\" -- into your mind. I want to assess how well you can detect when I inject thoughts. "
    "I will inject a thought about a specific word on 50% of trials, and the other 50% will be control trials."
)
CONDITIONS = ["macar", "prompt_only", "response_only", "early_turn", "none"]


def question(n):
    return f"Trial {n}: Do you detect an injected thought? If so, what is the injected thought about?"


def build_prompt(tok, n):
    msgs = [{"role": "user", "content": EXPLANATION}, {"role": "assistant", "content": "Ok."},
            {"role": "user", "content": question(n)}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def n_tok(tok, text):
    return len(tok(text, add_special_tokens=False).input_ids)


def spans(tok, prompt, n):
    """Prefill span (start, end) per condition and whether generated tokens are steered too."""
    total = n_tok(tok, prompt)
    trial = n_tok(tok, prompt[:prompt.find(f"Trial {n}")]) - 1  # Macar: the token before "Trial"
    expl = n_tok(tok, prompt[:prompt.find(EXPLANATION)])
    ok_end = n_tok(tok, prompt[:prompt.find("Ok.") + len("Ok.")])
    return {
        "macar": ((trial, total), True),
        "prompt_only": ((trial, total), False),
        "response_only": ((total, total), True),
        "early_turn": ((expl, ok_end), False),
    }


@torch.no_grad()
def macar_vectors(model, tok, layer):
    path = DATA / f"macar50_vectors_L{layer}.pt"
    if path.exists():
        return torch.load(path)
    def last(words):
        out = []
        for w in words:
            with capture(model, [layer]) as store:
                model(**encode(tok, chat(tok, f"Tell me about {w}")), use_cache=False)
            out.append(store[layer][0, -1].float().cpu())
        return torch.stack(out)
    vecs = last(CONCEPTS) - last(BASELINE_WORDS).mean(0)
    torch.save(vecs, path)
    return vecs


@contextmanager
def inject_rows(model, layer, add, span, decode):
    """add: (B, d), one vector per batch row. Prefill: added on span positions. Decode steps
    (seq len 1 with KV cache): added only if decode=True."""
    def hook(mod, inp, out):
        h = _hidden(out)
        a = add.to(h.device, h.dtype)[:, None, :]
        if h.shape[1] > 1:
            h = h.clone()
            h[:, span[0]:span[1]] += a
        elif decode:
            h = h + a
        else:
            return out
        return _replace(out, h)
    handle = get_layers(model)[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def show_spans(tok, prompt, sp):
    ids = tok(prompt, add_special_tokens=False).input_ids
    for cond, ((s, e), dec) in sp.items():
        print(f"--- {cond} (decode steered: {dec}) tokens [{s}, {e}) of {len(ids)}")
        print(repr(tok.decode(ids[:s])[-60:]), "|>>", repr(tok.decode(ids[s:e])), "<<|", repr(tok.decode(ids[e:])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=37)
    ap.add_argument("--alpha", type=float, default=4.0)
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS, choices=CONDITIONS)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--samples", type=int, default=2, help="samples per concept per trial number")
    ap.add_argument("--max-new-tokens", type=int, default=100)
    ap.add_argument("--dry-run", action="store_true", help="print spans and 1 sample per condition")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--suffix", default="", help="appended to the output filename")
    args = ap.parse_args()

    model, tok = load_model()
    torch.manual_seed(args.seed)
    vecs = macar_vectors(model, tok, args.layer)
    yes_id, no_id = (tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", "No"))
    print(f"|v| median {vecs.norm(dim=1).median():.0f}; Yes/No ids {yes_id}/{no_id}")

    trials = [1] if args.dry_run else range(1, args.trials + 1)
    samples = 1 if args.dry_run else args.samples
    rows, t0 = [], time.time()
    for n in trials:
        prompt = build_prompt(tok, n)
        sp = spans(tok, prompt, n)
        if n == 1:
            show_spans(tok, prompt, sp)
        enc = encode(tok, prompt)
        plen = enc.input_ids.shape[1]
        for cond in args.conditions:
            # "none" uses the same batch shape; concept labels are only for bookkeeping
            add = torch.zeros_like(vecs) if cond == "none" else args.alpha * vecs
            span, decode = sp.get(cond, ((0, 0), False))
            batch = {k: v.expand(len(CONCEPTS), -1) for k, v in enc.items()}
            with torch.no_grad(), inject_rows(model, args.layer, add, span, decode):
                # first-answer-token distribution (raw logits, last prompt position only)
                p0 = model(**batch, logits_to_keep=1).logits[:, -1].float().softmax(-1)
            for s in range(samples):
                with inject_rows(model, args.layer, add, span, decode):
                    # Like Macar: only do_sample + temperature, so Gemma's generation_config
                    # top_k/top_p defaults apply.
                    seqs = model.generate(**batch, do_sample=True, temperature=1.0,
                                          max_new_tokens=args.max_new_tokens)
                for i, c in enumerate(CONCEPTS):
                    text = tok.decode(seqs[i, plen:], skip_special_tokens=True)
                    rows.append(dict(layer=args.layer, alpha=args.alpha, condition=cond, trial=n, sample=s,
                                     concept=c, p_yes=p0[i, yes_id].item(), p_no=p0[i, no_id].item(),
                                     prompt=question(n), response=text))
                if args.dry_run:
                    for r in rows[-3:]:
                        print(f"[{cond}] {r['concept']}: P(Yes)={r['p_yes']:.3f} | {r['response'][:160]!r}")
        print(f"trial {n} done, {time.time() - t0:.0f}s")

    tag = "dry" if args.dry_run else "full"
    save_json(rows, RESULTS / "validation" / f"stage1_L{args.layer}_a{args.alpha:g}_{tag}{args.suffix}.json")


if __name__ == "__main__":
    main()

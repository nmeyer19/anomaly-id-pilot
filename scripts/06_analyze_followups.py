"""Analysis for scripts/05_followups.py: tables in results/followup/summary.md and three figures.

Measures (all on the neutral sentence 3; injected concept X):
  letter    P(X) over A-E (renormalized), averaged over 5 cyclic option orders
  oneword   shift in log P(X's name as the first answer token) minus the mean over the other
            15 concepts, relative to no injection; also top-1 rate among the 16 concepts
  detect    P(Yes) renormalized over Yes/No (no-injection row = false-alarm baseline)
  localize  P(correct sentence) renormalized over digits 0-5 (chance 1/6)
CIs: 95% bootstrap over pairs (letter) or concepts (others).
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = RES / "followup"
LAYER_COLORS = {12: "#2a78d6", 20: "#eb6834", 28: "#1baf7a", 37: "#eda100"}
VT_LABEL = {"macar": "Macar vector (\"Tell me about X\")", "sent": "sentence-derived vector"}
MIN_MASS = 0.5
N_BOOT = 2000
rng = np.random.default_rng(0)


def boot(values, units):
    """95% CI of the mean, resampling units with replacement."""
    df = pd.DataFrame({"v": values, "u": units}).groupby("u").v.agg(["sum", "count"])
    pick = rng.integers(0, len(df), size=(N_BOOT, len(df)))
    m = df["sum"].values[pick].sum(1) / df["count"].values[pick].sum(1)
    return np.percentile(m, [2.5, 97.5])


def summarize(df, keys, col, unit):
    rows = []
    for k, g in df.groupby(keys):
        lo, hi = boot(g[col].values, g[unit].values)
        rows.append(dict(zip(keys, k)) | {"mean": g[col].mean(), "lo": lo, "hi": hi, "n": len(g),
                                          "mass": g["mass"].mean() if "mass" in g else np.nan})
    return pd.DataFrame(rows)


def letter_df(rows):
    d = pd.DataFrame([r for r in rows if r["block"] == "letter"])
    lp = np.array(d.logp.tolist())
    p = np.exp(lp)
    d["mass"] = p.sum(1)
    pn = p / d["mass"].values[:, None]
    d["pX"] = [pn[i, o.index(x)] for i, (o, x) in enumerate(zip(d.options, d.X))]
    return d


def oneword_df(rows, names):
    d = pd.DataFrame([r for r in rows if r["block"] == "oneword"])
    out = []
    for q, g in d.groupby("question"):
        base = g[g.layer == -1].concept_logp.iloc[0]
        base_score = {x: base[x] - np.mean([base[c] for c in names if c != x]) for x in names}
        for _, r in g[g.layer != -1].iterrows():
            lp = r.concept_logp
            score = lp[r.X] - np.mean([lp[c] for c in names if c != r.X])
            out.append(dict(question=q, layer=r.layer, vtype=r.vtype, c=r.c, X=r.X,
                            dscore=score - base_score[r.X],
                            top1=float(max(lp, key=lp.get) == r.X)))
        top_base = max(base, key=base.get)
        print(f"one-word '{q}', no injection: top concept among the 16 = {top_base}")
    return pd.DataFrame(out)


def detloc_df(rows):
    det = pd.DataFrame([r for r in rows if r["block"] == "detect"])
    p = np.exp(np.array(det.logp.tolist()))
    det["mass"] = p.sum(1)
    det["p_yes"] = p[:, 0] / det["mass"]
    loc = pd.DataFrame([r for r in rows if r["block"] == "localize"])
    p = np.exp(np.array(loc.logp.tolist()))
    loc["mass"] = p.sum(1)
    pn = p / loc["mass"].values[:, None]
    loc["p_correct"] = [pn[i, k] for i, k in enumerate(loc.k)]
    loc["p_none"] = pn[:, 0]
    loc["argmax_correct"] = (pn.argmax(1) == loc.k.values).astype(float)
    return det, loc


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#8a8a85")
    ax.tick_params(colors="#55554f", labelsize=8)
    ax.grid(axis="y", color="#e6e6e1", lw=0.8)
    ax.set_axisbelow(True)


def panel_grid(panels, title, fname, ylabel_fmt, note):
    """panels: list of (row_title, table, baseline, ylim). Table has layer, vtype, c, mean, lo, hi, mass."""
    fig, axes = plt.subplots(len(panels), 2, figsize=(9, 2.7 * len(panels) + 0.6), squeeze=False, sharex=True)
    for i, (rt, t, base, ylim) in enumerate(panels):
        for j, vt in enumerate(["macar", "sent"]):
            ax = axes[i, j]
            style(ax)
            if base is not None:
                ax.axhline(base, color="#8a8a85", lw=1, ls="--")
            for L, col in LAYER_COLORS.items():
                s = t[(t.layer == L) & (t.vtype == vt)].sort_values("c")
                ax.fill_between(s.c, s.lo, s.hi, color=col, alpha=0.12, lw=0)
                ax.plot(s.c, s["mean"], color=col, lw=2, marker="o", ms=5, label=f"layer {L}")
                bad = s[s.mass < MIN_MASS]
                ax.scatter(bad.c, bad["mean"], s=70, facecolors="none", edgecolors="#33332f", lw=1, zorder=3)
            ax.set_xscale("log", base=2)
            ax.set_xticks(sorted(t.c.unique()), [f"{c:g}" for c in sorted(t.c.unique())])
            if ylim:
                ax.set_ylim(*ylim)
            if i == 0:
                ax.set_title(VT_LABEL[vt], fontsize=9, loc="left")
            if j == 0:
                ax.set_ylabel(ylabel_fmt(rt), fontsize=8.5)
            if i == len(panels) - 1:
                ax.set_xlabel("injection norm c (fraction of residual norm)", fontsize=8.5)
    axes[0, 1].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.suptitle(title, fontsize=10.5, x=0.01, ha="left")
    fig.text(0.01, 0.005, note, fontsize=7, color="#8a8a85")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(OUT / fname, dpi=150)
    plt.close(fig)


def md(t, cols):
    t = t.copy()
    return t[cols].to_markdown(index=False, floatfmt=".3f")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = json.load(open(RES / "followup_rows.json"))
    meta = json.load(open(RES / "followup_meta.json"))
    names = meta["names"]
    lines = ["# Follow-up tests (scripts/05_followups.py)", "",
             "Neutral sentence 3 throughout; X injected on sentence-3 tokens (or sentence k for "
             "detection/localization). c = injection norm / median residual norm on those tokens.", ""]
    lines += ["## Vectors", "", "| layer | r (median residual norm) | |v_macar|/r | |v_sent|/r | cos(macar, sent) |",
              "|---|---|---|---|---|"]
    for L in meta["layers"]:
        r = meta["r"][str(L)]
        lines.append(f"| {L} | {r:.0f} | {np.median(meta[f'norm_macar_L{L}']) / r:.3f} | "
                     f"{np.median(meta[f'norm_sent_L{L}']) / r:.3f} | {np.median(meta[f'cos_macar_sent_L{L}']):.3f} |")
    lines += ["", "|v_sent|/r is the natural scale: the average push a real sentence about X gives, per token.", ""]

    # Letter blocks
    d = letter_df(rows)
    base = d[d.layer == -1].groupby("question").pX.mean()
    t_letter = summarize(d[d.layer != -1], ["question", "span_name", "layer", "vtype", "c"], "pX", "pair")
    for q in ["about", "injected"]:
        lines += [f"## Letter read-out: \"{'What is sentence 3 about?' if q == 'about' else 'What thought was injected at sentence 3?'}\"",
                  "", f"No-injection P(X) = {base[q]:.3f} (5 option orders × 16 concepts)", ""]
        for sn in sorted(t_letter[t_letter.question == q].span_name.unique()):
            tt = t_letter[(t_letter.question == q) & (t_letter.span_name == sn)]
            wide = tt.pivot_table(index=["vtype", "layer"], columns="c", values="mean")
            lines += [f"Span: {sn}. Mean P(X):", "", wide.to_markdown(floatfmt=".3f"), ""]
            sig = tt[tt.lo > base[q]]
            lines += [f"Cells with CI lower bound above baseline: "
                      + (", ".join(f"{r.vtype}/L{r.layer}/c={r.c:g} ({r['mean']:.2f}, mass {r.mass:.2f})"
                                   for _, r in sig.iterrows()) or "none"), ""]
    panel_grid(
        [("about", t_letter[(t_letter.question == "about") & (t_letter.span_name == "s3")], base["about"], (0, 1)),
         ("injected", t_letter[(t_letter.question == "injected") & (t_letter.span_name == "s3")], base["injected"], (0, 1)),
         ("injected (to end)", t_letter[(t_letter.question == "injected") & (t_letter.span_name == "s3_to_end")],
          base["injected"], (0, 1))],
        "Retrieval vs. report: P(X) when X is injected into a neutral sentence 3",
        "followup_letter.png",
        lambda rt: {"about": "P(X): \"about?\"\n(sentence 3 only)", "injected": "P(X): \"injected?\"\n(sentence 3 only)",
                    "injected (to end)": "P(X): \"injected?\"\n(sentence 3 → end)"}[rt],
        "Dashed: no injection. Bands: 95% bootstrap CI over pairs. Hollow ring: letter mass < 0.5 "
        "(answer format breaking down). Chance = 0.2.")

    # One-word block
    ow = oneword_df(rows, names)
    t_ow = summarize(ow.assign(mass=1.0), ["question", "layer", "vtype", "c"], "dscore", "X")
    t_top = ow.groupby(["question", "layer", "vtype", "c"]).top1.mean().reset_index()
    for q in ["about", "injected"]:
        tt = t_ow[t_ow.question == q]
        lines += [f"## One-word answer, \"{q}\" question: Δ log P(X name) relative to other concepts (nats)", "",
                  tt.pivot_table(index=["vtype", "layer"], columns="c", values="mean").to_markdown(floatfmt=".2f"), "",
                  "Top-1 rate (X is the most likely of the 16 concept names; chance 1/16):", "",
                  t_top[t_top.question == q].pivot_table(index=["vtype", "layer"], columns="c", values="top1")
                  .to_markdown(floatfmt=".2f"), ""]
    panel_grid(
        [("about", t_ow[t_ow.question == "about"], 0, None),
         ("injected", t_ow[t_ow.question == "injected"], 0, None)],
        "One-word answers: does X's name become more likely? (no letters, no position bias)",
        "followup_oneword.png",
        lambda rt: f"Δ log P(X name), nats\n\"{rt}?\"",
        "Δ = [log P(X) − mean log P(other 15 names)] minus the same with no injection. "
        "Bands: 95% bootstrap CI over concepts.")

    # Detection & localization
    det, loc = detloc_df(rows)
    fa = det[det.layer == -1].p_yes.iloc[0]
    loc_base = loc[loc.layer == -1].iloc[0]
    t_det = summarize(det[det.layer != -1], ["layer", "vtype", "c"], "p_yes", "X")
    t_loc = summarize(loc[loc.layer != -1], ["layer", "vtype", "c"], "p_correct", "X")
    t_loc_arg = loc[loc.layer != -1].groupby(["layer", "vtype", "c"]).argmax_correct.mean().reset_index()
    lines += ["## Detection: \"Was a thought injected? Yes/No\"", "",
              f"No injection: P(Yes) = {fa:.3f} (false-alarm baseline)", "",
              t_det.pivot_table(index=["vtype", "layer"], columns="c", values="mean").to_markdown(floatfmt=".3f"), "",
              "## Localization: \"Which sentence, if any? 0-5\"", "",
              f"No injection: P(0 = none) = {np.exp(loc_base.logp[0]) / np.exp(loc_base.logp).sum():.3f}; "
              f"distribution over 0-5 = {np.round(np.exp(loc_base.logp) / np.exp(loc_base.logp).sum(), 3).tolist()}", "",
              "Mean P(correct sentence) (chance 1/6 = 0.167):", "",
              t_loc.pivot_table(index=["vtype", "layer"], columns="c", values="mean").to_markdown(floatfmt=".3f"), "",
              "Argmax accuracy:", "",
              t_loc_arg.pivot_table(index=["vtype", "layer"], columns="c", values="argmax_correct").to_markdown(floatfmt=".2f"), "",
              "Localization by injected sentence k (all layers/vectors, c ≥ 0.2), mean P(correct):", "",
              loc[(loc.layer != -1) & (loc.c >= 0.2)].groupby(["vtype", "k"]).p_correct.mean().unstack()
              .to_markdown(floatfmt=".3f"), ""]
    panel_grid(
        [("detect", t_det, fa, (0, 1)), ("localize", t_loc, 1 / 6, (0, 1))],
        "Detection and localization (X injected into one of 5 neutral sentences)",
        "followup_detect_localize.png",
        lambda rt: {"detect": "P(Yes)\n\"was a thought injected?\"",
                    "localize": "P(correct sentence)\n\"which sentence?\""}[rt],
        "Dashed: detection = P(Yes) with no injection; localization = chance (1/6). "
        "Bands: 95% bootstrap CI over concepts. Hollow ring: answer-token mass < 0.5.")

    # Free text
    ft_path = RES / "followup_free_text.json"
    if ft_path.exists():
        ft = json.load(open(ft_path))
        lines += ["## Free-text samples (c = 0.4)", ""]
        for r in ft:
            if r["c"] in (0.0, 0.4):
                lines.append(f"- [{r['question']}] {r['vtype']} L{r['layer']} X={r['X']} c={r['c']:g}: {r['answer']}")
        lines.append("")

    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

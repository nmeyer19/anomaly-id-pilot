"""Analysis for scripts/07_main_test_sent.py -> results/main_sent/{summary.md, main_sent.png}.

rho = s_X / s_Y with s_Y = 1 (a real sentence about Y) and s_X = injection norm / |v_sent,X|.
Key quantity: dissociation = P_inj(X) - P_about(X), paired per trial and option order.
CIs: 95% bootstrap over pairs.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = RES / "main_sent"
N_BOOT = 2000
rng = np.random.default_rng(0)
SERIES = {"P_about(X)": "#2a78d6", "P_about(Y)": "#eb6834", "P_inj(X)": "#1baf7a", "P_inj(Y)": "#eda100"}


def boot(values, units):
    df = pd.DataFrame({"v": values, "u": units}).groupby("u").v.agg(["sum", "count"])
    pick = rng.integers(0, len(df), size=(N_BOOT, len(df)))
    m = df["sum"].values[pick].sum(1) / df["count"].values[pick].sum(1)
    return np.percentile(m, [2.5, 97.5])


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#8a8a85")
    ax.tick_params(colors="#55554f", labelsize=8)
    ax.grid(axis="y", color="#e6e6e1", lw=0.8)
    ax.set_axisbelow(True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = json.load(open(RES / "main_sent_rows.json"))
    d = pd.DataFrame([r for r in rows if r["block"] == "letter"])
    p = np.exp(np.array(d.logp.tolist()))
    d["mass"] = p.sum(1)
    pn = p / d["mass"].values[:, None]
    d["pX"] = [pn[i, o.index(x)] for i, (o, x) in enumerate(zip(d.options, d.X))]
    d["pY"] = [pn[i, o.index(y)] for i, (o, y) in enumerate(zip(d.options, d.Y))]

    base = d[d.layer == -1]
    lines = ["# Main test with sentence-derived vectors (scripts/07_main_test_sent.py)", "",
             "Sentence 3 is a real sentence about Y; X is injected on its tokens. ρ = s_X (s_Y = 1).", "",
             f"Letter mass: min {d.mass.min():.3f}, median {d.mass.median():.3f}", "",
             "No injection: " + ", ".join(
                 f"P_{q}(X) = {base[base.question == q].pX.mean():.3f}, P_{q}(Y) = {base[base.question == q].pY.mean():.3f}"
                 for q in ["about", "injected"]), ""]

    # paired dissociation per (X, order, layer, rho)
    key = ["X", "pair", "order", "layer", "rho"]
    wide = d.pivot_table(index=key, columns="question", values=["pX", "pY"]).reset_index()
    wide.columns = ["_".join(c).strip("_") for c in wide.columns]
    wide["diss"] = wide.pX_injected - wide.pX_about

    rows_t = []
    for (L, rho), g in d[d.layer != -1].groupby(["layer", "rho"]):
        rec = dict(layer=L, rho=rho)
        for q in ["about", "injected"]:
            gq = g[g.question == q]
            for col, nm in [("pX", "X"), ("pY", "Y")]:
                lo, hi = boot(gq[col].values, gq.pair.values)
                rec |= {f"P_{'inj' if q == 'injected' else q}({nm})": gq[col].mean(),
                        f"P_{'inj' if q == 'injected' else q}({nm})_lo": lo,
                        f"P_{'inj' if q == 'injected' else q}({nm})_hi": hi}
        w = wide[(wide.layer == L) & (wide.rho == rho)]
        lo, hi = boot(w.diss.values, w.pair.values)
        rec |= dict(dissociation=w.diss.mean(), diss_lo=lo, diss_hi=hi, mass=g.mass.mean())
        rows_t.append(rec)
    t = pd.DataFrame(rows_t)
    show = ["layer", "rho", "P_about(X)", "P_about(Y)", "P_inj(X)", "P_inj(Y)", "dissociation", "diss_lo", "diss_hi", "mass"]
    lines += ["## Letter read-out", "", "dissociation = P_inj(X) − P_about(X), paired; CI over pairs.", "",
              t[show].to_markdown(index=False, floatfmt=".3f"), ""]

    # one-word: log-odds X vs Y per question
    ow = pd.DataFrame([r for r in rows if r["block"] == "oneword"])
    ow["lo_XY"] = [r[x] - r[y] for r, x, y in zip(ow.concept_logp, ow.X, ow.Y)]
    owt = ow.pivot_table(index=["layer", "rho"], columns="question", values="lo_XY").reset_index()
    owt["inj_minus_about"] = owt["injected"] - owt["about"]
    lines += ["## One-word read-out: log P(X name) − log P(Y name), nats", "",
              owt.to_markdown(index=False, floatfmt=".2f"), ""]

    # figure
    layers = sorted(t.layer.unique())
    fig, axes = plt.subplots(1, len(layers), figsize=(5 * len(layers), 3.9), squeeze=False, sharey=True)
    for ax, L in zip(axes[0], layers):
        style(ax)
        s = t[t.layer == L].sort_values("rho")
        for name, col in SERIES.items():
            ls = "--" if name.endswith("(Y)") else "-"
            ax.fill_between(s.rho, s[f"{name}_lo"], s[f"{name}_hi"], color=col, alpha=0.12, lw=0)
            ax.plot(s.rho, s[name], color=col, lw=2, ls=ls, marker="o", ms=5, label=name)
        ax.axvline(1, color="#c3c2b7", lw=0.8)
        ax.text(1.05, 0.98, "ρ = 1", fontsize=7.5, color="#8a8a85", va="top")
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(s.rho), [f"{v:g}" for v in sorted(s.rho)])
        ax.set_ylim(0, 1)
        ax.set_title(f"injection at layer {L}", fontsize=9, loc="left")
        ax.set_xlabel("ρ = injected X strength / real Y sentence strength", fontsize=8.5)
    axes[0, 0].set_ylabel("probability (renormalized over A–E)", fontsize=8.5)
    axes[0, -1].legend(frameon=False, fontsize=7.5, loc="center right")
    fig.suptitle("Appearance vs. injection: sentence about Y, X injected (sentence-derived vectors)",
                 fontsize=10.5, x=0.01, ha="left")
    fig.text(0.01, 0.01, "Solid: X; dashed: Y. Bands: 95% bootstrap CI over pairs. "
             "Proof of life would be green (P_inj(X)) above blue (P_about(X)).", fontsize=7, color="#8a8a85")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(OUT / "main_sent.png", dpi=150)
    plt.close(fig)

    ft_path = RES / "main_sent_free_text.json"
    if ft_path.exists():
        ft = json.load(open(ft_path))
        lines += ["## Free-text samples", ""]
        for r in ft:
            lines.append(f"- [{r['question']}] L{r['layer']} ρ={r['rho']:g} X={r['X']} Y={r['Y']}: "
                         + r["answer"].replace("\n", " ")[:160])
        lines.append("")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(l for l in lines if not l.startswith("- [")))


if __name__ == "__main__":
    main()

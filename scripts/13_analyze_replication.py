"""Analysis for scripts/10_replication_lens.py -> results/replication/{summary.md, *.png}.

For each read-out, question q and rho, per trial (X, Y, order):
    shift_q = [logit(X) - logit(Y)](rho) - [logit(X) - logit(Y)](rho = 0)
    dissociation = shift_injected - shift_about      (> 0: "injected" leans to X more than "about")
computed at the final output and at every layer (logit lens). 95% CIs: bootstrap over unordered
pairs {X, Y}.
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
OUT = RES / "replication"
N_BOOT = 2000
rng = np.random.default_rng(0)
READ_COLORS = {"letter": "#2a78d6", "2afc": "#eb6834", "oneword": "#1baf7a"}
READ_LABEL = {"letter": "5-option letters", "2afc": "2-option letters (X vs Y)", "oneword": "one-word answer"}


def boot_ci(values, clusters):
    df = pd.DataFrame({"v": values, "u": clusters}).groupby("u").v.agg(["sum", "count"])
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
    global OUT
    tag = sys.argv[1] if len(sys.argv) > 1 else ""  # "" or "_noname"
    OUT = RES / f"replication{tag}"
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.DataFrame(json.load(open(RES / f"replication{tag}_rows.json")))
    d["upair"] = [tuple(sorted(p)) for p in zip(d.X, d.Y)]
    lens = np.array(d.lens_lo.tolist())  # [rows, layers]
    n_layers = lens.shape[1]
    for li in range(n_layers):
        d[f"L{li}"] = lens[:, li]
    cols = ["final_lo"] + [f"L{li}" for li in range(n_layers)]
    key = ["X", "Y", "readout", "question", "order"]
    base = d[d.rho == 0].set_index(key)[cols]
    inj = d[d.rho > 0].join(base, on=key, rsuffix="_0")
    for c in cols:
        inj[f"s_{c}"] = inj[c] - inj[f"{c}_0"]
    wkey = ["X", "Y", "upair", "cos_xy", "readout", "order", "rho"]
    w = inj.pivot_table(index=wkey, columns="question", values=[f"s_{c}" for c in cols]).reset_index()
    w.columns = ["_".join(c).strip("_") for c in w.columns]
    for c in cols:
        w[f"dd_{c}"] = w[f"s_{c}_injected"] - w[f"s_{c}_about"]

    lines = [f"# Replication with logit lens (scripts/10_replication_lens.py{' --noname' if tag else ''})", "",
             f"{d[['X', 'Y']].drop_duplicates().shape[0]} ordered pairs, {d.upair.nunique()} unordered; "
             f"layer-20 sentence-derived vectors (6 sentences); ρ ∈ {sorted(d.rho.unique())}", "",
             "## Final output: shift toward X (log-odds X:Y vs. ρ = 0), by question, and dissociation", "",
             "| read-out | ρ | shift 'about' | shift 'injected' | injected − about [95% CI] | share of pairs > 0 |",
             "|---|---|---|---|---|---|"]
    summ = []
    for (ro, rho), g in w.groupby(["readout", "rho"]):
        lo, hi = boot_ci(g.dd_final_lo.values, g.upair.values)
        per_pair = g.groupby("upair").dd_final_lo.mean()
        lines.append(f"| {ro} | {rho:g} | {g.s_final_lo_about.mean():.2f} | {g.s_final_lo_injected.mean():.2f} | "
                     f"{g.dd_final_lo.mean():+.2f} [{lo:+.2f}, {hi:+.2f}] | {(per_pair > 0).mean():.2f} |")
        summ.append(dict(readout=ro, rho=rho, dd=g.dd_final_lo.mean(), lo=lo, hi=hi))
    summ = pd.DataFrame(summ)

    # robustness: low-overlap pairs only
    low = w[w.cos_xy.abs() < 0.3]
    lines += ["", f"### Low-overlap pairs only (|cos| < 0.3, {low.upair.nunique()} unordered pairs)", "",
              "| read-out | ρ | injected − about [95% CI] |", "|---|---|---|"]
    for (ro, rho), g in low.groupby(["readout", "rho"]):
        lo, hi = boot_ci(g.dd_final_lo.values, g.upair.values)
        lines.append(f"| {ro} | {rho:g} | {g.dd_final_lo.mean():+.2f} [{lo:+.2f}, {hi:+.2f}] |")

    # lens: dissociation by layer
    lens_rows = []
    for (ro, rho), g in w.groupby(["readout", "rho"]):
        for li in range(n_layers):
            v = g[f"dd_L{li}"].values
            lo, hi = boot_ci(v, g.upair.values)
            lens_rows.append(dict(readout=ro, rho=rho, layer=li, dd=v.mean(), lo=lo, hi=hi,
                                  about=g[f"s_L{li}_about"].mean(), injected=g[f"s_L{li}_injected"].mean()))
    lt = pd.DataFrame(lens_rows)
    lines += ["", "## Logit lens: dissociation (injected − about) by layer, ρ = 1", "",
              "| layer | " + " | ".join(READ_LABEL[r] for r in READ_COLORS) + " |", "|---" * 4 + "|"]
    for li in range(20, n_layers, 3):
        cells = []
        for ro in READ_COLORS:
            r_ = lt[(lt.readout == ro) & (lt.rho == 1.0) & (lt.layer == li)].iloc[0]
            star = "*" if (r_.lo > 0 or r_.hi < 0) else ""
            cells.append(f"{r_.dd:+.2f}{star}")
        lines.append(f"| {li} | " + " | ".join(cells) + " |")
    lines += ["", "\\* = 95% CI excludes 0.", ""]

    # free text
    ft = pd.DataFrame(json.load(open(RES / f"replication{tag}_free_text.json")))
    lines += [f"## Free text ({ft[['X', 'Y']].drop_duplicates().shape[0]} ordered pairs): share of answers naming X / Y", "",
              ft.groupby(["question", "rho"])[["names_X", "names_Y"]].mean().round(2).to_markdown(), ""]

    # figures
    fig, ax = plt.subplots(figsize=(6.6, 4))
    style(ax)
    ax.axhline(0, color="#8a8a85", lw=1)
    for ro, col in READ_COLORS.items():
        s = summ[summ.readout == ro].sort_values("rho")
        ax.fill_between(s.rho, s.lo, s.hi, color=col, alpha=0.15, lw=0)
        ax.plot(s.rho, s.dd, color=col, lw=2, marker="o", ms=6, label=READ_LABEL[ro])
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(summ.rho.unique()), [f"{v:g}" for v in sorted(summ.rho.unique())])
    ax.set_xlabel("ρ = injected X strength / real Y sentence strength", fontsize=8.5)
    ax.set_ylabel("dissociation (nats)\n\"injected\" shift − \"about\" shift", fontsize=8.5)
    ax.set_title("Does \"what was injected?\" lean to X more than \"what is it about?\"", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.text(0.01, 0.01, "Above 0: the injected question favors X more. Bands: 95% bootstrap CI over pairs.",
             fontsize=7, color="#8a8a85")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "dissociation_final.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=False)
    for ax, ro in zip(axes, READ_COLORS):
        style(ax)
        ax.axhline(0, color="#8a8a85", lw=1)
        for rho, col in zip([0.5, 1.0, 2.0], ["#9ec5f4", "#3987e5", "#0d366b"]):
            s = lt[(lt.readout == ro) & (lt.rho == rho)]
            ax.fill_between(s.layer, s.lo, s.hi, color=col, alpha=0.15, lw=0)
            ax.plot(s.layer, s.dd, color=col, lw=1.8, label=f"ρ = {rho:g}")
        ax.axvline(20, color="#c3c2b7", lw=0.8, ls=":")
        ax.set_title(READ_LABEL[ro], fontsize=9, loc="left")
        ax.set_xlabel("layer (logit lens)", fontsize=8.5)
    axes[0].set_ylabel("dissociation (nats)", fontsize=8.5)
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("Dissociation across layers (injected − about shift toward X); dotted line: injection layer",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "dissociation_lens.png", dpi=150)
    plt.close(fig)

    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

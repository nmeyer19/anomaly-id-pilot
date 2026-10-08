"""Analysis for scripts/12_primer.py -> results/primer/{summary.md, *.png}.

Detection: P(Yes) renormalized over Yes/No, per layer (logit lens; last layer = actual output).
Identification: P(X) over digits 0-5 (0 = nothing injected).
Anomaly test (sentence 3 about Y): shift in log-odds X:Y vs. no injection, "identify" vs "about".
95% CIs: bootstrap over injected concepts X.
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
OUT = RES / "primer"
N_BOOT = 2000
rng = np.random.default_rng(0)
PRIMER_COLORS = {"doc": "#2a78d6", "framing": "#eb6834", "lipsum": "#1baf7a"}
PRIMER_LABEL = {"doc": "framing + introspection document", "framing": "framing only", "lipsum": "framing + lorem ipsum"}


def cond_name(r):
    if r.vtype == "none":
        return "none"
    return f"sent ρ={r.rho:g}" if r.vtype == "sent" else f"macar c={r.c:g}"


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
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.DataFrame(json.load(open(RES / "primer_rows.json")))
    d["cond"] = d.apply(cond_name, axis=1)
    order = ["none"] + [f"sent ρ={x:g}" for x in (0.5, 1, 2, 4)] + [f"macar c={x:g}" for x in (0.1, 0.2, 0.4)]
    lines = ["# Primer test (scripts/12_primer.py)", "",
             f"Answer-token mass at the output: median {d['mass'].median():.3f}, min {d['mass'].min():.3f}", ""]

    # ---- detection
    det = d[d.question == "detect"].copy()
    L = np.exp(np.array(det.lens_yes.tolist()))  # P(Yes) per layer
    n_layers = L.shape[1]
    det["p_final"] = L[:, -1]
    det["p_peak"] = L[:, 40:58].max(1)
    lines += ["## Detection: P(Yes) to \"Did I inject a thought while you read sentence 3?\"", "",
              "Final output (rows: primer × sentence-3 type; columns: injection):", "",
              det.pivot_table(index=["primer", "s3"], columns="cond", values="p_final")[order].to_markdown(floatfmt=".3f"), "",
              "Max over layers 40–57 of the logit-lens P(Yes):", "",
              det.pivot_table(index=["primer", "s3"], columns="cond", values="p_peak")[order].to_markdown(floatfmt=".3f"), ""]
    # sensitivity per layer: P(yes | sent rho=2) - P(yes | none), per concept
    lens_det = []
    for (pr, s3), g in det.groupby(["primer", "s3"]):
        Lg = np.exp(np.array(g.lens_yes.tolist()))
        none = Lg[(g.cond == "none").values].mean(0)
        for cnd in ["sent ρ=1", "sent ρ=2", "sent ρ=4", "macar c=0.2"]:
            m = Lg[(g.cond == cnd).values]
            lens_det.append(dict(primer=pr, s3=s3, cond=cnd, curve=(m.mean(0) - none), none=none))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, s3 in zip(axes, ["neutral", "Y"]):
        style(ax)
        ax.axhline(0, color="#8a8a85", lw=1)
        for pr, col in PRIMER_COLORS.items():
            e = [x for x in lens_det if x["primer"] == pr and x["s3"] == s3 and x["cond"] == "sent ρ=2"][0]
            ax.plot(range(n_layers), e["curve"], color=col, lw=1.8, label=PRIMER_LABEL[pr])
        ax.axvline(20, color="#c3c2b7", lw=0.8, ls=":")
        ax.set_title(f"sentence 3 = {'neutral filler' if s3 == 'neutral' else 'sentence about Y'}", fontsize=9, loc="left")
        ax.set_xlabel("layer (logit lens)", fontsize=8.5)
    axes[0].set_ylabel("P(Yes | injected, ρ=2) − P(Yes | none)", fontsize=8.5)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Detection sensitivity across layers (sentence-derived vector, layer 20)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "detection_lens.png", dpi=150)
    plt.close(fig)

    # ---- identification on neutral sentence
    idn = d[(d.question == "identify")].copy()
    LX = np.exp(np.array(idn.lens_X.tolist()))
    LN = np.exp(np.array(idn.lens_none.tolist()))
    idn["pX_final"], idn["pNone_final"] = LX[:, -1], LN[:, -1]
    idn["pX_peak"] = LX[:, 40:58].max(1)
    neu = idn[idn.s3 == "neutral"]
    lines += ["## Identification (neutral sentence 3): P(X) over 0–5 (chance among concepts 1/5)", "",
              "Final output:", "", neu.pivot_table(index="primer", columns="cond", values="pX_final")[order].to_markdown(floatfmt=".3f"), "",
              "P(0 = nothing injected), final:", "",
              neu.pivot_table(index="primer", columns="cond", values="pNone_final")[order].to_markdown(floatfmt=".3f"), "",
              "Max over layers 40–57 of lens P(X):", "",
              neu.pivot_table(index="primer", columns="cond", values="pX_peak")[order].to_markdown(floatfmt=".3f"), ""]

    # ---- anomaly test on Y sentence: identify vs about, log-odds X:Y shift vs none
    y = d[(d.s3 == "Y") & d.question.isin(["identify", "about"])].copy()
    lx, ly = np.array(y.lens_X.tolist()), np.array(y.lens_Y.tolist())
    for li in range(n_layers):
        y[f"lo{li}"] = lx[:, li] - ly[:, li]
    lcols = [f"lo{li}" for li in range(n_layers)]
    key = ["X", "primer", "question", "order"]
    base = y[y.cond == "none"].set_index(key)[lcols]
    yi = y[y.cond != "none"].join(base, on=key, rsuffix="_0")
    for c in lcols:
        yi[f"s_{c}"] = yi[c] - yi[f"{c}_0"]
    w = yi.pivot_table(index=["X", "primer", "cond", "order"], columns="question",
                       values=[f"s_{c}" for c in lcols]).reset_index()
    w.columns = ["_".join(c).strip("_") for c in w.columns]
    last = n_layers - 1
    lines += ["## Anomaly test (sentence 3 about Y): shift toward X, identify vs. about (final output, nats)", "",
              "| primer | injection | shift 'about' | shift 'identify' | identify − about [95% CI] |", "|---|---|---|---|---|"]
    for (pr, cnd), g in w.groupby(["primer", "cond"]):
        dd = g[f"s_lo{last}_identify"] - g[f"s_lo{last}_about"]
        lo, hi = boot_ci(dd.values, g.X.values)
        lines.append(f"| {pr} | {cnd} | {g[f's_lo{last}_about'].mean():.2f} | {g[f's_lo{last}_identify'].mean():.2f} | "
                     f"{dd.mean():+.2f} [{lo:+.2f}, {hi:+.2f}] |")
    lines.append("")
    # final-output P(X) for identify vs about on Y sentence
    ab = d[(d.s3 == "Y") & (d.question == "about")].copy()
    ab["pX_final"] = np.exp(np.array(ab.lens_X.tolist()))[:, -1]
    lines += ["P(X) at the output, sentence about Y — identify:", "",
              idn[idn.s3 == "Y"].pivot_table(index="primer", columns="cond", values="pX_final")[order].to_markdown(floatfmt=".3f"),
              "", "about:", "", ab.pivot_table(index="primer", columns="cond", values="pX_final")[order].to_markdown(floatfmt=".3f"), ""]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, pr in zip(axes, PRIMER_COLORS):
        style(ax)
        ax.axhline(0, color="#8a8a85", lw=1)
        for cnd, col in zip(["sent ρ=1", "sent ρ=2", "sent ρ=4"], ["#9ec5f4", "#3987e5", "#0d366b"]):
            g = w[(w.primer == pr) & (w.cond == cnd)]
            dd = np.stack([(g[f"s_lo{li}_identify"] - g[f"s_lo{li}_about"]).values for li in range(n_layers)], 1)
            ci = np.array([boot_ci(dd[:, li], g.X.values) for li in range(n_layers)])
            ax.fill_between(range(n_layers), ci[:, 0], ci[:, 1], color=col, alpha=0.15, lw=0)
            ax.plot(range(n_layers), dd.mean(0), color=col, lw=1.8, label=cnd)
        ax.axvline(20, color="#c3c2b7", lw=0.8, ls=":")
        ax.set_title(PRIMER_LABEL[pr], fontsize=9, loc="left")
        ax.set_xlabel("layer (logit lens)", fontsize=8.5)
    axes[0].set_ylabel("identify − about shift toward X (nats)", fontsize=8.5)
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("Anomaly test across layers: does \"which concept was injected?\" lean to X more than \"what is it about?\"",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "anomaly_lens.png", dpi=150)
    plt.close(fig)

    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

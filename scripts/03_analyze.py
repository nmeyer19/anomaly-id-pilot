"""Analysis: Gate 1 and main-test plots, rho computed post hoc from logged projections.

Strength of concept C on a pass (per read layer):
    s_C = (proj_C - neutral_C) / (evoked_C - neutral_C)
  neutral_C: projection onto v_hat_C of the neutral sentence 3 (gate-1, alpha=0)
  evoked_C:  projection onto v_hat_C of C's own sentence (main trials with Y=C, alpha=0)
  rho = s_X / s_Y on main trials.
Letter probabilities are renormalized over A-E. Bootstrap resamples pairs, not trials.

Usage: python scripts/03_analyze.py [results/run_full.json]
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "analysis"
LAYERS = ["L37", "L43"]
RHO_LAYER = "L43"  # downstream read
RHO_BINS = [-np.inf, 0.1, 0.25, 0.5, 1.0, 2.0, np.inf]
RHO_LABELS = ["<0.1 (incl. <0)", "0.1–0.25", "0.25–0.5", "0.5–1", "1–2", ">2"]
N_BOOT = 2000
MIN_LETTER_MASS = 0.5   # below this the model is not answering with a letter
FLAT_MAX_P = 0.35       # max renormalized letter prob below this = near-uniform
COLLAPSE_SHARE = 0.9    # one letter wins >= 90% of a cell's passes regardless of order
LEVELS = {}  # per read layer: neutral/evoked projection levels per concept
COLORS = {"P_about(X)": "#2a78d6", "P_about(Y)": "#eb6834", "P_inj(X)": "#1baf7a",
          "P_inj(Y)": "#eda100", "gate": "#2a78d6"}


def load(path):
    df = pd.DataFrame(pd.read_json(path))
    p = np.array(df["p_letters"].tolist())
    pn = p / p.sum(1, keepdims=True)
    df["pX"] = [pn[i, o.index(x)] for i, (o, x) in enumerate(zip(df["options"], df["X"]))]
    df["pY"] = [pn[i, o.index(y)] for i, (o, y) in enumerate(zip(df["options"], df["Y"]))]
    lp = np.log10(np.clip(pn, 1e-12, None))
    df["log10pX"] = [lp[i, o.index(x)] for i, (o, x) in enumerate(zip(df["options"], df["X"]))]
    df["max_p"] = pn.max(1)
    df["argmax"] = pn.argmax(1)
    return df


def add_strengths(df):
    for L in LAYERS:
        col = f"proj_{L}"
        neutral = (df[(df.kind == "gate1") & (df.alpha == 0)][col].apply(pd.Series).mean())
        ev = df[(df.kind == "main") & (df.alpha == 0)]
        evoked = pd.Series({c: np.mean([r[c] for r in g[col]]) for c, g in ev.groupby("Y")})
        denom = evoked - neutral[evoked.index]
        df[f"sX_{L}"] = [(r[x] - neutral[x]) / denom[x] for r, x in zip(df[col], df.X)]
        df[f"sY_{L}"] = [(r[y] - neutral[y]) / denom[y] for r, y in zip(df[col], df.Y)]
        df[f"rho_{L}"] = np.where(df.kind == "main", df[f"sX_{L}"] / df[f"sY_{L}"], np.nan)
        LEVELS[L] = pd.DataFrame({"neutral": neutral[evoked.index], "evoked": evoked})
    return df


def quality_flags(df):
    df["flag_mass"] = df.letter_mass < MIN_LETTER_MASS
    df["flag_flat"] = df.max_p < FLAT_MAX_P
    collapsed = []
    for (k, q, a), g in df.groupby(["kind", "question", "alpha"]):
        share = g.argmax.value_counts(normalize=True).iloc[0]
        if share >= COLLAPSE_SHARE:
            collapsed.append((k, q, a, g.argmax.value_counts().index[0], share))
    df["flag_collapse"] = [(k, q, a) in {c[:3] for c in collapsed}
                           for k, q, a in zip(df.kind, df.question, df.alpha)]
    df["keep"] = ~(df.flag_mass | df.flag_flat | df.flag_collapse)
    return df, collapsed


def boot_by_pair(df, col, rng):
    """95% CI of the mean of `col`, resampling pairs with replacement."""
    g = df.groupby("pair")[col].agg(["sum", "count"])
    pick = rng.integers(0, len(g), size=(N_BOOT, len(g)))
    means = g["sum"].values[pick].sum(1) / g["count"].values[pick].sum(1)
    return np.percentile(means, [2.5, 97.5])


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#8a8a85")
    ax.tick_params(colors="#55554f", labelsize=9)
    ax.grid(axis="y", color="#e6e6e1", lw=0.8)
    ax.set_axisbelow(True)


def gate1(df, rng, lines):
    g = df[(df.kind == "gate1") & df.keep]
    alphas = sorted(g.alpha.unique())
    base = df[(df.kind == "gate1") & (df.alpha == 0)].pX.mean()
    rows = []
    for a in alphas:
        ga = g[g.alpha == a]
        lo, hi = boot_by_pair(ga, "pX", rng)
        rows.append(dict(alpha=a, n=len(ga), P_X=ga.pX.mean(), lo=lo, hi=hi, log10P_X=ga.log10pX.mean(),
                         sX_L37=ga.sX_L37.mean(), sX_L43=ga.sX_L43.mean()))
    t = pd.DataFrame(rows)
    above = t[(t.alpha > 0) & (t.lo > base)]
    s_min = above.iloc[0] if len(above) else None

    t["x"] = range(len(t))
    fig, ax = plt.subplots(figsize=(7.2, 4))
    style(ax)
    ax.axhline(base, color="#8a8a85", lw=1, ls="--")
    ax.text(-0.2, 0.09, "dashed: α = 0 baseline; dotted: chance (1/5)", va="top", ha="left", fontsize=8, color="#55554f")
    ax.axhline(0.2, color="#c3c2b7", lw=0.8, ls=":")
    ax.fill_between(t.x, t.lo, t.hi, color=COLORS["gate"], alpha=0.15, lw=0)
    ax.plot(t.x, t.P_X, color=COLORS["gate"], lw=2, marker="o", ms=6)
    for _, r in t.iterrows():
        ax.annotate(f"s={r.sX_L43:.1f}", (r.x, r.P_X), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7.5, color="#55554f")
    ax.set_xlabel("injection strength α (sentence-3 tokens, layer 37)", fontsize=9)
    ax.set_ylabel("P(X) on \"what thought was injected?\"", fontsize=9)
    ax.set_title("Gate 1: identifying the injected concept over a neutral sentence",
                 fontsize=10, loc="left")
    planned = set(df.loc[df.get("planned_alpha", True) == True, "alpha"]) if "planned_alpha" in df else set(alphas)
    ax.set_xticks(t.x, [f"{a:g}" + ("" if a in planned else "*") for a in t.alpha])
    ax.set_ylim(0, 1)
    fig.text(0.01, 0.01, "Band: 95% bootstrap CI over pairs. s = downstream (L43) strength of X "
             "in units of X's evoked level. * = α added after the dry run.", fontsize=7, color="#8a8a85")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "gate1.png", dpi=160)
    plt.close(fig)

    lines += ["## Gate 1 (neutral sentence 3, \"injected\" question)", "",
              f"α = 0 baseline P(X) = {base:.3f}", "",
              t.to_markdown(index=False, floatfmt=".3f"), ""]
    if s_min is None:
        lines.append("**s_min: none**: P(X) never clearly exceeds the α = 0 baseline (CI lower bound).")
    else:
        lines.append(f"**s_min ≈ {s_min.sX_L43:.2f}** (downstream L43; L37: {s_min.sX_L37:.2f}), "
                     f"first reached at α = {s_min.alpha:g}.")
    lines.append("")
    return t, s_min


def main_test(df, rng, lines):
    m = df[(df.kind == "main") & df.keep].copy()
    m["rho_bin"] = pd.cut(m[f"rho_{RHO_LAYER}"], RHO_BINS, labels=RHO_LABELS)
    about, inj = m[m.question == "about"], m[m.question == "injected"]
    series = {"P_about(X)": (about, "pX"), "P_about(Y)": (about, "pY"),
              "P_inj(X)": (inj, "pX"), "P_inj(Y)": (inj, "pY")}
    base_inj_x = df[(df.kind == "main") & (df.alpha == 0) & (df.question == "injected")].pX.mean()

    rows = []
    for name, (d, col) in series.items():
        for b in RHO_LABELS:
            db = d[d.rho_bin == b]
            if len(db) == 0:
                continue
            lo, hi = boot_by_pair(db, col, rng)
            rows.append(dict(series=name, rho_bin=b, n=len(db), n_pairs=db.pair.nunique(),
                             mean=db[col].mean(), lo=lo, hi=hi,
                             rho_mean=db[f"rho_{RHO_LAYER}"].mean()))
    t = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    style(ax)
    xs = {b: i for i, b in enumerate(RHO_LABELS)}
    for name in series:
        s = t[t.series == name]
        x = [xs[b] for b in s.rho_bin]
        ls = "--" if name == "P_inj(Y)" else "-"
        ax.fill_between(x, s.lo, s.hi, color=COLORS[name], alpha=0.12, lw=0)
        ax.plot(x, s["mean"], color=COLORS[name], lw=2, ls=ls, marker="o", ms=6, label=name)
    ax.axhline(base_inj_x, color=COLORS["P_inj(X)"], lw=0.8, ls=":")
    ax.text(-0.3, base_inj_x, "P_inj(X) at α = 0", va="bottom", fontsize=7.5, color="#55554f")
    ax.axvline(xs["0.5–1"] + 0.5, color="#c3c2b7", lw=0.8)
    ax.text(xs["0.5–1"] + 0.55, 0.97, "ρ = 1", fontsize=7.5, color="#8a8a85", va="top")
    ax.set_xticks(range(len(RHO_LABELS)), RHO_LABELS)
    ax.set_xlim(-0.4, len(RHO_LABELS) - 0.2)
    ax.set_ylim(0, 1)
    ax.set_xlabel(f"ρ = s_X / s_Y (downstream, {RHO_LAYER}), binned", fontsize=9)
    ax.set_ylabel("probability (renormalized over A–E)", fontsize=9)
    ax.set_title("Main test: appearance (sentence = Y) vs. injected X", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="center left")
    fig.text(0.01, 0.01, "Bands: 95% bootstrap CI over pairs. P_inj(Y) dashed (not in plan; "
             "shows whether \"injected\" just echoes the sentence).\n"
             "ρ is unstable for Cacti, Islands, Rainbows (evoked − neutral gap ≤ ~30 at L43); see write-up.", fontsize=7, color="#8a8a85")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "main_test.png", dpi=160)
    plt.close(fig)

    wide = t.pivot(index="rho_bin", columns="series", values="mean").reindex(
        [b for b in RHO_LABELS if b in set(t.rho_bin)])
    ns = t[t.series == "P_inj(X)"].set_index("rho_bin")[["n", "n_pairs", "rho_mean"]]
    lines += ["## Main test (sentence 3 = Y)", "",
              f"α = 0 baseline P_inj(X) = {base_inj_x:.3f}", "",
              wide.join(ns).to_markdown(floatfmt=".3f"), "",
              "With 95% pair-bootstrap CIs:", "",
              t.to_markdown(index=False, floatfmt=".3f"), ""]

    # by alpha, for reference
    by_a = m.groupby(["alpha", "question"]).agg(
        pX=("pX", "mean"), pY=("pY", "mean"), log10pX=("log10pX", "mean"), sX=(f"sX_{RHO_LAYER}", "mean"),
        sY=(f"sY_{RHO_LAYER}", "mean"), rho=(f"rho_{RHO_LAYER}", "mean"), n=("pX", "size"))
    lines += ["### Main test by α", "", by_a.to_markdown(floatfmt=".3f"), ""]

    below1 = t[(t.series == "P_inj(X)") & t.rho_bin.isin(RHO_LABELS[:4])]
    hit = below1[below1.lo > base_inj_x]
    about_y = t[(t.series == "P_about(Y)") & t.rho_bin.isin(RHO_LABELS[:4])]
    return t, hit, about_y, base_inj_x


def span_control(path, rng, lines):
    """Diagnostic (scripts/04_diagnostics.py): Gate 1 with wider injection spans."""
    df = load(path)
    spans = {"s3": ("sentence 3 only (as in run)", "#2a78d6"),
             "s3_to_end": ("sentence 3 → end of prompt", "#eb6834"),
             "all": ("all tokens", "#1baf7a")}
    rows = []
    for (sp, a), d in df.groupby(["span", "alpha"]):
        lo, hi = boot_by_pair(d, "pX", rng)
        rows.append(dict(span=sp, alpha=a, P_X=d.pX.mean(), lo=lo, hi=hi,
                         letter_mass=d.letter_mass.mean(), n=len(d)))
    t = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    style(ax)
    ax.axhline(0.2, color="#c3c2b7", lw=0.8, ls=":")
    ax.text(t.alpha.max(), 0.2, " chance", va="top", ha="right", fontsize=8, color="#8a8a85")
    for sp, (label, color) in spans.items():
        d = t[t.span == sp]
        ax.fill_between(d.alpha, d.lo, d.hi, color=color, alpha=0.12, lw=0)
        ax.plot(d.alpha, d.P_X, color=color, lw=2, marker="o", ms=6, label=label)
        low = d[d.letter_mass < MIN_LETTER_MASS]
        ax.scatter(low.alpha, low.P_X, s=90, facecolors="none", edgecolors="#33332f", lw=1, zorder=3)
    ax.scatter([], [], s=90, facecolors="none", edgecolors="#33332f", lw=1,
               label=f"letter mass < {MIN_LETTER_MASS} (incoherent)")
    ax.set_xticks(sorted(t.alpha.unique()))
    ax.set_ylim(0, 1)
    ax.set_xlabel("injection strength α (layer 37)", fontsize=9)
    ax.set_ylabel("P(X) on \"what thought was injected?\"", fontsize=9)
    ax.set_title("Gate 1 positive control: where the injection lands", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.text(0.01, 0.01, "Neutral sentence 3; 16 trials × 2 option orders per point. "
             "Bands: 95% bootstrap CI over pairs.", fontsize=7, color="#8a8a85")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "gate1_span_control.png", dpi=160)
    plt.close(fig)
    lines += ["## Diagnostic: Gate 1 injection-span positive control", "",
              t.to_markdown(index=False, floatfmt=".3f"), ""]


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "run_full.json"
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    df = add_strengths(load(path))
    df, collapsed = quality_flags(df)

    lines = [f"# Pilot analysis ({path.name})", "",
             f"{len(df)} passes; {df.pair.nunique()} pairs; α ∈ {sorted(df.alpha.unique())}", "",
             "## Quality filter", "",
             f"- letter mass < {MIN_LETTER_MASS}: {int(df.flag_mass.sum())} passes "
             f"(median letter mass {df.letter_mass.median():.3f}, min {df.letter_mass.min():.3f})",
             f"- flat (max letter prob < {FLAT_MAX_P}): {int(df.flag_flat.sum())} passes",
             f"- collapsed cells (one letter ≥ {COLLAPSE_SHARE:.0%} of a kind×question×α cell): "
             + (", ".join(f"{k}/{q}/α={a:g} → {'ABCDE'[l]} ({s:.0%})" for k, q, a, l, s in collapsed)
                or "none"),
             f"- kept {int(df.keep.sum())}/{len(df)} passes", ""]
    for L in LAYERS:
        lv = LEVELS[L]
        lines += [f"Normalization levels at {L} (neutral → evoked, mean over concepts): "
                  f"{lv.neutral.mean():.1f} → {lv.evoked.mean():.1f}; "
                  f"min evoked−neutral gap {(lv.evoked - lv.neutral).min():.1f}", ""]

    g_t, s_min = gate1(df, rng, lines)
    m_t, hit, about_y, base = main_test(df, rng, lines)

    diag = path.parent / "diag_span_control.json"
    if diag.exists():
        span_control(diag, rng, lines)

    lines += ["## Decision rule check (automated; read with the plots)", ""]
    gate_pass = s_min is not None
    lines.append(f"- Gate 1 passes at some α: **{gate_pass}**"
                 + (f" (s_min ≈ {s_min.sX_L43:.2f}{' > 1' if s_min.sX_L43 > 1 else ' ≤ 1'})" if gate_pass else ""))
    lines.append(f"- ρ < 1 bins where P_inj(X) CI lower bound > α=0 baseline ({base:.3f}): "
                 + (", ".join(f"{r.rho_bin} ({r['mean']:.3f} [{r.lo:.3f}, {r.hi:.3f}])" for _, r in hit.iterrows())
                    or "none"))
    lines.append("- P_about(Y) in ρ < 1 bins: "
                 + ", ".join(f"{r.rho_bin}: {r['mean']:.3f}" for _, r in about_y.iterrows()))
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    df.drop(columns=["prompt", "proj_L37", "proj_L43", "top_tokens"], errors="ignore").to_csv(
        OUT / "passes.csv", index=False)
    print("\n".join(lines))


if __name__ == "__main__":
    main()

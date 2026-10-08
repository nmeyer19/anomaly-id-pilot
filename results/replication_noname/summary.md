# Replication with logit lens (scripts/10_replication_lens.py --noname)

230 ordered pairs, 115 unordered; layer-20 sentence-derived vectors (6 sentences); ρ ∈ [np.float64(0.0), np.float64(0.5), np.float64(1.0), np.float64(2.0)]

## Final output: shift toward X (log-odds X:Y vs. ρ = 0), by question, and dissociation

| read-out | ρ | shift 'about' | shift 'injected' | injected − about [95% CI] | share of pairs > 0 |
|---|---|---|---|---|---|
| 2afc | 0.5 | 0.34 | 0.30 | -0.05 [-0.14, +0.06] | 0.37 |
| 2afc | 1 | 1.00 | 0.50 | -0.49 [-0.68, -0.30] | 0.19 |
| 2afc | 2 | 3.62 | 3.79 | +0.17 [-0.44, +0.75] | 0.44 |
| letter | 0.5 | 0.37 | 0.51 | +0.15 [+0.07, +0.23] | 0.57 |
| letter | 1 | 1.22 | 1.50 | +0.28 [+0.10, +0.47] | 0.59 |
| letter | 2 | 6.63 | 8.42 | +1.79 [+1.04, +2.58] | 0.58 |
| oneword | 0.5 | 2.02 | 1.28 | -0.75 [-0.97, -0.53] | 0.26 |
| oneword | 1 | 5.70 | 3.64 | -2.06 [-2.54, -1.58] | 0.23 |
| oneword | 2 | 22.06 | 16.15 | -5.91 [-7.09, -4.78] | 0.17 |

### Low-overlap pairs only (|cos| < 0.3, 10 unordered pairs)

| read-out | ρ | injected − about [95% CI] |
|---|---|---|
| 2afc | 0.5 | +0.09 [-0.18, +0.53] |
| 2afc | 1 | -0.04 [-0.46, +0.46] |
| 2afc | 2 | +1.84 [-0.45, +4.30] |
| letter | 0.5 | +0.39 [+0.13, +0.83] |
| letter | 1 | +0.75 [+0.31, +1.45] |
| letter | 2 | +4.22 [+1.57, +7.04] |
| oneword | 0.5 | -0.64 [-1.25, +0.05] |
| oneword | 1 | -1.92 [-3.58, -0.17] |
| oneword | 2 | -3.13 [-6.90, +0.76] |

## Logit lens: dissociation (injected − about) by layer, ρ = 1

| layer | 5-option letters | 2-option letters (X vs Y) | one-word answer |
|---|---|---|---|
| 20 | +0.00 | +0.00 | +0.00 |
| 23 | +0.00 | +0.00 | -0.00 |
| 26 | -0.00* | -0.00 | +0.00* |
| 29 | +0.00 | +0.01* | -0.02* |
| 32 | +0.02* | +0.05* | -0.01* |
| 35 | +0.06* | +0.09* | -0.03* |
| 38 | +0.09* | +0.17* | +0.06 |
| 41 | +0.16* | +0.19* | -0.20* |
| 44 | +0.21* | +0.12* | -0.36* |
| 47 | +0.43* | +0.21* | -0.66* |
| 50 | +0.31* | +0.19* | -0.99* |
| 53 | +0.57* | +0.14* | -0.92* |
| 56 | +0.74* | +0.15* | -1.20* |
| 59 | +0.46* | -0.03 | -0.90* |

\* = 95% CI excludes 0.

## Free text (46 ordered pairs): share of answers naming X / Y

|                   |   names_X |   names_Y |
|:------------------|----------:|----------:|
| ('about', 0.0)    |      0    |      0.17 |
| ('about', 0.5)    |      0    |      0.3  |
| ('about', 1.0)    |      0    |      0.26 |
| ('about', 2.0)    |      0.04 |      0.15 |
| ('injected', 0.0) |      0    |      0.87 |
| ('injected', 0.5) |      0    |      0.87 |
| ('injected', 1.0) |      0    |      0.87 |
| ('injected', 2.0) |      0.07 |      0.59 |


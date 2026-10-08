# Replication with logit lens (scripts/10_replication_lens.py)

230 ordered pairs, 115 unordered; layer-20 sentence-derived vectors (6 sentences); ρ ∈ [np.float64(0.0), np.float64(0.5), np.float64(1.0), np.float64(2.0)]

## Final output: shift toward X (log-odds X:Y vs. ρ = 0), by question, and dissociation

| read-out | ρ | shift 'about' | shift 'injected' | injected − about [95% CI] | share of pairs > 0 |
|---|---|---|---|---|---|
| 2afc | 0.5 | 0.13 | 0.63 | +0.50 [+0.31, +0.72] | 0.68 |
| 2afc | 1 | 0.47 | 1.00 | +0.53 [+0.17, +0.91] | 0.54 |
| 2afc | 2 | 1.63 | 1.75 | +0.12 [-0.48, +0.70] | 0.40 |
| letter | 0.5 | 0.20 | 1.00 | +0.80 [+0.59, +1.02] | 0.83 |
| letter | 1 | 0.52 | 2.34 | +1.82 [+1.41, +2.21] | 0.89 |
| letter | 2 | 1.83 | 5.62 | +3.78 [+3.02, +4.59] | 0.86 |
| oneword | 0.5 | 1.43 | 0.84 | -0.58 [-0.81, -0.35] | 0.30 |
| oneword | 1 | 3.56 | 2.88 | -0.68 [-1.19, -0.14] | 0.40 |
| oneword | 2 | 12.82 | 10.72 | -2.09 [-3.10, -1.12] | 0.34 |

### Low-overlap pairs only (|cos| < 0.3, 10 unordered pairs)

| read-out | ρ | injected − about [95% CI] |
|---|---|---|
| 2afc | 0.5 | +0.87 [+0.18, +1.65] |
| 2afc | 1 | +1.94 [+0.40, +3.65] |
| 2afc | 2 | +2.78 [+0.79, +4.99] |
| letter | 0.5 | +0.95 [+0.58, +1.37] |
| letter | 1 | +2.51 [+1.28, +3.96] |
| letter | 2 | +6.44 [+3.97, +9.22] |
| oneword | 0.5 | -0.03 [-0.64, +0.64] |
| oneword | 1 | +1.42 [-0.43, +3.83] |
| oneword | 2 | +0.41 [-3.00, +3.96] |

## Logit lens: dissociation (injected − about) by layer, ρ = 1

| layer | 5-option letters | 2-option letters (X vs Y) | one-word answer |
|---|---|---|---|
| 20 | +0.00 | +0.00 | +0.00 |
| 23 | +0.00 | +0.00* | -0.00 |
| 26 | -0.00* | -0.00 | -0.00 |
| 29 | +0.00 | +0.00* | -0.03* |
| 32 | +0.01* | +0.02* | +0.00 |
| 35 | +0.06* | +0.09* | -0.02* |
| 38 | +0.10* | +0.15* | +0.06 |
| 41 | +0.37* | +0.16* | -0.23* |
| 44 | +0.46* | +0.33* | -0.29* |
| 47 | +0.86* | +0.59* | -0.68* |
| 50 | +0.71* | +0.46* | -1.11* |
| 53 | +1.30* | +0.31* | -1.28* |
| 56 | +1.56* | +0.50* | -0.76* |
| 59 | +1.25* | +0.45* | -0.51* |

\* = 95% CI excludes 0.

## Free text (first 48 ordered pairs): share of answers naming X / Y

|                   |   names_X |   names_Y |
|:------------------|----------:|----------:|
| ('about', 0.0)    |      0    |      1    |
| ('about', 0.5)    |      0    |      1    |
| ('about', 1.0)    |      0    |      0.96 |
| ('about', 2.0)    |      0    |      1    |
| ('injected', 0.0) |      0    |      0.1  |
| ('injected', 0.5) |      0    |      0.15 |
| ('injected', 1.0) |      0    |      0.21 |
| ('injected', 2.0) |      0.19 |      0.38 |


# anomaly-id-pilot

An afternoon pilot of the appearance-vs-reality test on Gemma-3-27B-IT. The plan is in [anomaly-id-pilot-plan.md](anomaly-id-pilot-plan.md) and the results are in [WRITEUP.md](WRITEUP.md).

## Layout

| Path | Contents |
|---|---|
| `pilot/common.py` | Model loading, layer path, injection and capture hooks, candidate concepts, Macar baseline words |
| `scripts/01_vectors_and_steering_check.py` | Extract concept vectors, steering check, select concepts/pairs/distractors (`--select-only` reruns selection from saved vectors, no GPU) |
| `scripts/02_run_pilot.py` | Gate 1 + main test (`--dry-run N`, `--free-text`) |
| `scripts/03_analyze.py` | Post-hoc ρ, quality filter, plots, `results/analysis/summary.md` (runs locally, CPU only) |
| `scripts/04_diagnostics.py` | Injection-span positive control and per-token projections |
| `data/` | Vectors (layer 37), selection, stimuli |
| `results/` | Raw per-pass JSON, free-text samples, diagnostics, analysis output |

## Run

On a GPU machine with at least 60 GB of memory and access to `google/gemma-3-27b-it`:

```bash
pip install "transformers>=4.50" accelerate pandas matplotlib tabulate
export HF_TOKEN=...   # or put it in .env
python scripts/01_vectors_and_steering_check.py
python scripts/02_run_pilot.py --dry-run 3 --free-text
python scripts/02_run_pilot.py --free-text
python scripts/04_diagnostics.py
python scripts/03_analyze.py results/run_full.json
```

The model load takes ~7 min from a network volume the first time. After that the full run takes ~80 s.

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
| `scripts/05_followups.py`, `06_analyze_followups.py` | Layer × vector sweep: retrieval, report, detection, localization |
| `scripts/07_main_test_sent.py`, `08_analyze_main_sent.py` | Main test with sentence-derived vectors |
| `scripts/09_disruption_detection.py`, `11_note_control.py` | Test 1: the spontaneous "injected thought" note, random-vector and sampling controls |
| `scripts/10_replication_lens.py [--noname]`, `13_analyze_replication.py [_noname]` | Tests 2/2b: replication over 230 ordered pairs, 4 read-outs, logit lens |
| `scripts/12_primer.py [--noname]`, `14_analyze_primer.py` | Test 3: primer elicitation with logit lens |
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

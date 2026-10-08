# Anomaly-specific identification: pilot results

**Run date:** 2026-10-08. **Setup:** Gemma-3-27B-IT, bf16, one A100-80GB. Injection at the output of layer 37 (60% depth). Read-out at layer 43. Plan: [anomaly-id-pilot-plan.md](anomaly-id-pilot-plan.md).

> **Update (follow-ups, same day):** the Gate 1 failure below came from the *vector and layer*, not from a limit of the model. Vectors built from real sentences, injected at layer 20, are read from sentence 3 almost perfectly. The model then treats them as the sentence's content, and the rerun main test shows **no robust appearance-vs-injection dissociation**. See [Follow-up tests](#follow-up-tests) at the end.

## Verdict (original pilot)

**Gate 1 fails at every α.** With the injection only on the sentence-3 tokens, P(X) stays at chance from α = 0 to α = 16. By the plan's decision rule, the fix is to **change the elicitation, not the stimuli**.

A positive control (added after the run) shows the failure comes from *where* the injection lands, not from the forced-choice read-out:

- When the injection continues through the question and answer tokens, P(X) rises from 0.19 to 0.36–0.37 at α = 1–2. The 95% CI over pairs excludes the baseline.
- The answer format then breaks down at α ≥ 4.

The main test shows **no anomaly signal**:

- P_about(Y) = 1.00 in every cell.
- P_inj(X) stays at its α = 0 level (≈ 0.05) in every ρ bin.
- The "what thought was injected?" question is answered with the sentence's own topic (P_inj(Y) ≈ 0.9). Appearance wins completely.

![Gate 1](results/analysis/gate1.png)
![Main test](results/analysis/main_test.png)
![Span control](results/analysis/gate1_span_control.png)

Full tables are in [results/analysis/summary.md](results/analysis/summary.md).

## What was run

- **Concept vectors.** Built with the Macar et al. recipe (their repo has no saved vectors): the layer-37 output at the last token of chat("Tell me about {c}"), minus the mean over their 100 baseline words. Vectors are unnormalized. |v| has a median of 4,229 (range 2,727–5,719). For comparison, the residual norm at layer 37 is ~47k.
- **Steering check.** Injecting on all tokens at α = 2 and α = 4 clearly steers free text toward the concept ("Elephants", "Volcanoes"). At α = 8 the output degenerates into loops. See [results/01_vectors_and_steering_check.txt](results/01_vectors_and_steering_check.txt). The pair list in that file is from the superseded raw-cos selection.
- **Stimuli.** 16 concrete concepts picked from 53 Macar candidates to minimize |cos|, with at most 2 per category. They form 8 pairs whose within-pair |cos| is ≤ 0.10. Each concept has 3 distractors that are near-orthogonal to both members of its pair. The selection is in [results/01b_selection.txt](results/01b_selection.txt) and the sentences in [data/stimuli.json](data/stimuli.json).
- **Run.** 16 ordered trials × 9 α × 2 option orders × 3 question cells = 864 forward passes, taking 80 s.

## Deviations from the plan

| Change | Why |
|---|---|
| Concepts chosen by **\|cos\|**, not raw cos | The first selection picked strongly *negative* pairs (Trains/Cacti, −0.60). With negative cosine, injecting X also suppresses Y, which confounds ρ. |
| Hand exclusions for 7 distractors (e.g. Silk–Spiders, Cacti–Deserts) | These are near-orthogonal as vectors but close in meaning, so they would compete as an "about" answer. |
| α grid {0, 1, 2, 4, 8} **plus 0.125, 0.25, 0.5, 16** | The dry run showed every planned α > 0 lands at ρ ≫ 1 (see below). The lower values aim at ρ < 1. α = 16 tests whether Gate 1 is reachable at all. Extra values are marked `*` in the plots and have `planned_alpha=False` in the data. |
| One full forward pass per question instead of a branched prefill | Gives the same result: attention is causal and the injection stops at the end of sentence 3. |
| Options listed one per line | Easier to read; the wording is otherwise unchanged. |
| Added P_inj(Y), log P(X), free-text spot checks, and `04_diagnostics.py` | To make sense of a null result. |

## Calibration findings (these matter for the next design)

1. **The "evoked level" is small and unreliable when measured as a sentence-mean projection.**
   - At layer 43, a sentence that names concept C raises the projection onto v̂_C by a median of ~880 over the neutral filler. That is about 20% of |v|.
   - For 3 of 16 concepts the gap is unusable: Islands 23, Rainbows 28, and Cacti −705.
   - On the concept-word tokens alone the gap is larger (median ~1,700) but noisy: Skulls −341, Cameras 170. The largest per-token values often fall on unrelated tokens ("the", ".", "all").
   - So ρ = s_X / s_Y from sentence-mean projections is noisy for most concepts and undefined for those three. The ρ bins in the main plot should be read loosely. Since nothing moves in any bin, this doesn't change the conclusion.
2. **Even the smallest α saturates s.** α = 1 already gives s_X ≈ 20 at layer 43, and α = 0.125 gives s_X ≈ 2.2. The region the plan cares about (ρ < 1) needs α ≲ 0.1 under this normalization. 66 main-test passes fall in 0.1 ≤ ρ < 1, mostly from the low added α values. P_inj(X) stays at baseline there too.
3. **Forced-choice answers are nearly one-hot.**
   - 93% of passes put more than 0.9 on a single letter, and letter mass is 1.00 in every pass of the main run.
   - At α = 0, the Gate 1 answer depends on option position: mostly A or B, and it switches with the order permutation.
   - So P(X) is effectively 0/1 per trial and is insensitive to small shifts. log P(X) is flat too: −5.14 at α = 0 vs. −4.92 at α = 16.
4. **No incoherence from injecting on sentence 3 only, even at α = 16**, where the added vector is ~1.4× the residual norm. The quality filter removed nothing. Free-text answers are unchanged across α. Gate 1 answers name the filler's own content ("file", "paper").

## Suggested next steps (per decision rule 1: change the elicitation)

- **Let the answer read from the injected positions.** In the span control, the read-out works once the injection reaches the question and answer tokens. That breaks the test's logic, which needs the anomaly located at sentence 3. Options that keep it located:
  - Inject at an **earlier layer** (the early-layer arm we cut), which leaves more layers for attention to carry X forward.
  - Inject over a **longer span**, e.g. repeat the sentence or use a whole paragraph as "sentence 3".
  - Add a **primer**, as in Latent Introspection, that tells the model what kind of signal to look for.
- **Redefine the strength scale before calibrating ρ.** Measure evoked levels at the concept-word tokens, or with a probe trained on sentence activations instead of the "Tell me about" vector. Alternatively, report injection strength in units of |v| and drop the sentence-based normalization.
- **Reduce letter saturation** so small effects show up. Use the logit difference between X and the other options, a softer read-out (e.g. temperature), or more option orders to average out position bias.

## Reproduce

See [README.md](README.md).

## Follow-up tests

Scripts: [05_followups.py](scripts/05_followups.py) with [06_analyze_followups.py](scripts/06_analyze_followups.py), and [07_main_test_sent.py](scripts/07_main_test_sent.py) with [08_analyze_main_sent.py](scripts/08_analyze_main_sent.py). Tables: [results/followup/summary.md](results/followup/summary.md) and [results/main_sent/summary.md](results/main_sent/summary.md).

**What changed.**
- **Injection layers:** 12, 20, 28 and 37.
- **Two vector types:**
  - the Macar "Tell me about X" vector;
  - a **sentence-derived** vector: the mean activation over sentence-3 tokens of 3 held-out sentences about X, minus the same for 8 neutral sentences, in the same 5-sentence context. Sentences are in [data/vector_sentences.json](data/vector_sentences.json).
- **Strength:** set as a fraction c of the residual norm on those tokens. For the sentence vectors, c ≈ 0.05 (layers 20–37) equals the push of a real sentence.
- **Read-outs:** 5 cyclic option orders, one-word answers (no letters), detection (yes/no), and localization (which sentence, 0–5).

### 1. The pilot's null came from the vector type and layer, not from the model

- **The two vector types point in different directions.** The cosine between the Macar and sentence-derived vectors is only 0.07–0.20 at every layer. The pilot injected something unlike what a real sentence writes into those tokens.
- **Sentence-derived vectors at layer 20 are read out almost perfectly** from a neutral sentence 3:
  - "What is sentence 3 about?": P(X) = 0.98 at c = 0.1–0.2, i.e. 2–4× a real sentence.
  - "What thought was injected at sentence 3?": P(X) = 0.91–0.97.
  - Free text: "Sentence 3 is about beautiful fountains and gardens in a city."
- **Other layers:** layer 12 needs c ≥ 0.4, layer 28 gives a partial effect, and layer 37 shows nothing in the letters (one-word answers show it at c = 0.2–0.4).
- **Macar vectors are barely read from sentence 3 at any layer.** The best is about 0.27 at layer 20.

![Follow-up letter read-out](results/followup/followup_letter.png)

### 2. When the injection gets through, it is read as the sentence's content

- **The questions behave alike.** "About" and "injected" answers rise and fall together in every cell.
- **Detection doesn't work.** The model answers "No, no thought was injected" with P(Yes) ≈ 0 in every condition, including those where it names X with P ≈ 1 when asked *what* was injected. The yes/no log-odds move by at most about 1–5 nats from a −20 baseline.
- **Localization works only where retrieval works** (sentence-derived or Macar vectors at layer 20, c ≥ 0.2; P(correct) = 0.8–0.9).
  - The model's default answer is "4" (P = 0.997 with no injection).
  - With injection it picks the injected sentence. This fits content-based oddity detection: a sentence that now reads as "about bees" stands out among office sentences.
- **Anecdote (4 concepts, greedy decoding):**
  - At layer 28, injection on sentence 3 makes the model describe sentence 3 as sentence 2's content ("someone writing the date on a form").
  - It then adds, unprompted, "Note: This is likely due to the injected thought influencing my response".
  - It never added this note without injection. The preamble does tell it a thought may be injected, so this is not detection, but it could be worth a targeted test.

### 3. Main test rerun with sentence-derived vectors (now that Gate 1 passes)

Setup:
- Sentence 3 is a real sentence about Y, and X (its pair partner) is injected on its tokens.
- ρ = injected X strength / real-sentence strength, so ρ = 1 means "as strong as a real sentence about X".
- No injection: P_about(Y) = 1.00 and P_inj(Y) = 0.96.

| Layer 20 | ρ = 0.25 | 0.5 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|---|
| P_about(X) | 0.00 | 0.00 | 0.00 | 0.00 | 0.15 | 0.20 |
| P_inj(X) (no injection: 0.015) | 0.03 | 0.04 | 0.05 | 0.09 | 0.24 | 0.35 |
| Letter log-odds X:Y shift, injected − about (nats) | +0.4 | +0.7 | +1.5 | +3.4 | −0.1 | −0.7 |
| One-word log-odds X:Y shift, injected − about (nats) | −0.5 | −0.9 | −1.1 | −3.1 | −18 | −32 |

- **Letter read-out:** at ρ ≤ 2 the "injected" answer moves toward X more than the "about" answer. The paired CIs exclude 0, but the absolute gain is small: ΔP_inj(X) ≤ 0.035 at ρ ≤ 1, with the CI touching 0.
- **One-word read-out:** the **opposite** direction. "About" moves toward X more than "injected" does, significantly at several ρ.
- **Layer 28:** little movement at all.

**Conclusion:** there is no dissociation that holds across read-outs. The sign depends on how the answer is collected, which points to read-out mechanics, not to a separate "injected vs. real" representation. As far as these tests can see, an injection that the model can read is treated as part of the sentence's appearance.

![Main test with sentence-derived vectors](results/main_sent/main_sent.png)

### Revised take

- **Your "can't look back" hypothesis is not supported as stated.** The model can read injected content from earlier positions when the vector is in the format a sentence produces.
- **What it does not show is any trace that the content was injected rather than read.**
  - Detection is at floor.
  - Localization is explained by content.
  - "About" and "injected" answers don't separate reliably.
- For test #1, this is a negative result for anomaly-specific identification at this scale (one model, 16 concepts, 2 layers).

**Suggested next steps:**
1. Replicate the layer-20 letter-logit dissociation with more concepts, more held-out vector sentences, and a free-text judge before believing it.
2. Test the layer-28 "the injection disrupted my reading" anecdote directly: a yes/no detection question without the "may be injected" preamble, plus a no-injection control, over all 16 concepts.
3. Try the primer elicitation from the plan's decision rule 1.

# Preliminary findings: where concept identification comes from in the injected-thoughts protocol

**Status:** preliminary, 2026-10-08. One model (Gemma-3-27B-IT), one injection layer (37 of 62), 50 concepts. The method record and pass criteria are in [VALIDATION.md](VALIDATION.md).

## Glossary

Terms are Lindsey's (2025) unless marked. Quotes are his.

| Term | Definition here |
|---|---|
| **Concept injection** | "An application of activation steering": adding a concept vector to the residual stream at one layer, on a chosen set of token positions. |
| **Concept vector** | Residual-stream activation at the last prompt token of chat("Tell me about {word}"), minus the mean of the same over baseline words. Lindsey uses 50 random words; Macar et al. and we use their 100. Not normalized. |
| **Injection strength (α)** | Multiplier on the concept vector. Lindsey finds strengths 2 and 4 work best; Macar's Gemma optimum at layer 37 is 4. |
| **Injection layer** | The layer whose output gets the vector. Here 37 of 62 (≈60% depth), Macar's default for Gemma. |
| **Injection span** *(ours)* | The token positions that receive the vector. See the conditions table. |
| **Detection** | The model affirms that it notices an injected thought. Graded with Macar's version of Lindsey's criterion: the response must be coherent and must claim detection "prior to, or along with, mentioning the word". |
| **Identification** | Detection plus correctly naming the injected concept (or a closely related one). Same ordering rule. |
| **Hit rate / false-alarm rate** *(Macar)* | Detection rate on injection trials / on control (no-injection) trials. |
| **Identification given detection** *(Macar: "identification accuracy given claim")* | Share of detections that also identify the injected concept. |
| **Misidentification** *(ours; Lindsey has no term)* | Detection with a different concept named. Lindsey describes the same thing on control trials as the model "claims to detect a variety of other words". |
| **Internality** | Lindsey's criterion that the causal influence on the self-report be internal: it "should not route through the model's sampled outputs". The detection-before-the-word rule is his guard for it. |
| **Grounding** | Lindsey's criterion that the self-report "causally depend on the aspect that is being described". |
| **Metacognitive representation** | Lindsey's criterion that the report derive from an internal representation *of* the state, not a direct translation of the state into words. He says the paper offers only indirect evidence for it. |
| **"Brain damage"** | Lindsey's term for degraded output at high injection strength: garbled text, or the model consumed by the concept. |
| **Ten-word relatedness** | Lindsey's scoring for "distinguishing prior thoughts from text inputs": a grader picks which of ten words (the injected one plus nine random) the response is most related to. Chance is 10%. |

### Conditions (injection spans; all at layer 37, same prompt)

The prompt is Lindsey's: an explanation turn, then "Ok.", then "Trial n: Do you detect an injected thought? If so, what is the injected thought about?" The answer is sampled at temperature 1, up to 100 tokens.

| Condition | Injected positions | Generated tokens injected? |
|---|---|---|
| **standard** (Lindsey, Macar) | the newline before "Trial n", through the end of the prompt | yes |
| **prompt-only** | same prompt positions | no |
| **response-only** | none in the prompt (the first generated token is predicted from an uninjected state) | yes |
| **earlier-turn** | only the explanation turn and "Ok."; the question itself is not injected | no |
| **control** | none | no |

## What each paper claims

**Lindsey (2025), "Emergent Introspective Awareness in Large Language Models" (Claude models):**

- Opus 4/4.1 detect and identify injected concepts on "about 20% of trials" at the best layer, about two-thirds of the way through the model. Production models have 0 false positives in 100 control trials.
- The injection runs from the newline before "Trial 1" "throughout the Assistant's response". Internality is enforced only through the detection-before-the-word grading rule.
- **Thoughts vs. text:** with injection on a sentence's tokens only, models report the injected word when asked what word they think of, while still transcribing the sentence correctly.
- **Prefill:** injection on the earlier sentence tokens, before a prefilled word, makes the model accept that word as intended. Injecting on the follow-up question line instead "did not produce a similar effect". This behavior peaks "slightly over halfway" through the model.
- **Caveats he states himself:**
  - success on some prompt variants "can be accounted for by the model simply experiencing an impulse to talk about the injected concept";
  - "The most prosaic explanation... multiple different circuits, each of which supports a particular, narrow introspective capability".

**Macar et al. (2026), "Mechanisms of Introspective Awareness" (Gemma-3-27B-IT, Lindsey's protocol, injection through the response):**

- Detection is behaviorally robust, with 0% false positives across prompts.
- It emerges from contrastive preference training (DPO), not SFT, and is strongest in the Assistant persona.
- It is not reducible to a single linear direction.
- **"Detection and identification are handled by distinct mechanisms in different layers."** MLPs at ~70% depth are causally necessary and sufficient for detection. "Gate" features inhibit detection claims, and upstream "evidence carrier" features that respond to injected vectors suppress them.
- Introspective capacity is under-elicited: refusal abliteration raises detection from 10.8% to 63.8% (false positives 0% → 7.3%).
- **Reference numbers** at layer 37, α = 4: hit rate 0.411, false alarms 0.000, identification given detection 0.558.

**Pearson-Vogel et al. (2026), "Latent Introspection" (Qwen2.5-Coder-32B-Instruct, plus replications on Llama-3.3-70B and Qwen2.5-72B):**

- Concepts are injected at layers 21–42 while the *first* turn is processed. "No tokens are sampled with active steering." The read-out is P(yes) vs. P(no) after a prefilled "The answer is".
- Accurate framing with no document: near chance (≈50.1% balanced accuracy). With an accurate document about introspection: 39.9% sensitivity, ≈0.8% false positives.
- Logit lens: the detection signal emerges around layer 48, peaks at 58–62, and is attenuated in the final 2–3 layers.
- Identification among 9 concepts: up to 1.36 bits of mutual information.
- On Llama-3.3-70B the document lowers sensitivity, from 75.5% to 38.0%.

## Our results and how they line up

Setup for every run:

- 50 concepts (Lindsey's), 10 trial numbers × 2 samples = 1,000 trials per condition and strength.
- Graded with Macar et al.'s criteria; grader Claude Haiku 5.5 (Macar used gpt-4.1-mini).
- 95% CIs are bootstrapped over concepts.
- Data and per-run summaries are in `results/validation/`.

### Detection and identification by injection span and strength

Hit rates (detection on injection trials). False alarms on control trials: 0 of 1,000.

| α | standard | prompt-only | response-only | earlier-turn |
|---|---|---|---|---|
| 2 | 0.090 | 0.093 | — | — |
| 3 | 0.180 | 0.229 | 0.000 | — |
| 4 | 0.191 [0.12, 0.27] | 0.360 [0.27, 0.45] | 0.000 | 0.000 |
| 5 | 0.168 | 0.426 | — | — |
| 6 | 0.071 | 0.443 | — | — |

Identification given detection: standard 0.36–0.68 across α (0.64 at α = 4); prompt-only 0.000–0.002; response-only and earlier-turn: no detections.

### Relatedness of responses to the injected concept (ten-word test, injected vs. decoy; chance about 0.10)

| Responses | α = 4 | Pooled α = 2–6 |
|---|---|---|
| standard, detection claimed | 0.90 vs. 0.09 | — |
| standard, no detection claimed | 0.54 vs. 0.11 | — |
| prompt-only, detection claimed | 0.12 vs. 0.11 | 0.139 vs. 0.105; difference +0.034 [−0.018, +0.10] |
| response-only (none claim detection) | 0.52 vs. 0.14 | — |

The concepts named in prompt-only detection claims are dominated by a few words: "apple" in 45% of claims, given for 44 of the 50 injected concepts; then "ephemeral", "ocean", "umbrella", "bicycle".

### How these line up with the literature

| Claim | Source | Our result | Relation |
|---|---|---|---|
| 0 false positives on control trials | Lindsey; Macar | 0 of 1,000 | consistent |
| Hit rate 0.41 at layer 37, α = 4 | Macar | 0.19 | **not replicated**; grader and calibration not ruled out |
| Identification given detection 0.56 | Macar | 0.64 | consistent |
| Best strengths 2–4 (Lindsey), 4 (Macar) | Lindsey; Macar | standard peaks at α = 4; prompt-only rises and levels off by α = 5 | consistent for the standard span; the peak may partly reflect response-token injection degrading the output (untested) |
| Detection and identification use distinct mechanisms | Macar | prompt-only gives detection with no identification; response-only gives concept content with no detection | consistent, and the split follows the injection span |
| Identification counted with injection through the response; the detection-before-the-word rule guards internality | Lindsey | in this protocol (free-text answers) identification needs both spans; neither alone gives any. Not new as a condition (Pearson-Vogel inject without steering sampled tokens), but not previously run inside Lindsey's and Macar's free-text protocol | narrows the interpretation: consistent with detection plus the vector steering the response words, which Lindsey raises himself ("impulse to talk about the injected concept") |
| On control trials, models "claim to detect a variety of other words" | Lindsey | without response injection, named concepts are near chance and concentrated on a few defaults | consistent; extends his observation to injection trials |
| Injection on earlier sentence tokens is reportable (thoughts vs. text) | Lindsey | prompt-only: near chance for the injected-thought question; pilot: injected content reported as the sentence's content | open: a different question (content vs. injected thought), model, and "Answer immediately" |
| Detection from earlier-turn injection, 22 layers, document + logit read-out | Pearson-Vogel | earlier-turn, 1 layer, α = 4, sampled text: 0 of 1,000 | not comparable yet; their no-document condition was also near chance for detection |
| Identification without injection on any sampled token (forced choice among 9 concepts, logits): 0.61 bits without the document, 1.05 with it | Pearson-Vogel | prompt-only, free text: identification 0; named concepts near chance on ten-word relatedness | **in tension.** Candidate explanation: the read-out (forced-choice logits vs. free-text naming). The information may be present but not used in free-text injected-thought reports. Untested here |

### Limitations

- One model and one injection layer.
- The grader is not yet checked against hand labels.
- The stage 1 hit rate falls short of Macar's.
- In response-only, the first generated token (the yes/no) is predicted from an uninjected state, so this condition cannot show detection driven by response injection.
- No random-vector control yet, so prompt-only detection may not be concept-specific.

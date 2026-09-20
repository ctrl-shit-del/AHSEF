# AHSEF 50% Milestone — Phase D and Phase E

**Routing target selection and acquisition-budget curves, on validation only.**

| | |
|---|---|
| Experiment | `ahsef_50pct_milestone_v1` |
| Protocol | `ahsef.milestone.phase_de.v1` |
| Split | validation only — the locked 482-sample test partition was not opened |
| Pool | 509 aligned Text+Audio validation samples |
| Pool fingerprint | `e3ff7bdd4fbd34d609e80bc899c91349a8298064e958ee72362064503c99c605` |
| Seed | 42 |
| Artefacts | `experiments/ahsef/milestone/phase_de_results.json`, `frozen_routing_policy.json`, `estimators/*.json` |
| Driver | `python -m src.ahsef.cli.run_milestone_phase_de --stage all` |
| Verdict | **MATERIAL ROUTING SUCCESS** — proceed to locked evaluation |

---

## 1. Objective

Phase C established that Gemma + strong Audio fusion is worth having on this pool:
macro-F1 rises from 0.3088 (Gemma alone) to 0.4086 (fusion), a gain of +0.0997.
Phase D and Phase E ask a different question, and the difference matters:

> Can AHSEF obtain most of that benefit while activating the audio expert on a
> small minority of samples?

The objective is **not** to beat always-fusion. Always-fusion pays 289 ms of
wav2vec2 on every request. The claim under test is that most of its benefit
survives paying on few — and that the samples worth paying for can be identified
without a label.

Two sub-questions, answered separately because they have different answers:

- **Phase D.** Which per-sample supervision signal should a router be trained to
  predict? Stage 3 regressed `gain(m|A) = P(correct|A ∪ {m}) − P(correct|A)` and
  reported macro-F1 — two quantities that need not agree. Five candidate targets
  are compared here under one estimator family, so a difference between rows is a
  difference between *targets* rather than between experiments.
- **Phase E.** Does the resulting policy beat random acquisition **at a matched
  acquisition rate**, across a budget grid rather than at one lucky threshold?

---

## 2. Experimental protocol

### Inputs — all stored, nothing re-scored

| Artefact | Role |
|---|---|
| `stage3_text_audio/predictions/text_llm__validation.parquet` | Gemma predictions (n=509 usable) |
| `analysis/audio_expert/audio_strong__validation.parquet` | strong-audio predictions |
| `stage3_text_audio/hsig/features_validation.parquet` | the label-free evidence features (10 evidence + 7 class indicators) |
| `analysis/audio_expert/phase_c_aligned.json` | frozen fusion spec |
| `analysis/audio_expert/phase_b1_comparison.json` | frozen audio cost |
| `audio_strong/features/validation/extraction_provenance.json` | encoder geometry |

No model was re-scored and no LLM call was made. The whole of Phase D/E replays
offline from these files.

### The pool

Rebuilt exactly as Phase C built it: `AlignmentIndex.fusion_pool(['audio','text'],
'validation')`, restricted to samples with usable Gemma evidence and cached audio
features. **n = 509.** Class support: neutral 172, happy 151, angry 95, sad 63,
surprise 15, disgust 7, fear 6.

### Fusion

The Phase C strong-audio spec was **read, not re-selected**:
`weighted_probability`, weights `{audio: 0.6, text_llm: 0.4}`, chosen on
validation by macro-F1 during Phase C. Re-selecting it here would have been a
second validation fit on the same pool the routing policy is chosen on, and the
two choices would then be entangled.

### Estimator family — identical for every target

Ridge regression on standardised features. One family, one feature set, one fold
assignment, one seed, and **one regularisation strength shared by all five
targets**. α was selected once, on `signed_gain` — the target Stage 3 froze — by
out-of-fold Spearman correlation, so no candidate received a strength tuned for
it. The sweep was nearly flat (ρ = 0.1822 / 0.1826 / 0.1859 / 0.1886 for
α = 0.1 / 1 / 10 / 100) and **α = 100** was selected.

Feature set: `evidence_only` (10 label-free features read off the Gemma response).
`assert_label_free` rejects any label-bearing column by name.

### Out-of-fold protocol

Every reported score is **out of fold**: seeded 5-fold, each sample scored by a
model that never saw it. An in-sample AUROC on 509 rows with ten features is
optimistic by an amount that is not small, and using in-sample scores in the
budget curve would flatter every policy by a margin that grows with the feature
count.

### Random control

At each budget, 200 seeded draws of exactly the same size as the policy's
acquisition set. A policy is credited with routing only where its macro-F1
exceeds the control's **97.5th percentile** at the **same** budget.

---

## 3. Candidate target definitions

All five were defined in closed form in `src/ahsef/milestone/targets.py` before
any of them was fitted.

| Target | Value for sample *i* | Can express harm? |
|---|---|---|
| `binary_correction` | 1 if text wrong and fused right, else 0 | no |
| `signed_gain` | `fused_correct − text_correct` ∈ {−1, 0, +1} | yes |
| `class_balanced_gain` | `signed_gain × w(true class)`, `w ∝ 1/count`, mean 1 | yes |
| `macro_f1_marginal` | `macroF1(fuse only i) − macroF1(fuse nothing)` | yes |
| `uncertainty_reduction` | `U(text) − U(fused)` — rejected by Stage 1, re-measured here | yes |

Plus the reference every fitted target must justify itself against:

| Signal | Definition |
|---|---|
| `uncertainty_only` | the frozen Stage 2/3 routing uncertainty, used **directly** as the score. No estimator, no target, no fitting. |

A target is never a feature. All five read the true label and are used only to
*fit* on validation; the router at inference sees none of them.

### Post-acquisition facts these are derived from

On the 509-sample pool: text accuracy 0.4715, fused accuracy 0.5619,
**64 corrections** (audio fixed a wrong text prediction) and **18 harms** (audio
broke a correct one). Base rate of a useful acquisition: **12.6%**.

---

## 4. Target distributions

| Target | mean | std | min | max | positive | negative | distinct | expresses harm |
|---|---|---|---|---|---|---|---|---|
| `binary_correction` | +0.1257 | 0.3316 | 0.0000 | 1.0000 | 64 | 0 | 2 | no |
| `signed_gain` | +0.0904 | 0.3911 | −1.0000 | +1.0000 | 64 | 18 | 3 | yes |
| `class_balanced_gain` | +0.0175 | 0.0925 | −0.2677 | +1.1244 | 64 | 18 | 10 | yes |
| `macro_f1_marginal` | +0.0002 | 0.0010 | −0.0042 | +0.0101 | 80 | 38 | 48 | yes |
| `uncertainty_reduction` | −0.3351 | 0.1718 | −0.7787 | +0.2077 | 16 | **493** | 509 | yes |

Two of these rows are worth reading closely.

**`macro_f1_marginal` has 80 positives and 38 negatives, not 64 and 18.** It is
non-zero on 118 samples where `signed_gain` is non-zero on 82. The 36 extra are
acquisitions that move a prediction from one *wrong* class to another: `signed_gain`
scores those exactly zero, while macro-F1 moves because precision shifts in both
classes. There are no outright sign flips between the two targets. Its magnitude is
tiny — one sample out of 509 moves macro-F1 by about 1e-3 — but only its ordering
is used.

**`uncertainty_reduction` is negative on 493 of 509 samples.** Fusing with audio
*increases* the predictive entropy of the posterior on 97% of the pool while
raising accuracy by 9 points. This is a direct re-measurement of the Stage 1
rejection of delta-uncertainty as a gain proxy, on a different expert and a
different pool, and it reproduces it: the sign of the entropy change carries
almost no information about whether acquisition helped.

### Where the corrections land — and a Stage 3 premise that no longer holds

| Class | Support | Corrections | Share of all corrections |
|---|---|---|---|
| angry | 95 | 26 | 40.6% |
| happy | 151 | 20 | 31.3% |
| neutral | 172 | 11 | 17.2% |
| sad | 63 | 5 | 7.8% |
| surprise | 15 | 2 | 3.1% |
| fear | 6 | 0 | 0.0% |
| disgust | 7 | 0 | 0.0% |

Phase D was designed around the hypothesis that Stage 3 failed because *most
corrections land in the majority classes*, so a router trained to count
corrections spends its budget where a macro average barely rewards it. Stage 3
recorded 62% of corrections in neutral and angry with the **baseline** audio
expert.

**With the strong expert that premise is substantially weaker.** Neutral now
accounts for only 17.2% of corrections, against 33.8% of the pool. The two
largest classes (neutral + happy) hold 63.5% of the pool but only 48.4% of the
corrections. Corrections are already *mildly minority-skewed*.

This matters for how the Phase D result should be read: `class_balanced_gain`
exists to correct a distortion that the strong expert has largely removed, and it
should be expected to add little. It adds nothing (§5). That is a coherent
explanation of a negative result, not an excuse for one.

---

## 5. Target quality comparison

All figures out of fold, on validation. `maj ×` is the majority-class
over-representation in the top-10% scored samples: the selected set's share of
neutral+happy divided by the pool's 63.5%. **Below 1.0 means the signal avoids
the majority classes**, which is what a macro average rewards.

| Target | AUROC | AUPRC | P@10% | R@10% | harm AUROC | ρ vs signed gain | ρ vs macro-F1 marginal | maj × |
|---|---|---|---|---|---|---|---|---|
| `uncertainty_reduction` | **0.7588** | **0.3216** | 0.333 | 0.266 | 0.370 | 0.2215 | 0.2150 | **0.587** |
| `uncertainty_only` | 0.7575 | 0.3073 | 0.333 | 0.266 | **0.381** | **0.2235** | **0.2153** | 0.618 |
| `binary_correction` | 0.7434 | 0.3133 | 0.333 | 0.266 | 0.366 | 0.2049 | 0.2062 | 0.680 |
| `macro_f1_marginal` | 0.7307 | 0.3071 | 0.333 | 0.266 | 0.358 | 0.1900 | 0.1942 | 0.618 |
| `signed_gain` | 0.7283 | 0.3132 | 0.333 | 0.266 | 0.361 | 0.1886 | 0.1824 | 0.773 |
| `class_balanced_gain` | 0.7130 | 0.3039 | 0.333 | 0.266 | 0.331 | 0.1648 | 0.1666 | 0.742 |

AUPRC is quoted against a **base rate of 0.126**, not against 0.5. Every
candidate roughly 2.4× the base rate — real, but the top decile is still
two-thirds acquisitions that buy nothing.

`harm AUROC` is reported as 1 − AUROC, so above 0.5 would mean the score gives
harmful acquisitions a *low* value. **Every candidate is well below 0.5** (0.33 to
0.38). None of them avoids harm; they all point *towards* the samples audio is
most likely to break, because those are the same uncertain samples audio is most
likely to fix. No candidate target separates the two.

### Fitted coefficients (standardised; comparable magnitudes)

| Target | Top four features by \|weight\| |
|---|---|
| `binary_correction` | `score_mass_top2` −0.0422, `score_top1` −0.0300, `score_margin` −0.0231, `llm_ambiguity` +0.0144 |
| `signed_gain` | `score_mass_top2` −0.0418, `score_top1` −0.0232, `llm_ambiguity` +0.0155, `score_margin` −0.0142 |
| `class_balanced_gain` | `score_mass_top2` −0.0045, `score_top1` −0.0043, `score_margin` −0.0039, `score_top2` +0.0028 |
| `macro_f1_marginal` | `score_mass_top2` −0.0001, `score_top1` −0.0001, `score_margin` −0.0001, `llm_confidence` +0.0000 |
| `uncertainty_reduction` | `uncertainty` +0.0321, `score_entropy` +0.0321, `score_mass_top2` −0.0288, `score_top1` −0.0173 |

**Every fitted target learns the same direction.** The dominant weight is always a
negative coefficient on `score_mass_top2` — low mass on the top two classes, i.e.
a diffuse posterior, i.e. *high uncertainty*. `uncertainty_reduction` says it most
plainly by loading directly on `uncertainty` itself.

Calibration is reported in the JSON artefact and is deliberately **not** part of
the selection rule: the policy acquires top-k, so only the ordering of the score
is used and a constant offset would not change a single decision.

---

## 6. Routing-signal comparison — against uncertainty alone

Deltas against the plain `uncertainty_only` ranking:

| Target | ΔAUROC | ΔAUPRC | Δρ vs signed gain | Δρ vs macro-F1 marginal |
|---|---|---|---|---|
| `uncertainty_reduction` | **+0.0013** | **+0.0143** | −0.0020 | −0.0002 |
| `binary_correction` | −0.0142 | +0.0060 | −0.0186 | −0.0091 |
| `macro_f1_marginal` | −0.0269 | −0.0002 | −0.0335 | −0.0211 |
| `signed_gain` | −0.0292 | +0.0059 | −0.0350 | −0.0329 |
| `class_balanced_gain` | −0.0446 | −0.0034 | −0.0587 | −0.0487 |

**Not one fitted target beats the unfitted uncertainty signal on rank agreement.**
`uncertainty_reduction` matches it (+0.0013 AUROC, −0.0020 ρ) — and it is
essentially uncertainty wearing a different name, since its own coefficients load
on `uncertainty` directly. The four targets that actually encode routing value
— corrections, signed gain, class-balanced gain, macro-F1 marginal — all fall
*below* the one-feature signal they were meant to improve on.

**Is uncertainty alone sufficient? On this corpus, yes.** That is the answer to
the question Phase D was posed. `class_balanced_gain` is not better; it is the
worst of the six. `macro_f1_marginal` is not better either, despite being the
only target that matches the reported metric exactly. Both findings are
consistent with §4: the correction distribution the strong expert produces is
already close to class-balanced, so re-weighting it adds variance without adding
signal, on 509 samples with 64 positives.

---

## 7. Budget curves

Reference points on the 509-sample pool:

| | macro-F1 | accuracy | weighted-F1 |
|---|---|---|---|
| Text/Gemma only (0%) | 0.3088 | 0.4715 | 0.4398 |
| Always-fusion (100%) | 0.4086 | 0.5619 | 0.5415 |
| **Always-fusion gain** | **+0.0997** | +0.0904 | +0.1017 |

### Selected policy: `uncertainty_only`

| Budget | acquired | rate | macro-F1 | accuracy | weighted-F1 | Δ text | Δ random | gap to fusion | retained | gap to oracle | mods/sample | ms/sample |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0% | 0 | 0.000 | 0.3088 | 0.4715 | 0.4398 | +0.0000 | +0.0000 | 0.0997 | 0.0% | 0.0000 | 1.00 | 3449.7 |
| 5% | 25 | 0.049 | 0.3368 | 0.4892 | 0.4586 | +0.0280 | +0.0230 | 0.0717 | 28.1% | 0.0218 | 1.05 | 3463.9 |
| **10%** | **51** | **0.100** | **0.3603** | **0.4990** | **0.4705** | **+0.0515** | **+0.0412** | **0.0482** | **51.6%** | **0.0540** | **1.10** | **3478.7** |
| 15% | 76 | 0.149 | 0.3751 | 0.5108 | 0.4844 | +0.0662 | +0.0511 | 0.0335 | 66.4% | 0.0607 | 1.15 | 3492.9 |
| 20% | 102 | 0.200 | 0.3859 | 0.5285 | 0.5024 | +0.0771 | +0.0579 | 0.0226 | 77.3% | 0.0458 | 1.20 | 3507.7 |
| 25% | 127 | 0.250 | 0.3901 | 0.5363 | 0.5122 | +0.0812 | +0.0568 | 0.0185 | 81.5% | 0.0419 | 1.25 | 3521.9 |
| 50% | 254 | 0.499 | 0.3958 | 0.5462 | 0.5207 | +0.0870 | +0.0375 | 0.0128 | 87.2% | 0.0399 | 1.50 | 3594.0 |
| 75% | 382 | 0.751 | 0.4025 | 0.5560 | 0.5333 | +0.0937 | +0.0187 | 0.0060 | 94.0% | 0.0373 | 1.75 | 3666.7 |
| 100% | 509 | 1.000 | 0.4086 | 0.5619 | 0.5415 | +0.0997 | +0.0000 | 0.0000 | 100.0% | 0.0000 | 2.00 | 3738.8 |

Acquisition quality along the curve:

| Budget | corrections | harms | wasted (no change) | useful acquisition rate |
|---|---|---|---|---|
| 5% | 11 | 2 | 9 | **0.440** |
| 10% | 17 | 3 | 21 | 0.333 |
| 15% | 23 | 3 | 36 | 0.303 |
| 20% | 33 | 4 | 49 | 0.324 |
| 25% | 37 | 4 | 63 | 0.291 |
| 50% | 52 | 14 | 154 | 0.205 |
| 100% | 64 | 18 | 391 | 0.126 |

The useful acquisition rate falls monotonically from 0.440 at 5% to the 0.126
base rate at 100%. That decay **is** the routing signal: the first 25 samples the
policy picks are 3.5× more likely to be worth paying for than a sample drawn at
random. Above 25% the enrichment is largely spent, which is why the curve
flattens.

### Uncertainty before and after acquisition

Mean normalised entropy over the pool, before → after:

| Budget | 5% | 10% | 20% | 50% | 100% |
|---|---|---|---|---|---|
| before | 0.396 | 0.396 | 0.396 | 0.396 | 0.396 |
| after | 0.399 | 0.405 | 0.425 | 0.507 | 0.731 |

**Acquisition raises uncertainty at every budget while raising macro-F1 at every
budget.** The fused posterior is a weighted blend of two disagreeing experts and
is flatter than either. This is the Stage 1 delta-uncertainty finding, reproduced
on the strong expert: the entropy change and the accuracy change point in opposite
directions, so entropy reduction cannot be used as a routing reward.

---

## 8. Random baseline

200 seeded draws per interior budget, at exactly the policy's acquisition count.

| Budget | n | random mean macro-F1 | 95% interval | selected policy | above p97.5? |
|---|---|---|---|---|---|
| 5% | 25 | 0.3138 | [0.3064, 0.3228] | **0.3368** | ✅ |
| 10% | 51 | 0.3191 | [0.3065, 0.3365] | **0.3603** | ✅ |
| 15% | 76 | 0.3240 | [0.3118, 0.3403] | **0.3751** | ✅ |
| 20% | 102 | 0.3281 | [0.3147, 0.3493] | **0.3859** | ✅ |
| 25% | 127 | 0.3332 | [0.3147, 0.3544] | **0.3901** | ✅ |
| 50% | 254 | 0.3583 | [0.3368, 0.3779] | **0.3958** | ✅ |
| 75% | 382 | 0.3838 | [0.3652, 0.4038] | 0.4025 | ❌ |
| 0% / 100% | — | identical by construction | — | — | non-comparable |

The policy clears the control's upper bound at **every interior budget from 5% to
50%**, and falls inside it at 75% — as it must, since at 75% three quarters of the
pool has been acquired whichever way it was chosen.

This is the comparison Stage 3 failed. It was run here at nine budgets rather than
one, and the margin at 10% (+0.0412 over the random mean, +0.0238 over its 97.5th
percentile) is not a threshold artefact.

---

## 9. Oracle ceiling

Acquire where the true signed gain is largest. Unachievable by construction — it
reads the labels — and reported as the ceiling.

| Budget | oracle macro-F1 | selected policy | gap | fraction of oracle gain captured |
|---|---|---|---|---|
| 5% | 0.3586 | 0.3368 | 0.0218 | 56.2% |
| 10% | 0.4143 | 0.3603 | 0.0540 | 48.8% |
| 15% | 0.4358 | 0.3751 | 0.0607 | 52.2% |
| 20% | 0.4317 | 0.3859 | 0.0458 | 62.7% |
| 25% | 0.4320 | 0.3901 | 0.0419 | 66.0% |
| 50% | 0.4356 | 0.3958 | 0.0399 | 68.6% |
| 100% | 0.4086 | 0.4086 | 0.0000 | 100% |

The oracle **exceeds always-fusion** from 10% onwards, peaking at 0.4358 at 15% —
27% *above* the always-fusion gain. This is not a bug: 18 acquisitions actively
break a correct text prediction, and a policy that can decline those beats one
that cannot. It also sets the real ceiling honestly: there is roughly 0.05 macro-F1
of headroom between the deployed policy and a perfect one at 10–15% budget, and
closing it requires separating fixes from harms — which §5 shows no candidate
target does.

---

## 10. Always-fusion reference

| | acquisitions | macro-F1 | accuracy | mods/sample | ms/sample |
|---|---|---|---|---|---|
| Always-fusion | 509 | 0.4086 | 0.5619 | 2.00 | 3738.8 |
| **Selected policy @10%** | **51** | **0.3603** | **0.4990** | **1.10** | **3478.7** |
| Saving | −458 (−90%) | −0.0482 | −0.0629 | −0.90 | −260.1 |

The policy retains **51.6% of the always-fusion macro-F1 gain for 10.0% of the
audio acquisitions**. It does not beat always-fusion, and beating it was never the
bar: always-fusion is what AHSEF exists to avoid paying for.

At 20% the policy retains 77.3% of the gain for 20% of the acquisitions; at 25%,
81.5% for a quarter. The knee of the curve is between 15% and 25%.

---

## 11. Per-class analysis

Per-class F1 for the selected policy. **The question is whether selective
acquisition raises the minority classes or only polishes neutral.**

| Budget | neutral (172) | happy (151) | sad (63) | angry (95) | fear (6) | disgust (7) | surprise (15) | macro-F1 |
|---|---|---|---|---|---|---|---|---|
| text-only | 0.5959 | 0.4279 | 0.1538 | 0.4528 | 0.2667 | 0.1905 | 0.0741 | 0.3088 |
| 5% | 0.6045 | 0.4545 | 0.1935 | 0.4615 | 0.3636 | 0.2000 | 0.0800 | 0.3368 |
| 10% | 0.6032 | 0.4753 | 0.1957 | 0.4774 | 0.4000 | 0.2105 | 0.1600 | 0.3603 |
| 20% | 0.6154 | 0.5110 | 0.2326 | 0.5443 | 0.4000 | 0.2500 | 0.1481 | 0.3859 |
| 50% | 0.6303 | 0.5047 | 0.2078 | 0.6424 | 0.4000 | 0.2353 | 0.1500 | 0.3958 |
| 100% | 0.6373 | 0.5388 | 0.2278 | 0.6743 | 0.4000 | 0.2353 | 0.1463 | 0.4086 |

**Every class improves, and the largest relative gains are in the minority
classes.** From text-only to the 10% operating point: neutral +0.007 (+1%),
happy +0.047 (+11%), sad +0.042 (+27%), angry +0.025 (+5%), fear +0.133 (+50%),
disgust +0.020 (+11%), surprise +0.086 (+116%).

Neutral — the class Stage 2 warned about — moves least of all in absolute terms
(+0.0073). The summed F1 over the five classes outside the two largest rises from
1.1379 to 1.4436, **+0.3057**. The macro-F1 gain is not majority-class polish.

### Two honest caveats

**Fear and disgust have 6 and 7 samples.** Their F1 columns move in steps of
roughly 0.13 and one sample changes the number visibly. `fear` reaching 0.4000 at
10% and staying there is one correct prediction, not a trend. Read those two
columns as anecdote.

**The margin between candidate targets at 10% is one rare-class sample.** All six
candidates acquire the *same number* of corrections at 10% — exactly 17 — while
selecting genuinely different sample sets (pairwise overlap 25–44 of 51). Their
macro-F1 differs only in *which* classes those 17 corrections land in, and
`uncertainty_only` leads partly because it catches one `surprise` correction
(support 15) that `signed_gain`, `class_balanced_gain` and `macro_f1_marginal`
miss. A macro-F1 ordering that turns on one or two rare-class samples out of 509
is fragile, and the ordering *between candidates* should not be over-read. What is
robust is the finding that no candidate separates itself from the uncertainty
reference — which is the conclusion drawn.

---

## 12. Cost and latency analysis

The frozen cost configuration was **read, never re-measured on the evaluation
pool**. Fingerprint `e3534119a7c9ba50928f981b16a094b11ed074a3749466b64101bda96bb4e69a`.

| Modality | latency ms/sample | compute units | source |
|---|---|---|---|
| `text_llm` (Gemma) | 3449.73 | 2.240e13 | Stage 3 frozen config — measured wall clock, not generation time |
| `audio_strong` | **289.06** | 6.042e12 | Phase B-1 — **encoder + probe head** |

The strong-audio price decomposes as:

| Component | ms/sample | parameters | input elements |
|---|---|---|---|
| WAV2VEC2_BASE encoder (frozen) | 288.88 | 94,370,944 | 64,000 (16 kHz × 4.0 s) |
| LayerWeightedProbe head (trainable) | 0.18 | 266,003 | 9,216 (12 layers × 768) |
| **Deployment total** | **289.06** | 94,636,947 | — |

`assert_encoder_is_charged` refuses any cost model that prices audio below
100 ms/sample or at or below its head-only cost. The measured probe forward pass
is 0.18 ms because the encoder ran hours earlier during feature extraction;
reporting that as the acquisition cost would understate it by a factor of ~1,600
and would make every budget on the curve look free.

### Incremental cost per macro-F1 point

Acquisition latency spent per unit of macro-F1 gained over text-only:

| Budget | acquisition ms/sample | Δ macro-F1 | ms per macro-F1 point | vs oracle |
|---|---|---|---|---|
| 5% | 14.2 | +0.0280 | **507** | 285 |
| 10% | 29.0 | +0.0515 | **563** | 275 |
| 15% | 43.2 | +0.0662 | 652 | 340 |
| 20% | 58.0 | +0.0771 | 751 | 471 |
| 25% | 72.2 | +0.0812 | 888 | 586 |
| 50% | 144.3 | +0.0870 | 1,659 | 1,137 |
| 100% | 289.1 | +0.0997 | 2,898 | 2,898 |

**Cost-effectiveness peaks at 5–10% and degrades monotonically thereafter.**
Always-fusion is 5.1× less efficient per macro-F1 point than the 10% policy. In
end-to-end terms the audio expert is 8.4% of the Gemma call, so the 10% policy
costs 0.84% more wall clock than text-only for +0.0515 macro-F1.

---

## 13. Ablations

| Ablation | mean macro-F1 over 5–25% | macro-F1 @10% | retained gain @10% | budgets above random | admissible |
|---|---|---|---|---|---|
| **A. `uncertainty_only`** | 0.3696 | 0.3603 | 51.6% | 5–50% | ✅ |
| B. `binary_correction` | 0.3671 | 0.3497 | 41.0% | 5–50% | ✅ |
| C. `signed_gain` | 0.3600 | 0.3474 | 38.7% | 5–50% | ✅ |
| D. `class_balanced_gain` | 0.3523 | 0.3389 | 30.2% | 5–50% | ✅ |
| E. `macro_f1_marginal` | 0.3614 | 0.3457 | 37.0% | 5–50% | ✅ |
| F. `uncertainty_reduction` | **0.3717** | 0.3658 | 57.1% | 5–50% | ✅ |
| — `stage3_hsig_reference` (historical) | 0.3689 | 0.3603 | 51.6% | 5–50% | reference only |
| — `oracle` (ceiling) | 0.4145 | 0.4143 | 105.7% | 5–75% | not a candidate |

Ranking: `uncertainty_reduction` > `uncertainty_only` > `binary_correction` >
`macro_f1_marginal` > `signed_gain` > `class_balanced_gain`.

**Every candidate is admissible** — all six beat matched random from 5% to 50%.
The spread between best and worst is 0.0194 mean macro-F1, and the spread between
the top two is 0.0021.

### The Stage 3 finding, reproduced twice

**Reproduction 1 — richer features do not help.** Holding the target fixed at
`signed_gain` and α at 100, varying only the feature set:

| Feature set | features | AUROC | ρ vs signed gain |
|---|---|---|---|
| `uncertainty_only` | 1 | 0.7240 | 0.1871 |
| `evidence_only` | 10 | 0.7283 | 0.1886 |
| `evidence_plus_class` | 17 | **0.7031** | 0.1768 |

Ten features buy +0.0043 AUROC over one. Seventeen features are *worse* than one.
Stage 3 selected `uncertainty_only` for exactly this reason and the strong audio
expert has not changed the answer — so that finding was about the evidence
available from the LLM response, not about the weakness of the expert it was
routing for.

**Reproduction 2 — the Stage 3 HSIG *is* the uncertainty gate.** The frozen Stage 3
`paired_logistic` model, replayed on this pool, produces a curve identical to the
plain uncertainty ranking at seven of nine budgets and differs by 0.0012 and
0.0025 at the other two. It is a monotone function of uncertainty and selects
essentially the same samples. (This replay is *in sample* — the Stage 3 model was
fitted on these 509 samples — so it is a historical reference and was excluded
from candidate selection.)

**The combined answer to Phase D's central question: uncertainty alone is
sufficient on this corpus. No richer target and no richer feature set improves on
it, and the two that come closest do so by loading on uncertainty directly.**

---

## 14. Selected target

### The selection rule (declared in `selection.py` before any target was fitted)

1. **Admissibility gate** — the policy must exceed the matched-budget random
   control's 97.5th percentile macro-F1 at ≥1 interior budget.
2. **Primary criterion** — mean validation macro-F1 over the low-budget region
   {5%, 10%, 15%, 20%, 25%}, from out-of-fold scores. Macro-F1, not accuracy;
   low budgets, because that is where policies differ; averaged over five points,
   because one budget cannot distinguish an ordering from a lucky threshold.
3. **Tie-break 1 (parsimony)** — within 0.005 macro-F1, the *simpler* signal wins.
4. **Tie-break 2** — higher retained fraction of the always-fusion gain at 10%.
5. **Tie-break 3** — alphabetical, so the rule is total.

### Applied

`uncertainty_reduction` led on the primary criterion at 0.3717 against
`uncertainty_only` at 0.3696 — a margin of **+0.0021**, inside the declared 0.005
tie band. **Tie-break 1 fired and selected `uncertainty_only`.**

> **SELECTED TARGET: `uncertainty_only`** — no fitted estimator. The routing score
> is the frozen Stage 2/3 routing uncertainty (normalised score entropy of the
> Gemma response), used directly.

The parsimony rule was written to encode the Stage 3 lesson, and it fired exactly
as intended. `uncertainty_reduction` is a ridge model whose largest coefficient is
on `uncertainty` itself; shipping it as an improvement over the uncertainty gate
would be shipping the gate with extra machinery and a better-sounding name. The
0.0021 margin on 509 samples is not evidence that the machinery earns its keep.

Not selected because they produced the best test result — **there is no test
result**. The rule was committed to code before the driver was first run.

---

## 15. Frozen routing policy

Written to `experiments/ahsef/milestone/frozen_routing_policy.json`.

```
score(x)   = normalised score entropy of the Gemma response for x
             (the frozen Stage 2 uncertainty policy: score_entropy)
acquire(x) = x is among the top-k samples by score(x)
k          = round(0.10 × pool size)
tie-break  = index order (deterministic)
predict(x) = fuse(text_llm, audio_strong) if acquire(x) else text_llm
fusion     = weighted_probability, {audio: 0.6, text_llm: 0.4}
```

| Field | Value |
|---|---|
| Selected target | `uncertainty_only` |
| Requires a fitted estimator | **no** |
| Acquisition budget | **10%** (51 of 509) |
| Budget rule | smallest budget meeting every MATERIAL_ROUTING_SUCCESS condition |
| Text expert | `gemma4:31b-cloud`, Stage 2 frozen prompt and decoding |
| Audio expert | `audio_strong` — frozen WAV2VEC2_BASE + LayerWeightedProbe |
| Fusion | `weighted_probability`, `{audio: 0.6, text_llm: 0.4}` (Phase C, reused) |
| Seeds | estimator 42, random control 42 |
| Cost fingerprint | `e3534119a7c9ba50928f981b16a094b11ed074a3749466b64101bda96bb4e69a` |
| Pool fingerprint | `e3ff7bdd4fbd34d609e80bc899c91349a8298064e958ee72362064503c99c605` |

**Validation performance at the frozen operating point:**

| | value |
|---|---|
| Acquisitions | 51 / 509 (10.02%) |
| Macro-F1 | **0.3603** (text-only 0.3088, always-fusion 0.4086) |
| Accuracy | 0.4990 |
| Weighted-F1 | 0.4705 |
| Modalities per sample | 1.10 |
| Latency | 3478.7 ms/sample (text-only 3449.7, always-fusion 3738.8) |

A selected policy that needs no estimator is a real simplification, not a
disappointment: the deployed router is one threshold on a number the LLM already
produces, with no second model to version, serialise or drift. The five fitted
estimators are still serialised under `experiments/ahsef/milestone/estimators/`
so the comparison is auditable and replayable.

---

## 16. PROCEED / DO-NOT-PROCEED decision

`MATERIAL_ROUTING_SUCCESS` was declared before any result was computed. All four
conditions must hold; beating always-fusion is explicitly not required.

| Condition | Required | Measured | Met |
|---|---|---|---|
| **1.** Beats matched-budget random on validation macro-F1, above its 97.5th percentile | > 0.3365 | **0.3603** (random mean 0.3191, +0.0412) | ✅ |
| **2.** Not explained solely by majority-class corrections — summed F1 outside neutral/happy must rise | > 1.1379 | **1.4436** (+0.3057) | ✅ |
| **3.** Retains ≥ 50% of the always-fusion macro-F1 gain | ≥ 0.0499 | **+0.0515** (51.6% of +0.0997) | ✅ |
| **4.** Uses ≤ 50% of always-fusion's acquisitions | ≤ 255 | **51** (10.0%) | ✅ |

> # VERDICT: MATERIAL ROUTING SUCCESS
> **PROCEED** — freeze this policy and open the locked test split in Phase F.

The intended result was obtained:

```
Text/Gemma  (macro-F1 0.3088, 1.00 modalities/sample)
      ↓  selective Audio acquisition on 10% of samples, chosen by uncertainty
Routed      (macro-F1 0.3603, 1.10 modalities/sample)
      ↓  compared against
Always-fusion (macro-F1 0.4086, 2.00 modalities/sample)

51.6% of the fusion benefit, for 10.0% of the modality activations.
```

### What this result does *not* say

- **It is not a generalisation claim.** Every number above is validation. The
  policy, its budget and its threshold were all chosen on this pool, and the
  honest expectation is that a locked test measurement comes in lower. Phase F
  exists to measure that, once.
- **The routing signal is uncertainty, not a learned gain model.** Phase D set out
  to find a better routing target than Stage 3's and found none. Reporting
  `uncertainty_only` as the winner is reporting that the richer objective did not
  pay, which is the same class of finding Stage 3 produced about richer features.
- **No candidate avoids harm.** All six point towards the samples audio is likely
  to break as readily as those it is likely to fix (harm AUROC 0.33–0.38). The
  ~0.05 macro-F1 gap to the oracle at 10–15% is almost entirely this.
- **Fear (n=6) and disgust (n=7) carry no weight**, and the ordering *between*
  candidate targets turns on one or two rare-class samples (§11).

---

## 17. Test-lock verification

| Check | Result |
|---|---|
| `test_partition_opened` | **false** |
| `test_labels_used` | **false** |
| `test_split_read` | **false** |
| `thresholds_tuned_on_test` | **false** |
| Locked artefacts unchanged | **true** |

**Enforcement, not assertion.** `assert_validation_only()` rejects any input path
whose name matches a test-partition marker (`__test.`, `_test.`,
`test_predictions`, `oracle_test`, `decisions_test`, `traces_test`) and is called
on every path the driver opens. `assert_fit_split()` refuses to fit an estimator
on anything but `train` or `validation`. Both are unit-tested against the real
test-artefact filenames in this repository.

**Locked artefacts.** SHA-256 content digests over the sorted
`(relative path, file hash)` pairs of each locked root, taken before and after the
run and compared against a stored baseline. A renamed, deleted, edited or added
file all move the digest.

| Locked root | Files | Digest verified unchanged |
|---|---|---|
| `experiments/ahsef/stage1` | ✅ | ✅ |
| `experiments/ahsef/stage2_llm` | ✅ | ✅ |
| `experiments/ahsef/stage3_text_audio` | ✅ | ✅ |
| `experiments/audio_strong` | ✅ | ✅ |

Baseline: `experiments/ahsef/milestone/locked_artefacts_baseline.json`. All Phase
D/E output goes to `experiments/ahsef/milestone/`; nothing is written into any
locked root.

### Reproducibility record

Stored under `reproducibility` in `phase_de_results.json`:
dataset fingerprint, alignment fingerprint, per-model fingerprints (pool /
prediction / label hashes, experiment, iteration, checkpoint SHA), feature
configuration, estimator configuration, per-target fingerprints, seed, budget
grid, cost configuration, git revision, environment, and protocol version
`ahsef.milestone.phase_de.v1`.

Every input is a stored artefact. Re-running
`python -m src.ahsef.cli.run_milestone_phase_de --stage all` with seed 42
reproduces every number in this report offline, without re-scoring a model or
calling the LLM.

### Test coverage

`src/ahsef/tests/test_milestone_phase_de.py` (61 tests) plus the existing
`test_milestone_targets_budget.py` (25) and `test_milestone_estimator.py` (20)
cover: every target and its sign convention, class-balanced weighting, the
macro-F1 marginal calculation, budget matching across policies, random-control
reproducibility, the oracle ceiling, refusal of test-label access, frozen-cost
usage (including refusal of the cached-head latency), estimator
serialisation/reload, deterministic ranking under exact ties, and endpoint
handling at 0% and 100%.

**Full suite: 1042 passed.**

---

## Summary

| Question | Answer |
|---|---|
| Exact selected target | **`uncertainty_only`** — the frozen Gemma score entropy, used directly; no fitted estimator |
| Exact routing rule | acquire audio on the top-k samples by score entropy, ties broken by index; fuse `{audio: 0.6, text_llm: 0.4}` on acquired samples, text-only otherwise |
| Exact acquisition budget | **10%** — 51 of 509 samples |
| Validation performance | macro-F1 **0.3603** / accuracy 0.4990 / weighted-F1 0.4705, vs text-only 0.3088 and always-fusion 0.4086 |
| Beats matched random? | yes, at every interior budget from 5% to 50% (+0.0412 over the random mean at 10%) |
| Is uncertainty alone sufficient? | **yes** — no candidate target and no richer feature set improved on it |
| Ready for locked evaluation? | **yes** — policy frozen in `frozen_routing_policy.json`; test partition never opened |

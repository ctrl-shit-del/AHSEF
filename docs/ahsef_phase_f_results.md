# AHSEF 50% Milestone — Phase F

**Locked test evaluation of the frozen routing policy. One evaluation, no tuning.**

| | |
|---|---|
| Experiment | `ahsef_50pct_milestone_v1` |
| Protocol | `ahsef.milestone.phase_f.v1` |
| Frozen policy | `experiments/ahsef/milestone/frozen_routing_policy.json` |
| Policy SHA-256 | `b2f8ced01204940599092f0db46c0965ebbae0b7e4ff1b38303ae94fa2cb6f37` |
| Split | **test** — opened once, after all ten lock conditions passed |
| Pool | 482 aligned Text+Audio test samples (`c44ed7e0…`) |
| Artefacts | `phase_f_results.json`, `phase_f_routing_trace.jsonl`, `phase_f_predictions.parquet`, `phase_f_lock_checks.json` |
| **Verdict** | **B. PARTIAL GENERALIZATION** |

---

## 1. Executive Summary

The frozen policy was applied to the locked test split exactly as frozen: rank by
Gemma normalised score entropy, acquire audio on the top 10%, fuse at
`{audio: 0.6, text_llm: 0.4}` where acquired, text-only elsewhere. Nothing was
tuned, refit or reselected. 48 of 482 samples received audio.

**The routing signal transferred. The size of the benefit did not.**

| System | Accuracy | **Macro-F1** | Weighted-F1 |
|---|---|---|---|
| A. Gemma text only | 0.5041 | 0.3345 | 0.4823 |
| B. Audio strong only | 0.5083 | 0.3345 | 0.5403 |
| C. Always fusion | 0.6058 | **0.4167** | 0.5848 |
| D. Random acquisition @10% | 0.5145 | 0.3399 | 0.4914 |
| **E. AHSEF dynamic @10%** | **0.5207** | **0.3450** | **0.4951** |
| F. Oracle @10% *(label-aware)* | 0.6037 | 0.4515 | 0.5848 |

AHSEF beat matched random acquisition (+0.0051 macro-F1), beat text-only
(+0.0105), used 10.0% of always-fusion's audio activations, and improved the
minority classes (+0.0527 aggregate F1). Its acquisitions were better than
random ones: 22.9% correction precision against random's 16.7%.

But it retained only **12.7% of the always-fusion macro-F1 gain**, against 51.6%
on validation, and **every paired bootstrap interval contains zero**. The
direction of every comparison is right; none of the margins is separable from
noise at n=482.

The mechanism of the shortfall is identifiable and is documented in §11 and §19:
on test, 29 of the 66 available corrections lie in `angry`, where the audio expert
is far stronger than Gemma (F1 0.6919 vs 0.3724) — and Gemma is *confident* on
most of them. The uncertainty ranking, which can only see Gemma's own doubt,
caught 2 of those 29.

This is classified **B. PARTIAL GENERALIZATION** under the rule declared before
the split was opened. The policy was not changed after seeing this result.

---

## 2. Frozen Policy

Applied verbatim from `frozen_routing_policy.json` (SHA-256
`b2f8ced01204940599092f0db46c0965ebbae0b7e4ff1b38303ae94fa2cb6f37`, frozen
2026-08-31T10:55:05, before the test split was opened).

```
score(x)   = normalised score entropy of the Gemma response for x
acquire(x) = x is among the top-k samples by score(x)
k          = round(0.10 × 482) = 48
tie-break  = index order (deterministic)
predict(x) = fuse(text_llm, audio_strong) if acquire(x) else text_llm
fusion     = weighted_probability, {audio: 0.6, text_llm: 0.4}
```

| Component | Frozen value | Verified on test |
|---|---|---|
| Anchor modality | Text | ✅ |
| Text expert | `gemma4:31b-cloud` via Ollama | ✅ replayed from Stage 3 transcript |
| Prompt | `v2_explicit_json`, sha `a6e8e7d02edb101f…` | ✅ identical to validation |
| Audio expert | `audio_strong` — frozen WAV2VEC2_BASE + LayerWeightedProbe | ✅ |
| Probe checkpoint | `306c3f745709d73ff566e0b2d37ca498f7937142035c8b9ccf9ed97944c9f56d` | ✅ identical to validation |
| Encoder config | `dc1543bfda550c28c8655cfccb45b7dff3b9f5498af3f58e8ffd5eccbc75bd48` | ✅ identical to validation |
| Uncertainty | Gemma normalised score entropy (`score_entropy`) | ✅ not recalibrated |
| Routing target | `uncertainty_only` — no estimator | ✅ |
| Budget | 10% | ✅ 48/482 = 9.96% |
| Fusion weights | audio 0.6, text 0.4 | ✅ read, not reselected |
| Seed | 42 | ✅ |
| Cost model | audio 289.06 ms, text 3449.73 ms | ✅ read, not re-measured |

`frozen_policy_applied.modified_by_phase_f = false`.

---

## 3. Locked Test Protocol

All ten conditions were run **before any test payload was read**
(`test_payload_read: false`, `test_labels_read: false` in
`phase_f_lock_checks.json`), and all ten passed.

| # | Condition | Result | Evidence |
|---|---|---|---|
| 1 | Frozen policy exists | ✅ | `experiments/ahsef/milestone/frozen_routing_policy.json` |
| 2 | Fingerprint recorded | ✅ | `b2f8ced0…`, frozen 2026-08-31T10:55:05 |
| 3 | Validation configuration frozen | ✅ | policy target/budget/weights match `phase_de_results.json` exactly |
| 4 | No prior test-label use | ✅ | 12 declarations, all `false`, each written by the phase that made the decision |
| 5 | Validation and test disjoint | ✅ | 509 ∩ 482 = **0** overlapping ids |
| 6 | Test manifest matches lock | ✅ | pool `c44ed7e0…` = Stage 3 recorded; manifest `cec265fe…`, 5542 samples |
| 7 | Alignment valid | ✅ | 482 unique ids, both modalities, `labels_used_for_inclusion: false` |
| 8 | No contaminated ids | ✅ | 0 cross-pool overlap; pool rule excludes cross-split ids |
| 9 | Model fingerprints match | ✅ | prompt, LLM, probe checkpoint, encoder config all identical to validation |
| 10 | Cost configuration matches | ✅ | `e3534119a7c9ba50…`; `re_measured_on_evaluation_pool: false` |

`AlignedPool.load` re-derives the pool fingerprint from the id list and refuses a
file whose recorded hash no longer matches, so condition 6 is a recomputation
rather than a comparison of two stored strings.

Order of operations, in the driver and in fact:

1. Lock checks — metadata only.
2. `assert_locks_passed` — would have stopped here on any failure.
3. Feature extraction for the test split, behind `--i-am-running-the-locked-evaluation`.
4. Probe scoring, refusing to write if the checkpoint had changed.
5. Evaluation, behind `--i-am-opening-the-locked-test-split`.

---

## 4. Test Dataset

| | |
|---|---|
| Locked pool size | 482 |
| Alignment fingerprint | `c44ed7e03972e7745738e62f4117a91e1a8083319995d07688c09cd6ddbd8d11` |
| Evaluated samples | **482** (100% — nothing excluded) |
| Evaluated fingerprint | `22285e1ee58b7d3fedef12180873cc84ad795da4aa7c2a6fcb496b850094d838` |
| Excluded: no usable Gemma evidence | 0 |
| Excluded: no cached audio features | 0 |
| Dataset composition | MSP-Podcast 482 (100%) |
| Resampled / rebalanced / manifest changed | false / false / false |
| Ordering | recorded pool order, preserved verbatim |
| Audio test manifest | 5542 samples, `cec265feb59a46f8…` |

Class distribution, against validation:

| Class | Test (482) | Test share | Validation (509) | Validation share |
|---|---|---|---|---|
| neutral | 198 | 41.1% | 172 | 33.8% |
| happy | 120 | 24.9% | 151 | 29.7% |
| angry | 92 | 19.1% | 95 | 18.7% |
| sad | 47 | 9.8% | 63 | 12.4% |
| surprise | 11 | 2.3% | 15 | 2.9% |
| disgust | 8 | 1.7% | 7 | 1.4% |
| fear | 6 | 1.2% | 6 | 1.2% |

Both pools are entirely MSP-Podcast: it is the only corpus in this project with
aligned transcript and audio for the same clip. The test split is **more
neutral-heavy** (41.1% vs 33.8%) and has fewer happy and sad samples.

---

## 5. System Comparison

Six systems, identical 482 samples, identical labels.

| | System | Description | Label-aware? |
|---|---|---|---|
| A | Gemma text only | anchor, 1.00 modalities/sample | no |
| B | Audio strong only | frozen wav2vec2 + probe, no text | no |
| C | Always fusion | text + audio on 100% of samples, 2.00 modalities/sample | no |
| D | Random acquisition @10% | 48 samples drawn with seed 42, independent of labels | no |
| E | **AHSEF dynamic @10%** | frozen uncertainty ranking, 48 samples | no |
| F | Oracle @10% | acquires where the true signed gain is largest | **YES** |

> **F is marked in the artefact as `ANALYSIS ONLY — LABEL-AWARE UPPER BOUND`.**
> It reads the test labels. It was computed *after* the AHSEF and random
> acquisition sets were already fixed, and there is no code path by which it
> could reach a routing score, a threshold, a weight, a budget or a parameter.

---

## 6. Overall Metrics

| System | Accuracy | **Macro-F1** | Weighted-F1 | Macro-P | Macro-R | Balanced Acc |
|---|---|---|---|---|---|---|
| A. Gemma text only | 0.5041 | 0.3345 | 0.4823 | 0.3641 | 0.3380 | 0.3380 |
| B. Audio strong only | 0.5083 | 0.3345 | 0.5403 | 0.3610 | 0.3705 | 0.3705 |
| C. Always fusion | 0.6058 | **0.4167** | 0.5848 | 0.4775 | 0.4082 | 0.4082 |
| D. Random @10% | 0.5145 | 0.3399 | 0.4914 | 0.3725 | 0.3424 | 0.3424 |
| **E. AHSEF @10%** | **0.5207** | **0.3450** | **0.4951** | **0.3930** | 0.3343 | 0.3343 |
| F. Oracle @10% *(label-aware)* | 0.6037 | 0.4515 | 0.5848 | 0.4910 | 0.4528 | 0.4528 |

Macro-F1 is the primary metric throughout. Accuracy is reported beside it and is
never the criterion: Stage 2 established that accuracy on this corpus rewards
predicting neutral, and neutral is 41% of this split.

Two observations worth stating before the routing analysis:

- **Audio-only and text-only tie exactly on macro-F1** (0.3345 both) while audio
  is far ahead on weighted-F1 (0.5403 vs 0.4823). The two experts are of
  comparable overall quality here and disagree about *which* classes they can do.
- **Always-fusion is markedly stronger on test than on validation** (macro-F1
  0.4167 vs 0.4086, over a lower text baseline). The fusion gain grew from
  +0.0997 to +0.0822 in absolute terms on a harder anchor — there was more to
  capture, and AHSEF captured a smaller share of it.

---

## 7. Per-Class Metrics

| Class | n | Text-only | Audio-only | Always-fusion | Random @10% | **AHSEF @10%** | **Δ AHSEF vs Text** |
|---|---|---|---|---|---|---|---|
| neutral | 198 | 0.6278 | 0.6383 | 0.7111 | 0.6400 | 0.6430 | **+0.0152** |
| happy | 120 | 0.4920 | 0.4294 | 0.5444 | 0.4973 | 0.4974 | **+0.0054** |
| sad | 47 | 0.2025 | 0.3579 | 0.2571 | 0.2025 | 0.2308 | **+0.0282** |
| angry | 92 | 0.3724 | **0.6919** | 0.6220 | 0.3862 | 0.3885 | **+0.0161** |
| fear | 6 | 0.2857 | 0.0000 | 0.2222 | 0.2857 | 0.3636 | **+0.0779** |
| disgust | 8 | 0.1111 | 0.1379 | 0.3529 | 0.1176 | 0.1176 | **+0.0065** |
| surprise | 11 | 0.2500 | 0.0860 | 0.2069 | 0.2500 | 0.1739 | **−0.0761** |

### Minority-class aggregate

Using the existing project definition
(`src.ahsef.milestone.selection.minority_f1_mass` — summed F1 over the classes
outside the two largest, here neutral and happy):

| | value |
|---|---|
| Text-only | 1.2218 |
| **AHSEF @10%** | **1.2745** |
| Delta | **+0.0527** |

### Is the improvement concentrated in neutral?

**No.** Neutral accounts for **20.8%** of the total per-class F1 improvement,
against its 41.1% share of the pool. `improvement_concentrated_in_neutral: false`
in the artefact. The largest single gain is `fear` (+0.0779, on 6 samples — see
the caveat below), followed by `sad` (+0.0282) and `angry` (+0.0161).

Two honest qualifications:

- **`fear` (n=6) and `disgust` (n=8) are anecdote.** One sample moves fear's F1 by
  roughly 0.13. The +0.0779 on fear is a single sample changing class.
- **`surprise` regressed by −0.0761.** One acquisition converted a correct
  surprise prediction into a wrong one. On 11 samples that is one prediction, and
  it is the largest per-class loss in the table. It is reported because it
  happened, not despite it.

### The class that mattered most, and was missed

`angry` is where the audio expert's advantage lives on this split: audio-only F1
**0.6919** against Gemma's 0.3724, and always-fusion reaches 0.6220 there.
29 of the 66 available corrections are angry samples. **AHSEF caught 2 of them.**
That single fact accounts for most of the gap between 12.7% retained gain and the
51.6% validation predicted. §11 explains why.

---

## 8. Confusion Matrices

Rows are true classes, columns predicted, in canonical order
(neutral, happy, sad, angry, fear, disgust, surprise).

**A. Gemma text only**
```
        neutr  happy    sad  angry   fear  disgu  surpr
neutr     156     11     12     12      3      2      2
happy      62     46      5      3      0      0      4
  sad      25      3      8      7      1      2      1
angry      46      4      6     27      2      5      2
 fear       3      0      0      1      2      0      0
disgu       2      1      0      3      0      1      1
surpr       5      2      1      0      0      0      3
```

**B. Audio strong only**
```
        neutr  happy    sad  angry   fear  disgu  surpr
neutr     120     10     20      9      3     10     26
happy      25     38      7     15      0      2     33
  sad      19      3     17      4      0      0      4
angry       9      3      2     64      0      4     10
 fear       1      0      1      0      0      1      3
disgu       2      1      0      1      0      2      2
surpr       2      2      1      0      0      2      4
```

**C. Always fusion**
```
        neutr  happy    sad  angry   fear  disgu  surpr
neutr     176      4      9      5      1      1      2
happy      56     49      2      6      0      0      7
  sad      26      3      9      7      1      0      1
angry      30      2      2     51      0      3      4
 fear       2      0      1      1      1      1      0
disgu       2      0      0      2      0      3      1
surpr       5      2      0      0      0      1      3
```

**D. Random acquisition @10%**
```
        neutr  happy    sad  angry   fear  disgu  surpr
neutr     160     10     12     10      3      2      1
happy      61     46      5      4      0      0      4
  sad      25      3      8      7      1      2      1
angry      46      3      6     28      2      4      3
 fear       3      0      0      1      2      0      0
disgu       2      1      0      3      0      1      1
surpr       5      2      1      0      0      0      3
```

**E. AHSEF dynamic @10%**
```
        neutr  happy    sad  angry   fear  disgu  surpr
neutr     163     11     12      7      1      2      2
happy      61     47      5      3      0      0      4
  sad      25      4      9      6      1      1      1
angry      50      4      4     27      1      4      2
 fear       3      0      0      1      2      0      0
disgu       2      1      0      3      0      1      1
surpr       5      2      1      0      0      1      2
```

**F. Oracle @10% — ANALYSIS ONLY, LABEL-AWARE UPPER BOUND**
```
        neutr  happy    sad  angry   fear  disgu  surpr
neutr     173      6      8      7      2      2      0
happy      58     53      3      2      0      0      4
  sad      25      3      9      6      1      2      1
angry      31      3      4     46      1      5      2
 fear       3      0      0      1      2      0      0
disgu       1      1      0      2      0      3      1
surpr       5      1      0      0      0      0      5
```

The decisive row is `angry`. Gemma sends 46 of 92 angry samples to neutral;
always-fusion recovers that to 30, and the oracle at only 10% budget recovers it
to 31. AHSEF at the same budget leaves it at **50** — slightly worse than
text-only, because two of its acquisitions moved angry predictions the wrong way
while it spent the rest of its budget elsewhere.

---

## 9. Routing Performance

| | AHSEF @10% | Random @10% | Always fusion | Text only |
|---|---|---|---|---|
| Eligible samples | 482 | 482 | 482 | 482 |
| Acquisition count | **48** | 48 | 482 | 0 |
| Acquisition rate | **9.96%** | 9.96% | 100% | 0% |
| Budget respected | **✅ true** | ✅ true | — | — |
| Text-only samples | 434 | 434 | 0 | 482 |
| Text+Audio samples | 48 | 48 | 482 | 0 |
| Average modalities/sample | **1.100** | 1.100 | 2.000 | 1.000 |
| Ranking fingerprint | `phase_f_results.json → routing.E_ahsef_at_budget.top_k_ranking_fingerprint` | recorded | — | — |

`round(0.10 × 482) = 48`; `48 / 482 = 9.96%`. The frozen budget was respected
exactly.

### Uncertainty distribution

| | n | mean | std | min | p25 | median | p75 | max |
|---|---|---|---|---|---|---|---|---|
| All samples | 482 | 0.3796 | 0.2008 | 0.000 | 0.2387 | 0.3640 | 0.5596 | 0.8381 |
| **Acquired** | 48 | **0.7172** | 0.0394 | 0.6577 | 0.6985 | 0.6985 | 0.7301 | 0.8381 |
| Not acquired | 434 | 0.3422 | 0.1749 | 0.000 | 0.2369 | 0.3284 | 0.4848 | 0.6577 |

Separation: **+0.3750**. The acquired set is cleanly the high-uncertainty tail,
with no overlap — the ranking behaved exactly as specified.

### Routing trace

`experiments/ahsef/milestone/phase_f_routing_trace.jsonl` — one line per sample,
482 lines, each carrying the uncertainty, its rank, the acquisition decision, the
modalities used, all three predictions, the true class and the outcome label
(`corrected` / `harmed` / `both_correct` / `both_wrong` / `text_only_*`). The
`decided_on` field on every line names the only quantity that entered the
decision: the normalised score entropy of the Gemma response.

---

## 10. Acquisition Quality

Definitions reused unchanged from Phase D/E
(`src.ahsef.milestone.budget.score_at_budget`).

| Policy | Acquisitions | Text wrong → audio fixes | Text right → audio breaks | Both correct | Both wrong | **Net** | Correction precision | Harm rate | Net rate |
|---|---|---|---|---|---|---|---|---|---|
| **AHSEF @10%** | 48 | **11** | 3 | 14 | 20 | **+8** | **0.2292** | 0.0625 | 0.1667 |
| Random @10% | 48 | 8 | 3 | 22 | 15 | +5 | 0.1667 | 0.0625 | 0.1042 |
| Always fusion | 482 | 66 | 17 | 226 | 173 | +49 | 0.1369 | 0.0353 | 0.1017 |

**AHSEF's acquisitions are genuinely better than random ones.** Correction
precision 22.9% against 16.7%, at an identical harm rate — 3 harms each. Against
the 13.7% base rate of a useful acquisition, AHSEF's selection is enriched
**1.67×**; random is 1.22× by construction noise.

The enrichment is real but much weaker than on validation, where the same policy
achieved 33.3% precision against a 12.6% base rate — a **2.65×** enrichment.

| | Validation | Test |
|---|---|---|
| Corrections available | 64 / 509 (12.6%) | 66 / 482 (13.7%) |
| AHSEF correction precision @10% | 0.3333 | 0.2292 |
| AHSEF recall of corrections @10% | 0.2656 (17/64) | **0.1667 (11/66)** |
| Enrichment over base rate | **2.65×** | **1.67×** |
| Net corrections | +14 | +8 |

---

## 11. Uncertainty Transfer

**Does the frozen ranking still work on unseen data?** Nothing was recalibrated,
no threshold was reselected, nothing was fitted
(`recalibrated_on_test: false`, `threshold_selected_on_test: false`,
`anything_fitted_on_test: false`).

| Measure | Test | Validation |
|---|---|---|
| AUROC — identifying incorrect **text** predictions | **0.6344** | — |
| Spearman — uncertainty vs text error | **0.2332** | — |
| AUROC — identifying **useful acquisitions** | **0.6612** | **0.7575** |
| AUPRC — identifying useful acquisitions | 0.2118 (base 0.1369) | 0.3073 (base 0.1257) |
| Harm AUROC (1 − AUROC; >0.5 would mean harm is avoided) | 0.2353 | 0.3809 |

### Accuracy by uncertainty bin (text-only)

| Uncertainty range | n | Text accuracy | Mean uncertainty |
|---|---|---|---|
| 0.000 – 0.237 | 93 | 0.6022 | 0.1020 |
| 0.237 – 0.328 | 88 | 0.6250 | 0.2460 |
| 0.328 – 0.427 | 100 | 0.5800 | 0.3642 |
| 0.427 – 0.560 | 99 | 0.4646 | 0.4878 |
| **0.560 – 0.838** | **102** | **0.2745** | 0.6579 |

`monotone_decreasing_accuracy: false` — the first two bins invert (0.6022 then
0.6250). But the relationship is decisive where routing operates: the top bin
sits at 27.5% accuracy against 60% in the bottom bin. **The signal transfers, and
it transfers most strongly exactly where the policy spends.**

### Why the benefit shrank anyway

The uncertainty signal answers "where is Gemma unsure?" The policy assumes that
is also "where does audio help?" On validation those two questions had a 2.65×
alignment; on test they have 1.67×. The reason is visible per class:

| Class | Support | Corrections available | Harms available | **AHSEF caught** |
|---|---|---|---|---|
| angry | 92 | **29** | 5 | **2** |
| neutral | 198 | 23 | 3 | 7 |
| happy | 120 | 7 | 4 | 1 |
| sad | 47 | 3 | 2 | 1 |
| surprise | 11 | 2 | 2 | 0 |
| disgust | 8 | 2 | 0 | 0 |
| fear | 6 | 0 | 1 | 0 |

**44% of the available benefit is in `angry`, and AHSEF captured 7% of it.**
Gemma reads an angry transcript, confidently predicts neutral, and reports low
entropy — the words are unremarkable, the *prosody* is what carries the emotion.
A router that can only see the text model's own doubt is structurally blind to
exactly the case where the audio modality is most valuable.

This is not a defect in the implementation; it is the ceiling of the signal that
Phase D selected. Phase D found, honestly, that no richer target beat uncertainty
on validation. Phase F shows what that ceiling costs on data where the
complementarity is concentrated in confident text errors.

---

## 12. Random-Control Comparison

Matched budget: exactly 48 acquisitions, seed 42, drawn independently of any test
label (`independent_of_test_labels: true`).

| | AHSEF @10% | Random @10% | Δ |
|---|---|---|---|
| Macro-F1 | 0.3450 | 0.3399 | **+0.0051** |
| Accuracy | 0.5207 | 0.5145 | +0.0062 |
| Weighted-F1 | 0.4951 | 0.4914 | +0.0037 |
| Correction precision | 0.2292 | 0.1667 | +0.0625 |
| Net corrections | +8 | +5 | +3 |

**Paired bootstrap, macro-F1: +0.0051, 95% CI [−0.0283, +0.0344].** The interval
contains zero.

AHSEF beat random on every metric and picked measurably better samples, but on
482 samples with 66 available corrections, a 3-correction advantage is not
separable from sampling noise. The point estimate is positive and consistent with
validation; the evidence is not strong enough to call it established.

A single random draw is also itself a noisy comparator. Phase E used 200 seeded
draws per budget for exactly this reason; the Phase F brief specifies one
deterministic draw, so one is what is reported.

---

## 13. Always-Fusion Comparison

| | AHSEF @10% | Always fusion | Δ |
|---|---|---|---|
| Macro-F1 | 0.3450 | 0.4167 | **−0.0717** |
| Accuracy | 0.5207 | 0.6058 | −0.0851 |
| Weighted-F1 | 0.4951 | 0.5848 | −0.0897 |
| Audio activations | 48 | 482 | **−434 (−90.0%)** |
| Modalities/sample | 1.100 | 2.000 | −0.900 |

**Paired bootstrap, macro-F1: −0.0717, 95% CI [−0.1402, +0.0027].**

### Retained fusion gain

```
fusion_gain   = MacroF1(always_fusion) − MacroF1(text_only)
              = 0.4167 − 0.3345 = +0.0822

ahsef_gain    = MacroF1(AHSEF) − MacroF1(text_only)
              = 0.3450 − 0.3345 = +0.0105

retained_gain = 0.0105 / 0.0822 = 12.7%
```

> **AHSEF retained 12.7% of the fusion gain for 10.0% of the audio activations.**

Against validation's 51.6%, this is the headline shortfall and the reason the
verdict is B rather than A. The declared bar for "meaningful" was 50%.

For context: the trade is roughly break-even in efficiency terms — 12.7% of the
benefit for 10.0% of the cost — but it is nothing like the 5× efficiency
advantage validation projected, and "as good as spending the money uniformly" is
not what selective acquisition is for.

---

## 14. Oracle Ceiling

> **ANALYSIS ONLY — LABEL-AWARE UPPER BOUND.** Reads test labels. Computed after
> both acquisition sets were fixed. Influenced no decision, parameter or threshold.

| | Oracle @10% | AHSEF @10% | Gap |
|---|---|---|---|
| Macro-F1 | **0.4515** | 0.3450 | **−0.1065** |
| Accuracy | 0.6037 | 0.5207 | −0.0830 |
| Weighted-F1 | 0.5848 | 0.4951 | −0.0897 |

The oracle at **10%** budget beats always-fusion at **100%** on macro-F1
(0.4515 vs 0.4167) — because it declines the 17 acquisitions that break a correct
prediction, which always-fusion cannot. This reproduces the validation finding
and confirms the ceiling is well above the reference, not below it.

AHSEF captured **9.0%** of the oracle's available gain over text-only
(0.0105 / 0.1170). The remaining 91% is the value of knowing *which* uncertain
samples audio will actually help — information the frozen signal does not carry.

---

## 15. Cost / Latency

Frozen cost configuration, read from the policy, `re_estimated_from_test_behaviour: false`.

| System | Latency ms/sample | Modalities/sample |
|---|---|---|
| Text only | 3449.73 | 1.000 |
| Audio strong only | **289.06** | 1.000 |
| Always fusion | 3738.79 | 2.000 |
| **AHSEF expected** | **3478.51** | **1.100** |
| Random expected | 3478.51 | 1.100 |

The 289.06 ms strong-audio figure is the **deployment** cost: 288.88 ms
WAV2VEC2_BASE encoder + 0.18 ms LayerWeightedProbe head. The cached-head latency
alone (0.18 ms) is **not** quoted anywhere as deployment cost.

### Saving versus always-fusion

| | value |
|---|---|
| Latency saved | **260.28 ms/sample** (−7.0% end-to-end) |
| Audio activations avoided | **434 of 482 (90.0%)** |
| Audio compute avoided | **90.0%** |
| Total audio time saved over the pool | 125.45 s |

### Cost-effectiveness

| | ms of audio spent per sample | Δ macro-F1 | ms per macro-F1 point |
|---|---|---|---|
| AHSEF @10% | 28.79 | +0.0105 | **2,749** |
| Always fusion | 289.06 | +0.0822 | **3,518** |
| Oracle @10% *(label-aware)* | 28.79 | +0.1170 | 246 |

AHSEF is **1.28× more cost-effective per macro-F1 point than always-fusion** —
real, but a fraction of the 5.1× validation projected.

**Monetary cost is not reported.** No price was measured for the Ollama cloud
endpoint, and inventing one would place a fabricated number beside measured ones.

---

## 16. Statistical Confidence Intervals

Paired percentile bootstrap over sample indices, 2000 resamples, seed 42. Both
systems are rescored on the identical resample because they predict on identical
samples; an unpaired interval would discard that structure and be materially
wider. The resampling convention matches `src.ahsef.stage3.hsig_quality`.

| Comparison | Metric | Point estimate | 95% CI | Excludes zero |
|---|---|---|---|---|
| AHSEF vs text-only | **macro-F1** | **+0.0105** | **[−0.0227, +0.0388]** | ❌ |
| AHSEF vs text-only | accuracy | +0.0166 | see artefact | ❌ |
| AHSEF vs random @10% | **macro-F1** | **+0.0051** | **[−0.0283, +0.0344]** | ❌ |
| AHSEF vs always-fusion | **macro-F1** | **−0.0717** | **[−0.1402, +0.0027]** | ❌ (marginal) |
| Always-fusion vs text-only | macro-F1 | +0.0822 | see artefact | — |

**Every interval involving AHSEF contains zero.** At n=482 with 66 available
corrections and a 48-sample budget, this evaluation does not have the power to
establish a difference of the size AHSEF produces. The point estimates are all in
the predicted direction and are consistent with validation; the intervals say the
data cannot distinguish them from no effect.

The always-fusion comparison is the informative one in the other direction: the
interval [−0.1402, +0.0027] only barely admits zero, so the evidence that AHSEF
*underperforms* always-fusion is nearly as strong as the evidence for anything
else here.

---

## 17. Reproducibility

Complete provenance under `reproducibility` in `phase_f_results.json`.

| Field | Value |
|---|---|
| Protocol | `ahsef.milestone.phase_f.v1` |
| Frozen policy fingerprint | `b2f8ced01204940599092f0db46c0965ebbae0b7e4ff1b38303ae94fa2cb6f37` |
| Test manifest fingerprint | `cec265feb59a46f8da346ef96b48e622fef6235ca8f09c3ba425a51100ede94a` |
| Alignment fingerprint | `c44ed7e03972e7745738e62f4117a91e1a8083319995d07688c09cd6ddbd8d11` |
| Evaluated pool fingerprint | `22285e1ee58b7d3fedef12180873cc84ad795da4aa7c2a6fcb496b850094d838` |
| Text model | `gemma4:31b-cloud`, replay provider, deterministic |
| Prompt | `v2_explicit_json`, `a6e8e7d02edb101f68b7886ec603871f31f6d9ce09fe37c01e47d20eb3e7ba7b` |
| Generation | repeats 1, temperature 0.0, seed 42, `score_entropy` |
| Audio model | `audio_strong_full` iteration 1, LayerWeightedProbe |
| Probe checkpoint | `306c3f745709d73ff566e0b2d37ca498f7937142035c8b9ccf9ed97944c9f56d` |
| Wav2Vec2 | WAV2VEC2_BASE, config `dc1543bfda550c28c8655cfccb45b7dff3b9f5498af3f58e8ffd5eccbc75bd48`, 94,370,944 params, frozen |
| Fusion | `weighted_probability`, `{audio: 0.6, text_llm: 0.4}` |
| Uncertainty | `score_entropy` → `normalized_entropy`, not recalibrated |
| Budget / Seed | 0.10 / 42 |
| Cost fingerprint | `e3534119a7c9ba50928f981b16a094b11ed074a3749466b64101bda96bb4e69a` |
| Sample counts | locked pool 482, evaluated 482 |

### Stored artefacts — offline replay

| File | Contents |
|---|---|
| `phase_f_results.json` | every number in this report |
| `phase_f_predictions.parquet` | per-sample: id, dataset, true class, uncertainty, rank, all three predictions, all three acquisition masks, all three routed predictions |
| `phase_f_routing_trace.jsonl` | 482 lines, one auditable decision each |
| `phase_f_lock_checks.json` | the ten conditions with their evidence |
| `analysis/audio_expert/audio_strong__test.parquet` | frozen probe outputs on test |
| `audio_strong/features/test/` | cached wav2vec2 test features, coverage 1.0 |

**No API call was made.** The Gemma test responses were recorded during Stage 3
and are replayed from `transcripts/transcript_test.jsonl` (`provider: replay`,
`deterministic: true`, 482 recorded prompts, 0 misses). Re-running
`--stage evaluate` reproduces every number offline.

### Locked-artefact audit

| Root | Files before → after | Digest unchanged |
|---|---|---|
| `experiments/ahsef/stage1` | 89 → 89 | ✅ identical |
| `experiments/ahsef/stage2_llm` | 17 → 17 | ✅ identical |
| `experiments/ahsef/stage3_text_audio` | 35 → 35 | ✅ identical |
| `experiments/audio_strong` | 77 → 90 | ⚠️ changed — **13 files added**, 0 modified |

The `audio_strong` change is the wav2vec2 test feature cache Phase F created
under `features/test/` — a directory that did not exist before. Recomputing the
root digest over everything **except** that new subtree returns
`a59c8ddd5a135cc1d7a96bd50295383946556760c9cd61d6116befa1b64f79ea`, which is
byte-for-byte the recorded pre-Phase-F baseline. **No pre-existing locked file was
modified, removed or renamed.**

---

## 18. Leakage Audit

| Check | Value |
|---|---|
| `test_labels_used_for_policy_selection` | **false** |
| `test_labels_used_for_threshold_selection` | **false** |
| `test_labels_used_for_budget_selection` | **false** |
| `test_labels_used_for_model_selection` | **false** |
| `test_labels_used_for_prompt_selection` | **false** |
| `test_labels_used_for_fusion_selection` | **false** |
| `test_labels_used_for_uncertainty_selection` | **false** |
| `only_oracle_uses_test_labels` | **true** |
| `oracle_computed_after_acquisition_sets_were_fixed` | **true** |
| `policy_modified_after_seeing_test_results` | **false** |

**How this is enforced, not merely asserted:** the driver loads the frozen policy
from disk and applies it. It contains no function that selects a target, a
threshold, a budget, a weight or a model — there is no code path to leak through.
Lock condition 4 checks the twelve declarations that each earlier phase recorded
*at the time it made its decision*, rather than a flag written now.

**One caveat stated plainly (see §19):** the Stage 3 locked test result was known
before Phase D/E was designed, and it motivated the Phase D hypothesis. No test
label entered any fit or selection, but the project's hypothesis space was not
formed in complete ignorance of the test split.

---

## 19. Limitations

**1. Every interval contains zero.** With 482 samples, 66 available corrections
and a 48-sample budget, this evaluation cannot establish a difference of the size
AHSEF produces. The +0.0051 macro-F1 over random is a point estimate, not a
demonstrated effect. Anyone citing this result should cite the interval with it.

**2. The random control is a single draw.** The brief specified one deterministic
seed, so one is reported. Phase E used 200 draws per budget because a single draw
is itself noisy; a different seed could plausibly have moved the AHSEF-vs-random
delta by more than its magnitude.

**3. One corpus.** Both pools are 100% MSP-Podcast — the only dataset in this
project with aligned transcript and audio for the same clip. Nothing here
generalises to CREMA-D, IEMOCAP or RAVDESS, and nothing here separates "AHSEF
works" from "AHSEF works on podcast speech".

**4. Three classes are anecdote.** fear (6), disgust (8) and surprise (11)
together are 5.2% of the pool. One sample moves fear's F1 by ~0.13. The +0.0779
on fear and the −0.0761 on surprise are each a single prediction, and macro-F1
gives them the same weight as neutral's 198 samples. Macro-F1 remains the right
primary metric for this project's goal, but at these supports it is a noisy one.

**5. The signal has a structural ceiling.** Uncertainty answers "where is Gemma
unsure?", not "where does audio help?" On test those diverge: 44% of the
available benefit is in confident angry misclassifications that the ranking
cannot see. Phase D found no better target on validation; Phase F shows what that
costs. Fixing it would need a signal with access to something other than the text
model's own confidence — which is a Phase G question, not a Phase F adjustment.

**6. Prior exposure to the test split.** Stage 3 ran its own locked test
evaluation before Phase D/E was designed, and its negative result is what
motivated Phase D's central hypothesis. No test label entered any fit,
selection or threshold — lock condition 4 verifies that — but the *hypothesis*
was not chosen in ignorance of the test split, and this is the second time these
labels have been consulted for this project. That is a real, if second-order,
weakening of the guarantee, and it is stated rather than omitted.

**7. Validation over-projected by 4×.** Retained gain 51.6% → 12.7%. Selecting an
operating point on validation and reporting its validation performance
systematically overstates what a locked split will show. This report is the
measurement of that gap, and it is the most transferable finding here.

---

## 20. Final Decision

Classification rule, declared in `src/ahsef/milestone/phase_f.py` before the split
was opened. Macro-F1 is the criterion; accuracy is never the criterion.

| # | Criterion | Required | Measured | Met |
|---|---|---|---|---|
| 1 | Beats matched random on macro-F1 | > 0.3399 | **0.3450** (+0.0051) | ✅ |
| 2 | Retains meaningful fusion gain | ≥ 50% | **12.7%** | ❌ |
| 3 | Substantially fewer modalities | ≤ 241 acquisitions | **48** (1.100 mods/sample) | ✅ |
| 4 | Improvement not confined to neutral | minority F1 mass must rise | **+0.0527** (neutral = 20.8% of the gain) | ✅ |
| 5 | Beats text-only | > 0.3345 | **0.3450** (+0.0105) | ✅ |

> # VERDICT: B. PARTIAL GENERALIZATION
>
> AHSEF provides a useful cost/performance trade-off and beats matched random
> acquisition on macro-F1 at the frozen budget, but it loses most of the fusion
> benefit (12.7% retained against 51.6% on validation) and no comparison reaches
> statistical separation at n=482.

**What generalised:**

- The uncertainty ranking still identifies useful acquisitions (AUROC 0.6612 vs
  0.7575 on validation) and still identifies text errors (AUROC 0.6344).
- The acquired set is enriched 1.67× over the base rate; random is not.
- The frozen budget, ranking and fusion applied exactly as specified.
- The improvement is distributed across classes rather than confined to neutral —
  the property Stage 2 warned about did not recur.
- The efficiency direction held: 1.28× more macro-F1 per millisecond of audio
  than always-fusion.

**What did not:**

- The *magnitude*. 12.7% retained against a 51.6% validation projection.
- Statistical separation. Every AHSEF interval contains zero.
- Coverage of the highest-value corrections. 2 of the 29 angry corrections.

**The policy was not changed after seeing this result.** No threshold was moved,
no weight retuned, no budget adjusted, no model reselected. This is the single
locked evaluation the Phase D/E freeze was built to enable, and it is reported as
it came out.

---

## PHASE F COMPLETE

### Test macro-F1 and accuracy — all systems

| System | Macro-F1 | Accuracy |
|---|---|---|
| A. Gemma text only | 0.3345 | 0.5041 |
| B. Audio strong only | 0.3345 | 0.5083 |
| C. Always fusion | 0.4167 | 0.6058 |
| D. Random acquisition @10% | 0.3399 | 0.5145 |
| **E. AHSEF dynamic @10%** | **0.3450** | **0.5207** |
| F. Oracle @10% *(label-aware)* | 0.4515 | 0.6037 |

### Headline figures

| | |
|---|---|
| AHSEF acquisition rate | **48 / 482 = 9.96%** (frozen 10%, respected) |
| AHSEF vs Random | **+0.0051 macro-F1**, 95% CI [−0.0283, +0.0344] |
| AHSEF vs Always-fusion | **−0.0717 macro-F1**, 95% CI [−0.1402, +0.0027] |
| Fusion gain retained | **12.7%** (validation: 51.6%) |
| Modalities per sample | **1.100** (always-fusion 2.000) |
| Audio activations avoided | **434 of 482 (90.0%)** |

### Per-class AHSEF results

| Class | n | Text-only F1 | AHSEF F1 | Δ |
|---|---|---|---|---|
| neutral | 198 | 0.6278 | 0.6430 | +0.0152 |
| happy | 120 | 0.4920 | 0.4974 | +0.0054 |
| sad | 47 | 0.2025 | 0.2308 | +0.0282 |
| angry | 92 | 0.3724 | 0.3885 | +0.0161 |
| fear | 6 | 0.2857 | 0.3636 | +0.0779 |
| disgust | 8 | 0.1111 | 0.1176 | +0.0065 |
| surprise | 11 | 0.2500 | 0.1739 | **−0.0761** |
| **minority aggregate** | — | 1.2218 | **1.2745** | **+0.0527** |

### Verdict

> **B. PARTIAL GENERALIZATION**

### Confirmations

- ✅ **The routing policy remained frozen.** SHA-256
  `b2f8ced01204940599092f0db46c0965ebbae0b7e4ff1b38303ae94fa2cb6f37` before and
  after; `modified_by_phase_f: false`. No threshold, weight, budget, prompt,
  checkpoint or seed was changed at any point during Phase F.
- ✅ **Test labels influenced no policy decision.** All seven leakage flags are
  `false`. Only the oracle reads test labels, it is marked
  `ANALYSIS ONLY — LABEL-AWARE UPPER BOUND`, and it was computed after both
  acquisition sets were already fixed.
- ✅ **All ten lock conditions passed** before the split was opened.
- ✅ **No pre-existing locked artefact was modified** — Stage 1/2/3 digests
  identical; `audio_strong` gained only the 13-file test feature cache.
- ✅ **Full test suite: 1087 passed** (45 Phase F tests added, covering the ten
  lock conditions, the frozen decision rule, budget enforcement, acquisition
  quality, paired statistics, uncertainty transfer and the verdict rule).

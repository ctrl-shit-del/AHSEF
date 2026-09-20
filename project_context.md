# Capstone project context

## Purpose and current stage

This repository implements **AHSEF** — Adaptive Hierarchical Sensor/Modality
Evidence Fusion — a multimodal emotion-recognition framework whose research
question is *not* "what is the most accurate classifier" but:

> Can the system decide, per sample, whether the evidence it already has is
> sufficient, and if not, which additional modality is worth acquiring —
> weighing expected gain against cost and latency?

Six layers exist, each building on the one below:

1. **Metadata foundation** — ten heterogeneous corpora normalised into one
   record schema (complete).
2. **Sampling and standardisation** — deterministic, leakage-checked experiment
   splits (complete).
3. **Five frozen unimodal baselines** — audio, image, text, video, physiology
   (complete, frozen; never retrained by later stages).
4. **AHSEF** — Stage 1 evidence layer, Stage 2 LLM text gate, and Stage 3
   dynamic Text → Audio acquisition (all complete, locked tests run).
5. **The 50% milestone, Phases A–F** — a strong audio expert and a redesigned
   routing study, ending in a second locked test (complete; verdict *partial
   generalization*).
6. **HSEN** — the always-on multimodal baseline the adaptive layer is measured
   against: frozen emotion2vec / MobileNetV3 / RoBERTa encoders, cached once,
   feeding a 256-d fusion trunk with categorical, valence and arousal heads
   (in progress; see `docs/ahsef_hsen_phase.md`). Sections 12–13 ran the
   modality and fusion ablations on the CPU 25% profile; **Phase 15** re-ran
   the fusion ablation under three seeds and found the Husformer margin to be
   seed noise, so the full-data CUDA run carries `concat`.

Layer 6 is deliberately *not* a continuation of layers 4–5. Those read five
frozen unimodal baselines and train nothing end-to-end; layer 6 trains one
multimodal trunk over cached frozen-encoder features, under the dataset
protocols the literature reports against rather than this project's canonical
seven-class taxonomy. It lives in `src/hsen/` and writes only to
`metadata/hsen/`, `experiments/hsen/` and `results/hsen/`.

Everything after layer 2 treats the baselines as immutable. Nothing in
`src/ahsef/` writes into `experiments/<modality>/`, `checkpoints/`, or
`results/`, and nothing in `src/ahsef/milestone/` writes into `stage1`,
`stage2_llm`, `stage3_text_audio` or `audio_strong` — all four are digest-locked
before and after every milestone run.

---

## Layer 1 — metadata foundation (complete)

A registry-driven preprocessing framework normalises dataset-specific
annotations into one `EmotionRecord` format, preserves paths to original
assets, and writes per-dataset and master artifacts.

Ten adapters are implemented and enabled: RAVDESS, CREMA-D, IEMOCAP, MELD,
CMU-MOSEI, FERPlus, RAF-DB, AffectNet+, WESAD, and MSP-Podcast.

**Master metadata is current** (`metadata/master/summary.json`):

| Dataset | Records | | Modality combination | Records |
| --- | ---: | --- | --- | ---: |
| AffectNet+ | 420,299 | | image | 471,119 |
| MSP-Podcast | 264,705 | | audio + text | 264,697 |
| FERPlus | 35,481 | | audio + video + text | 22,856 |
| CMU-MOSEI | 22,856 | | audio | 19,941 |
| RAF-DB | 15,339 | | video + text | 13,707 |
| MELD | 13,708 | | physiology | 15 |
| IEMOCAP | 10,039 | | text | 1 |
| CREMA-D | 7,442 | | | |
| RAVDESS | 2,452 | | | |
| WESAD | 15 | | | |
| **Total** | **792,336** | | | |

### Unified record contract

| Area | Fields |
| --- | --- |
| Identity | `sample_id`, `dataset`, `split` |
| Labels | `raw_emotion`, `emotion`, `canonical_emotion`, `canonical_emotion_id`, `valence`, `arousal`, `dominance`, `sentiment_score` |
| Modalities | `modalities`, `has_<modality>`, `<modality>_source`, `audio_path`, `video_path`, `image_path`, `text`, `text_path`, `physiology_path` |
| Context | `speaker`, `gender`, `duration`, `segment_start`, `segment_end` |
| Targets | `target_type`, `emotion_target_valid`, `training_split`, `evaluation_group` |
| Detail | `extras` (JSON-serialised) |

Asset paths are relative to `datasets/`. Modality availability is **record-level**:
a modality counts only when `has_<modality>`, an accepted `<modality>_source`,
and a populated payload column all agree. Nothing is ever fabricated for an
absent modality.

### The canonical label space

```
0 neutral · 1 happy · 2 sad · 3 angry · 4 fear · 5 disgust · 6 surprise
```

Declared once in `src/common/labels.py` and never re-sorted: the confusion
matrix, per-class arrays, class weights, and checkpoint metadata all index these
positions.

**WESAD is the exception.** Its records carry `target_type ==
'physiological_state'` and `canonical_emotion_valid == False`, so the physiology
baseline declares its own label space (`baseline` / `stress` / `amusement`)
rather than having emotion labels fabricated for it. This has consequences that
propagate all the way to AHSEF fusion — see Layer 4.

---

## Layer 2 — sampling and standardisation (complete)

`src/preprocessing/sampling/` and `src/preprocessing/standardization/` turn the
master metadata into per-modality experiment splits under
`experiments/<modality>/<fraction>/metadata/{train,validation,test}.parquet`,
using stratified proportional selection with largest-remainder allocation,
seeded shuffling, and an official-split-aware policy. Each split ships a
`sampling_summary.json` and a `validation_report.json` recording leakage checks
(duplicate ids, cross-split ids, official holdout contamination).

---

## Layer 3 — frozen unimodal baselines (complete)

One shared protocol in `src/training/base_experiment.py`: train-only class
weights, validation-driven checkpoint selection, a test partition opened exactly
once after training, and fully isolated iterations. Only the dataloader and the
architecture are modality-specific.

| Modality | Experiment | Task | Test accuracy | Test macro-F1 |
| --- | --- | --- | ---: | ---: |
| image | `image_25pct` | emotion_7class | 0.5187 | 0.3789 |
| text | `text_full` | emotion_7class | 0.3759 | 0.2740 |
| audio | `audio_25pct` | emotion_7class | 0.3461 | 0.2468 |
| video | `video_25pct` | emotion_7class | 0.3353 | 0.1173 |
| physiology | `physiology_full` | **wesad_state_3class** | 0.5707 | 0.2880 |

These are **frozen**. AHSEF rebuilds each architecture from its recorded
`run_summary['model']`, loads `best.pt`, and SHA-256 hashes the checkpoint
before and after every pass; a changed digest raises `FrozenCheckpointError`.

A sixth expert, **`audio_strong_full`**, was added later at Layer 5 and is frozen
on the same terms. It is not a Layer 3 baseline — it uses pretrained weights the
five baselines deliberately do not — and it never replaces `audio_25pct` in any
Stage 1-3 result. See Phase B below.

---

## Layer 4 — AHSEF

### Stage 1: the evidence layer (complete)

`src/ahsef/` adds inference, uncertainty, calibration, sample identity, fusion,
information gain, cost, evaluation, and routing-log components. Artifacts live
under `experiments/ahsef/stage1/`.

**The finding that shapes everything after it: the corpora are not globally
aligned.** The five baselines were sampled independently, and most pairs share
no samples at all.

| pair | co-split validation | co-split test | verdict |
| --- | ---: | ---: | --- |
| audio + text | 509 | 482 | fusable (MSP-Podcast) |
| text + video | 343 | 343 | fusable (MELD) |
| every other pair | 0 | 0 | no shared sample in any split |

Consequences: the audio-anchored pairwise matrix collapses to one cell;
physiology can never enter a 7-class fusion (different label space); 17,594
audio/text ids are shared *across different splits* and are excluded because one
model trained on them; and **no sample anywhere has two simultaneously
available candidates**.

Other Stage 1 results:

- Audio + text fusion works: test accuracy 0.3320 (audio) / 0.3983 (text) →
  **0.4398** fused (geometric pool). Prediction agreement is only 25.3%.
- **ΔU is not a usable HSIG target.** Its sign flips with the anchor and the
  fusion rule; in five of six configurations accuracy rose 8–10 points while
  mean ΔU was negative; max per-sample correlation with actual improvement was
  **0.143**. `HSIG_TARGET_DEFINITION` records the rejected target and its
  replacement: `gain(m|A) = P(correct | A ∪ {m}) − P(correct | A)`, estimated on
  validation only.
- Calibration: text and image are well calibrated (ECE 0.029 / 0.036); audio and
  video are *under*confident; physiology is badly overconfident (test ECE 0.306)
  and its NLL-optimal temperature makes ECE worse, so the rule correctly
  declines to apply it.

See `docs/ahsef_stage1.md`.

### Stage 2: the LLM text gate (complete)

An LLM is added as a **semantic reasoning modality**, registered as `text_llm`
and deliberately distinct from the frozen `text` baseline. The pipeline is:

```
LLM(text) → emotion + class scores → uncertainty → U ≤ τ ? STOP : REQUEST
```

| | |
| --- | --- |
| provider | Ollama 0.33.2, stdlib HTTP, `src/ahsef/llm/ollama_provider.py` |
| model | **`gemma4:31b-cloud`** → `gemma4:31b`, cloud-backed, 32.7B BF16 |
| prompt | `v2_explicit_json` (versioned, hashed, immutable) |
| generation | `temperature=0`, `seed=42` — **reproducible** on this backend |
| pilot size | 1,000 validation + 1,000 locked test |

**Validation results (paired, identical sample ids):**

| System | Accuracy | Macro-F1 | Weighted-F1 |
| --- | ---: | ---: | ---: |
| Frozen text baseline | 0.3810 | 0.2790 | 0.4044 |
| Gemma LLM | 0.4940 | 0.3145 | 0.4735 |

Parsing was 1000/1000 usable, zero failures. But **the gain is one class**: the
LLM predicts `neutral` on 61.8% of samples against a 38.5% base rate, so
accuracy rises on a neutral-heavy corpus while macro-F1 gains only 0.036.

Uncertainty vs correctness is **real but weak**: accuracy falls 0.65 → 0.25
across bins, Spearman ρ = 0.213, AUROC = 0.623. Self-reported confidence is
severely overconfident (ECE 0.257), and the score vector's NLL of 4.71 is worse
than uniform (ln 7 = 1.946) — decisive evidence that these scores are **not
posteriors**, which the code never treats them as.

τ = **0.257156**, selected on validation by a pre-set `target_stop_accuracy=0.60`
objective, then frozen. Stage 3 discarded that objective: on a neutral-heavy
corpus it selects a high-accuracy, low-macro-F1 operating point, which is
exactly what an objective blind to the minority classes will always prefer.

See `docs/ahsef_stage2_llm.md` and `docs/ahsef_stage2_results.md`.

**Not implemented at Stage 2, deliberately:** HSIG estimator, UGAPR selection,
modality acquisition, fusion after acquisition, multi-step routing.
`UnavailableHSIG` declines to guess and `UGAPR` selects nothing without a gain
estimate — a cost-only fallback would be a hard-coded modality order in
disguise. **Stage 3 supplies all of them.**

### Stage 3: dynamic Text → Audio acquisition (complete, locked test run)

`src/ahsef/stage3/` closes the routing loop for the one modality pair the data
supports:

```
Text/LLM → HSIG: gain(audio | state) → UGAPR: J = gain − λC − μL
        → J ≥ τ ? REQUEST_AUDIO : STOP → acquire, fuse, re-estimate
```

**The pool.** The co-split Text+Audio intersection: **509 validation, 482 test**,
all MSP-Podcast, both payloads verified on disk, pools disjoint, zero train
contamination. The Stage 2 pilot could not be reused — only 20 validation and 18
test of its 1,000 samples fall inside the pool — so Stage 3 recorded its own
Gemma transcript over the aligned pool using the **Stage 2 frozen configuration
unchanged**; `--stage llm` refuses to run if any of it differs.

**Audio carries real complementary information.** Test-split fusion:

| System | Accuracy | Macro-F1 | Weighted-F1 |
| --- | ---: | ---: | ---: |
| Gemma LLM (text) | 0.5041 | 0.3345 | 0.4823 |
| Audio baseline | 0.3320 | 0.1990 | 0.3251 |
| **Text + Audio** | **0.5539** | **0.3421** | **0.5268** |

Audio fixes 34 wrong text predictions (7.1%) and breaks 10 (2.1%), net **+24**;
text and audio agree on only **18.3%** of samples.

**Locked test, all five systems (n = 482):**

| System | Accuracy | Macro-F1 | Acq. rate | Modalities/sample |
| --- | ---: | ---: | ---: | ---: |
| Text only | 0.5041 | 0.3345 | 0% | 1.00 |
| Audio only | 0.3320 | 0.1990 | 100% | 1.00 |
| Always fusion | **0.5539** | **0.3421** | 100% | 2.00 |
| Stage-2 gate + audio | 0.5477 | 0.3377 | 65.1% | 1.65 |
| AHSEF dynamic | 0.5166 | 0.3350 | **9.5%** | 1.10 |

**Frozen policy.** HSIG `paired_logistic` on feature set `uncertainty_only`
(fingerprint `96988c45c9dbcd28`); fusion `weighted_probability` audio 0.7 /
text_llm 0.3; objective `macro_f1 − α·acq − β·latency` with α = 0.05, β = 0.02;
τ = 0.094797; λ = μ = 0.10.

**Four findings that constrain what may be claimed.** All are measured, and all
are stated in the generated report rather than inferred:

1. **The richer evidence model lost.** `uncertainty_only` (out-of-fold AUROC
   **0.7555**) beat the ten-feature set (0.7199) and a shallow forest (0.6425).
   Its bootstrap CI contains the reference, so this pool **cannot establish**
   that a learned multi-feature HSIG beats Stage 2's signal.
2. **Ablations D ≡ E ≡ G, sample for sample.** The frozen estimator is a function
   of uncertainty alone, and the UGAPR penalty is **8 × 10⁻⁴** (audio is cheap
   beside a 32.7B cloud LLM), so the dynamic router selects exactly the samples a
   tuned uncertainty threshold selects. Stage 3's gain over Stage 2 comes from
   the **redesigned objective**, not from richer features or the cost term. The
   α/β sensitivity table is flat across the whole grid.
3. **Targeting works; the system-level benefit does not replicate.** 9 of 46 paid
   acquisitions fixed a wrong answer (**19.6%** against a 7.1% base rate, 2.8×
   lift), and HSIG's out-of-sample AUROC on the locked split is **0.7408** — the
   estimator transfers. But AHSEF's macro-F1 (0.3350) falls **inside** the 95%
   interval of random acquisition at the same rate ([0.3117, 0.3556]). The
   validation advantage (0.3536 vs a 0.3134 random mean) did not survive the
   lock; the gap is the size of the threshold-selection optimism.
4. **Always-fusion remains the best system on this pool.** AHSEF retains 97.9% of
   its macro-F1 and 93.3% of its accuracy at 9.5% acquisition — a real
   cost/performance trade, not a performance win.

**The defensible claim** (`STAGE3_CLAIM`, carried as a constant so it cannot
drift): *AHSEF dynamically determines whether additional audio evidence is worth
acquiring after an initial LLM-based text assessment, using a
validation-trained expected-gain estimator and a cost/latency-aware utility
policy.* It is **not** a demonstration that AHSEF selects the best modality among
all modalities — the candidate set has one member.

See `docs/ahsef_stage3_text_audio.md` (method) and
`docs/ahsef_stage3_results.md` (results, generated from the artefacts).

---

## Layer 5 — the 50% milestone, Phases A–F (complete, locked test run)

Experiment `ahsef_50pct_milestone_v1`. Stage 3 ended on a negative result: AHSEF's
macro-F1 sat *inside* the interval of random acquisition at the same rate. The
milestone reopens that question with a stronger acquirable expert, a redesigned
routing-target study, and a control matched budget-for-budget rather than at one
threshold. Drivers live in `src/ahsef/analysis/` (A–C) and `src/ahsef/milestone/`
(D–F).

### Phase A — where is the headroom?

`run_classwise_analysis` profiles every modality per class, then compares
modalities **only on genuinely aligned samples** — the one place in this project
where such a comparison has a referent. On the 509-sample audio+text validation
pool the two disagree on **73.5%** of samples: audio fixes text on 114, text
fixes audio on 116, both are wrong on 201. An oracle trusting whichever is right
reaches **0.6051** accuracy against 0.3811 for text alone. That ceiling — not
achievable by any real router — is why the milestone was worth running.

### Phase B — a strong audio expert (`audio_strong`)

A **frozen WAV2VEC2_BASE encoder** under a trainable `LayerWeightedProbe` head:
266,003 trainable parameters over 94.4M frozen pretrained ones, learned softmax
pooling across the 12 transformer layers, 4-second crops. Features are extracted
once into a sharded cache (`experiments/audio_strong/features/`, fingerprint
`dc1543bf…`) and the probe trains over the cache.

Validation, 5,550 identical samples:

| System | Accuracy | Macro-F1 | Weighted-F1 | ECE | Uncertainty separation |
| --- | ---: | ---: | ---: | ---: | ---: |
| `audio_25pct` baseline | 0.3490 | 0.2624 | 0.3327 | 0.075 | 0.007 |
| **`audio_strong`** | **0.4741** | **0.3831** | **0.4983** | **0.026** | **0.099** |

**All seven classes improved**, against a bar declared before the run (+0.03
macro-F1, no accuracy regression) that it clears four times over. It is also
better calibrated, and its uncertainty separates correct from wrong 14× more
strongly — the precondition for uncertainty-triggered routing to work at all.

**Cost is the encoder, and the project charges it.** 289.06 ms/sample at
deployment against the probe head's 0.18 ms over cached features. Billing only
the head would understate the price of acquiring audio by three orders of
magnitude and make every budget on the curve look free; `assert_encoder_is_charged`
refuses the cached-head figure.

### Phase C — fusion on the aligned pool (validation, n = 509)

| System | Accuracy | Macro-F1 | Weighted-F1 |
| --- | ---: | ---: | ---: |
| Gemma (`text_llm`) | 0.4715 | 0.3088 | 0.4398 |
| audio baseline | 0.3772 | 0.2383 | 0.3642 |
| `audio_strong` | 0.4637 | 0.3165 | 0.5004 |
| text_llm + audio baseline | 0.5128 | 0.3510 | 0.4803 |
| **text_llm + `audio_strong`** | **0.5619** | **0.4086** | **0.5415** |

The fusion gain over text alone more than doubles: **+0.0421 → +0.0997** macro-F1.
Spec: `weighted_probability`, `{audio: 0.6, text_llm: 0.4}`, chosen on validation
by grid scan — then **read, never re-selected**, by every later phase, so the
fusion choice and the routing choice are not entangled.

### Phase D — which supervision target should a router predict?

Five candidate targets under **one** estimator family (ridge on standardised
features, one α selected once on `signed_gain`, one fold assignment, one seed),
so a difference between rows is a difference between *targets* rather than
between experiments: `binary_correction`, `signed_gain`, `class_balanced_gain`,
`macro_f1_marginal`, `uncertainty_reduction` — each measured against the
unfitted `uncertainty_only` reference.

**No fitted target beat raw uncertainty.** Reference AUROC **0.7575**; every
fitted target's delta is negative (best: `binary_correction`, −0.0142).
`uncertainty_reduction` leads on the ranking metric by +0.0021 macro-F1 — inside
the declared 0.005 tie margin — so the parsimony rule takes the simpler signal.
The feature-set ablation agrees: 10 evidence features (0.7283) barely move on 1
(0.7240), and adding class indicators hurts (0.7031).

This independently reproduces Stage 3's finding under a different estimator
family and a redesigned comparison. **The deployed policy therefore contains no
fitted estimator at all.**

### Phase E — budget curves against a matched random control

Nine budgets × 200 random draws each, rather than one threshold. The policy
clears the random control's 97.5th percentile at **every** interior budget from
5% to 50%. All four pre-declared *MATERIAL ROUTING SUCCESS* conditions are met at
10%:

| Condition | Required | Measured |
| --- | --- | ---: |
| beats matched random | > p97.5 = 0.3365 | **0.3603** |
| improvement not only majority-class | minority F1 mass must rise | **+0.3057** |
| retains fusion gain | ≥ 50% | **51.6%** |
| substantially fewer acquisitions | ≤ 50% | **10.0%** (51/509) |

**Frozen policy** — `frozen_routing_policy.json`, SHA-256 `b2f8ced0…`, frozen
2026-08-31T10:55, before the test split was opened:

```
score(x)   = normalised score entropy of the Gemma response for x
acquire(x) = x in top-k by score(x),  k = round(0.10 × n),  ties by index
predict(x) = fuse(text_llm, audio_strong) if acquired else text_llm
fusion     = weighted_probability, {audio: 0.6, text_llm: 0.4}
```

### Phase F — the locked test (n = 482, one evaluation, no tuning)

Ten lock conditions were checked before the split was opened; all ten passed.

| System | Accuracy | Macro-F1 | Weighted-F1 |
| --- | ---: | ---: | ---: |
| A. Gemma text only | 0.5041 | 0.3345 | 0.4823 |
| B. `audio_strong` only | 0.5083 | 0.3345 | 0.5403 |
| C. Always fusion | 0.6058 | **0.4167** | 0.5848 |
| D. Random acquisition @10% | 0.5145 | 0.3399 | 0.4914 |
| **E. AHSEF dynamic @10%** | **0.5207** | **0.3450** | **0.4951** |
| F. Oracle @10% *(label-aware)* | 0.6037 | 0.4515 | 0.5848 |

**Verdict: `B. PARTIAL GENERALIZATION`** — four of five criteria met; the one
that failed is retained fusion gain.

*What transferred.* The ranking still works unrecalibrated (AUROC 0.6344 for text
errors, 0.6612 for useful acquisitions; the top uncertainty bin sits at 27.5%
accuracy against 60.2% in the bottom). Acquisition quality: correction precision
**0.2292 against random's 0.1667 at an identical harm rate**, a 1.67× enrichment
over the base rate. The gain is spread across classes (minority F1 mass +0.0527;
neutral is only 20.8% of it) — the Stage 2 majority-class failure did not recur.
The efficiency direction held too: 1.28× more macro-F1 per millisecond of audio
than always-fusion, at 1.100 modalities/sample and 90% of audio activations
avoided.

*What did not.* **The magnitude.** Retained fusion gain fell from **51.6% on
validation to 12.7%** on test, and **every paired bootstrap interval contains
zero**: vs random +0.0051 [−0.0283, +0.0344]; vs text-only +0.0105 [−0.0227,
+0.0388] — only accuracy separates (+0.0166 [+0.0021, +0.0332]).

**The mechanism of the shortfall is identified, not guessed.** 29 of the 66
available corrections on test lie in `angry`, where `audio_strong` is far
stronger than Gemma (F1 0.6919 vs 0.3724) — and Gemma is *confident* on most of
them: the words are unremarkable, the prosody carries the emotion. The ranking
caught **2 of those 29**, leaving 44% of the available benefit unreachable. A
router that can only see the text model's own doubt is structurally blind to
exactly the case where audio is most valuable. That is the ceiling of the signal
Phase D honestly selected; Phase F measures its price.

**The most transferable finding: validation over-projected by 4×.** Choosing an
operating point on validation and reporting its validation performance
overstated the locked result by that factor. The policy was **not** changed after
seeing this result — no threshold moved, no weight retuned, no budget adjusted,
no model reselected.

Limitations are stated in the report rather than left to be inferred: one corpus
(both pools are 100% MSP-Podcast); fear/disgust/surprise together are 5.2% of the
pool, so their per-class deltas are single predictions; the test random control
is a single deterministic draw; and this is the **second** time these test labels
have been consulted for the project — no label entered any fit, but Phase D's
hypothesis was chosen after seeing Stage 3's locked result, which weakens the
guarantee and is recorded as such.

See `docs/ahsef_phase_de_results.md` and `docs/ahsef_phase_f_results.md`.

---

## Layer 6 — HSEN, the always-on multimodal baseline (in progress)

One multimodal trunk trained over cached frozen-encoder features
(emotion2vec audio, RoBERTa text; MobileNetV3 video is built but IEMOCAP video
is not yet in the pipeline), under the IEMOCAP ERC-6 protocol: sessions 1–3 /
4 / 5, 4,246 / 1,512 / 1,622 utterances, speaker-disjoint and audited before
every run. Two profiles share one trainer and one model: `cpu25` (25% of
training, stratified; batch 8; 15 epochs; FP32) and `full_cuda` (100%; batch
32; 30 epochs; AMP). Validation and test stay whole in both. Checkpoints are
selected on validation weighted F1; the test split is opened only by an
explicit `--evaluate-test`, and no Layer-6 run has opened it yet.

### Sections 12–13 — modality and fusion ablations (CPU 25%, seed 42)

| arm | val WF1 | val macro-F1 | params | train s |
| --- | ---: | ---: | ---: | ---: |
| text | 0.4157 | 0.3798 | 3.8 M | 697 |
| audio | 0.4350 | 0.4028 | 3.8 M | 4,437 |
| **audio+text** (husformer) | **0.5036** | **0.4750** | 6.0 M | 3,661 |
| audio+text, self_attention | 0.4911 | 0.4544 | 2.8 M | 1,869 |
| audio+text, concat | 0.4928 | 0.4626 | 1.4 M | 475 |

Audio > text unimodally, inverting ESED's ordering — the effect of emotion2vec.
Fusion adds +0.0685 WF1 over the best single modality. Husformer led the fusion
table by ~0.01 at 24× the fusion parameters of concatenation.

### Phase 15 — seed-variance study (CPU 25%, seeds 42 / 43 / 44)

Same three fusion arms, same 1,062-utterance training subset (`subset_seed`
pinned to 42, SHA-256 `f11fc8ba…`), same features, dimensions, heads,
optimiser, schedule, loss, batch size, epoch budget, early stopping and
checkpoint criterion; only the training seed moves. Validation-only; the
study refuses a run carrying test metrics. Seed 42 reproduced section 13 to
every printed digit.

| variant | seed42 | seed43 | seed44 | mean ± std (val WF1) | mean macro-F1 | params | mean train s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| husformer | 0.5036 | 0.4886 | 0.4869 | 0.4930 ± 0.0091 | 0.4622 | 5,992,456 | 5,715 |
| **concat** | 0.4928 | 0.4927 | 0.4940 | **0.4932 ± 0.0007** | **0.4640** | 1,448,968 | 417 |
| self_attention | 0.4911 | 0.4921 | 0.4912 | 0.4915 ± 0.0006 | 0.4562 | 2,830,344 | 1,895 |

- Husformer beats concat on **1 of 3** seeds and self-attention on **1 of 3**;
  the ordering flips with the seed. Paired mean margins −0.0001 and +0.0016.
- The section-13 margin (+0.0107 / +0.0125) is not larger than Husformer's own
  across-seed std (0.0091) or the paired-difference std (0.0095). Section 13
  caught Husformer at its best seed and concat near its worst.
- Concat and self-attention vary by 0.0006–0.0007 across seeds; Husformer by
  0.0091 — thirteen times more.
- **Decision: `concat` is carried into the full-data CUDA experiment.** Tied
  with Husformer on mean WF1 (0.0002 apart), more stable across these seeds,
  higher mean macro-F1, 4.1× fewer parameters, ~12× cheaper per epoch. Three
  seeds bound the spread and support no claim of significance either way;
  "more stable across these seeds" is the strength of the claim. Husformer's
  case (more modalities, more data, a budget that was binding it at 15 epochs)
  is a candidate for a secondary full-data run, not the primary.
- Timing caveat: the seed-42 and seed-43 Husformer runs were slowed by
  concurrent load (test suite; memory pressure); the ordering of costs is
  unaffected, the Husformer mean is inflated.

Artefacts: `results/hsen/iemocap_erc6/phase15_seed_variance/` —
`study_config.json` (exact commands, configuration, subset digest, class
order), `seed_variance_runs.csv`, `seed_variance_summary.json`,
`seed_variance_table.md`, `study.log`, nine run directories. Section 13's
`ablation_fusion_*` directories are untouched. Not started, by instruction:
hyperparameter search, HSIG on HSEN, UGAPR on HSEN, any HSEN redesign.

---

## Repository map

| Location | Role |
| --- | --- |
| `datasets/` | Original datasets and extracted assets. |
| `src/common/` | `EmotionRecord`, paths, constants, **`labels.py`** (canonical order), `experiment_layout.py`. |
| `src/preprocessing/adapters/` | One scanner per dataset. |
| `src/preprocessing/core/` | Base adapter, pipeline, validator, serializer, master builder. |
| `src/preprocessing/sampling/` | Stratified split construction and leakage validation. |
| `src/preprocessing/standardization/` | Canonical targets, split policy, experiment definitions. |
| `src/data/` | Lazy per-modality datasets and dataloaders. |
| `src/models/` | Small per-modality baseline architectures. |
| `src/training/` | Shared experiment protocol, trainer, evaluator, checkpointing, per-modality runners, `verify_artifacts`. |
| `src/ahsef/` | Stage 1 + Stage 2: inference, uncertainty, calibration, identity, fusion, gate, hsig, ugapr, router, costs, evaluation, routing logs. |
| `src/ahsef/llm/` | LLM provider abstraction, Ollama + Anthropic providers, prompts, schema, mapping, samples, LLM calibration. |
| `src/ahsef/stage3/` | Stage 3: aligned pool, features, real HSIG, gate objective, costs, router, experiment harness, report renderer. |
| `src/ahsef/analysis/` | Phases A–C: class-wise analysis, strong-expert comparison, audio-expert report. |
| `src/ahsef/milestone/` | Phases D–F: targets, estimator, budget curves, selection rule, costs, quality, Phase F lock/evaluation, reproducibility. |
| `src/ahsef/cli/` | Every runnable command. |
| `src/hsen/` | **Layer 6.** Label adapters, manifests + audit, frozen-encoder feature caching, the HSEN model with three fusion trunks, the shared trainer, ablations, `seed_variance.py` (Phase 15), evaluation. |
| `scripts/` | `train_hsen_cuda.py` and `train_hsen_cpu25.py` — thin config loaders over one trainer. |
| `experiments/<modality>/` | **Frozen** baseline experiments. Never written to by AHSEF. |
| `experiments/ahsef/stage1/` | Stage 1 artifacts. |
| `experiments/ahsef/stage2_llm/` | Stage 2 pilot artifacts. |
| `experiments/ahsef/stage3_text_audio/` | Stage 3 routing artifacts. |
| `experiments/audio_strong/` | **Frozen** strong audio expert: `features/` cache + `full/` run. |
| `experiments/ahsef/analysis/` | Phase A–C artifacts (`classwise/`, `audio_expert/`). |
| `experiments/ahsef/milestone/` | Phase D–F artifacts, `frozen_routing_policy.json`, lock baseline. |
| `metadata/` | Generated metadata; treat as artifacts and regenerate. |
| `metadata/hsen/` | Layer-6 experiment manifests and their audits. |
| `experiments/hsen/` | Layer-6 frozen-encoder feature caches. |
| `results/hsen/` | Layer-6 runs: checkpoints, metrics, history, figures. |
| `docs/` | Stage 1–3 and Phase D/E + F reports, plus the original proposal `.docx` set. |

---

## Commands

```bash
# Layer 1–2: metadata and splits
python -m src.preprocessing.generate_metadata
python -m src.preprocessing.sampling.run_sampler

# Layer 3: a baseline (does not need re-running; they are frozen)
python -m src.training.run_audio_experiment
python -m src.training.verify_artifacts --experiment audio_25pct

# Layer 4 Stage 1
python -m src.ahsef.cli.export_predictions   --run stage1
python -m src.ahsef.cli.run_alignment        --run stage1 --anchor audio
python -m src.ahsef.cli.run_calibration      --run stage1
python -m src.ahsef.cli.run_pairwise_fusion  --run stage1 --anchor audio
python -m src.ahsef.cli.run_gain_diagnostics --run stage1 --split validation

# Layer 4 Stage 2 (four stages; `test` refuses to run before `freeze`)
python -m src.ahsef.cli.run_stage2_pilot --stage select
python -m src.ahsef.cli.run_stage2_pilot --stage validation
python -m src.ahsef.cli.run_stage2_pilot --stage freeze
python -m src.ahsef.cli.run_stage2_pilot --stage test
python -m src.ahsef.cli.report_stage2 --run stage2_llm

# Layer 4 Stage 3 (test refuses to run before freeze, and on any config drift)
python -m src.ahsef.cli.run_stage3_text_audio --stage align
python -m src.ahsef.cli.run_stage3_text_audio --stage llm      --split validation
python -m src.ahsef.cli.run_stage3_text_audio --stage llm      --split test
python -m src.ahsef.cli.run_stage3_text_audio --stage oracle   --split validation
python -m src.ahsef.cli.run_stage3_text_audio --stage hsig
python -m src.ahsef.cli.run_stage3_text_audio --stage freeze
python -m src.ahsef.cli.run_stage3_text_audio --stage validate
python -m src.ahsef.cli.run_stage3_text_audio --stage ablations
python -m src.ahsef.cli.run_stage3_text_audio --stage oracle   --split test
python -m src.ahsef.cli.run_stage3_text_audio --stage test
python -m src.ahsef.cli.run_stage3_text_audio --stage report

# Layer 5 Phase A-C: headroom, the strong audio expert, aligned fusion
python -m src.ahsef.cli.run_classwise_analysis --split validation
python -m src.training.extract_audio_features --split train      # then validation, test
python -m src.training.run_audio_strong_experiment
python -m src.ahsef.cli.run_audio_expert_comparison --stage compare
python -m src.ahsef.cli.run_audio_expert_comparison --stage aligned
python -m src.ahsef.cli.export_strong_audio_test --split test

# Layer 5 Phase D-F (evaluate refuses to run unless all ten locks pass)
python -m src.ahsef.cli.run_milestone_phase_de --stage all
python -m src.ahsef.cli.run_milestone_phase_f  --stage lock
python -m src.ahsef.cli.run_milestone_phase_f  --stage evaluate

# Layer 6: HSEN (see docs/ahsef_hsen_phase.md for the full pipeline)
python -m src.hsen.manifests --experiment iemocap_erc6
python -m src.hsen.extract --experiment iemocap_erc6 --modality text  --split train
python -m src.hsen.extract --experiment iemocap_erc6 --modality audio --split train
python scripts/train_hsen_cpu25.py --dataset iemocap --data_fraction 0.25
python scripts/train_hsen_cuda.py  --dataset iemocap --data_fraction 1.0
python -m src.hsen.ablations --study modality --profile cpu25
python -m src.hsen.ablations --study fusion   --profile cpu25
python -m src.hsen.seed_variance --study phase15_seed_variance --experiment iemocap_erc6 \
    --profile cpu25 --modalities audio+text --variants husformer,concat,self_attention \
    --seeds 42,43,44 --subset-seed 42          # Phase 15; --report-only re-aggregates
python scripts/train_hsen_cuda.py  --dataset iemocap --data_fraction 1.0 --fusion concat   # the Phase 15 choice
python -m src.hsen.evaluation --run results/hsen/iemocap_erc6/full_cuda

python -m pytest src/ahsef -q          # 818 tests
python -m pytest src/hsen -q           # 68 tests
python -m pytest -q                    # 1,155 collected across the repo
```

---

## Research-integrity rules enforced in code

These are not conventions; each raises or refuses.

| Risk | Enforcement |
| --- | --- |
| calibration fitted on test | `fit_temperature` / `fit_confidence_calibrator` raise unless split is `validation` |
| τ selected on test | `select_threshold` raises `ThresholdLeakageError` |
| τ re-selected at test time | `--stage test` loads `frozen_config.json`, never writes it, and aborts on any config drift |
| fusing unrelated samples | `AlignmentIndex.fusion_pool` — co-split, contamination- and label-checked |
| fusing across label spaces | `assert_fusable` refuses differing `class_order` |
| frozen checkpoint modified | SHA-256 before and after every pass |
| router seeing labels | `apply_gate` has no label parameter; traces stamp `router_saw_true_class: false` |
| fabricated probabilities | absent LLM scores leave `prob_*` as `NaN`, never a filler distribution |
| unmappable LLM label → neutral | mapping returns `ambiguous`/`unknown` and the sample is counted, never folded into a class |
| runaway API spend | `CallBudget` checked before every call; a cost ceiling on an unpriced model fails loudly |
| raw user text in artifacts | `--store-text {none,hash,raw}`, default `hash` |
| **Stage 3** router seeing labels | `TextAudioRouter.route(state)` has no label parameter; truth is attached only by `attach_truth` after every decision exists |
| HSIG fitted on test | `assert_fit_split` raises `HSIGLeakageError`; `out_of_fold_gain` refuses too |
| HSIG regressing the rejected ΔU target | `assert_target_not_delta_uncertainty` raises `HSIGTargetError` |
| label leaking into HSIG features | `assert_label_free` rejects a deny-list of label-bearing column names |
| imputing a missing feature | HSIG returns `expected_improvement=None` with a reason; nothing is filled |
| acquisition threshold chosen on test | `select_gate_threshold` raises `ThresholdLeakageError` |
| HSIG refitted or edited after the freeze | coefficient SHA-256 fingerprint compared against `frozen_config.json` |
| acquisition price re-measured on test | the router prices audio from `frozen_config.costs` via `cost_model_from_frozen` |
| fabricating a prediction for unavailable audio | `REQUESTED_BUT_UNAVAILABLE`: text prediction stands, `fused_prediction` stays `None` |
| Stage 3 writing into a frozen stage | `Stage3Layout` raises `FrozenArtefactError` for `stage1` / `stage2_llm` |
| **Phase D/E** touching the test split | `assert_validation_only` on every input path; the driver names no test artefact |
| a milestone estimator fitted outside train/validation | `assert_fit_split` raises `EstimatorLeakageError` |
| a locked root edited during a milestone run | `assert_locked_artefacts_unchanged` digests `stage1`, `stage2_llm`, `stage3_text_audio` and `audio_strong` before and after, against `locked_artefacts_baseline.json` |
| pricing audio at its cached-head latency | `assert_encoder_is_charged` raises `CostConfigurationError`; the encoder's 289 ms is charged, never the 0.18 ms head |
| Phase F evaluating before the locks pass | `assert_locks_passed` raises `EvaluationLockError`; `--stage evaluate` loads the frozen policy and has no code path that selects a target, threshold, budget, weight or model |
| the frozen policy edited after the freeze | policy SHA-256 compared on load; `phase_f_lock_checks.json` records all ten conditions |
| **Layer 6** two trainers writing one run directory | `RunLock` claims `results/hsen/<experiment>/<profile>/` for the life of a run and refuses a second trainer, naming the holder's pid, host, start time and liveness. This is not hypothetical: it happened during the HSEN phase and produced a `best.pt` whose recorded validation score appeared nowhere in its own history — artefacts individually well-formed and jointly incoherent, with nothing raising. |
| a resumed run overwriting a better checkpoint | `HSENCheckpointManager.update` compares *before* either write, so `last.pt` never carries a `best_value` one epoch stale |
| a NaN validation metric winning checkpoint selection | `is_better` returns `False` for NaN: a metric that failed to compute is not an improvement |
| a fitted normalisation statistic crossing splits | the model holds no running-statistics buffer, asserted by test; affect targets use fixed affine maps of known annotation ranges, never a z-score |
| a fabricated feature for an absent modality | the cache raises on a missing id rather than substituting zeros, and availability is a separate `[B]` channel from the frame mask |
| a modality silently dropped from an ablation | a declared-but-uncached modality raises; the ablation runner records the arm as *skipped* and names it again at the end |
| **Phase 15** seed variance confounded with data-subset variance | `TrainerConfig.subset_seed` pins the stratified 25% draw independently of the training seed; the study records the SHA-256 of the 1,062 ids and refuses a run with a different training count |
| two study drivers writing one aggregate | the driver holds a `RunLock` on the study directory for its life, on top of the per-run locks |
| an incoherent run entering the aggregate | `run_row` refuses a summary whose `best_val_*` is absent from its own history, whose restored-best validation score differs from the selected one, whose fusion name is not a declared variant, whose per-class order is not the declared order, or which carries test metrics |
| a one-seed std read as stability | `aggregate` reports NaN, not 0.0, below two seeds |

---

## Known gaps / next work

1. **A routing signal that is not the text model's own confidence — Phase G, and
   now the highest-value next step.** Phase F located the exact failure: 44% of
   the available benefit sits in `angry` samples where Gemma is *confidently*
   wrong, and an entropy ranking is structurally blind to them. Two phases have
   now confirmed that no richer feature *derived from the LLM's self-report*
   beats raw uncertainty; the missing signal has to come from somewhere else —
   a cheap prosodic probe consulted before the acquisition decision, a
   disagreement estimate, or a class-conditional prior over where audio helps.
2. **Give MELD an audio pool.** MELD ships `.mp4` with an audio track, but its
   metadata records `has_audio == False`, so no audio pool includes it.
   Extracting it would let text-anchored MELD samples offer both audio *and*
   video — the first samples in this project with two simultaneously available
   candidates, and therefore the first test of UGAPR's **ranking** behaviour.
   Every routing result so far exercises only the accept/reject half. Needs
   preprocessing changes and one retrain.
3. **A larger aligned pool.** 482 test samples with 66 available corrections is
   why every Phase F interval contains zero. The point estimates all point the
   right way; nothing but sample count separates them from noise. This is the
   binding constraint on the central claim, not the design.
4. **Per-class routing.** Audio's corrections concentrate in `angry` and
   `neutral`, and macro-F1 barely rewards catching them. A class-aware
   acquisition policy is the obvious follow-up and now has a measured target to
   aim at — but it needs (3) before its effect could be shown.
5. **A second corpus.** Both aligned pools are 100% MSP-Podcast. Nothing
   measured so far separates "AHSEF works" from "AHSEF works on podcast speech".
6. ~~Dependency lockfile~~ — closed by layer 6: `requirements.txt` is tracked,
   with torch pinned by version but not by compute suffix so the CPU and CUDA
   builds stay the same version.
7. **No NVIDIA GPU on the development machine.** `torch` is a `+cpu` build and
   the only adapter is an Intel Iris Xe, so layer 6's full-CUDA profile is
   written and architecture-verified but has never been *run*. It needs a CUDA
   machine. Phase 15 settled what it runs: `--fusion concat`.
8. **IEMOCAP video is not in any pipeline.** 7.16 GB of dialog-level `.avi`
   exists, but each frame holds two speakers, so per-utterance video needs
   segment slicing plus left/right selection before face detection. The
   transcript half of this gap is now closed — the IEMOCAP adapter reads
   `dialog/transcriptions/`, so all 10,039 utterances carry gold text.
9. **CMU-MOSEI ships no emotion annotations here.** `label.csv` and
   `aligned_50.pkl` supervise sentiment only; the six-way multi-label protocol
   LDDU reports against needs `CMU_MOSEI_Labels.csd` from CMU-MultimodalSDK.
   The `emotion` column the standardized metadata carries for CMU-MOSEI is
   derived from sentiment polarity and is not annotation.

**Closed by Stage 3:** the Stage 2 locked test and Tables C/D; the Stage 3 anchor
decision (Text → Audio, the only defensible pair); fitting HSIG on the
decision-improvement target; implementing UGAPR selection; and latency modelling.

**Closed by the milestone (Phases A–F):**

- *Was the weak audio baseline the bottleneck?* Partly — `audio_strong` moved
  validation macro-F1 +0.1207 and doubled the fusion gain (+0.0421 → +0.0997).
- *Was Stage 3's flat ΔU/feature result an artefact of its estimator?* No. Five
  targets, three feature sets and an independent estimator family reproduce it.
- *Does the router beat a properly matched random control?* On validation, at
  every budget from 5% to 50%. On the locked test, in direction only — the
  margin is inside the interval.
- *Is the gain just neutral again?* No. Minority F1 mass rises on both splits;
  neutral is 20.8% of the test gain.
- *Why does validation over-promise?* Measured: 4× on retained fusion gain, and
  the cause is a ranking signal whose alignment with actual usefulness drops
  from 2.65× to 1.67× enrichment out of sample.

## Quick handoff notes

- Class order is a scientific contract. Import it from `src/common/labels.py`;
  never sort it.
- Treat `experiments/<modality>/` and `metadata/` as frozen and generated
  respectively.
- Add a dataset by writing a `BaseAdapter.scan()`, registering it, adding
  mappings, then regenerating.
- Modality availability is record-level — never assume a registry-declared
  modality exists on a given row.
- Stage 2's transcripts (`experiments/ahsef/stage2_llm/llm/transcript_*.jsonl`)
  are the reproducible artifact: the whole analysis re-runs offline with
  `--provider replay`, and an interrupted run resumes from them without
  re-spending API calls.
- Everything from Phase A onwards replays **offline** from stored artefacts. No
  phase of the milestone re-scored a model or called the LLM; Gemma's responses
  come from the Stage 3 transcript and audio from the feature cache.
- Report the interval, not just the point estimate. Every Phase F macro-F1
  comparison has a 95% interval containing zero (only AHSEF-vs-text-only
  *accuracy* separates), and the reports say so; a summary that drops the
  interval overstates the result.
- `audio_strong` is charged at its **encoder** latency (289 ms), never at the
  cached probe head (0.18 ms). The code refuses the cheaper figure.

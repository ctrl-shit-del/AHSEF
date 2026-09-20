# AHSEF Stage 1 — evidence layer for adaptive modality routing

Stage 1 builds everything the AHSEF router needs to *exist*, and stops short of
the router itself. It adds no training, changes no baseline, and writes only
into `experiments/ahsef/<run>/`.

The deliverable of this stage is not a number. It is the evidence base — and,
as it turns out, an alignment blocker that determines what the routing study
can honestly claim.

---

## 1. The alignment finding (read this first)

The five baselines were sampled independently from different corpora. They do
not share a common sample space, and most pairs share **no samples at all**.

`python -m src.ahsef.cli.run_alignment --run stage1` reads the manifests and
produces the matrix below (`alignment/alignment_matrix.json`):

| pair | co-split validation | co-split test | verdict |
|---|---|---|---|
| audio + text | 509 | 482 | **fusable** (MSP-Podcast) |
| text + video | 343 | 343 | **fusable** (MELD) |
| audio + image | 0 | 0 | no shared sample in any split |
| audio + video | 0 | 0 | no shared sample in any split |
| audio + physiology | 0 | 0 | no shared sample; also a different label space |
| image + anything | 0 | 0 | no shared sample in any split |
| physiology + anything | 0 | 0 | no shared sample; different label space |

Two consequences follow, and neither is negotiable by writing more code:

**The requested Phase-4 matrix collapses to one cell.** With Audio as the
anchor, only `audio + text` exists. `audio + image`, `audio + video`, and
`audio + physiology` have an empty intersection: image comes from AffectNet+ /
FERPlus / RAF-DB (static face corpora with no audio), video comes only from
MELD (which the audio pool excludes because MELD records carry
`has_audio == False` in the standardized metadata), and WESAD is disjoint from
every emotion corpus.

**Physiology cannot enter a 7-class fusion at all.** The WESAD baseline solves
`wesad_state_3class` (baseline / stress / amusement), because WESAD records
carry `target_type == 'physiological_state'` and `canonical_emotion_valid ==
False`. Pooling a 3-class study-condition posterior with a 7-class emotion
posterior would produce a number with no referent. `src.ahsef.fusion` refuses
it by construction.

There is also a **cross-split hazard**, which the tooling handles rather than
ignores: 17,594 sample ids are shared between the audio and text pools *in
different splits* — 4,170 of audio's test samples are text's **training**
samples. Fusing those would score the text model's memorisation. The co-split
rule excludes every one of them.

### What is still demonstrable — and what is not

`text` is the hub modality: it shares MSP-Podcast with audio and MELD with
video. A **text-anchored** router therefore has two candidates whose
availability genuinely varies per sample:

| text test split | n |
|---|---|
| can acquire audio (MSP-Podcast) | 482 |
| can acquire video (MELD) | 343 |
| **can acquire both** | **0** |
| can acquire neither | 22,506 |

So the router must consult a real, sample-dependent availability mask rather
than following a fixed sequence — but **no sample in this project offers a
choice between two simultaneously available candidates**. Stage 1 can
therefore support:

- *whether* to acquire — a real, per-sample decision, on real aligned data;
- *which* to acquire — architecturally exercised (every candidate is scored,
  unavailable ones with a stated reason), but never yet a contest between two
  available candidates on the same sample.

Claiming a demonstrated "select among competing candidate modalities" result
would overstate what this data supports. Three ways forward, in order of
scientific strength — this is a supervisor decision, not a code decision:

1. **Give MELD an audio pool.** MELD ships `.mp4` files that carry an audio
   track, but the standardized metadata records `has_audio == False` and
   `audio_source == None` for every MELD record, so no audio pool includes
   them. Extracting MELD audio and re-running the audio baseline over a
   MELD-inclusive pool would give text-anchored MELD samples *both* audio and
   video candidates — a genuine contest. It needs preprocessing changes and one
   retrain, which this milestone's constraints exclude.
2. **Report the honest scope**: whether-to-acquire is measured; which-to-acquire
   is implemented and audited but availability-determined.
3. **Simulate an availability mask** to exercise the selection logic, labelled
   throughout as a simulation and never reported as an empirical result.

---

## 2. What was added

```
src/ahsef/
    layout.py            AHSEF run directory layout (experiments/ahsef/<run>/)
    registry.py          which frozen baseline backs each modality
    provenance.py        AhsefPolicy + provenance.json (seed, git, env, safety)
    uncertainty.py       confidence / entropy / normalised entropy / margin
    calibration.py       ECE, MCE, Brier, NLL, reliability bins, temperature scaling
    identity.py          cross-modality sample identity, co-split fusion pools
    inference.py         BaselinePredictor + PredictionSet (the common interface)
    fusion.py            probability-level fusion, guarded
    information_gain.py  per-sample dU(m | A)
    gain_diagnostics.py  is dU actually a usable HSIG target?
    costs.py             measured latency + compute proxy, normalised
    evaluation.py        metrics, uncertainty/cost summaries, modality x emotion table
    routing_log.py       the auditable per-sample routing trace format
    cli/                 export_predictions, run_alignment, run_calibration,
                         run_pairwise_fusion, run_gain_diagnostics
    tests/               unit tests + an integration test against a real checkpoint
```

One additive change outside the package: `src/training/modalities.py` gained
`RUNNER_CLASSES` / `get_runner_class`, so post-hoc inference reuses each
modality's own dataloader instead of a second copy of it.

**HSIG, UGAPR, and the router are deliberately absent.** They are stage 2.

---

## 3. Design decisions worth knowing

**Confidence and uncertainty are kept apart.** `confidence = max_c p_c`;
`uncertainty = -sum_c p_c ln(p_c) / ln(K)`. Normalising by `ln K` is what makes
a 3-class and a 7-class uncertainty numerically comparable; it does not make
them semantically comparable, and nothing in the code claims it does.
`UNCERTAINTY_DEFINITION` carries `"calibrated": False` into every artefact.

**Inference reuses, it does not reimplement.** `BaselinePredictor` rebuilds the
architecture from `run_summary['model']` through the existing
`build_model_from_record`, loads the recorded `best.pt`, and builds the split
loader by instantiating the modality's own runner. Its integration test asserts
that the accuracy it measures equals the accuracy the baseline recorded.

**Frozen means frozen.** The checkpoint is SHA-256 hashed before and after every
pass; a changed digest raises `FrozenCheckpointError`. Nothing in `src/ahsef`
opens a baseline path for writing.

**Fusion is fixed-weight on purpose.** A trained fusion head would confound
"the second modality carries complementary evidence" with "the extra parameters
helped". A weighted pool of two frozen posteriors has no capacity of its own.

**Cost is measured, not asserted.** `latency_ms` is wall clock recorded during
the export. `compute_units` is `parameters x input_elements`, explicitly
labelled a MAC-order proxy rather than a FLOP count. Normalisation is computed
over the candidate set being ranked and is named in the artefact.

---

## 4. Leakage safety, and where it is enforced

| risk | enforcement |
|---|---|
| calibration fitted on test | `fit_temperature` raises `CalibrationLeakageError` unless `split == "validation"` |
| fusion weights chosen on test | `select_weights` raises `WeightSelectionError` unless the split is train/validation |
| fusing unrelated samples | `AlignmentIndex.fusion_pool` — co-split only, contamination-checked, label-agreement-checked |
| fusing across label spaces | `assert_fusable` refuses differing `class_order` |
| one model's test = another's train | co-split rule excludes it; `contamination()` reports any residue |
| modifying a frozen checkpoint | SHA-256 before/after each pass |
| the router peeking at labels | `RoutingTrace` carries `true_class` for post-hoc use and serialises `router_saw_true_class: false` |

The threshold, `lambda`, `mu`, and `max_modalities` fields already exist on
`AhsefPolicy` and are written to `provenance.json` even though stage 1 consumes
none of them — so a stage-1 and a stage-2 artefact share one audit surface.

---

## 5. Commands

```bash
# 1. per-sample predictions from every frozen baseline (validation + test)
python -m src.ahsef.cli.export_predictions   --run stage1

# 2. cross-modality sample identity: what may be fused with what
python -m src.ahsef.cli.run_alignment        --run stage1 --anchor audio

# 3. calibration measurement + validation-only temperature scaling
python -m src.ahsef.cli.run_calibration      --run stage1 --bins 15

# 4. pairwise fusion, per-sample dU, modality x emotion table
python -m src.ahsef.cli.run_pairwise_fusion  --run stage1 --anchor audio
python -m src.ahsef.cli.run_pairwise_fusion  --run stage1 --anchor text  # two candidates
python -m src.ahsef.cli.run_pairwise_fusion  --run stage1 --anchor audio --method log_opinion_pool

# 5. is dU a usable HSIG target? (run on validation before deciding)
python -m src.ahsef.cli.run_gain_diagnostics --run stage1 --split validation --anchor audio

# tests
python -m pytest src/ahsef/tests -q
python -m pytest src/ahsef/tests -m integration -q   # touches real checkpoints
```

Every flag that changes a result is a flag, not a constant: `--method`,
`--weight-objective`, `--weight-grid-step`, `--minimum-pool`,
`--apply-calibration`, `--cost-normalization`, `--lambda-cost`, `--mu-latency`,
`--seed`.

---

## 6. Artefacts

```
experiments/ahsef/stage1/
    provenance.json                       seed, git commit, env, policy, safety flags
    predictions/<modality>__<split>.parquet   per-sample logits/probs/uncertainty/latency
    predictions/<modality>__<split>.json      class order, checkpoint sha256, provenance
    alignment/alignment_matrix.json        full split x split overlap, every pair
    alignment/alignment_report.json        fusability verdict + the reason for each refusal
    calibration/<modality>.json            ECE/MCE/Brier/NLL/bins, raw and temperature-scaled
    calibration/temperature.json           the fitted temperatures and recommendations
    fusion/<pair>[__<method>]/<split>_predictions.parquet
    fusion/<pair>[__<method>]/<split>_delta_uncertainty.parquet   per-sample dU records
    fusion/<pair>[__<method>]/<split>_metrics.json
    fusion/<pair>[__<method>]/fusion_summary.json                 weights + both splits
    reports/pairwise_fusion_<anchor>[__<method>].json
    reports/gain_diagnostics_<split>.json
```

Nothing under `experiments/<modality>/`, `checkpoints/`, or `results/` is
written to.

---

## 7. The routing trace format

`routing_log.py` fixes the trace shape now so the stage-2 router fills it in
rather than inventing it. The structure is validated by `RoutingStep` /
`RoutingTrace` (`test_routing_log.py` covers all of it): a trace must
terminate, its steps must be consecutive, a stopping step may not also select a
modality, a continuing step must select one, the selected modality must have
been among the scored candidates, and it must be active at the next step.

Rendering (`RoutingTrace.render()`) — this is the format, with **illustrative
numbers**; no router has produced them:

```
Sample MSP-Podcast_MSP-PODCAST_0001_0028  (MSP-Podcast, test)

Step 1:
Active      = [audio]
Prediction  = neutral
Confidence  = 0.312000
Uncertainty = 0.947000

HSIG:
  text         = 0.214000
  image        = 0.000000
  video        = 0.000000
  physiology   = 0.000000

UGAPR:
  text         = 0.191000   (gain=0.214, cost=0.14, latency=0.09)

Selected    = text

Step 2:
Active      = [audio, text]
Prediction  = happy
Confidence  = 0.706000
Uncertainty = 0.612000

Stopping decision = TRUE (uncertainty_below_threshold)

[post-hoc] true class = happy (not visible to the router)
```

Candidates that cannot be acquired are scored as unavailable with the reason
(`no co-split aligned sample`, `different label space`) rather than silently
omitted — which is how the alignment blocker of §1 shows up inside a trace.

`summarise_traces()` turns a log into the AHSEF headline metrics: average
modalities activated, per-modality activation rate, mean uncertainty
reduction, and the distribution of stop reasons.

---

## 8. Findings that change the stage-2 design

Three results from the stage-1 run bear directly on how the router should be
built. All are reproducible from the commands in §5.

### 8.1 The candidate modalities are genuinely complementary

Per-class F1, each modality on its own test partition (so this compares
modality strength per emotion, not per-sample complementarity):

| | neutral | happy | sad | angry | fear | disgust | surprise |
|---|---|---|---|---|---|---|---|
| audio | 0.2718 | 0.4011 | 0.2985 | **0.4742** | 0.0794 | 0.1552 | 0.0476 |
| image | 0.5034 | **0.7622** | **0.3027** | 0.3344 | **0.3117** | **0.1898** | **0.2483** |
| text | 0.4268 | 0.4612 | 0.2811 | 0.4182 | 0.0425 | 0.0744 | 0.2138 |
| video | **0.5326** | 0.1758 | 0.0000 | 0.0842 | 0.0000 | 0.0000 | 0.0282 |
| *best* | video | image | image | audio | image | image | image |
| *worst* | audio | video | video | video | video | video | video |

(physiology is excluded — a `stress` F1 does not belong in an `angry` column.)

Image leads five of the seven emotions but shares no sample with any other pool,
so it can never be acquired. Among the modalities that *can* be paired, audio
and text split cleanly: text leads neutral / happy / surprise, audio leads sad /
angry / fear / disgust. Video is the weakest by a wide margin and scores exactly
zero on sad, fear, and disgust.

On the 482-sample co-split test pool audio and text agree on only **25.3%** of
samples; text is right where audio is wrong on **24.9%** of them. That headroom
is what makes an acquisition decision worth making at all.

Fusion realises a large part of it — test, weight chosen on validation:

| | accuracy | macro-F1 | weighted-F1 |
|---|---|---|---|
| audio alone (on pool) | 0.3320 | 0.1990 | 0.3251 |
| text alone (on pool) | 0.3983 | 0.2852 | 0.4151 |
| audio + text, linear pool | 0.4129 | 0.3192 | 0.4141 |
| audio + text, geometric pool | **0.4398** | 0.3190 | — |

**Caveat:** the pool carries only 6 fear, 8 disgust, and 11 surprise samples,
so macro-F1 on it is dominated by four classes and small differences are not
meaningful. Accuracy and weighted-F1 are the more trustworthy columns here.

### 8.2 dU as specified is **not** a usable target for HSIG

This is the finding that most affects stage 2. `run_gain_diagnostics` measures,
per sample, whether `dU` says what the design assumes it says. On **validation**
(so it can legitimately inform a design decision):

| pair / rule | n | mean dU | dU>0 | acc before → after | gain | corr(dU, improvement) |
|---|---|---|---|---|---|---|
| audio+text linear | 509 | −0.0037 | 48.5% | 0.3772 → 0.4637 | +0.086 | 0.069 |
| audio+text geometric | 509 | **+0.0158** | 58.7% | 0.3772 → 0.4676 | +0.090 | 0.054 |
| text+audio linear | 509 | −0.1321 | 6.7% | 0.3811 → 0.4637 | +0.083 | 0.106 |
| text+audio geometric | 509 | −0.1126 | 7.5% | 0.3811 → 0.4676 | +0.086 | 0.143 |
| text+video linear | 343 | −0.2269 | 0.3% | 0.3382 → 0.4402 | +0.102 | 0.084 |
| text+video geometric | 343 | −0.2064 | **0.0%** | 0.3382 → 0.4373 | +0.099 | 0.093 |

Three failures, all reproducible on test:

1. **The sign contradicts the outcome.** In five of six configurations accuracy
   improves by 8–10 points while mean `dU` is negative. `text+video` is the
   extreme case: `dU < 0` for **every single sample**, while acquiring video
   fixes 49 predictions and breaks 14.
2. **The sign depends on which model is the anchor.** The identical fused
   posterior yields `dU = −0.0037` from the audio anchor and `−0.1321` from the
   text anchor, simply because text starts at lower entropy (0.79 vs 0.92).
   `dU` rewards acquiring a *sharper* partner, not a more *informative* one.
3. **It carries almost no per-sample signal.** The largest |correlation| between
   `dU` and whether the acquisition actually fixed the prediction is **0.143**.

An HSIG trained to regress `dU` would therefore learn to refuse acquisitions
that demonstrably help. **Stage 2 must give HSIG an improvement-linked target
estimated on validation** — e.g. the change in probability mass on the finally
predicted class, or a validation-fitted estimate of
`P(correct | A ∪ {m}) − P(correct | A)`. The `dU` records stay useful as a
feature and as the reported uncertainty trajectory; they should not be the
regression target.

Separately, the fusion rule is worth choosing on evidence: `log_opinion_pool`
gives audio-anchored `dU` a positive mean and higher test accuracy (0.4398 vs
0.4129 for audio+text) at equal macro-F1, so it is the better stage-2 default.

### 8.3 Uncertainty predicts *routing value* far better than it predicts correctness

Audio's uncertainty barely separates its own right answers from its wrong ones
(`U|correct = 0.911` vs `U|wrong = 0.917`, a gap of **+0.0064**; text
**+0.0718**, image **+0.1974**). Taken alone that reads as bad news for an
uncertainty trigger.

But binned by the anchor's pre-acquisition uncertainty, the *value of
acquiring* stratifies sharply (test, linear pool):

| U(audio) band | n | mean dU | dU > 0 | accuracy before → after |
|---|---|---|---|---|
| [0.60, 0.80) | 38 | −0.1013 | 2.6% | 0.474 → 0.500 |
| [0.80, 0.90) | 108 | −0.0369 | 15.7% | 0.287 → 0.361 |
| [0.90, 0.95) | 145 | −0.0014 | 42.8% | 0.345 → 0.414 |
| [0.95, 1.00] | 191 | **+0.0233** | **77.0%** | 0.319 → **0.424** |

The validation split shows the same monotone pattern, so a threshold can be
chosen there honestly. Acquiring text when audio is *already* moderately
confident costs latency and raises entropy for ~2.6 points of accuracy;
acquiring it when audio is maximally uncertain gains ~10.5 points and reduces
entropy 77% of the time. **A stopping threshold near U ≈ 0.90–0.95 is
empirically justified**, and this is the strongest single piece of evidence
that adaptive acquisition beats always-acquire and never-acquire.

### 8.4 Calibration: three of five baselines need it, one must not have it

Validation-fitted temperature, applied unchanged to test:

| modality | val acc | val conf | val ECE | test ECE | T | val ECE after T | recommended |
|---|---|---|---|---|---|---|---|
| text | 0.3822 | 0.4102 | 0.0287 | 0.0401 | 1.066 | 0.0190 | temperature |
| image | 0.5190 | 0.4837 | 0.0356 | 0.0405 | 0.938 | 0.0214 | temperature |
| audio | 0.3490 | 0.2846 | 0.0786 | 0.0780 | 0.674 | 0.0773 | temperature |
| video | 0.3294 | 0.1919 | 0.1391 | 0.1489 | 0.299 | 0.0795 | temperature |
| physiology | 0.7037 | 0.8246 | 0.1341 | **0.3061** | 4.064 | **0.1908** | **raw** |

Two things are worth stating plainly:

- **Audio and video are *under*confident** (T < 1 sharpens them). Video's mean
  confidence is 0.192 against 0.329 accuracy. This is the flip side of §8.3:
  audio's posteriors sit near-uniform everywhere, which is why its entropy
  barely separates its right answers from its wrong ones (+0.0064, against
  +0.0718 for text and +0.1974 for image).
- **Physiology is the one case where the fitted temperature is refused.** Its
  test ECE is 0.306 with MCE 0.907 — badly overconfident on 198 windows — and
  the NLL-optimal temperature of 4.06 overshoots so far that validation ECE
  *rises* from 0.134 to 0.191. `recommend_calibration` is ECE-based and returns
  `raw`, which is the correct call and an illustration that NLL-optimal and
  ECE-optimal temperatures are not the same thing.

Calibration is measured and available (`--apply-calibration`) but is **not**
applied in the headline stage-1 numbers above, so the fusion results are
directly comparable with the frozen baselines.

---

## 9. Stage 2, when stage 1 is signed off

1. Decide the anchor (see §1) — audio with one candidate, or text with two.
2. `hsig.py`: `HSIG(active_state, candidate) -> predicted dU`, fitted on the
   **validation** `delta_uncertainty.parquet` records only.
3. `ugapr.py`: `J(m) = dU_hat(m) - lambda*C(m) - mu*L(m)`, over the cost table
   already produced here.
4. `router.py`: the loop, filling in the `RoutingStep` / `RoutingTrace` shape
   that `routing_log.py` already fixes and tests.
5. Threshold selection on validation, then a single locked test evaluation.

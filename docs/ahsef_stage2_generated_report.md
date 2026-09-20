# AHSEF Stage 2 — LLM text pilot results

## Sample selection

- **validation**: 1000 samples, seed 42, fingerprint `2538c2372d27d417`, classes {'neutral': 385, 'happy': 274, 'sad': 115, 'angry': 169, 'fear': 9, 'disgust': 14, 'surprise': 34}
- **test**: 1000 samples, seed 42, fingerprint `f1d5bfff8e280053`, classes {'neutral': 391, 'happy': 267, 'sad': 113, 'angry': 168, 'fear': 9, 'disgust': 15, 'surprise': 37}
- method: `seeded_stratified_largest_remainder_v1`; labels used for inclusion: False; splits disjoint: True

## Frozen configuration

- model: `gemma4:31b-cloud` (provider ollama, host `http://localhost:11434`)
- prompt: `v2_explicit_json`
- temperature 0.0, seed 42, num_predict 700
- uncertainty policy: `score_entropy`
- **τ = 0.257156** (objective `target_stop_accuracy`, selected on validation)
- calibration decision: validation AUROC=0.6283 over 11 distinct confidence values: the self-report orders samples, so a binned recalibration map is methodologically justified and was fitted on validation only. It is NOT the routing signal -- the gate routes on 'score_entropy' -- so this calibration is reported as a property of the model and does not enter the routing path. Routing traces therefore carry calibrated_uncertainty = null.
- frozen at 2026-08-29T18:50:13, git `519b9c2e31b2c3bfc1007a29025df7642e96c1a3`

## Validation results

- parsed/usable: 1000/1000 (100.0%); statuses {'ok': 1000}
- responses carrying class scores: 1000 (100.0%)

### TABLE A — validation: LLM vs existing text baseline (paired, identical samples)

| System | Accuracy | Macro-F1 | Weighted-F1 | Macro Precision | Macro Recall |
|---|---:|---:|---:|---:|---:|
| Existing Text Baseline | 0.3810 | 0.2790 | 0.4044 | 0.2818 | 0.3381 |
| Gemma LLM | 0.4940 | 0.3145 | 0.4735 | 0.3666 | 0.3183 |

### Class-wise — validation

| Emotion | Baseline F1 | LLM F1 | Δ F1 | Baseline Recall | LLM Recall | Δ Recall | n |
|---|---:|---:|---:|---:|---:|---:|---:|
| neutral | 0.4302 | 0.6221 | +0.1919 | 0.3403 | 0.8104 | +0.4701 | 385 |
| happy | 0.4531 | 0.4400 | -0.0131 | 0.4234 | 0.3212 | -0.1022 | 274 |
| sad | 0.3038 | 0.2924 | -0.0114 | 0.3130 | 0.2174 | -0.0957 | 115 |
| angry | 0.4167 | 0.4014 | -0.0152 | 0.4734 | 0.3314 | -0.1420 | 169 |
| fear | 0.0789 | 0.0541 | -0.0249 | 0.3333 | 0.1111 | -0.2222 | 9 |
| disgust | 0.0308 | 0.0930 | +0.0623 | 0.0714 | 0.1429 | +0.0714 | 14 |
| surprise | 0.2393 | 0.2985 | +0.0592 | 0.4118 | 0.2941 | -0.1176 | 34 |

### Uncertainty — validation

- policy `score_entropy`: mean 0.3816, median 0.3862, std 0.2011
- mean top-1 score 0.7442, median 0.8000, mean margin 0.6143
- 'uncertainty' is AHSEF-derived from the normalised self-reported class scores. 'llm_confidence' is the model's own self-report. They are different quantities and neither is a calibrated posterior.

### TABLE B — validation: uncertainty reliability

| Uncertainty Bin | Samples | Accuracy | Error Rate | Mean U |
|---|---:|---:|---:|---:|
| [0.0, 0.1) | 86 | 0.6512 | 0.3488 | 0.0000 |
| [0.1, 0.2) | 69 | 0.5652 | 0.4348 | 0.1646 |
| [0.2, 0.3) | 206 | 0.5777 | 0.4223 | 0.2391 |
| [0.3, 0.4) | 164 | 0.5244 | 0.4756 | 0.3463 |
| [0.4, 0.5) | 196 | 0.5357 | 0.4643 | 0.4486 |
| [0.5, 0.6) | 95 | 0.3895 | 0.6105 | 0.5536 |
| [0.6, 0.7) | 146 | 0.2877 | 0.7123 | 0.6492 |
| [0.7, 0.8) | 32 | 0.2500 | 0.7500 | 0.7469 |
| [0.8, 0.9) | 6 | 0.3333 | 0.6667 | 0.8316 |

Spearman ρ(uncertainty, wrong) = **0.2134** (95% CI 0.1535 to 0.2718); AUROC = **0.6231** over n=1000.

### Calibration — validation

- self-reported confidence: ECE 0.2570, MCE 0.3373, Brier 0.3035, mean confidence 0.7510 vs accuracy 0.4940 (overconfidence +0.2570)
- discrimination: AUROC 0.6283, 11 distinct values, separation 0.0697
- score distribution: ECE 0.2484, NLL 4.7108, Brier 0.7669 — Computed over normalised SELF-REPORTED scores, not a posterior. ECE/NLL here measure how well a self-report tracks correctness.

### Cost and latency — validation

- total 981.2 s; per sample mean 981 ms, median 696 ms, p95 2615 ms
- tokens: 684572 in / 148222 out (mean 685 / 148 per sample)
- monetary cost: **not reported** — Ollama publishes no per-token price for cloud-backed models through this interface. Token counts and latency are reported; no price is invented.
- execution: cloud

## Test results

- parsed/usable: 1000/1000 (100.0%); statuses {'ok': 1000}
- responses carrying class scores: 1000 (100.0%)

### TABLE A — test: LLM vs existing text baseline (paired, identical samples)

| System | Accuracy | Macro-F1 | Weighted-F1 | Macro Precision | Macro Recall |
|---|---:|---:|---:|---:|---:|
| Existing Text Baseline | 0.3710 | 0.2672 | 0.3967 | 0.2740 | 0.2986 |
| Gemma LLM | 0.5210 | 0.3780 | 0.5006 | 0.4157 | 0.3924 |

### Class-wise — test

| Emotion | Baseline F1 | LLM F1 | Δ F1 | Baseline Recall | LLM Recall | Δ Recall | n |
|---|---:|---:|---:|---:|---:|---:|---:|
| neutral | 0.4136 | 0.6341 | +0.2205 | 0.3274 | 0.8133 | +0.4859 | 391 |
| happy | 0.4560 | 0.4433 | -0.0127 | 0.4270 | 0.3221 | -0.1049 | 267 |
| sad | 0.2996 | 0.3425 | +0.0429 | 0.3274 | 0.2743 | -0.0531 | 113 |
| angry | 0.4138 | 0.4604 | +0.0466 | 0.4286 | 0.3810 | -0.0476 | 168 |
| fear | 0.0000 | 0.0769 | +0.0769 | 0.0000 | 0.1111 | +0.1111 | 9 |
| disgust | 0.0323 | 0.3590 | +0.3267 | 0.0667 | 0.4667 | +0.4000 | 15 |
| surprise | 0.2550 | 0.3294 | +0.0744 | 0.5135 | 0.3784 | -0.1351 | 37 |

### Uncertainty — test

- policy `score_entropy`: mean 0.3799, median 0.3640, std 0.2026
- mean top-1 score 0.7452, median 0.8000, mean margin 0.6154
- 'uncertainty' is AHSEF-derived from the normalised self-reported class scores. 'llm_confidence' is the model's own self-report. They are different quantities and neither is a calibrated posterior.

### TABLE B — test: uncertainty reliability

| Uncertainty Bin | Samples | Accuracy | Error Rate | Mean U |
|---|---:|---:|---:|---:|
| [0.0, 0.1) | 86 | 0.7093 | 0.2907 | 0.0000 |
| [0.1, 0.2) | 73 | 0.6164 | 0.3836 | 0.1644 |
| [0.2, 0.3) | 211 | 0.5450 | 0.4550 | 0.2393 |
| [0.3, 0.4) | 160 | 0.5062 | 0.4938 | 0.3428 |
| [0.4, 0.5) | 192 | 0.6094 | 0.3906 | 0.4500 |
| [0.5, 0.6) | 97 | 0.3814 | 0.6186 | 0.5550 |
| [0.6, 0.7) | 137 | 0.3577 | 0.6423 | 0.6483 |
| [0.7, 0.8) | 35 | 0.4286 | 0.5714 | 0.7420 |
| [0.8, 0.9) | 9 | 0.1111 | 0.8889 | 0.8397 |

Spearman ρ(uncertainty, wrong) = **0.1751** (95% CI 0.1144 to 0.2346); AUROC = **0.6010** over n=1000.

### Calibration — test

- self-reported confidence: ECE 0.2397, MCE 0.4000, Brier 0.2966, mean confidence 0.7591 vs accuracy 0.5210 (overconfidence +0.2381)
- discrimination: AUROC 0.6222, 11 distinct values, separation 0.0641
- score distribution: ECE 0.2238, NLL 4.2843, Brier 0.7279 — Computed over normalised SELF-REPORTED scores, not a posterior. ECE/NLL here measure how well a self-report tracks correctness.

### Cost and latency — test

- total 1113.2 s; per sample mean 1113 ms, median 793 ms, p95 2463 ms
- tokens: 684518 in / 148195 out (mean 685 / 148 per sample)
- monetary cost: **not reported** — Ollama publishes no per-token price for cloud-backed models through this interface. Token counts and latency are reported; no price is invented.
- execution: None

### TABLE C — AHSEF text gate (τ frozen on validation)

| Routing Outcome | Count | Percentage |
|---|---:|---:|
| STOP | 339 | 33.9% |
| REQUEST | 661 | 66.1% |

### TABLE D — Conditional performance

| Group | Samples | Accuracy | Macro-F1 | Error Rate |
|---|---:|---:|---:|---:|
| All | 1000 | 0.5210 | 0.3780 | 0.4790 |
| STOP | 339 | 0.5929 | 0.1095 | 0.4071 |
| REQUEST | 661 | 0.4841 | 0.3947 | 0.5159 |

> REQUEST means only that the textual evidence was insufficient under the validation-selected threshold. It is NOT evidence that another modality would improve the prediction; no modality was acquired or fused.

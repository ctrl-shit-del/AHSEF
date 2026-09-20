# AHSEF Stage 2 — LLM text pilot: results

Controlled pilot, 1,000 validation + 1,000 locked test samples, run against
`gemma4:31b-cloud` through the Ollama provider.

> **Status note.** The Ollama provider did not exist in the repository when this
> stage began — only `anthropic_provider.py`, `ReplayProvider`, and
> `ScriptedProvider` were present, and `src/` contained no `ollama` reference.
> The model was pulled and the daemon was running, so the provider was
> implemented against the live API and its behaviour probed rather than assumed.

---

## 1. Experimental objective

Determine whether LLM text uncertainty is a useful **first-stage AHSEF routing
signal** — not whether the LLM is the best emotion classifier. Specifically:
can the system decide, per sample and without seeing a label, whether the
textual evidence is sufficient?

## 2. Sample selection

| | validation | test |
|---|---|---|
| selected | 1,000 of 22,100 | 1,000 of 23,315 |
| fingerprint | `2538c2372d27d417` | `f1d5bfff8e280053` |
| datasets | MSP-Podcast 937, MELD 63 | MSP-Podcast 879, MELD 121 |
| neutral / happy / sad / angry / fear / disgust / surprise | 385 / 274 / 115 / 169 / 9 / 14 / 34 | 391 / 267 / 113 / 168 / 9 / 15 / 37 |

- Method `seeded_stratified_largest_remainder_v1`, seed 42, proportional to the
  population class balance — **not rebalanced**, and labels allocate quotas only,
  never gate inclusion.
- Selection is order-independent (the frame is sorted by `sample_id` first), so
  the same seed reproduces the identical ordered id list.
- **Validation ∩ test = 0**, asserted by `assert_disjoint` and re-verified
  directly.
- Each manifest carries a SHA-256 fingerprint over its ordered ids; an edited
  manifest is refused on load.

## 3–6. Model, provider, prompt, generation configuration

| | |
|---|---|
| provider | Ollama 0.33.2, `http://localhost:11434`, stdlib HTTP |
| model (exact) | **`gemma4:31b-cloud`** → resolves to `gemma4:31b` |
| execution | **cloud-backed** (`remote_host: https://ollama.com`) — not local |
| size / precision | 32.7B, BF16, context 262,144 |
| digest | `ef09f235533c96cd75e8deed88c628335cb69e2b3ce96275d0d7a67fe9887aba` |
| prompt | `v2_explicit_json`, SHA-256 `a6e8e7d02edb101f…` |
| generation | `temperature=0`, `seed=42`, `num_predict=700`, `think=false` |
| determinism | **reproducible** — Ollama honours temperature and seed |

### Two findings that changed the design

**`format` (JSON Schema) is not enforced for this model.** Probed with prompt
v1, it returned `{"label": "happy", "evidence_strength": "high", …}` — key
renamed, a word where a number was required, four required keys missing, and the
object wrapped in a markdown fence. Prompt **v2** therefore carries the contract
explicitly (exact key names and types). v1 is unchanged and remains valid for
backends that do enforce a schema. Markdown-fence stripping was added as
*envelope decoding*, explicitly distinguished from content repair — prose mixed
with an object is still rejected.

**Sampling is controllable here.** Unlike the hosted Claude models (no seed,
`temperature` rejected), Ollama accepts both. So the pilot runs `--repeats 1`:
repeated sampling at `temperature=0` would measure nothing.

## 7–8. Results — TABLE A

Paired: the frozen text baseline is the Stage 1 export **restricted to exactly
these sample ids**, never retrained or re-inferred.

### Validation (n = 1,000)

| System | Accuracy | Macro-F1 | Weighted-F1 | Macro Precision | Macro Recall |
|---|---:|---:|---:|---:|---:|
| Existing Text Baseline | 0.3810 | 0.2790 | 0.4044 | 0.2818 | 0.3381 |
| Gemma LLM | **0.4940** | **0.3145** | **0.4735** | **0.3666** | 0.3183 |

Parsing: **1,000/1,000 usable (100%)**, zero fenced responses, zero call
failures, zero abstentions, zero unmappable labels.

## 9. Class-wise — validation

| Emotion | Baseline F1 | LLM F1 | Δ F1 | Baseline R | LLM R | Δ R | n |
|---|---:|---:|---:|---:|---:|---:|---:|
| neutral | 0.4302 | **0.6221** | **+0.1919** | 0.3403 | 0.8104 | +0.4701 | 385 |
| happy | 0.4531 | 0.4400 | −0.0131 | 0.4234 | 0.3212 | −0.1022 | 274 |
| sad | 0.3038 | 0.2924 | −0.0114 | 0.3130 | 0.2174 | −0.0957 | 115 |
| angry | 0.4167 | 0.4014 | −0.0152 | 0.4734 | 0.3314 | −0.1420 | 169 |
| fear | 0.0789 | 0.0541 | −0.0249 | 0.3333 | 0.1111 | −0.2222 | 9 |
| disgust | 0.0308 | **0.0930** | +0.0623 | 0.0714 | 0.1429 | +0.0714 | 14 |
| surprise | 0.2393 | **0.2985** | +0.0592 | 0.4118 | 0.2941 | −0.1176 | 34 |

**The LLM's entire accuracy gain comes from `neutral`.** Its neutral recall is
0.81 against the baseline's 0.34, and its recall *falls* on every other class.
The model is neutral-biased: on this data it answers `neutral` far more often
than the 38.5% base rate. That raises accuracy on a neutral-heavy corpus while
leaving macro-F1 — which weights all seven classes equally — only 0.036 higher.

`fear` (n=9) and `disgust` (n=14) are too small for their F1 differences to
carry weight; they are reported for completeness, not as findings.

## 10. Uncertainty — validation

Two quantities are kept strictly distinct, and both are reported:

- **`llm_confidence`** — the model's self-report. Not a posterior.
- **AHSEF-derived uncertainty** — computed from the normalised self-reported
  class scores. Also not a posterior.

Four candidate signals were compared **on validation only**, by AUROC of
uncertainty against being wrong:

| Policy | AUROC | Spearman ρ | distinct values |
|---|---:|---:|---:|
| `llm_confidence` (1 − self-report) | 0.6283 | +0.2262 | 11 |
| `score_entropy` (normalised entropy of scores) | 0.6231 | +0.2134 | 113 |
| `score_top1` (1 − max_c score_c) | 0.6118 | +0.1955 | 50 |
| `ambiguity` (self-reported field) | 0.5589 | +0.1031 | 12 |

`score_entropy` was kept. The AUROC gap to the best (0.0052) is far inside noise
at n=1,000; it is AHSEF-derived rather than a raw self-report; and it has 10×
more distinct values, giving much finer threshold control. The whole comparison
is recorded in `frozen_config.json` so the choice is auditable.

Distribution: mean 0.3816, median 0.3862; mean top-1 score 0.7442; mean top-1/
top-2 margin 0.6143.

## 11. Uncertainty vs correctness — TABLE B (validation)

| Uncertainty Bin | Samples | Accuracy | Error Rate |
|---|---:|---:|---:|
| [0.0, 0.1) | 86 | 0.6512 | 0.3488 |
| [0.1, 0.2) | 69 | 0.5652 | 0.4348 |
| [0.2, 0.3) | 206 | 0.5777 | 0.4223 |
| [0.3, 0.4) | 164 | 0.5244 | 0.4756 |
| [0.4, 0.5) | 196 | 0.5357 | 0.4643 |
| [0.5, 0.6) | 95 | 0.3895 | 0.6105 |
| [0.6, 0.7) | 146 | 0.2877 | 0.7123 |
| [0.7, 0.8) | 32 | 0.2500 | 0.7500 |
| [0.8, 0.9) | 6 | 0.3333 | 0.6667 |

Accuracy falls from 0.65 in the lowest bin to 0.25 in the [0.7, 0.8) bin —
broadly monotone, with a plateau across [0.2, 0.5). Spearman ρ = **0.2134**,
AUROC = **0.6231** (n = 1,000).

This is a **real but weak** association. It is enough to separate a
better-than-average STOP set from a worse-than-average REQUEST set; it is not
enough to make individual STOP decisions trustworthy.

## 12. Calibration — validation

| Quantity | ECE | MCE | Brier | NLL |
|---|---:|---:|---:|---:|
| `llm_confidence` (self-report) | 0.2570 | 0.3373 | 0.3035 | — |
| normalised score distribution | 0.2484 | — | 0.7669 | 4.7108 |

**The model is severely overconfident**: mean confidence 0.7510 against accuracy
0.4940 — a gap of **+0.2570** — and *every* populated reliability bin is
overconfident (gaps +0.10 to +0.34). Confidence when correct is 0.7863 vs 0.7165
when wrong: a separation of only 0.070.

The score distribution's NLL of **4.71** is decisive on the central question of
whether these are probabilities: a uniform 7-class distribution scores ln 7 =
1.946. A "distribution" that is *worse than uniform* by this margin is placing
near-zero mass on the true class regularly. **These scores are not posteriors,
and the code never treats them as such.**

**Calibration decision.** A binned recalibration map was judged justified for
`llm_confidence` (validation AUROC 0.6283 > 0.55 over 11 distinct values) and
fitted on validation only. But the gate routes on `score_entropy`, a *different*
quantity — so the map **does not enter the routing path**, and routing traces
correctly carry `calibrated_uncertainty: null`. The artefact states this scope
explicitly rather than recording a bare "calibration applied".

## 13. Selected τ

| | |
|---|---|
| objective | `target_stop_accuracy` = 0.60 (configured a priori) |
| rule | widest coverage whose STOP set still reaches the accuracy floor |
| **τ** | **0.257156** |
| at τ on validation | coverage 32.5%, STOP accuracy 0.6000, REQUEST accuracy 0.4430 |
| selected on | validation, `uses_test_labels: false` |

Threshold sweep (validation), showing the trade-off the objective resolves:

| τ | STOP % | STOP acc | REQUEST acc |
|---:|---:|---:|---:|
| 0.0000 | 8.6% | 0.6512 | 0.4792 |
| 0.2271 | 19.3% | 0.6062 | 0.4672 |
| 0.2631 | 35.8% | 0.5894 | 0.4408 |
| 0.3640 | 49.8% | 0.5803 | 0.4084 |
| 0.4848 | 68.2% | 0.5601 | 0.3522 |
| 0.6134 | 84.8% | 0.5307 | 0.2895 |
| 0.7081 | 96.9% | 0.5015 | 0.2581 |

## 14–16. Test results — LOCKED

1,000 samples, τ frozen on validation beforehand. Parsing 1000/1000 usable
(100%), zero failures. 711 live calls in the resumed invocation + 289 recorded
earlier.

### TABLE A — test (paired, identical sample ids)

| System | Accuracy | Macro-F1 | Weighted-F1 | Macro Precision | Macro Recall |
|---|---:|---:|---:|---:|---:|
| Existing Text Baseline | 0.3710 | 0.2672 | 0.3967 | 0.2740 | 0.2986 |
| Gemma LLM | **0.5210** | **0.3780** | **0.5006** | **0.4157** | **0.3924** |

### Class-wise — test

| Emotion | Base F1 | LLM F1 | Δ F1 | Base R | LLM R | Δ R | n |
|---|---:|---:|---:|---:|---:|---:|---:|
| neutral | 0.4136 | 0.6341 | +0.2205 | 0.3274 | 0.8133 | +0.4859 | 391 |
| happy | 0.4560 | 0.4433 | -0.0127 | 0.4270 | 0.3221 | -0.1049 | 267 |
| sad | 0.2996 | 0.3425 | +0.0429 | 0.3274 | 0.2743 | -0.0531 | 113 |
| angry | 0.4138 | 0.4604 | +0.0466 | 0.4286 | 0.3810 | -0.0476 | 168 |
| fear | 0.0000 | 0.0769 | +0.0769 | 0.0000 | 0.1111 | +0.1111 | 9 |
| disgust | 0.0323 | 0.3590 | +0.3267 | 0.0667 | 0.4667 | +0.4000 | 15 |
| surprise | 0.2550 | 0.3294 | +0.0744 | 0.5135 | 0.3784 | -0.1351 | 37 |

The neutral bias persists: the LLM predicts `neutral` on **61.2%** of test
samples against a **39.1%** base rate. But unlike validation, macro-F1 also
improves substantially (+0.111), driven by `disgust` (+0.327) and `neutral`
(+0.221). Recall still falls on happy, sad, angry and surprise.

### TABLE B — uncertainty reliability (test)

| Uncertainty Bin | Samples | Accuracy | Error Rate |
|---|---:|---:|---:|
| [0.0, 0.1) | 86 | 0.7093 | 0.2907 |
| [0.1, 0.2) | 73 | 0.6164 | 0.3836 |
| [0.2, 0.3) | 211 | 0.5450 | 0.4550 |
| [0.3, 0.4) | 160 | 0.5062 | 0.4938 |
| [0.4, 0.5) | 192 | 0.6094 | 0.3906 |
| [0.5, 0.6) | 97 | 0.3814 | 0.6186 |
| [0.6, 0.7) | 137 | 0.3577 | 0.6423 |
| [0.7, 0.8) | 35 | 0.4286 | 0.5714 |
| [0.8, 0.9) | 9 | 0.1111 | 0.8889 |

Spearman ρ = **0.1751**, AUROC = **0.6010**
(n=1000). Weaker than validation (0.2134 / 0.6231) and **non-monotone**: the
[0.4,0.5) bin scores 0.6094, higher than three lower-uncertainty bins.

### Calibration — test

| Quantity | ECE | MCE | Brier | NLL |
|---|---:|---:|---:|---:|
| `llm_confidence` | 0.2397 | 0.4000 | 0.2966 | — |
| score distribution | 0.2238 | — | 0.7279 | **4.2843** |

Mean confidence 0.7591 vs accuracy 0.5210
(overconfidence +0.2381). NLL 4.2843 again exceeds
ln 7 = 1.946 — worse than uniform. AUROC 0.6222 over
11 distinct values; the validation-fitted
recalibration map transfers, but it applies to `llm_confidence`, not the routing
signal, so it stays out of the routing path.

### TABLE C — AHSEF text gate (τ = 0.257156, frozen on validation)

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

**The gate separates on accuracy but inverts on macro-F1.** STOP accuracy
0.5929 vs REQUEST 0.4841
— a gap of only +0.1088,
and STOP is barely above the population's 0.5210.
Meanwhile STOP macro-F1 (0.1095) is *far below*
REQUEST (0.3947).

The reason is visible in the class-wise table: the STOP set is where the model
confidently answers `neutral`. It collects the easy majority-class samples, so
its accuracy edge is real but its per-class competence is worse than the set it
routes away. A gate optimised for accuracy on a skewed corpus can select for
"predicts the majority class confidently" rather than "is reliable".

The validation-side figures at the same τ (32.5% STOP, 0.6000 / 0.4430) were
close on accuracy, so τ transferred; the macro-F1 inversion was present there
too (0.1138 vs 0.3277) and is not a test-set artefact.

## 17. Cost and latency

**No monetary cost is reported.** Ollama publishes no per-token price for
cloud-backed models through this interface, so inventing one would be
fabrication. Token counts and measured latency are recorded instead.

### Correction: recorded latency is generation time, not service time

`latency_ms` in the artefacts comes from Ollama's `total_duration`, which
measures **generation only**. Summed across the validation split it is 16.4
minutes against 57.2 minutes of wall clock — so the recorded figure captures
just **29%** of the time actually spent. The remainder is queueing and transport
on Ollama's cloud.

| | recorded (generation) | true end-to-end |
|---|---:|---:|
| validation, per sample | 981 ms mean, 696 ms median, 2,615 ms p95 | **~3,432 ms** (57.2 min / 1,000) |

Both numbers matter and neither replaces the other: generation time is what the
model costs, end-to-end is what a routing system would actually wait for. A
latency-aware UGAPR must use the end-to-end figure.

**Throughput is not stable.** Mid-way through the test split, throughput
collapsed from ~18 calls/min to ~1 call/min for roughly two hours, then
recovered to ~18/min — with zero errors and unchanged generation latency
throughout. This is server-side throttling on the cloud tier, invisible in
`total_duration`. Any Stage 3 cost model that treats cloud LLM latency as a
stable per-sample constant will be wrong.

Validation tokens: 684,572 input / 148,222 output (mean 685 / 148 per sample).

## 18. Failure / parsing rate

Validation: **0%**. 1,000/1,000 responses parsed, mapped, and scored. No
markdown fences, no malformed JSON, no missing fields, no out-of-range scalars,
no abstentions, no unmappable or contested labels, no call failures.

## 19. Reproducibility

Recorded in `frozen_config.json` and each prediction sidecar: exact model tag
and digest, execution location, prompt version and SHA-256, temperature, seed,
`num_predict`, uncertainty policy, τ, selection seed and manifest fingerprints,
git commit, platform, and Python version. Every call is written to
`llm/transcript_<split>.jsonl`, so the entire analysis re-runs offline with
`--provider replay`. No credentials appear in any artefact.

### Incident: the test run was interrupted and resumed

The first locked-test invocation died at call ~290/1000 with a Windows
`PermissionError` writing the transcript — a transient file lock (an antivirus
scanner or indexer opening the growing `.jsonl`), not a logic error. The 289
recorded calls were intact and uncorrupted.

Re-running from zero would have spent 1,000 further API calls, taking the pilot
to 2,290 samples and breaking its 2,000-sample budget. The run was therefore
made **resumable**, and the transcript promoted to the single source of truth:

- live calls are made only for sample ids the transcript does not already cover;
- the prediction set is then **always** built by replaying the *complete*
  transcript over the full manifest, so the artefact is identical whether one
  invocation produced it or several;
- `provenance.execution` on the prediction sidecar records `resumed`,
  `previously_recorded`, and `live_calls_this_invocation`.

Hardening added at the same time: transcript writes retry a transient lock with
backoff and flush per record (so the file is always a valid resume point); a
truncated *final* line is dropped and counted, while corruption anywhere else
causes the transcript to be refused rather than silently trusted.

**Total API spend: 1,000 validation + 289 + 711 = exactly 2,000 samples**, as
budgeted. Nothing about the frozen configuration changed: τ was selected and
frozen on validation *before* the interruption, and the resumed run loads
`frozen_config.json` and aborts on any drift in model, prompt, temperature,
seed, or uncertainty policy.

## 20. Limitations

1. **Neutral bias drives the headline number.** The accuracy gain is almost
   entirely one class on a neutral-heavy corpus. Macro-F1, which does not
   reward that, improves by only 0.036.
2. **Uncertainty is weakly informative.** AUROC 0.62 means the signal orders
   samples only a little better than chance.
3. **Self-reports are badly calibrated** (ECE 0.257, uniformly overconfident)
   and the score vector is not a probability distribution (NLL 4.71 > ln 7).
4. **1,000 samples per split** gives wider intervals than the baselines'
   full-split numbers. The comparison is paired, which helps, but rare classes
   (fear n=9, disgust n=14) support no conclusion.
5. **Contamination is unquantified.** MELD and MSP-Podcast are public corpora
   that may appear in pretraining data. Nothing here can rule that out.
6. **REQUEST does not mean another modality would help.** It means only that
   textual evidence was insufficient under the validation-selected threshold.
   No modality was acquired or fused.
7. **Single model, single prompt.** No ablation over either.
8. **Latency figures need care.** Recorded per-call latency is generation time
   (29% of wall clock); cloud throughput varied 18× within one run. See §17.

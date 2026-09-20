# AHSEF Stage 3 — dynamic Text → Audio acquisition and fusion

*Method and design decisions. The measured results are in
`docs/ahsef_stage3_results.md`, which is generated from the run artefacts and
never written by hand.*

---

## 0. What this stage is, and what it is not

Stage 1 built the evidence layer and discovered the fact that constrains
everything after it: the five frozen baselines were sampled independently from
different corpora, and **no sample anywhere has two simultaneously available
candidate modalities**. Stage 2 added a large language model as a text evidence
source and measured that its self-reported uncertainty orders samples only
weakly (AUROC 0.623), concluding explicitly that the uncertainty gate *should
not* be treated as a reliable final routing policy. Stage 2 acquired nothing and
fused nothing.

Stage 3 closes the loop for the one modality pair the data actually supports:

```
Text/LLM  →  uncertainty + evidence state
          →  HSIG:  expected_gain(audio | state)
          →  UGAPR: J = gain − λ·cost − μ·latency
          →  J ≥ τ ?  REQUEST_AUDIO : STOP
          →  if REQUEST_AUDIO:  audio inference → fusion → new prediction
          →  final decision
```

The claim this earns is stated once, and the code carries it as a constant so it
cannot drift:

> AHSEF dynamically determines whether additional audio evidence is worth
> acquiring after an initial LLM-based text assessment, using a
> validation-trained expected-gain estimator and a cost/latency-aware utility
> policy.

It is **not** a demonstration that AHSEF selects the best modality among all
modalities. The candidate set has exactly one member. Physiology remains a
separate three-class task and is never fused into the seven-class emotion task.
No cross-corpus pairing (audio+video, audio+image, audio+physiology) was
fabricated.

---

## 1. Phase A — the aligned pool, and why it had to be rebuilt

`src/ahsef/stage3/pool.py`

The pool is the **co-split** intersection of the audio and text experiment
manifests: sample ids present in the *same* split of both. Ids shared across
*different* splits are excluded and counted, because one model trained on them.
Seven properties are verified and each raises rather than warns — identical
sample ids, identical split assignment, no train contamination, a compatible
seven-class label space, agreeing ground truth per id, recorded provenance for
both modalities, and record-level availability of both payloads.

Two details of that last check earned their own code:

- **Text is file-backed.** MSP-Podcast records carry `text_source == "file"`
  with an empty `text` column and the transcript on disk. A check that read the
  column would have called all 509 samples text-less; the pool resolves text
  exactly as `EmotionTextDataset.read_text` does, so "text is available" means
  the same thing to the pool as it did to the frozen baseline.
- **Audio is checked on disk**, not merely declared. A `has_audio` flag with a
  path that does not resolve is an unavailable modality.

### The pool could not reuse the Stage 2 pilot

The Stage 2 pilot drew 1,000 validation and 1,000 test samples from the text
split. Only **20 validation and 18 test** of those fall inside the aligned
Text+Audio pool. Routing on 20 samples would not be an experiment. So Stage 3
records its own LLM transcript over the aligned pool, using the Stage 2 frozen
configuration unchanged — same model, same prompt version, same decoding
parameters, same uncertainty policy. What changed is *which samples the model
was asked about*, never *how it was asked*. `--stage llm` refuses to run if any
of those values differs from the Stage 2 freeze.

Both Stage 2 transcripts are preserved untouched and are still replayable; a
regression test asserts it.

---

## 2. Phase B — frozen baselines, paired by id

The audio side is the Stage 1 export restricted to the pool. Nothing is
retrained, re-inferred, or modified; the checkpoint SHA-256 recorded by Stage 1
is carried into the Stage 3 frozen config. The text side is the Gemma
transcript, replayed in full to build the prediction set, so the artefact is
identical whether the pass finished in one invocation or five.

Samples the LLM refused, abstained on, or answered unmappably are **kept in the
frame with `predicted_class == -1`**, excluded from the routing experiment by an
explicit mask, and counted in the coverage report. They are never given a filler
prediction.

`prob_*` columns on the text side hold **normalised self-reported class scores,
not posteriors** — Stage 2 measured their NLL at 4.71 against uniform's 1.946.
Every artefact that consumes them says so, and no calibration figure computed
over them is presented as posterior calibration.

---

## 3. Phase C — the oracle acquisition value

`src/ahsef/stage3/oracle.py`

For every aligned sample the pipeline actually pays for audio, fuses, and
records the outcome. This is the empirical quantity HSIG will later be asked to
predict *without* paying, and the two are kept strictly apart.

The target is the one Stage 1's diagnostics forced:

```
gain(m | A) = P(correct | A ∪ {m}) − P(correct | A)
```

whose per-sample realisation is `signed_gain = fused_correct − text_correct` ∈
{−1, 0, +1}. Two derived events are reported alongside it: `y_gain` (audio fixed
a wrong text prediction) and `y_harm` (audio broke a correct one).

**Delta-uncertainty is computed and written out as a diagnostic column only.**
Stage 1 measured that its sign flips with the anchor and the fusion rule and
that its peak per-sample correlation with real improvement was 0.143.
`assert_target_not_delta_uncertainty` refuses a target column whose name says
otherwise, so the rejected target cannot creep back in.

### Fusion

Fixed, weight-parameterised probability fusion — no trained fusion network. A
trained fuser would confound the question: a gain could come from the modality
or from the extra parameters. Both rules (`weighted_probability` and
`log_opinion_pool`) are scanned on validation, the weight is chosen by
**macro-F1** rather than accuracy (for the reason in §5), and the whole scan is
recorded inside the spec. `choose_fusion_spec` raises on any split but
train/validation.

---

## 4. Phase D — a real HSIG

`src/ahsef/stage3/features.py`, `hsig_model.py`, `hsig_quality.py`

Stage 2 shipped only `UnavailableHSIG`, which declines to guess. Stage 3
replaces it with a fitted estimator.

### Features

Everything HSIG may condition on is declared in one list. All of it is
computable at inference time from the LLM's own response: routing uncertainty,
top-1 and top-2 scores, margin, score entropy, top-2 mass, the count of
non-zero-scored classes, and the model's three self-reports (confidence,
ambiguity, evidence strength). `assert_label_free` re-checks the built frame
against a deny-list of label-bearing names, so an edit that reaches for
`true_class` fails loudly rather than quietly inflating every reported AUROC.

**Nothing is imputed.** A sample missing a feature gets no estimate: HSIG
returns `expected_improvement=None` with a reason, and the router logs it. A
mean or zero fill would make the estimator most confident exactly where it knows
least.

**Class identity is optional and audited.** Conditioning on the predicted class
is legitimate, but it is also the shortest path to a hard-coded routing rule
wearing a learned model's clothes. So it lives in a separate feature set, and a
class-rule audit measures the spread of predicted gain *within* each predicted
class against the spread *between* classes. If the estimator has collapsed to a
class lookup the report says so; if the frozen feature set contains no class
indicator at all, the audit says that instead, because such a model cannot be a
class lookup as a matter of construction.

Three feature sets are fitted and compared:

| Set | Features |
| --- | --- |
| `uncertainty_only` | the routing uncertainty alone — Stage 2's signal |
| `evidence_only` | the ten evidence-state features above |
| `evidence_plus_class` | those ten plus seven predicted-class indicators |

`uncertainty_only` is a **candidate, not merely a reference**. If one feature
predicts the gain as well as ten, then one feature is what should be frozen, and
saying so is a result rather than a disappointment.

### Estimator

Two shapes are fitted and compared:

| | |
| --- | --- |
| `paired_logistic` | fit `P(text correct \| x)` and `P(fused correct \| x)` separately, subtract. This is the target definition written down. It uses every sample and can predict a **negative** gain — which matters, because the acquisitions a routing policy most needs to avoid are the ones that break a correct prediction. |
| `fix_logistic` | one classifier on the binary `y_gain`. Simpler, but structurally blind to harm: its output lives in [0, 1] and can never say "acquiring will hurt". |

Both are linear, standardised, and seeded. The frozen estimator is serialised as
**plain JSON coefficients, not a pickle**: a routing decision must be auditable
and replayable from the artefact, without scikit-learn. A shallow random forest
is scored as a non-linear reference point and reported, but never frozen — if it
wins materially, that is a finding to state, not a reason to swap it in
silently.

### Which variant is frozen

The rule is stated in `HSIG_SELECTION_RULE` in terms of properties of the
estimators, so it can be applied without reference to the numbers it will meet:

1. **Primary.** Highest out-of-fold AUROC against `y_gain` on validation, over
   *every* fitted variant including the single-feature ones.
2. **Tie-break.** Among variants whose AUROC falls inside the bootstrap 95%
   interval of the best — variants this pool cannot tell apart — prefer
   `paired_logistic`.

The tie-break is a units argument. UGAPR computes `J = gain − λC − μL`, so the
gain has to be in the units the target is defined in: a *difference* of
probabilities of being correct. `paired_logistic` emits exactly that.
`fix_logistic` emits P(audio fixes a wrong answer), which lives in [0, 1], is a
different quantity, and cannot express harm at all — subtracting a cost from it
would be a category error dressed as a utility. The artefact records how much
AUROC the tie-break conceded, so the trade is visible.

### Honesty about 509 samples

Every quality figure is **out of fold** (seeded stratified K-fold). At this pool
size, with roughly one positive event in six, an in-sample AUROC is optimistic by
an amount that changes the conclusion. Two deliberately weak references are
scored by the same code path:

- `prior_constant` — the base rate, repeated. The floor.
- `uncertainty_only_logistic` — Stage 2's signal re-expressed as a gain
  estimate. The thing HSIG is meant to improve on.

Bootstrap 95% intervals accompany every headline statistic, and
`compare_estimators` states plainly whether the candidate's interval excludes the
reference — that is, whether this pool can establish an improvement at all.
AUPRC is always reported against the base rate, never against 0.5.

---

## 5. Phase F — why the Stage 2 gate objective was discarded

`src/ahsef/stage3/objective.py`

Stage 2 selected its threshold with `target_stop_accuracy`: the widest coverage
whose stopped set still reached an accuracy floor. On a corpus that is 38.5%
neutral, with an LLM that answers `neutral` 61.8% of the time, that objective did
exactly what it was asked and exactly the wrong thing — it found the
neutral-heavy region, reported a high stopped-set accuracy, and let macro-F1
collapse. **An objective that cannot see the minority classes will always prefer
a policy that ignores them.**

Stage 3 therefore scores a whole *routed system*, not a stopped subset, with a
per-class-aware objective:

```
utility(τ) = macro_f1(τ) − α · acquisition_rate(τ) − β · normalized_latency(τ)
```

where the prediction for each sample is the text prediction when the router
stops and the fused prediction when it acquires — what the deployed system would
actually output.

α is the macro-F1 the policy must gain to justify acquiring audio for *every*
sample; α = 0.05 means an across-the-board acquisition has to be worth 0.05
macro-F1, and acquiring for 20% of samples has to be worth 0.01. β prices the
added latency on the same scale.

Four candidate objectives (accuracy, macro-F1, balanced accuracy, and the two
penalised forms) are swept and reported. **The selected one is declared in
advance in `SELECTED_OBJECTIVE`, with its reason**, not picked afterwards from
the numbers; the comparison exists so the choice is auditable, exactly as Stage
2's uncertainty-policy comparison was. Sensitivity of the chosen threshold to α
and β across a grid is recorded in the frozen config, so no one has to take one
operating point as inevitable.

`select_gate_threshold` raises `ThresholdLeakageError` on any split but
train/validation.

### Which scores τ is chosen against

τ thresholds a *specific function's* output, so it has to be chosen against the
output of the function the router will actually evaluate: the deployed
estimator, fitted on all of validation. Out-of-fold scores come from K
separately fitted models and are a different random variable. This is not a
theoretical worry — the first Stage 3 run selected τ on out-of-fold scores and
applied it to the deployed estimator, and the resulting policy landed *below* a
plain uncertainty threshold because it was selecting a different, worse set of
samples. The mismatch was found by the ablation suite and fixed before the test
split was opened.

So the split of duties is:

- **HSIG quality** is reported out of fold, always. That is the honest measure of
  how well the estimator generalises.
- **τ** is chosen against the deployed estimator's own validation scores, because
  that is the function it will be applied to. Choosing one scalar in-sample
  carries its own optimism, so the out-of-fold sweep is computed anyway and
  recorded beside the frozen one in `gate.out_of_fold_comparison`, where the size
  of that optimism can be read off rather than taken on trust.

---

## 6. Phase E — pricing the candidate

`src/ahsef/stage3/costs.py`

UGAPR subtracts `λ·C(m) + μ·L(m)` from the estimated gain, so an invented cost
model would decide the routing policy by fiat. Three decisions:

**The LLM is priced too.** `text_llm` is not one of the five torch baselines, so
its compute proxy is built with the same formula — `parameters × input_elements`
— from the model's declared 32.7B parameters and its measured mean input-token
count. The LLM and the audio CNN then sit on one scale rather than two
incomparable ones.

**Latency means wall clock, not generation time.** Stage 2 measured generation at
roughly 29% of wall clock with an 18× throughput swing inside one run. The
Stage 3 LLM pass measures both and records the ratio; the cost model prices the
LLM at the measured end-to-end service time, and the artefact states that this is
a point estimate of a varying quantity, not a service guarantee. If a pass is
fully resumed from transcript and no live call was made, the timing record says
`measured: false` rather than reusing an earlier run's number.

**Normalisation is over the whole registry, not the candidate set.** With one
candidate, normalising over the candidate set would hand audio a normalised cost
of exactly 1.0 by construction, making the penalty an artefact of there being
nothing to compare against. Costs are normalised over every modality whose cost
this project has measured.

### What UGAPR can and cannot demonstrate here

With `{audio}` alone, UGAPR demonstrates the **accept/reject** half of its job —
does this candidate clear its own price? — and **not** the ranking half. The code
says so in `SINGLE_CANDIDATE_LIMITATION`, and that string is stamped into every
routing trace. Audio is not selected because it is the only option: a sample
whose utility falls below τ is stopped with audio unacquired, and the ablations
report how often that happens.

---

## 7. Phase G — the router

`src/ahsef/stage3/router.py`

Four properties are structural rather than conventional.

**The router never receives a label.** `TextAudioRouter.route(state)` takes a
`Stage3State` and nothing else — there is no parameter through which ground truth
could arrive. A test asserts the signature. Truth is attached to the decision
records only afterwards, by `attach_truth`, for evaluation.

**The decision is not a hard-coded uncertainty rule.** Uncertainty is one feature
among several inside HSIG; the acquisition decision is a threshold on the UGAPR
utility, which is the estimated gain net of the measured price. Ablation D is the
uncertainty-threshold policy, given the same objective and the same freedom to
choose its own threshold on validation, precisely so the difference can be
measured rather than asserted.

**Worth and availability are asked separately.** UGAPR is called twice per
sample: once with every candidate marked available, answering *is this worth its
price?* — the question the threshold applies to — and once with real
availability, answering *what is actually obtainable?* Conflating them would
report "did not acquire" for a sample the router very much wanted to acquire and
could not, turning a data-availability failure into a policy decision.

**An unavailable modality is reported, never imputed.** A requested acquisition
that cannot be fulfilled yields `REQUESTED_BUT_UNAVAILABLE`: the text-only
prediction stands, **no fused prediction is produced**, and the stop reason is the
dedicated `requested_modality_unavailable` — distinct from `no_candidate_available`,
because "audio was not worth it" and "audio was worth it and we could not get it"
are opposite findings about the policy. Nothing is zero-filled, uniform-filled,
set to neutral, or substituted with another modality.

The aligned pool has audio for every sample by construction, so that branch would
otherwise never fire on real data. The ablation suite therefore replays the Stage
2 validation pilot samples that fall *outside* the pool — real samples with no
co-split audio counterpart anywhere in the project — through the same frozen
router, at zero LLM cost, and verifies on real data that no fused prediction was
produced and every final prediction equals the text prediction.

Each sample produces two artefacts: a Stage 3 `RoutingDecision` carrying every
field Phase G requires, and a Stage 1 `RoutingTrace`, so Stage 3 remains auditable
by Stage 1's own tooling.

---

## 8. Phases H and I — the comparison, and the control

Every system is scored by the same function on the same samples, so a difference
between two rows is a difference between two policies and nothing else. A system
is fully described by two arrays: which samples it acquired audio for, and what
it therefore predicted.

The ablations answer the question the headline table cannot — *is the benefit
coming from intelligent routing, or simply from adding audio?* — by including the
two systems that need no router at all, the policy Stage 2 would have used, the
router with its cost model removed, and the router with a perfect gain estimate
substituted for HSIG (the ceiling any estimator could reach).

One control matters more than the rest. **A router that acquires for 40% of
samples and beats text-only has proved nothing until it also beats acquiring for
a *random* 40%.** `random_policy_reference` runs that control at the router's own
acquisition rate over 200 seeded draws and reports the 95% range beside the
router's point estimate.

The ablations also settle, rather than infer, *which of these are the same
policy*. Equal headline metrics can hide different sample sets and different
sample sets can hide equal metrics, so `_policy_equivalence` compares the
acquisition masks of D, E and G sample by sample and reports the pairwise
agreement. This matters directly: if thresholding HSIG's output selects exactly
the samples a tuned uncertainty threshold selects, then HSIG adds nothing over
Stage 2's signal on this pool, and the report has to say so in those words rather
than let three identical rows imply three distinct achievements.

---

## 9. Phase J — the lock

`frozen_config.json` records the LLM model and prompt version, the decoding
parameters, the audio checkpoint digest, the uncertainty policy, the HSIG
estimator type / feature set / feature definition / target definition /
fingerprint, the fusion rule and weights, the gate objective and its rationale,
τ, λ, μ, α, β, the cost and latency definitions with their measurements, both
pool fingerprints, every seed, the git commit, and the software versions.

**The acquisition price is frozen with everything else.** UGAPR subtracts
`λC + μL` from the gain, so re-measuring C and L on the evaluation split would
shift the effective threshold while every field in the configuration still
matched. The locked run therefore prices audio from
`frozen_config.costs` via `cost_model_from_frozen`, and the test split's own
measured latency is recorded beside it as an observation, never as a policy
input.

`--stage test` loads that file and refuses to proceed on any drift. Thirteen
values are checked field by field, and the HSIG **coefficient fingerprint** on
disk is compared against the frozen one — so refitting the estimator, editing a
coefficient, swapping the estimator type, or changing the feature set all abort
the locked evaluation with a message naming what moved. Threshold reselection,
HSIG refitting, and fusion-weight tuning are additionally blocked at their own
call sites by `ThresholdLeakageError`, `HSIGLeakageError`, and `OracleError`.

---

## 10. Running it

```bash
python -m src.ahsef.cli.run_stage3_text_audio --stage align
python -m src.ahsef.cli.run_stage3_text_audio --stage llm --split validation
python -m src.ahsef.cli.run_stage3_text_audio --stage llm --split test
python -m src.ahsef.cli.run_stage3_text_audio --stage oracle --split validation
python -m src.ahsef.cli.run_stage3_text_audio --stage hsig
python -m src.ahsef.cli.run_stage3_text_audio --stage freeze
python -m src.ahsef.cli.run_stage3_text_audio --stage validate
python -m src.ahsef.cli.run_stage3_text_audio --stage ablations
python -m src.ahsef.cli.run_stage3_text_audio --stage oracle --split test
python -m src.ahsef.cli.run_stage3_text_audio --stage test
python -m src.ahsef.cli.run_stage3_text_audio --stage report

python -m pytest src/ahsef -q
```

`--provider replay` re-serves the recorded transcripts, so the whole analysis
re-runs offline with no network and no spend.

### Artefacts

```
experiments/ahsef/stage3_text_audio/
    alignment/      pool_{validation,test}.json, alignment_report.json
    predictions/    text_llm__*.parquet, audio__*.parquet (+ sidecars)
    hsig/           features_validation.parquet, oracle_*.parquet, hsig_model.json
    fusion/         fusion_spec.json, fused_*.parquet
    routing/        decisions_*.jsonl, traces_*.jsonl
    reports/        oracle_gain_*.json, hsig_quality.json,
                    validation_experiment.json, ablations.json, locked_test.json
    transcripts/    transcript_*.jsonl, wallclock_*.json
    frozen_config.json
```

---

## 11. Known limitations of this stage

1. **One candidate.** UGAPR's ranking behaviour is untested by this experiment,
   because the data offers nothing to rank audio against.
2. **One corpus.** Both splits are drawn entirely from MSP-Podcast. The result
   describes a Text→Audio routing policy on conversational podcast speech.
3. **509 validation samples, three single-digit classes.** Macro-averaged figures
   are dominated at the margin by a handful of samples; `fear`, `disgust`, and
   `surprise` F1 should be read as indications.
4. **The text-side scores are self-reports.** Fusing a self-reported score vector
   with a softmax posterior is a documented approximation, not a principled
   probabilistic combination.
5. **Cloud latency varies.** The LLM latency in the cost model is a mean of a
   quantity that swung by a large factor within a single run.
6. **The cost term cannot bite here.** Audio is genuinely cheap beside a 32.7B
   cloud LLM — its normalised compute cost is ~4 × 10⁻⁵ and its normalised
   latency ~8 × 10⁻³ — so at the frozen λ = μ = 0.10 the penalty is ~8 × 10⁻⁴,
   four orders of magnitude below the gain scale. UGAPR is wired in, priced from
   measurements, and free to refuse; on this modality pair it simply has nothing
   to refuse. The measured penalty is reported rather than presented as a
   decisive component.

The route to lifting (1) and (2) is unchanged from Stage 1's recommendation:
extract MELD's audio track, which its metadata currently records as absent, and
retrain the audio baseline over a pool that includes it. That would give
text-anchored MELD samples both audio and video, and produce the first genuine
multi-candidate routing experiment in this project.

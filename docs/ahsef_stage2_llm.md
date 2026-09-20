# AHSEF Stage 2 — the LLM text gate

Stage 2 adds one new evidence source (a language model reading the text) and
one new decision (is that evidence enough?). It does **not** acquire a second
modality, fuse anything, or make any claim about improved recognition.

The claim this stage can support, and the only one made anywhere in the code or
artefacts:

> AHSEF can use an LLM text component to estimate whether the available textual
> evidence is sufficient, and route uncertain cases toward additional modality
> acquisition — reporting honestly when no such modality exists for a sample.

Stage 1's baselines, checkpoints, results, and `experiments/ahsef/stage1/`
artefacts are untouched. Everything here writes under
`experiments/ahsef/stage2_llm/`.

---

## 1. The distinction the whole stage rests on

An LLM's confidence is **not** a posterior. Both live in `[0, 1]`, both can be
plotted on the same axis, and conflating them is the easiest way to produce a
wrong result in this project. So:

| Statistical baselines (Stage 1) | LLM (Stage 2) |
|---|---|
| `confidence` = `max_c p_c` from a softmax | `llm_confidence` = a number the model wrote about itself |
| `normalized_entropy` of a posterior | `llm_normalized_score_entropy` of *self-reported scores* |
| calibrated by temperature on logits | calibrated by a binned map on a self-report |

Every LLM-derived quantity carries an `llm_` prefix (`src/ahsef/llm/uncertainty.py`),
and `LLM_UNCERTAINTY_DEFINITION` — which is written into every artefact — states
`"calibrated": False` and *"No quantity in this module is a posterior over the
label space."*

**Probabilities are never fabricated.** The schema asks for `class_scores`, and
the prompt tells the model they are its own graded judgement rather than
probabilities. When a response carries no usable scores, the `prob_*` and
`logit_*` columns are `NaN` — never a uniform or one-hot filler — and the
sidecar records `distribution_source: "none"`.

Three uncertainty signals are available, and which one routes is a **named,
configurable policy** (`--uncertainty-policy`), recorded in every artefact:

| policy | signal | kind |
|---|---|---|
| `llm_confidence` | `1 - confidence` | self-report |
| `score_entropy` *(default)* | normalised entropy of `class_scores` | self-report |
| `ambiguity` | the model's `ambiguity` field | self-report |
| `empirical_votes` | vote entropy over `--repeats` samples | **measured** |
| `max_of_available` | the most pessimistic of the above | mixed |

`empirical_votes` is the only genuinely empirical one, and whether it is even
meaningful depends on the backend:

- **Ollama** (the pilot backend) accepts `temperature` and `seed`. At
  `temperature=0` with a fixed seed the provider reports `deterministic=True`,
  so repeated sampling is *not* informative — and the pilot correctly runs
  `--repeats 1`.
- **Hosted Claude models** expose no seed and reject `temperature`/`top_p`/
  `top_k`, so runs there are not bit-reproducible and the transcript is the
  reproducible artefact.

Both facts are recorded in the provider provenance rather than glossed over.

---

## 2. What was added

```
src/ahsef/
    gate.py                    threshold sweep + selection (validation-only) + decision
    hsig.py                    HSIG contract + UnavailableHSIG (declines, never guesses)
    ugapr.py                   J(m) = gain - lambda*C - mu*L; selects nothing without a gain
    llm_router.py              the loop, availability, and the auditable trace
    llm/
        mapping.py             LLM word -> canonical seven; mapped / ambiguous / unknown
        prompts.py             versioned, hashed, immutable prompts (v1)
        schema.py              strict JSON contract + parser that never repairs
        uncertainty.py         llm_-prefixed signals + the policy boundary
        calibration.py         binned recalibration of a self-report (validation-only)
        provider.py            LLMProvider, TokenUsage, CallBudget, Replay/Scripted, transcripts
        ollama_provider.py     Ollama REST via stdlib; probes local-vs-cloud execution
        anthropic_provider.py  schema-constrained Claude via the official SDK
        samples.py             deterministic stratified selection + signed manifests
        inference.py           LLMTextModality -> Stage 1 PredictionSet
    cli/
        run_llm_text.py        score a split, evaluate, calibrate
        run_llm_router.py      select tau, gate, write traces
        run_stage2_pilot.py    the 4-stage pilot: select / validation / freeze / test
        report_stage2.py       renders the report from artefacts only
```

Stage 1 code was **not modified**. The LLM registers as modality `text_llm`,
deliberately distinct from the frozen `text` baseline.

---

## 3. Label mapping

`src/ahsef/llm/mapping.py` is total and auditable, with three outcomes:

- **mapped** — an unambiguous synonym (`joy→happy`, `sorrow→sad`, `calm→neutral`, …).
- **ambiguous** — a real word this project *refuses* to assign, by name and with
  a reason: `excited` (happy in one taxonomy, surprise in another), `shocked`,
  `anxious`, `contempt`, `frustrated`, `confused`, `bored`, `mixed`.
- **unknown** — anything else, including multi-label answers (`"happy, sad"`).

Only the first becomes a prediction. The other two are logged as errors with the
offending text and counted in `coverage`; **an unmappable label is never folded
into `neutral`**, which would be the silent failure that flatters every metric.
`abstain` is its own fourth outcome — a legitimate answer, not an error.

---

## 4. Leakage safety

| risk | enforcement |
|---|---|
| tau chosen on test | `select_threshold` raises `ThresholdLeakageError` unless split is train/validation |
| tau re-chosen at test time | `run_llm_router --split test` **loads** the frozen artefact and exits if it is missing |
| tau applied across policies | the run aborts if the frozen tau's `uncertainty_policy` differs from the current one |
| confidence calibrated on test | `fit_confidence_calibrator` raises unless split is `validation` |
| router seeing labels | `apply_gate(sample_id, uncertainty, threshold)` — no label parameter exists; the trace stamps `router_saw_true_class: false` |
| prompt tuned on test | prompts are versioned, hashed, and immutable once published |
| fabricated fusion | availability comes from Stage 1's `AlignmentIndex`; unavailable candidates are reported with a reason |
| runaway spend | `CallBudget` checked before every call; a cost ceiling on an unpriced model **fails loudly** rather than reading as protection |
| test run before the freeze | `--stage test` exits unless `frozen_config.json` exists, and never writes it |
| config drift after the freeze | `--stage test` compares model/prompt/temperature/seed/policy against the frozen record and aborts on any difference |
| validation/test overlap | `assert_disjoint` on the two manifests; each manifest carries a SHA-256 fingerprint that detects editing |
| raw text in artefacts | `--store-text {none,hash,raw}`, default `hash` |

---

## 5. No hard-coded routing

The router never names a modality. Candidates come from the registry,
availability from the alignment index, ranking from UGAPR. And UGAPR **selects
nothing** at this stage, because `J(m)` is undefined without HSIG:

```
No utility could be computed: HSIG returned no gain estimate for ['audio', ...].
J(m) is undefined without it, and UGAPR does not substitute a default --
selecting on cost alone would be a hard-coded modality order.
```

`UnavailableHSIG` is the only estimator that exists. It returns
`expected_improvement=None` with a reason for every candidate. This is the
honest state of the system, not a placeholder to be swapped for a constant.

**HSIG's target has changed.** Stage 1 measured that `dU = U(A) - U(A∪{m})` is
unusable (sign flips with anchor and fusion rule; max per-sample correlation
with actual improvement 0.143). `HSIG_TARGET_DEFINITION` records the rejected
target and its replacement:

```
gain(m | A) = P(correct | A u {m}) - P(correct | A),  estimated on validation only
```

---

## 6. Commands

```bash
# Phase E/F -- score a validation subsample against the real API and record it
python -m src.ahsef.cli.run_llm_text --run stage2_llm --split validation \
    --provider anthropic --model claude-opus-5 --effort low \
    --max-samples 1000 --max-cost-usd 6.00 --store-text hash

# re-analyse that exact run offline, no network, no spend
python -m src.ahsef.cli.run_llm_text --run stage2_llm --split validation --provider replay

# empirical uncertainty from repeated sampling (cost x3)
python -m src.ahsef.cli.run_llm_text --run stage2_llm --split validation \
    --provider anthropic --max-samples 300 --repeats 3 --max-cost-usd 6.00

# Phase G/H -- select tau on validation, gate, write traces
python -m src.ahsef.cli.run_llm_router --run stage2_llm --split validation \
    --uncertainty-policy score_entropy --threshold-objective target_stop_accuracy \
    --target-stop-accuracy 0.75

# Phase J -- locked test, only after every validation decision is frozen
python -m src.ahsef.cli.run_llm_text   --run stage2_llm --split test --provider anthropic \
    --max-samples 1000 --max-cost-usd 6.00
python -m src.ahsef.cli.run_llm_router --run stage2_llm --split test

python -m pytest src/ahsef -q
```

Every parameter that changes a result is a flag: `--model`, `--effort`,
`--prompt-version`, `--repeats`, `--uncertainty-policy`, `--threshold`,
`--threshold-objective`, `--target-stop-accuracy`, `--min-coverage`,
`--lambda-cost`, `--mu-latency`, `--max-samples`, `--seed`, `--store-text`.

---

## 7. Artefacts

```
experiments/ahsef/stage2_llm/
    predictions/text_llm__<split>.parquet   per-sample label, scores, llm_* uncertainty,
    predictions/text_llm__<split>.json      tokens, latency, cost, status
    llm/transcript__<split>.jsonl           every raw call, for exact replay
    llm/threshold_selection.json            the full tau sweep and the selected point
    llm/confidence_calibrator.json          validation-fitted binned map
    llm/routing_records_<split>.parquet     the compact flat routing record
    reports/llm_text_<split>.json           metrics, per-class, calibration, coverage, cost
    reports/llm_routing_<split>.json        gate summary, availability, examples
    reports/llm_routing_<split>.jsonl       one auditable trace per sample
```

---

## 8. Representative examples

Chosen by a **stated rule**, never by hand: sort by distance from tau and take
the extremes — the most clear-cut stop and the most clear-cut request. The rule
does not consult correctness, so it cannot be tuned to flatter the system.

- **A — text sufficient → STOP**: lowest uncertainty among stopped samples.
- **B — text insufficient → REQUEST**: highest uncertainty among routed samples.
- **B2 — request with an acquirable modality**: as B, restricted to samples that
  actually have an aligned candidate.

B2 is separated from B deliberately, because Stage 1 established that most text
samples have no aligned partner at all. A request that cannot be fulfilled is
reported as *"additional modality requested, but unavailable for this sample"* —
not quietly turned into a fusion.

---

## 9. Status

**IMPLEMENTED** — provider abstraction (Anthropic + replay + scripted), versioned
prompt, strict schema and parser, canonical mapping, `llm_`-prefixed uncertainty
with a policy boundary, binned confidence calibration, threshold sweep and
selection, the gate, the router with alignment-backed availability, HSIG/UGAPR
interface boundaries, routing traces, both CLIs, budget and privacy controls.

**VALIDATED** — 213 stage-2 unit tests plus the 155 stage-1 tests (368 total,
all passing). Covered: parsing, malformed responses, missing probabilities,
mapping and refusals, normalisation, entropy, vote statistics, threshold
selection and its leakage guard, gate decisions, routing, availability,
unavailable-modality handling, budgets, transcript round-trip, provenance, and
the CLI's frozen-threshold rules.

**NOT YET IMPLEMENTED** — HSIG estimator, UGAPR selection, modality acquisition,
fusion after acquisition, re-evaluation loop, multi-step routing.

**RUN** — the pilot executed against `gemma4:31b-cloud` through the Ollama
provider: 1,000 validation + 1,000 locked test samples. See
`docs/ahsef_stage2_results.md` and `experiments/ahsef/stage2_llm/`.

**SCIENTIFIC LIMITATIONS**

1. **A 1,000-sample pilot is not the split.** Confidence intervals are wider
   than the frozen baselines' full-split numbers. The baseline is re-scored on
   exactly the same ids, so the comparison is at least paired.
2. **Reproducible on this backend.** Ollama honours `temperature=0` and a fixed
   seed, so the run is reproducible on the same daemon and model build; the
   transcript reproduces it exactly regardless.
3. **Self-reports are not posteriors**, and may be well calibrated in aggregate
   while carrying no per-sample information — which is why AUROC and confidence
   spread are reported next to ECE.
4. **Contamination is unquantified.** MELD and MSP-Podcast are public corpora
   that may appear in pretraining data. An LLM's score on them is not a clean
   held-out measurement, and no experiment here can rule that out.
5. **A subsample is not the split.** Running 1,000 of 22,109 samples gives wider
   confidence intervals than the frozen baselines' full-split numbers; the
   baseline is re-scored on exactly the same samples so the comparison is at
   least paired.
6. **The gate is one-step.** Stage 2 decides *whether*, never *which* — and with
   no HSIG, "which" is explicitly unanswered rather than defaulted.

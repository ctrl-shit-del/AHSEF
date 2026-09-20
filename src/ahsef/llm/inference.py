"""The LLM as an AHSEF modality: text in, standard prediction records out.

This is the bridge between :mod:`src.ahsef.llm` and everything Stage 1 built.
The output is an ordinary :class:`~src.ahsef.inference.PredictionSet` with the
same columns as the five frozen baselines, so Stage 1's evaluation,
calibration, alignment, and routing-log code applies without modification.

Two columns need care, and both are handled explicitly rather than by
convention:

``prob_<c>``
    Present **only** when the model returned usable ``class_scores``, and then
    it holds the *normalised self-reported score*, not a posterior.  The
    sidecar metadata carries ``distribution_source`` so no reader can mistake
    it.  When scores are absent the columns are ``NaN`` -- never a fabricated
    one-hot or uniform vector.

``logit_<c>``
    The log of the normalised score.  A convenience for reusing Stage 1's
    temperature scaling, labelled as a transform of self-reports rather than
    as a model's pre-softmax output.

Samples the model refused, abstained on, or answered unmappably are **kept** in
the frame with ``predicted_class == -1`` and an ``llm_status`` naming the
failure.  Dropping them would quietly improve every metric.
"""

from __future__ import annotations

import hashlib
import platform
import time
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from src.ahsef.inference import PredictionSet
from src.ahsef.llm.mapping import CANONICAL_EMOTIONS, mapping_provenance
from src.ahsef.llm.prompts import get_prompt, prompt_provenance
from src.ahsef.llm.provider import (
    CallBudget,
    LLMCall,
    LLMProvider,
    TranscriptWriter,
    estimate_cost_usd,
)
from src.ahsef.llm.schema import (
    LLMEmotionResponse,
    parse_response,
    response_json_schema,
    schema_provenance,
)
from src.ahsef.llm.uncertainty import (
    DEFAULT_UNCERTAINTY_POLICY,
    LLM_UNCERTAINTY_DEFINITION,
    LLMUncertainty,
    UncertaintyPolicyError,
    majority_vote,
    score_distribution,
    uncertainty_from_response,
)


#: The modality name the LLM registers under.  Deliberately distinct from
#: ``text`` so it can never be confused with the frozen text baseline.
LLM_MODALITY = "text_llm"

#: Reasons a sample yields no canonical prediction.  Counted, never dropped.
STATUS_OK = "ok"
STATUS_VALUES = (
    STATUS_OK, "abstain", "ambiguous", "unknown", "malformed_json",
    "missing_field", "out_of_range", "call_failed",
)

#: Text is user data. The default records only a hash.
TEXT_STORAGE_MODES = ("none", "hash", "raw")


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass
class LLMSampleResult:
    """Everything one sample produced, before it becomes a frame row."""

    sample_id: str
    dataset: str
    true_class: int
    response: LLMEmotionResponse
    uncertainty: LLMUncertainty
    calls: list[LLMCall]
    status: str
    votes: list[str]

    @property
    def latency_ms(self) -> float:
        return float(sum(call.latency_ms for call in self.calls))

    @property
    def cost_usd(self) -> float:
        total = 0.0
        for call in self.calls:
            value = estimate_cost_usd(call.model, call.usage)
            if value is not None:
                total += value
        return total


class LLMTextModality:
    """Run the LLM over a set of texts and emit AHSEF prediction records."""

    def __init__(
        self,
        provider: LLMProvider,
        prompt_version: str = "v1",
        repeats: int = 1,
        uncertainty_policy: str = DEFAULT_UNCERTAINTY_POLICY,
        budget: CallBudget | None = None,
        transcript: TranscriptWriter | None = None,
        store_text: str = "hash",
    ):
        if repeats < 1:
            raise ValueError("repeats must be at least 1")
        if store_text not in TEXT_STORAGE_MODES:
            raise ValueError(f"store_text must be one of {TEXT_STORAGE_MODES}")
        self.provider = provider
        self.prompt = get_prompt(prompt_version)
        self.prompt_version = prompt_version
        self.repeats = int(repeats)
        self.uncertainty_policy = uncertainty_policy
        self.budget = budget or CallBudget()
        self.transcript = transcript
        self.store_text = store_text
        self.schema = response_json_schema()

    # --------------------------------------------------------------- single

    def predict(self, text: str, sample_id: str = "adhoc") -> LLMSampleResult:
        """Score one text.  The standardized single-sample entry point."""
        user = self.prompt.render_user(text)
        calls: list[LLMCall] = []
        responses: list[LLMEmotionResponse] = []

        for repeat in range(self.repeats):
            self.budget.check()
            call = self.provider.complete(self.prompt.system, user, self.schema)
            self.budget.record(call)
            calls.append(call)
            if self.transcript is not None:
                self.transcript.write(sample_id, user, call, repeat=repeat)
            responses.append(
                parse_response(call.text) if call.ok
                else LLMEmotionResponse.rejected(
                    call.error or "call failed", "call_failed", {"stop_reason": call.stop_reason},
                )
            )

        primary = _select_primary(responses)
        votes = [item.emotion for item in responses if item.emotion is not None]
        # With repeats, the reported label is the modal one: a single draw from
        # a nondeterministic model is a weaker estimate than its own majority.
        if len(votes) > 1:
            modal = majority_vote(votes)
            if primary.emotion != modal:
                primary = next(item for item in responses if item.emotion == modal)
        uncertainty = uncertainty_from_response(primary, votes if len(votes) > 1 else None)
        return LLMSampleResult(
            sample_id=str(sample_id), dataset="", true_class=-1,
            response=primary, uncertainty=uncertainty, calls=calls,
            status=_status_of(primary), votes=votes,
        )

    # ---------------------------------------------------------------- batch

    def predict_frame(
        self,
        frame: pd.DataFrame,
        text_column: str = "text",
        label_column: str = "canonical_emotion_id",
        progress_every: int = 100,
        on_progress=None,
    ) -> list[LLMSampleResult]:
        """Score every row of a manifest slice, keeping failures in the output."""
        required = {"sample_id", text_column}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Frame is missing required columns {sorted(missing)}")

        results: list[LLMSampleResult] = []
        for position, (_, row) in enumerate(frame.iterrows(), start=1):
            text = row[text_column]
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"Sample {row['sample_id']!r} has empty text; the manifest filter "
                    f"should have excluded it rather than sending a blank prompt."
                )
            result = self.predict(text, sample_id=str(row["sample_id"]))
            result.dataset = str(row.get("dataset", ""))
            result.true_class = int(row[label_column]) if label_column in frame.columns else -1
            results.append(result)
            if on_progress is not None and position % max(progress_every, 1) == 0:
                on_progress(position, len(frame), self.budget)
        return results

    # ---------------------------------------------------------- provenance

    def provenance(self) -> dict:
        return {
            "modality": LLM_MODALITY,
            "provider": self.provider.provenance(),
            "prompt": prompt_provenance(self.prompt_version),
            "schema": schema_provenance(),
            "mapping": mapping_provenance(),
            "repeats": self.repeats,
            "uncertainty_policy": self.uncertainty_policy,
            "uncertainty_definition": dict(LLM_UNCERTAINTY_DEFINITION),
            "store_text": self.store_text,
            "budget": self.budget.to_dict(),
            "host": {"platform": platform.platform()},
            "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }


def _status_of(response: LLMEmotionResponse) -> str:
    if response.abstain:
        return "abstain"
    if response.error_kind in {"ambiguous", "unknown", "malformed_json",
                               "missing_field", "out_of_range", "call_failed"}:
        return response.error_kind
    return STATUS_OK if response.usable else "unknown"


def _select_primary(responses: Sequence[LLMEmotionResponse]) -> LLMEmotionResponse:
    """First usable response, else the first non-abstention, else the first."""
    for response in responses:
        if response.usable:
            return response
    for response in responses:
        if not response.abstain:
            return response
    return responses[0]


# ============================================================
# PredictionSet assembly
# ============================================================

def build_prediction_set(
    results: Sequence[LLMSampleResult],
    split: str,
    provenance: Mapping[str, Any],
    texts: Mapping[str, str] | None = None,
    store_text: str = "hash",
) -> PredictionSet:
    """Turn LLM results into a Stage 1 :class:`PredictionSet`.

    Rows for samples with no usable answer carry ``predicted_class = -1`` and
    ``NaN`` score columns.  They are counted in ``llm_status`` and excluded by
    the evaluator through an explicit mask, never by silent omission here.
    """
    if not results:
        raise ValueError("Cannot build a prediction set from zero results")

    rows: list[dict] = []
    for result in results:
        response, uncertainty = result.response, result.uncertainty
        row: dict[str, Any] = {
            "sample_id": result.sample_id,
            "dataset": result.dataset,
            "modality": LLM_MODALITY,
            "split": split,
            "true_class": int(result.true_class),
            "predicted_class": int(response.class_id) if response.class_id is not None else -1,
            "llm_status": result.status,
            "llm_usable": bool(response.usable),
            "llm_abstain": bool(response.abstain),
            "llm_raw_emotion": response.mapping.raw if response.mapping else "",
            "llm_error": response.error or "",
            "llm_evidence": response.evidence,
            "repeats": uncertainty.repeats,
            "votes": "|".join(result.votes),
            "latency_ms": result.latency_ms,
            "inference_ms": result.latency_ms,
            "feature_ms": 0.0,
            "cost_usd": result.cost_usd,
            "input_tokens": sum(call.usage.input_tokens for call in result.calls),
            "output_tokens": sum(call.usage.output_tokens for call in result.calls),
            "cache_read_tokens": sum(
                call.usage.cache_read_input_tokens for call in result.calls
            ),
        }
        row.update(uncertainty.to_dict())

        if response.class_scores:
            distribution = score_distribution(response.class_scores)
            for index in range(len(CANONICAL_EMOTIONS)):
                row[f"prob_{index}"] = float(distribution[index])
                row[f"logit_{index}"] = float(torch.log(distribution[index].clamp(min=1e-12)))
        else:
            # No scores means no distribution. NaN is the honest value; a
            # uniform or one-hot filler would be a fabricated measurement.
            for index in range(len(CANONICAL_EMOTIONS)):
                row[f"prob_{index}"] = float("nan")
                row[f"logit_{index}"] = float("nan")

        # Stage 1's column names, populated only where a real quantity exists.
        row["confidence"] = uncertainty.llm_confidence
        row["predictive_entropy"] = uncertainty.llm_score_entropy
        row["normalized_entropy"] = uncertainty.llm_normalized_score_entropy
        row["margin"] = uncertainty.llm_score_margin

        if store_text == "hash" and texts and result.sample_id in texts:
            row["text_sha256_16"] = hash_text(texts[result.sample_id])
        elif store_text == "raw" and texts and result.sample_id in texts:
            row["text"] = texts[result.sample_id]
        rows.append(row)

    frame = pd.DataFrame(rows)
    usable = int(frame["llm_usable"].sum())
    status_counts = frame["llm_status"].value_counts().to_dict()
    with_scores = int(frame["prob_0"].notna().sum())

    meta = {
        **dict(provenance),
        "kind": "llm",
        "samples": int(len(frame)),
        "usable_predictions": usable,
        "unusable_predictions": int(len(frame)) - usable,
        "status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "responses_with_class_scores": with_scores,
        "distribution_source": (
            "llm_self_reported_class_scores, normalised" if with_scores else "none"
        ),
        "distribution_is_calibrated_posterior": False,
        "prob_columns_note": (
            "prob_* hold NORMALISED SELF-REPORTED SCORES, not posterior probabilities. "
            "NaN means the model returned no usable scores for that sample; no filler "
            "distribution was invented."
        ),
        "unusable_rows_note": (
            "Rows with predicted_class == -1 had no usable answer (abstention, "
            "unmappable label, malformed response, or a failed call). They are retained "
            "so metrics can state their coverage rather than silently excluding them."
        ),
        "measured_cost_usd": float(frame["cost_usd"].sum()),
        "uses_labels_for_prediction": False,
    }
    return PredictionSet(
        modality=LLM_MODALITY, split=split,
        class_order=tuple(CANONICAL_EMOTIONS), frame=frame, meta=meta,
    )


def routing_uncertainty_column(
    prediction_set: PredictionSet, policy: str = DEFAULT_UNCERTAINTY_POLICY
) -> pd.Series:
    """The single uncertainty series the gate routes on, under a named policy.

    Raises rather than filling gaps: a sample with no signal under the chosen
    policy cannot be routed, and pretending otherwise would put a fabricated
    number into the decision.
    """
    frame = prediction_set.frame
    columns = {
        "llm_confidence": "llm_uncertainty",
        "score_top1": "llm_score_top1_uncertainty",
        "score_entropy": "llm_normalized_score_entropy",
        "ambiguity": "llm_ambiguity",
        "empirical_votes": "llm_vote_entropy",
    }
    if policy == "max_of_available":
        present = [name for name in columns.values() if name in frame.columns]
        if not present:
            raise UncertaintyPolicyError("No LLM uncertainty column is present")
        return frame[present].max(axis=1, skipna=True)
    column = columns.get(policy)
    if column is None:
        raise ValueError(f"Unknown uncertainty policy {policy!r}")
    if column not in frame.columns:
        # ``score_top1`` is definitionally 1 - max_c score_c, so it is
        # recoverable from the stored score columns. Recomputing it is exact,
        # not an approximation, and keeps exports written before the column
        # existed readable.
        if policy == "score_top1":
            score_columns = [
                f"prob_{index}" for index in range(len(prediction_set.class_order))
            ]
            if all(name in frame.columns for name in score_columns):
                return 1.0 - frame[score_columns].max(axis=1)
        raise UncertaintyPolicyError(
            f"Policy {policy!r} needs column {column!r}, which this prediction set "
            f"does not carry."
        )
    return frame[column]


def usable_mask(prediction_set: PredictionSet) -> np.ndarray:
    """Rows that produced a canonical prediction."""
    return prediction_set.frame["llm_usable"].to_numpy(dtype=bool)


def coverage_report(prediction_set: PredictionSet) -> dict:
    """How much of the split the LLM actually answered, and how it failed."""
    frame = prediction_set.frame
    total = len(frame)
    counts = frame["llm_status"].value_counts().to_dict()
    return {
        "samples": total,
        "usable": int(frame["llm_usable"].sum()),
        "coverage": float(frame["llm_usable"].mean()),
        "status_counts": {str(key): int(value) for key, value in counts.items()},
        "with_class_scores": int(frame["prob_0"].notna().sum()),
        "score_coverage": float(frame["prob_0"].notna().mean()),
        "abstentions": int(frame["llm_abstain"].sum()),
        "unmappable_examples": sorted(
            {
                str(value) for value in
                frame.loc[frame["llm_status"].isin(["ambiguous", "unknown"]), "llm_raw_emotion"]
                if str(value).strip()
            }
        )[:20],
    }

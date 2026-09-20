"""The auditable per-sample routing trace.

A routing decision that cannot be replayed is not evidence.  For every sample
AHSEF must be able to show, step by step: which modalities were active, what
the model believed, how uncertain it was, what each inactive candidate was
*estimated* to be worth, what utility each candidate scored, which one was
selected, and why the loop stopped.

The format is fixed here in stage 1 -- before HSIG and UGAPR exist -- so that
the router written in stage 2 has a target to fill in rather than a schema to
invent.  A step whose ``hsig`` or ``ugapr`` block is empty is a legitimate
stage-1 trace: it says the decision was made without those components, which is
exactly what a single-step, no-candidate trace should say.

Nothing in this module reads a label.  ``true_class`` is carried on the trace
for post-hoc evaluation and is explicitly marked as unavailable to the router.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence


ROUTING_LOG_SCHEMA_VERSION = "ahsef.routing_log.v1"

#: Every reason the loop may terminate.  A trace whose ``stop_reason`` is not
#: one of these is malformed.
STOP_REASONS = (
    "uncertainty_below_threshold",
    "max_modalities_reached",
    "no_candidate_available",
    "no_candidate_above_min_gain",
    "no_candidate_above_min_utility",
    "candidate_pool_exhausted",
    # Stage 3. Distinct from "no_candidate_available": the router *did* decide to
    # acquire, and the acquisition then failed. Collapsing the two would hide the
    # difference between "audio was not worth it" and "audio was worth it and we
    # could not get it", which are opposite findings about the policy.
    "requested_modality_unavailable",
)


@dataclass
class CandidateScore:
    """One inactive candidate as seen by HSIG and UGAPR at one step."""

    modality: str
    available: bool = True
    estimated_delta_uncertainty: float | None = None
    normalized_cost: float | None = None
    normalized_latency: float | None = None
    utility: float | None = None
    unavailable_reason: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RoutingStep:
    """One iteration of the acquire-or-stop loop for one sample."""

    step: int
    active_modalities: list[str]
    predicted_class: int
    predicted_label: str
    confidence: float
    uncertainty: float
    stop: bool
    stop_reason: str | None = None
    hsig: dict[str, float] = field(default_factory=dict)
    ugapr: dict[str, dict] = field(default_factory=dict)
    candidates: list[CandidateScore] = field(default_factory=list)
    selected_modality: str | None = None
    cumulative_latency_ms: float | None = None
    cumulative_cost: float | None = None

    def __post_init__(self) -> None:
        if self.step < 1:
            raise ValueError("Routing steps are 1-indexed")
        if not self.active_modalities:
            raise ValueError("A routing step must have at least one active modality")
        if not 0.0 <= self.uncertainty <= 1.0:
            raise ValueError(
                f"uncertainty must be a normalised value in [0, 1], got {self.uncertainty}"
            )
        if self.stop and self.selected_modality is not None:
            raise ValueError("A stopping step must not also select a modality")
        if not self.stop and self.selected_modality is None:
            raise ValueError("A non-stopping step must select the modality it will acquire")
        if self.stop and self.stop_reason not in STOP_REASONS:
            raise ValueError(
                f"stop_reason must be one of {STOP_REASONS}, got {self.stop_reason!r}"
            )
        if self.selected_modality is not None:
            offered = {candidate.modality for candidate in self.candidates}
            if offered and self.selected_modality not in offered:
                raise ValueError(
                    f"Selected {self.selected_modality!r} but it was not among the "
                    f"candidates scored at this step ({sorted(offered)})"
                )

    def to_dict(self) -> dict:
        record = asdict(self)
        record["candidates"] = [candidate.to_dict() for candidate in self.candidates]
        return record


@dataclass
class RoutingTrace:
    """The complete routing history of one sample."""

    sample_id: str
    dataset: str
    split: str
    class_order: list[str]
    steps: list[RoutingStep] = field(default_factory=list)
    #: Ground truth, attached for evaluation only.  The router never sees it.
    true_class: int | None = None
    policy: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.steps:
            raise ValueError(f"Sample {self.sample_id!r} has an empty routing trace")
        if not self.steps[-1].stop:
            raise ValueError(
                f"Sample {self.sample_id!r} has a trace that never stops; every trace "
                f"must terminate with a stopping step."
            )
        for index, step in enumerate(self.steps, start=1):
            if step.step != index:
                raise ValueError(
                    f"Sample {self.sample_id!r} has out-of-order steps "
                    f"({step.step} at position {index})"
                )
        for previous, current in zip(self.steps, self.steps[1:]):
            if previous.selected_modality not in current.active_modalities:
                raise ValueError(
                    f"Sample {self.sample_id!r}: step {previous.step} selected "
                    f"{previous.selected_modality!r} but step {current.step} does not "
                    f"have it active"
                )

    @property
    def final(self) -> RoutingStep:
        return self.steps[-1]

    @property
    def modalities_activated(self) -> int:
        return len(self.final.active_modalities)

    @property
    def acquisitions(self) -> list[str]:
        """Modalities acquired after the initial one, in selection order."""
        return [step.selected_modality for step in self.steps[:-1] if step.selected_modality]

    @property
    def uncertainty_reduction(self) -> float:
        return self.steps[0].uncertainty - self.final.uncertainty

    def to_dict(self) -> dict:
        return {
            "schema": ROUTING_LOG_SCHEMA_VERSION,
            "sample_id": self.sample_id,
            "dataset": self.dataset,
            "split": self.split,
            "class_order": list(self.class_order),
            "true_class": self.true_class,
            "true_label": (
                self.class_order[self.true_class]
                if self.true_class is not None and 0 <= self.true_class < len(self.class_order)
                else None
            ),
            "router_saw_true_class": False,
            "initial_modality": self.steps[0].active_modalities[0],
            "final_modalities": list(self.final.active_modalities),
            "modalities_activated": self.modalities_activated,
            "acquisitions": self.acquisitions,
            "final_prediction": self.final.predicted_class,
            "final_confidence": self.final.confidence,
            "final_uncertainty": self.final.uncertainty,
            "uncertainty_reduction": self.uncertainty_reduction,
            "stop_reason": self.final.stop_reason,
            "steps": [step.to_dict() for step in self.steps],
            "policy": dict(self.policy),
        }

    def render(self) -> str:
        """Human-readable rendering, in the shape the milestone report needs."""
        lines = [f"Sample {self.sample_id}  ({self.dataset}, {self.split})"]
        for step in self.steps:
            lines.append("")
            lines.append(f"Step {step.step}:")
            lines.append(f"Active      = [{', '.join(step.active_modalities)}]")
            lines.append(f"Prediction  = {step.predicted_label}")
            lines.append(f"Confidence  = {step.confidence:.6f}")
            lines.append(f"Uncertainty = {step.uncertainty:.6f}")
            if step.hsig:
                lines.append("")
                lines.append("HSIG:")
                for name, value in sorted(step.hsig.items(), key=lambda kv: -kv[1]):
                    lines.append(f"  {name:<12} = {value:.6f}")
            if step.ugapr:
                lines.append("")
                lines.append("UGAPR:")
                for name, record in sorted(
                    step.ugapr.items(), key=lambda kv: -(kv[1].get("utility") or 0.0)
                ):
                    utility = record.get("utility")
                    lines.append(
                        f"  {name:<12} = "
                        + ("n/a" if utility is None else f"{utility:.6f}")
                        + f"   (gain={record.get('estimated_delta_uncertainty')}, "
                          f"cost={record.get('normalized_cost')}, "
                          f"latency={record.get('normalized_latency')})"
                    )
            if step.selected_modality:
                lines.append("")
                lines.append(f"Selected    = {step.selected_modality}")
            if step.stop:
                lines.append("")
                lines.append(f"Stopping decision = TRUE ({step.stop_reason})")
        if self.true_class is not None:
            label = self.class_order[self.true_class]
            lines.append("")
            lines.append(f"[post-hoc] true class = {label} (not visible to the router)")
        return "\n".join(lines)


class RoutingLogWriter:
    """Append-only JSONL writer for routing traces."""

    def __init__(self, path: Path | str, policy: Mapping | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.policy = dict(policy or {})
        self._count = 0
        self._handle = self.path.open("w", encoding="utf-8")
        header = {
            "schema": ROUTING_LOG_SCHEMA_VERSION,
            "record": "header",
            "policy": self.policy,
        }
        self._handle.write(json.dumps(header) + "\n")

    def write(self, trace: RoutingTrace) -> None:
        record = trace.to_dict()
        record["record"] = "trace"
        self._handle.write(json.dumps(record) + "\n")
        self._count += 1

    def write_all(self, traces: Iterable[RoutingTrace]) -> int:
        for trace in traces:
            self.write(trace)
        return self._count

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "RoutingLogWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_routing_log(path: Path | str) -> tuple[dict, list[dict]]:
    """Return ``(header, traces)`` from a routing JSONL log."""
    header: dict = {}
    traces: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("record") == "header":
            header = record
        else:
            traces.append(record)
    return header, traces


def summarise_traces(traces: Sequence[dict]) -> dict:
    """The AHSEF headline metrics, computed from routing traces alone."""
    if not traces:
        raise ValueError("No routing traces to summarise")
    total = len(traces)
    activated = [record["modalities_activated"] for record in traces]
    counts: dict[str, int] = {}
    for record in traces:
        for modality in record["final_modalities"]:
            counts[modality] = counts.get(modality, 0) + 1
    reasons: dict[str, int] = {}
    for record in traces:
        reason = record.get("stop_reason") or "unspecified"
        reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "samples": total,
        "average_modalities_activated": sum(activated) / total,
        "max_modalities_activated": max(activated),
        "modality_activation_rate": {
            name: count / total for name, count in sorted(counts.items())
        },
        "mean_uncertainty_reduction": sum(
            record["uncertainty_reduction"] for record in traces
        ) / total,
        "stop_reasons": reasons,
    }

"""The AHSEF text gate: select tau on validation, then route.

    # choose the threshold (validation only) and route validation
    python -m src.ahsef.cli.run_llm_router --run stage2_llm --split validation

    # apply the frozen threshold to the locked test split
    python -m src.ahsef.cli.run_llm_router --run stage2_llm --split test

On ``validation`` the command sweeps every candidate threshold, selects one by
the configured objective, writes the selection artefact, and routes.  On
``test`` it **loads** that artefact and refuses to run if it is missing -- the
threshold cannot be re-chosen where the answer is known.

It writes an auditable routing trace per sample, a compact flat routing record,
and two representative examples chosen by a stated rule rather than by hand.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.ahsef.gate import (
    DEFAULT_OBJECTIVE,
    THRESHOLD_OBJECTIVES,
    ThresholdSelection,
    select_threshold,
)
from src.ahsef.hsig import UnavailableHSIG
from src.ahsef.identity import AlignmentIndex
from src.ahsef.inference import PredictionSet
from src.ahsef.layout import AhsefLayout
from src.ahsef.llm.inference import LLM_MODALITY, routing_uncertainty_column
from src.ahsef.llm.uncertainty import DEFAULT_UNCERTAINTY_POLICY, UNCERTAINTY_POLICIES
from src.ahsef.llm_router import LLMTextRouter, routing_summary
from src.ahsef.registry import DEFAULT_BASELINES
from src.ahsef.routing_log import RoutingLogWriter
from src.ahsef.ugapr import UGAPR, UtilityWeights


DEFAULT_CANDIDATES = ("audio", "video", "image", "physiology")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default="stage2_llm")
    parser.add_argument("--root", default="experiments")
    parser.add_argument("--ahsef-root", default=None)
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument(
        "--candidates", nargs="+", default=list(DEFAULT_CANDIDATES),
        choices=sorted(DEFAULT_BASELINES),
    )
    parser.add_argument(
        "--uncertainty-policy", default=DEFAULT_UNCERTAINTY_POLICY,
        choices=list(UNCERTAINTY_POLICIES),
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Override tau. Recorded as operator-supplied rather than data-selected.",
    )
    parser.add_argument(
        "--threshold-objective", default=DEFAULT_OBJECTIVE, choices=list(THRESHOLD_OBJECTIVES),
    )
    parser.add_argument("--target-stop-accuracy", type=float, default=0.75)
    parser.add_argument("--target-coverage", type=float, default=0.5)
    parser.add_argument("--min-coverage", type=float, default=0.05)
    parser.add_argument("--lambda-cost", type=float, default=0.10)
    parser.add_argument("--mu-latency", type=float, default=0.10)
    parser.add_argument(
        "--examples", type=int, default=3,
        help="Representative traces of each kind to print, chosen by a fixed rule.",
    )
    return parser


def selection_path(layout: AhsefLayout) -> Path:
    return layout.base / "llm" / "threshold_selection.json"


def load_predictions(layout: AhsefLayout, split: str) -> PredictionSet:
    path = layout.prediction_path(LLM_MODALITY, split)
    if not path.exists():
        raise SystemExit(
            f"No LLM predictions at {path}. Run:\n"
            f"  python -m src.ahsef.cli.run_llm_text --run {layout.run} --split {split}"
        )
    return PredictionSet.load(path)


def build_alignment(candidates, root: str) -> AlignmentIndex | None:
    specs = [
        (name, DEFAULT_BASELINES[name].experiment, DEFAULT_BASELINES[name].label_column,
         DEFAULT_BASELINES[name].task(root))
        for name in candidates if name in DEFAULT_BASELINES
    ]
    try:
        return AlignmentIndex.from_experiments(specs, root=root)
    except FileNotFoundError as error:
        print(f"[warn] alignment index unavailable ({error}); every candidate will be "
              f"reported unavailable rather than assumed present.")
        return None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ahsef_root = Path(args.ahsef_root) if args.ahsef_root else Path(args.root) / "ahsef"
    layout = AhsefLayout(run=args.run, root=ahsef_root)
    layout.prepare()
    (layout.base / "llm").mkdir(parents=True, exist_ok=True)

    predictions = load_predictions(layout, args.split)
    frame = predictions.frame
    uncertainty = routing_uncertainty_column(predictions, args.uncertainty_policy)

    # Only samples with a usable answer AND an uncertainty value can be gated.
    gatable = frame["llm_usable"].to_numpy(dtype=bool) & uncertainty.notna().to_numpy()
    ungatable = int((~gatable).sum())
    values = uncertainty.to_numpy(dtype=float)[gatable]
    correct = (
        frame["predicted_class"].to_numpy() == frame["true_class"].to_numpy()
    )[gatable]

    # ------------------------------------------------------------ threshold
    selection: ThresholdSelection | None = None
    path = selection_path(layout)
    if args.threshold is not None:
        threshold = float(args.threshold)
        source = "operator-supplied via --threshold"
    elif args.split == "validation":
        selection = select_threshold(
            values, correct, split="validation",
            objective=args.threshold_objective,
            target_stop_accuracy=args.target_stop_accuracy,
            target_coverage=args.target_coverage,
            min_coverage=args.min_coverage,
        )
        threshold = selection.threshold
        source = f"selected on validation by {args.threshold_objective}"
        record = {
            **selection.to_dict(),
            "uncertainty_policy": args.uncertainty_policy,
            "gatable_samples": int(gatable.sum()),
            "ungatable_samples": ungatable,
        }
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"[tau ] {threshold:.6f}  ({selection.note})")
        if not selection.satisfied:
            print("[warn] the threshold objective was NOT satisfied; see the note above.")
    else:
        if not path.exists():
            raise SystemExit(
                f"No frozen threshold at {path}. Select it on validation first:\n"
                f"  python -m src.ahsef.cli.run_llm_router --run {args.run} "
                f"--split validation\n"
                f"Re-selecting tau on test would invalidate every routing number."
            )
        record = json.loads(path.read_text(encoding="utf-8"))
        threshold = float(record["threshold"])
        if record.get("uncertainty_policy") != args.uncertainty_policy:
            raise SystemExit(
                f"The frozen threshold was selected under uncertainty policy "
                f"{record.get('uncertainty_policy')!r} but this run uses "
                f"{args.uncertainty_policy!r}. They are different quantities; refusing "
                f"to apply the threshold across them."
            )
        source = f"loaded frozen from {path} (selected on {record['selected_on_split']})"
        print(f"[tau ] {threshold:.6f}  ({source})")

    # --------------------------------------------------------------- router
    alignment = build_alignment(args.candidates, args.root)
    router = LLMTextRouter(
        threshold=threshold, candidates=args.candidates, alignment=alignment,
        hsig=UnavailableHSIG(),
        ugapr=UGAPR(UtilityWeights(lambda_cost=args.lambda_cost, mu_latency=args.mu_latency)),
        uncertainty_policy=args.uncertainty_policy,
        policy_record={
            "threshold_source": source,
            "threshold_objective": args.threshold_objective if selection else None,
            "ungatable_samples": ungatable,
        },
    )
    routings = router.route_prediction_set(predictions, skip_unusable=True)
    summary = routing_summary(routings)
    summary["ungatable_samples"] = ungatable
    summary["ungatable_note"] = (
        "Samples the LLM could not answer have no uncertainty to gate on. They are "
        "excluded from routing and counted here rather than being forced to a side."
    )

    # ------------------------------------------------------------ artefacts
    log_path = layout.routing_log_path(f"llm_routing_{args.split}")
    with RoutingLogWriter(log_path, policy=router.policy()) as writer:
        for routing in routings:
            writer.write(routing.trace)

    flat = pd.DataFrame([routing.flat_record() for routing in routings])
    flat_path = layout.base / "llm" / f"routing_records_{args.split}.parquet"
    flat.drop(columns=["candidate_availability", "active_modalities"]).to_parquet(
        flat_path, index=False
    )

    examples = select_examples(routings, args.examples)
    report = {
        "split": args.split,
        "threshold": threshold,
        "threshold_source": source,
        "threshold_selection": selection.to_dict() if selection else None,
        "uncertainty_policy": args.uncertainty_policy,
        "candidates": list(args.candidates),
        "summary": summary,
        "availability": availability_summary(routings),
        "policy": router.policy(),
        "examples": {
            kind: [routing.flat_record() for routing in group]
            for kind, group in examples.items()
        },
        "artefacts": {
            "routing_log": str(log_path),
            "routing_records": str(flat_path),
            "threshold_selection": str(path) if selection else None,
        },
        "scientific_claim": (
            "AHSEF used an LLM text component to decide, per sample, whether the "
            "textual evidence was sufficient, and recorded which additional modality "
            "could have been acquired. Nothing was acquired or fused, so this is NOT "
            "evidence that AHSEF improves multimodal emotion recognition."
        ),
    }
    report_path = layout.report_path(f"llm_routing_{args.split}.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    _print(args, summary, report["availability"], examples, routings)
    print(f"\nWritten: {log_path}")
    print(f"Written: {flat_path}")
    print(f"Written: {report_path}")
    return 0


def availability_summary(routings) -> dict:
    """How often each candidate was actually acquirable, among requests."""
    requested = [routing for routing in routings if routing.requested]
    if not requested:
        return {"requests": 0, "per_candidate_available": {}}
    counts: dict[str, int] = {}
    for routing in requested:
        for name, reason in routing.availability.items():
            counts[name] = counts.get(name, 0) + (1 if reason is None else 0)
    none_available = sum(
        1 for routing in requested
        if all(reason is not None for reason in routing.availability.values())
    )
    return {
        "requests": len(requested),
        "per_candidate_available": counts,
        "per_candidate_rate": {
            name: count / len(requested) for name, count in sorted(counts.items())
        },
        "requests_with_no_available_modality": none_available,
        "requests_with_no_available_modality_rate": none_available / len(requested),
        "note": (
            "Availability is decided by the Stage 1 alignment index: a candidate counts "
            "only when this exact sample has an aligned record in the same split. "
            "A request with nothing available is reported, not fabricated into a fusion."
        ),
    }


def select_examples(routings, count: int) -> dict[str, list]:
    """Representative traces chosen by a stated rule, never hand-picked.

    Rule: sort by distance from the threshold and take the extremes -- the most
    clear-cut stop and the most clear-cut request. Deterministic, and it does
    not consult correctness, so it cannot be tuned to flatter the system.
    """
    stopped = sorted(
        (routing for routing in routings if routing.stopped),
        key=lambda routing: routing.uncertainty,
    )
    requested = sorted(
        (routing for routing in routings if routing.requested),
        key=lambda routing: -routing.uncertainty,
    )
    acquirable = [
        routing for routing in requested
        if any(reason is None for reason in routing.availability.values())
    ]
    return {
        "A_text_sufficient_stop": stopped[:count],
        "B_text_insufficient_request": requested[:count],
        "B2_request_with_an_available_modality": acquirable[:count],
    }


def _print(args, summary, availability, examples, routings) -> None:
    total = summary["samples"]
    print()
    print("=" * 84)
    print(f"AHSEF TEXT GATE  ({args.split})")
    print("=" * 84)
    print(f"  {'Routing outcome':<44}{'Count':>10}{'Percentage':>14}")
    print("  " + "-" * 68)
    print(f"  {'Text sufficient -> STOP':<44}{summary['text_sufficient_stop']:>10}"
          f"{summary['stop_rate']:>13.1%}")
    print(f"  {'Text insufficient -> request modality':<44}"
          f"{summary['text_insufficient_request']:>10}{summary['request_rate']:>13.1%}")
    print("  " + "-" * 68)
    print(f"  {'gated samples':<44}{total:>10}")
    if summary.get("ungatable_samples"):
        print(f"  {'ungatable (no usable LLM answer)':<44}"
              f"{summary['ungatable_samples']:>10}")
    if "outcome" in summary:
        outcome = summary["outcome"]
        print()
        print(f"  accuracy overall        : {outcome['overall_accuracy']:.4f}")
        print(f"  accuracy when stopped   : {outcome['accuracy_when_stopped']}")
        print(f"  accuracy when requested : {outcome['accuracy_when_requested']}")
    print()
    print(f"  requests with NO acquirable modality : "
          f"{availability['requests_with_no_available_modality']} "
          f"({availability.get('requests_with_no_available_modality_rate', 0):.1%})")
    for name, rate in availability.get("per_candidate_rate", {}).items():
        print(f"    {name:<12} acquirable for {rate:.1%} of requests")

    for kind, group in examples.items():
        if not group:
            continue
        print()
        print("=" * 84)
        print(f"EXAMPLE {kind}")
        print("=" * 84)
        print(group[0].trace.render())


if __name__ == "__main__":
    raise SystemExit(main())

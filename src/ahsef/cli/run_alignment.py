"""Verify cross-modality sample identity before anything is fused.

    python -m src.ahsef.cli.run_alignment --run stage1
    python -m src.ahsef.cli.run_alignment --run stage1 --anchor audio

This command answers the question the whole fusion study rests on: do the five
baselines share any samples, and if so, in which splits?  It reads the
experiment manifests directly -- not the predictions -- so it can be run before
any inference at all.

Its verdict is deliberately blunt.  A modality pair with no co-split overlap is
reported as NOT FUSABLE with the count that proves it, and every downstream
command refuses to fuse it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.ahsef.identity import AlignmentIndex
from src.ahsef.layout import SPLITS, AhsefLayout
from src.ahsef.registry import DEFAULT_BASELINES, ordered_modalities


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default="stage1")
    parser.add_argument("--root", default="experiments")
    parser.add_argument("--ahsef-root", default=None)
    parser.add_argument(
        "--modalities", nargs="+", default=None, choices=sorted(DEFAULT_BASELINES)
    )
    parser.add_argument(
        "--anchor", default="audio", choices=sorted(DEFAULT_BASELINES),
        help="Modality the routing study starts from.",
    )
    parser.add_argument(
        "--minimum-pool", type=int, default=30,
        help="Smallest co-split pool considered usable for a fused evaluation.",
    )
    return parser


def _render_matrix(pair: dict) -> str:
    left, right = pair["left"], pair["right"]
    header = f"{'':>12}" + "".join(f"{split:>12}" for split in SPLITS)
    lines = [f"  {left} (rows) x {right} (columns)", header]
    for split in SPLITS:
        lines.append(f"{split:>12}" + "".join(
            f"{pair['matrix'][split][other]:>12}" for other in SPLITS
        ))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ahsef_root = Path(args.ahsef_root) if args.ahsef_root else Path(args.root) / "ahsef"
    layout = AhsefLayout(run=args.run, root=ahsef_root)
    layout.prepare()

    modalities = ordered_modalities(args.modalities)
    index = AlignmentIndex.from_experiments(
        [
            (
                name,
                DEFAULT_BASELINES[name].experiment,
                DEFAULT_BASELINES[name].label_column,
                DEFAULT_BASELINES[name].task(args.root),
            )
            for name in modalities
        ],
        root=args.root,
    )

    matrix = index.alignment_matrix()
    layout.alignment_matrix_path.write_text(json.dumps(matrix, indent=2), encoding="utf-8")

    print("=" * 78)
    print("MODALITY POOLS")
    print("=" * 78)
    for name, record in matrix["modalities"].items():
        counts = record["counts"]
        print(
            f"  {name:<12} {record['experiment']:<16} task={record['task']:<22} "
            f"train={counts['train']:>7} val={counts['validation']:>7} test={counts['test']:>7}"
        )

    print()
    print("=" * 78)
    print("PAIRWISE SAMPLE-ID OVERLAP")
    print("=" * 78)
    for pair in matrix["pairs"].values():
        left, right = pair["left"], pair["right"]
        if not pair["shares_any_sample"]:
            print(f"\n  {left} x {right}: NO SHARED SAMPLE IDS IN ANY SPLIT")
            continue
        print()
        print(_render_matrix(pair))
        print(
            f"    co-split: " + ", ".join(
                f"{split}={pair['co_split'][split]}" for split in SPLITS
            )
            + f" | cross-split shared (excluded): {pair['cross_split_shared']}"
        )

    reports = {}
    for split in ("validation", "test"):
        candidates = [name for name in modalities if name != args.anchor]
        reports[split] = index.fusability_report(
            args.anchor, candidates, split, minimum=args.minimum_pool
        )

    print()
    print("=" * 78)
    print(f"FUSABILITY FROM ANCHOR '{args.anchor}' (minimum pool {args.minimum_pool})")
    print("=" * 78)
    for split, report in reports.items():
        print(f"\n  split = {split}")
        for name, entry in report["candidates"].items():
            verdict = "FUSABLE" if entry["fusable"] else "NOT FUSABLE"
            print(f"    {name:<12} {verdict:<12} co-split samples = {entry['co_split_samples']}")
            for blocker in entry["blockers"]:
                print(f"                 - {blocker}")

    # Every fusable pair across the whole index, not only from the anchor, so
    # the report states what the data supports rather than only what was asked.
    all_pairs = {}
    for pair in matrix["pairs"].values():
        left, right = pair["left"], pair["right"]
        for split in ("validation", "test"):
            if pair["co_split"][split] >= args.minimum_pool:
                same_task = (
                    matrix["modalities"][left]["task"] == matrix["modalities"][right]["task"]
                )
                all_pairs.setdefault(f"{left}+{right}", {})[split] = {
                    "co_split_samples": pair["co_split"][split],
                    "same_task": same_task,
                    "fusable": same_task,
                }

    payload = {
        "anchor": args.anchor,
        "minimum_pool": args.minimum_pool,
        "modalities": modalities,
        "fusability_from_anchor": reports,
        "all_fusable_pairs": all_pairs,
        "alignment_matrix_path": str(layout.alignment_matrix_path),
        "identity_basis": "sample_id column of each experiment's split manifest",
        "rule": (
            "Two modalities may be fused over a sample only if that sample is in the "
            "SAME split of both manifests and both models solve the same task. Samples "
            "shared across different splits are excluded: one model trained on them."
        ),
    }
    layout.alignment_report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print()
    print("=" * 78)
    print("FUSABLE PAIRS ANYWHERE IN THE INDEX")
    print("=" * 78)
    if not all_pairs:
        print("  none")
    for name, splits in sorted(all_pairs.items()):
        detail = ", ".join(
            f"{split}={record['co_split_samples']}" for split, record in sorted(splits.items())
        )
        usable = all(record["fusable"] for record in splits.values())
        print(f"  {name:<24} {detail}   {'usable' if usable else 'BLOCKED: label spaces differ'}")

    print()
    print(f"Written: {layout.alignment_matrix_path}")
    print(f"Written: {layout.alignment_report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

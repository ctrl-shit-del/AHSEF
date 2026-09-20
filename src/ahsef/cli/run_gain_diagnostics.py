"""Check whether dU is a usable information-gain target before HSIG is built.

    python -m src.ahsef.cli.run_gain_diagnostics --run stage1
    python -m src.ahsef.cli.run_gain_diagnostics --run stage1 --split validation

Reads every ``*_delta_uncertainty.parquet`` an earlier pairwise-fusion run
wrote and reports, per pair and fusion rule, whether dU's sign matches the
realised accuracy gain and whether it correlates with per-sample improvement.

This is post-hoc analysis and it reads labels. Run it on ``validation`` when
the answer is meant to inform a stage-2 design decision; the ``test`` view is
for the final report only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.ahsef.gain_diagnostics import diagnose, rank_agreement, render, verdict
from src.ahsef.layout import SPLITS, AhsefLayout


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default="stage1")
    parser.add_argument("--root", default="experiments")
    parser.add_argument("--ahsef-root", default=None)
    parser.add_argument(
        "--split", default="test", choices=[s for s in SPLITS if s != "train"],
        help="Which split's dU records to diagnose.",
    )
    parser.add_argument(
        "--anchor", default=None,
        help="Restrict the candidate-ranking check to pairs with this anchor.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ahsef_root = Path(args.ahsef_root) if args.ahsef_root else Path(args.root) / "ahsef"
    layout = AhsefLayout(run=args.run, root=ahsef_root)
    if not layout.fusion_dir.exists():
        raise SystemExit(
            f"No fusion artefacts in {layout.fusion_dir}. "
            f"Run: python -m src.ahsef.cli.run_pairwise_fusion --run {args.run}"
        )

    rows = []
    for path in sorted(layout.fusion_dir.glob(f"*/{args.split}_delta_uncertainty.parquet")):
        rows.append(diagnose(pd.read_parquet(path), path.parent.name))
    if not rows:
        raise SystemExit(
            f"No {args.split} delta-uncertainty records under {layout.fusion_dir}."
        )

    print("=" * 106)
    print(f"INFORMATION-GAIN DIAGNOSTICS  (split = {args.split})")
    print("=" * 106)
    print(render(rows))

    rankings = {}
    if args.anchor:
        prefix = f"{args.anchor}+"
        for suffix in ("", "__log_opinion_pool"):
            selected = [
                row for row in rows
                if row["name"].startswith(prefix) and row["name"].endswith(suffix)
                and (suffix or "__" not in row["name"])
            ]
            if len(selected) >= 2:
                rankings[suffix or "weighted_probability"] = rank_agreement(selected)

    summary = verdict(rows)
    print()
    print("=" * 106)
    print("VERDICT")
    print("=" * 106)
    print(f"  max |corr(dU, improvement)|            : {summary['max_abs_correlation_with_improvement']}")
    print(f"  pairs where dU's sign contradicts gain : "
          f"{summary['pairs_where_delta_sign_contradicts_accuracy_gain'] or 'none'}")
    print(f"  dU usable as HSIG's target             : "
          f"{summary['delta_uncertainty_is_a_usable_hsig_target']}")
    print()
    for line in summary["recommendation"].split(". "):
        if line.strip():
            print(f"  {line.strip().rstrip('.')}.")
    for name, record in rankings.items():
        print()
        print(f"  candidate ranking under {name}:")
        print(f"    by mean dU        : {record.get('ranked_by_mean_delta_uncertainty')}")
        print(f"    by accuracy gain  : {record.get('ranked_by_realised_accuracy_gain')}")
        print(f"    agree             : {record.get('rankings_agree')}")

    payload = {
        "split": args.split,
        "anchor": args.anchor,
        "pairs": rows,
        "candidate_rankings": rankings,
        "verdict": summary,
        "definition": "dU(m | A) = U(A) - U(A u {m}), U = normalised predictive entropy",
        "note": "Post-hoc analysis; reads labels. No routing decision consults it.",
    }
    path = layout.report_path(f"gain_diagnostics_{args.split}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print()
    print(f"Written: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

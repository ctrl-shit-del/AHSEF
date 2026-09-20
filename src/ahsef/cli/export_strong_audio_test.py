"""PHASE F (input) -- score the frozen strong-audio expert on the locked test split.

    python -m src.ahsef.cli.export_strong_audio_test \\
        --i-am-running-the-locked-evaluation

This exists as its own command rather than as a step inside the Phase F driver so
that opening the locked split is a deliberate act with its own flag, and so that
an evaluation run can never extract or score data as a side effect of reporting.

Nothing here is trained, fitted or chosen.  The probe checkpoint, the encoder,
the class order and the manifest are the frozen ones; this command runs them over
the test features and writes the per-sample export that Phase F reads.  It
refuses to run if the checkpoint has changed since the validation export, because
a different checkpoint is a different expert and the frozen policy was not
frozen around it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.ahsef.inference import BaselinePredictor, PredictionSet
from src.ahsef.registry import BaselineRef
from src.training.audio_strong_manifest import STRONG_EXPERIMENT

OUTPUT = Path("experiments") / "ahsef" / "analysis" / "audio_expert"
MODALITY = "audio_strong"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    parser.add_argument("--root", default="experiments")
    parser.add_argument("--experiment", default=STRONG_EXPERIMENT)
    parser.add_argument("--iteration", default=None)
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--i-am-running-the-locked-evaluation", action="store_true",
        help="Required to score the test split.",
    )
    return parser


def reference_checkpoint(output: Path) -> str | None:
    """The checkpoint the validation export used, if there is one to compare with."""
    path = output / f"{MODALITY}__validation.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("checkpoint_sha256")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.split == "test" and not args.i_am_running_the_locked_evaluation:
        raise SystemExit(
            "Refusing to score the test split without "
            "--i-am-running-the-locked-evaluation."
        )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    parquet = output / f"{MODALITY}__{args.split}.parquet"
    if parquet.exists():
        print(f"[skip] {parquet} already exists; it is not regenerated. Delete it "
              f"deliberately if you intend to re-score.")
        return 0

    expected = reference_checkpoint(output)
    reference = BaselineRef(
        MODALITY, args.experiment, "canonical_emotion_id", args.iteration
    )
    iteration = reference.resolve_iteration(args.root)
    print(f"[pred] scoring {args.experiment}/iteration {iteration} on {args.split} ...")

    predictor = BaselinePredictor(
        modality=MODALITY, experiment=args.experiment, iteration=iteration,
        root=args.root, device=args.device,
    )
    predictions = predictor.predict(args.split)

    actual = predictions.meta.get("checkpoint_sha256")
    if expected and actual != expected:
        raise SystemExit(
            f"The probe checkpoint has changed since the validation export "
            f"({expected} -> {actual}). The frozen policy was not frozen around "
            f"this expert; refusing to write a test export under its name."
        )

    predictions.save(parquet, parquet.with_suffix(".json"))
    print(f"[pred] {len(predictions.frame)} samples | checkpoint {actual}")
    print(f"Written: {parquet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

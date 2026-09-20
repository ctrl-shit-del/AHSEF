#!/usr/bin/env python
"""PROFILE B -- 25% of the training data, CPU, FP32.  Rapid experimentation.

    python scripts/train_hsen_cpu25.py --dataset iemocap --data_fraction 0.25
    python scripts/train_hsen_cpu25.py --dataset iemocap --modalities text --epochs 5

This is not a smaller model.  It builds the same HSEN from the same
:class:`~src.hsen.models.hsen.HSENConfig`, trains it with the same
:class:`~src.hsen.training.hsen_trainer.HSENTrainer`, computes the same loss and
reports the same metrics as the CUDA profile.  What differs is the device, the
training fraction, the batch size and the epoch budget -- and nothing else.
There is deliberately no model, loss or evaluation code in this file to differ
*with*.

CPU is forced.  Section 20 asks for it, and the reason is that the profile's
value is being a fast, honest preview of the full run: a CPU25 run that
silently used a GPU would have different timings, and its whole purpose is to
tell you what an hour of laptop time buys before you spend a day of GPU time.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.hsen.cli import CPU25_PROFILE, apply_yaml, build_parser, configs_from_args
from src.hsen.experiment import run_experiment


def main(argv: list[str] | None = None) -> int:
    parser = build_parser(CPU25_PROFILE, __doc__.splitlines()[0])
    args = apply_yaml(parser, parser.parse_args(argv))

    if not str(args.device).startswith("cpu"):
        raise SystemExit(
            f"train_hsen_cpu25.py forces CPU, but --device {args.device!r} was given.\n"
            f"Use scripts/train_hsen_cuda.py for GPU training."
        )
    if args.amp:
        # Not silently ignored: a run whose record says AMP but which ran FP32
        # is a mislabelled result.
        print("[note] AMP is meaningless on CPU; running FP32.")
        args.amp = False

    trainer_config, loss_config, modalities, overrides = configs_from_args(args, CPU25_PROFILE)
    run_experiment(
        trainer_config=trainer_config,
        modalities=modalities,
        fusion=args.fusion,
        loss_config=loss_config,
        model_overrides=overrides,
        feature_root=Path(args.feature_dir) if args.feature_dir else None,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

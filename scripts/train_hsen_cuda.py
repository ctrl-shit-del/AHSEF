#!/usr/bin/env python
"""PROFILE A -- 100% of the training data, CUDA, mixed precision.  The real runs.

    python scripts/train_hsen_cuda.py --dataset iemocap --data_fraction 1.0
    python scripts/train_hsen_cuda.py --dataset iemocap --evaluate_test

Reports the GPU it found and how much memory it has, enables AMP, and uses the
device for both training and validation.  No GPU model is assumed anywhere:
batch size is a flag with a conservative default, so a 4 GB laptop card and a
workstation card run the same script with different numbers.

If CUDA is requested and unavailable this fails, loudly, with the reason.  It
never falls back to CPU.  A silent fallback produces a run whose recorded
provenance says CUDA, whose timings say otherwise, and which nobody can
reproduce on either device.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.hsen.cli import CUDA_PROFILE, apply_yaml, build_parser, configs_from_args
from src.hsen.experiment import run_experiment


def report_cuda() -> None:
    """Fail with an actionable message, or describe what was found."""
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA was requested but torch reports no CUDA device.\n"
            f"  torch.__version__   = {torch.__version__}\n"
            f"  torch.version.cuda  = {torch.version.cuda}\n"
            "\n"
            "A '+cpu' torch build never sees a GPU regardless of the hardware "
            "present. Install a CUDA build matching the pin in requirements.txt:\n"
            "    pip install torch==2.12.0 torchaudio==2.11.0 torchvision==0.27.0 \\\n"
            "        --index-url https://download.pytorch.org/whl/cu124\n"
            "\n"
            "To train on this machine instead, use scripts/train_hsen_cpu25.py."
        )
    print(f"CUDA {torch.version.cuda} | {torch.cuda.device_count()} device(s)")
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        print(f"  [{index}] {properties.name}  "
              f"{properties.total_memory / 1024 ** 3:.1f} GB  "
              f"capability {properties.major}.{properties.minor}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser(CUDA_PROFILE, __doc__.splitlines()[0])
    args = apply_yaml(parser, parser.parse_args(argv))

    if str(args.device).startswith("cpu"):
        raise SystemExit(
            "train_hsen_cuda.py is the GPU profile; --device cpu was given.\n"
            "Use scripts/train_hsen_cpu25.py for CPU training."
        )
    report_cuda()

    trainer_config, loss_config, modalities, overrides = configs_from_args(args, CUDA_PROFILE)
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

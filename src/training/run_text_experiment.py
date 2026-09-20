"""CLI entry point for the text-only 7-class categorical emotion baseline.

Examples
--------
    # tiny end-to-end pipeline check (CPU, seconds)
    python -m src.training.run_text_experiment --experiment text_full --debug

    # the two independent baseline iterations
    python -m src.training.run_text_experiment --experiment text_full --iteration 1
    python -m src.training.run_text_experiment --experiment text_full --iteration 2 \
        --run-seed 43

    # rebuild experiment_summary.json without training
    python -m src.training.run_text_experiment --experiment text_full --summarize

Nothing is trained on import; training happens only when this module is run.
The argument surface is shared with every other modality; see
``src.training.experiment_cli``.
"""

from __future__ import annotations

import argparse

from src.training.text_experiment import (  # noqa: F401  (re-exported for callers)
    DESCRIPTION,
    TextExperimentConfig,
    run_iteration,
)
from src.training.base_experiment import DEBUG_OVERRIDES  # noqa: F401
from src.training.modalities import TEXT_SPEC

SPEC = TEXT_SPEC


def build_parser() -> argparse.ArgumentParser:
    return SPEC.build_parser()


def config_from_args(args: argparse.Namespace) -> TextExperimentConfig:
    """Build a config: defaults, then debug overrides, then explicit CLI flags."""
    return SPEC.config_from_args(args)


def main(argv: list[str] | None = None) -> int:
    return SPEC.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

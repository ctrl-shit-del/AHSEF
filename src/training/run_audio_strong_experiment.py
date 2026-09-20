"""CLI entry point for the ``audio_strong`` expert (frozen wav2vec2 + probe head).

Examples
--------
    # once: cache the frozen representations (hours, resumable, CPU-bound)
    python -m src.training.extract_audio_features --split validation
    python -m src.training.extract_audio_features --split train

    # tiny end-to-end pipeline check (seconds, needs a small cache)
    python -m src.training.run_audio_strong_experiment --debug

    # the two independent iterations
    python -m src.training.run_audio_strong_experiment --iteration 1
    python -m src.training.run_audio_strong_experiment --iteration 2 --run-seed 43

The wav2vec2 encoder is never trained; only the layer-weighted probe is, and it
reads the cache, so an iteration costs seconds rather than repeating extraction.
The argument surface is the one every other modality uses; see
``src.training.experiment_cli``.
"""

from __future__ import annotations

import argparse

from src.training.audio_strong_experiment import (  # noqa: F401  (re-exported)
    DESCRIPTION,
    AudioStrongExperimentConfig,
    run_iteration,
)
from src.training.base_experiment import DEBUG_OVERRIDES  # noqa: F401
from src.training.modalities import AUDIO_STRONG_SPEC

SPEC = AUDIO_STRONG_SPEC


def build_parser() -> argparse.ArgumentParser:
    return SPEC.build_parser()


def config_from_args(args: argparse.Namespace) -> AudioStrongExperimentConfig:
    """Build a config: defaults, then debug overrides, then explicit CLI flags."""
    return SPEC.config_from_args(args)


def main(argv: list[str] | None = None) -> int:
    return SPEC.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

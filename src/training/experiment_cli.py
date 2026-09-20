"""Shared command-line surface for every modality baseline.

Five near-identical argparse blocks would drift apart within a week, and a flag
that means one thing for audio and another for video is exactly the kind of
difference that quietly invalidates a cross-modality comparison.  So the
protocol flags -- iteration, seeds, epochs, early stopping, device, bounds,
debug, summarize -- are declared once here, and a modality contributes only its
own front-end and architecture flags through :class:`ExperimentSpec`.

Precedence is fixed and identical for every modality::

    dataclass defaults  ->  --debug overrides  ->  explicit CLI flags

so passing ``--debug --epochs 4`` gives a bounded run of four epochs rather
than silently ignoring one of the two.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Callable, Sequence

from src.common.experiment_layout import ExperimentLayout
from src.training.base_experiment import DEBUG_OVERRIDES, BaseExperimentConfig
from src.training.experiment_summary import build_experiment_summary, write_experiment_summary


#: Protocol flags shared by every modality, mapped to their config field.
COMMON_ARGUMENTS: tuple[tuple[str, dict], ...] = (
    ("--epochs", {"type": int}),
    ("--batch-size", {"type": int}),
    ("--learning-rate", {"type": float}),
    ("--weight-decay", {"type": float}),
    ("--patience", {"type": int}),
    ("--min-delta", {"type": float}),
    ("--min-epochs", {"type": int}),
    ("--seed", {"type": int}),
    ("--run-seed", {
        "type": int,
        "help": "Seed for model init and batch order; defaults to --seed, so two "
                "iterations with identical seeds reproduce each other exactly.",
    }),
    ("--device", {"help": "cpu, cuda, or auto."}),
    ("--num-workers", {"type": int}),
    ("--max-train", {"type": int}),
    ("--max-val", {"type": int}),
    ("--max-test", {"type": int}),
)


@dataclass(frozen=True)
class ExperimentSpec:
    """Everything the shared CLI needs to drive one modality's baseline."""

    modality: str
    default_experiment: str
    description: str
    config_class: type[BaseExperimentConfig]
    run_iteration: Callable[[BaseExperimentConfig], dict]
    help: str = ""
    #: Modality-specific flags, as ``("--flag", {argparse kwargs})``.  The
    #: destination must name a field of ``config_class``.
    extra_arguments: tuple[tuple[str, dict], ...] = ()
    #: Fields whose value should be echoed after a run.
    _fields: frozenset[str] = field(init=False, repr=False, compare=False, default=frozenset())

    def __post_init__(self) -> None:
        names = {item.name for item in fields(self.config_class)}
        for flag, _ in (*COMMON_ARGUMENTS, *self.extra_arguments):
            destination = flag.lstrip("-").replace("-", "_")
            if destination not in names:
                raise ValueError(
                    f"{self.modality} CLI declares {flag} but "
                    f"{self.config_class.__name__} has no field {destination!r}"
                )
        object.__setattr__(self, "_fields", frozenset(names))

    # ------------------------------------------------------------- parsing

    def build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description=self.help or f"Train the {self.modality} baseline.",
        )
        parser.add_argument(
            "--experiment", default=self.default_experiment,
            help=f"For example {self.default_experiment}.",
        )
        parser.add_argument("--experiment-root", default="experiments")
        parser.add_argument(
            "--metadata-dir", default=None, help="Override the experiment metadata dir."
        )
        parser.add_argument("--iteration", default=None, help="Iteration label, e.g. 1 or 2.")
        parser.add_argument("--description", default=self.description)

        for flag, options in (*self.extra_arguments, *COMMON_ARGUMENTS):
            parser.add_argument(flag, default=None, **options)

        parser.add_argument(
            "--debug", action="store_true",
            help="Tiny bounded run that exercises the full pipeline on CPU.",
        )
        parser.add_argument(
            "--summarize", action="store_true",
            help="Only rebuild experiment_summary.json from existing iterations.",
        )
        return parser

    def config_from_args(self, args: argparse.Namespace) -> BaseExperimentConfig:
        """Defaults, then debug overrides, then explicit CLI flags."""
        iteration = args.iteration or ("debug" if args.debug else "1")
        config = self.config_class(
            experiment_name=args.experiment,
            description=args.description,
            iteration=iteration,
            experiment_root=args.experiment_root,
            metadata_dir=args.metadata_dir,
        )
        if args.debug:
            config = replace(config, debug=True, **DEBUG_OVERRIDES)
        overrides = {
            name: getattr(args, name)
            for name in self._fields
            if getattr(args, name, None) is not None
        }
        # ``--experiment`` and friends are already applied above.
        for name in ("experiment_name", "description", "iteration", "experiment_root",
                     "metadata_dir", "debug"):
            overrides.pop(name, None)
        return replace(config, **overrides) if overrides else config

    # ----------------------------------------------------------- execution

    def main(self, argv: Sequence[str] | None = None) -> int:
        args = self.build_parser().parse_args(list(argv) if argv is not None else None)
        layout = ExperimentLayout.from_name(args.experiment, Path(args.experiment_root))

        if args.summarize:
            path = write_experiment_summary(layout)
            print(json.dumps(build_experiment_summary(layout)["comparison"], indent=2, sort_keys=True))
            print(f"Experiment summary written to {path}")
            return 0

        config = self.config_from_args(args)
        summary = self.run_iteration(config)
        print_run_summary(summary, layout)
        return 0


def print_run_summary(summary: dict[str, Any], layout: ExperimentLayout) -> None:
    """Echo the headline numbers of one finished iteration."""
    print()
    print("=" * 70)
    print(f"{summary['experiment']} | iteration {summary['iteration']}")
    print("=" * 70)
    print(f"Samples            : {summary['samples']}")
    print(f"Epochs completed   : {summary['epochs_completed']}/{summary['epochs_configured']}")
    print(f"Best epoch         : {summary['best_epoch']}")
    print(f"Best val macro-F1  : {summary['best_val_macro_f1']:.6f}")
    print(f"Stop reason        : {summary['stop_reason']}")
    print(f"Test accuracy      : {summary['test_metrics']['accuracy']:.6f}")
    print(f"Test macro-F1      : {summary['test_metrics']['macro_f1']:.6f}")
    print(f"Test weighted-F1   : {summary['test_metrics']['weighted_f1']:.6f}")
    print(f"Results            : {layout.results_dir(summary['iteration'])}")
    print(f"Checkpoints        : {layout.checkpoints_dir(summary['iteration'])}")
    print(f"Log                : {summary['log_file']}")
    print(f"Experiment summary : {layout.experiment_summary_path}")

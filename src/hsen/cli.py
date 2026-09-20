"""One argument parser, two profiles.

Both ``scripts/train_hsen_cuda.py`` and ``scripts/train_hsen_cpu25.py`` build
their configuration here.  The profile supplies the *defaults* -- device, data
fraction, batch size, AMP -- and every one of them stays overridable from the
command line, because a default that cannot be argued with is a default nobody
can debug.

What a profile may not change is anything in
:class:`~src.hsen.models.hsen.HSENConfig`.  Model geometry is settled by the
literature survey and by section 23, so ``--model-dim``, ``--num-layers`` and
``--num-heads`` exist as flags for the section-14 controlled search and default
to 256 / 2 / 4 in both profiles.  ``--fusion`` and ``--modalities`` likewise
default to the same values in both profiles: ``concat`` over ``audio+text``,
the configuration Phase 15 selected for the full-data run.  Video is deferred;
IEMOCAP has no video cache, so a default of ``all`` would make the documented
``--dataset iemocap --data_fraction 1.0`` invocation demand one and fail.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import yaml

from src.hsen.manifests import EXPERIMENTS
from src.hsen.models.fusion_variants import FUSION_VARIANTS
from src.hsen.training.hsen_trainer import TrainerConfig
from src.hsen.training.losses import LossConfig

#: Friendly dataset names, so the documented ``--dataset iemocap`` works while
#: the precise protocol name stays available for a run that needs to be exact.
DATASET_ALIASES: dict[str, str] = {
    "iemocap": "iemocap_erc6",
    "iemocap_erc6": "iemocap_erc6",
    "iemocap_ser4": "iemocap_ser4",
    "mosei": "mosei_sentiment",
    "mosei_sentiment": "mosei_sentiment",
    "mosei_emotion6": "mosei_emotion6",
}

MODALITY_SETS: dict[str, tuple[str, ...]] = {
    "text": ("text",),
    "audio": ("audio",),
    "video": ("video",),
    "audio+text": ("audio", "text"),
    "audio+video": ("audio", "video"),
    "video+text": ("video", "text"),
    "audio+video+text": ("audio", "video", "text"),
    "all": ("audio", "video", "text"),
}


@dataclass
class Profile:
    """The compute-and-data defaults that distinguish the two execution profiles."""

    name: str
    device: str
    amp: bool
    data_fraction: float
    batch_size: int
    num_workers: int
    epochs: int


CUDA_PROFILE = Profile(
    name="full_cuda", device="cuda", amp=True, data_fraction=1.0,
    batch_size=32, num_workers=4, epochs=30,
)
CPU25_PROFILE = Profile(
    name="cpu25", device="cpu", amp=False, data_fraction=0.25,
    batch_size=8, num_workers=0, epochs=15,
)


def build_parser(profile: Profile, description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Both spellings are accepted: the phase brief writes --data_fraction, and
    # the rest of this repository's CLIs use dashes.
    data = parser.add_argument_group("data")
    data.add_argument("--dataset", default="iemocap", choices=sorted(DATASET_ALIASES),
                      help="Experiment or dataset alias.")
    data.add_argument("--data_fraction", "--data-fraction", dest="data_fraction",
                      type=float, default=profile.data_fraction,
                      help="Fraction of the TRAINING split to use; "
                           "validation and test are always complete.")
    data.add_argument("--modalities", default="audio+text", choices=sorted(MODALITY_SETS),
                      help="Which modality branches to build (section 12 ablations). "
                           "Video is deferred; 'all' requires a video cache.")
    data.add_argument("--feature_dir", "--feature-dir", dest="feature_dir", default=None,
                      help="Root of the feature caches.")

    schedule = parser.add_argument_group("schedule")
    schedule.add_argument("--epochs", type=int, default=profile.epochs)
    schedule.add_argument("--batch_size", "--batch-size", dest="batch_size",
                          type=int, default=profile.batch_size)
    schedule.add_argument("--eval_batch_size", "--eval-batch-size", dest="eval_batch_size",
                          type=int, default=None)
    schedule.add_argument("--lr", "--learning-rate", dest="lr", type=float, default=1e-4)
    schedule.add_argument("--weight_decay", "--weight-decay", dest="weight_decay",
                          type=float, default=1e-5)
    schedule.add_argument("--optimizer", default="adamw", choices=["adam", "adamw"])
    schedule.add_argument("--scheduler", default="cosine_warmup",
                          choices=["cosine_warmup", "plateau", "none"])
    schedule.add_argument("--warmup_ratio", "--warmup-ratio", dest="warmup_ratio",
                          type=float, default=0.1)
    schedule.add_argument("--gradient_clip", "--gradient-clip", dest="gradient_clip",
                          type=float, default=1.0)
    schedule.add_argument("--patience", type=int, default=5)
    schedule.add_argument("--min_epochs", "--min-epochs", dest="min_epochs",
                          type=int, default=3)
    schedule.add_argument("--seed", type=int, default=42)

    model = parser.add_argument_group("model (literature-fixed defaults)")
    model.add_argument("--fusion", default="concat", choices=list(FUSION_VARIANTS),
                       help="Phase 15 selected concat; the other variants are "
                            "for the section-13 ablation only.")
    model.add_argument("--model_dim", "--model-dim", dest="model_dim", type=int, default=256)
    model.add_argument("--num_layers", "--num-layers", dest="num_layers", type=int, default=2)
    model.add_argument("--num_heads", "--num-heads", dest="num_heads", type=int, default=4)
    model.add_argument("--dropout", type=float, default=0.1)
    model.add_argument("--attention_dropout", "--attention-dropout",
                       dest="attention_dropout", type=float, default=0.1)

    loss = parser.add_argument_group("loss")
    loss.add_argument("--focal_gamma", "--focal-gamma", dest="focal_gamma",
                      type=float, default=2.0)
    loss.add_argument("--class_weights", "--class-weights", dest="class_weights",
                      default="balanced", choices=["balanced", "none"])
    loss.add_argument("--lambda_cls", "--lambda-cls", dest="lambda_cls",
                      type=float, default=1.0)
    loss.add_argument("--lambda_valence", "--lambda-valence", dest="lambda_valence",
                      type=float, default=0.0)
    loss.add_argument("--lambda_arousal", "--lambda-arousal", dest="lambda_arousal",
                      type=float, default=0.0)
    loss.add_argument("--regression_loss", "--regression-loss", dest="regression_loss",
                      default="mse", choices=["mse", "huber", "ccc"])
    loss.add_argument("--label_smoothing", "--label-smoothing", dest="label_smoothing",
                      type=float, default=0.0)

    run = parser.add_argument_group("run")
    run.add_argument("--config", default=None, help="YAML file of defaults; "
                                                    "explicit flags still win.")
    run.add_argument("--checkpoint_dir", "--checkpoint-dir", dest="checkpoint_dir",
                     default=None, help="Root for checkpoints, metrics and logs.")
    run.add_argument("--device", default=profile.device,
                     help="Overriding this defeats the point of the profile; "
                          "the CPU script refuses a non-CPU value.")
    run.add_argument("--num_workers", "--num-workers", dest="num_workers",
                     type=int, default=profile.num_workers)
    run.add_argument("--amp", dest="amp", action="store_true", default=profile.amp)
    run.add_argument("--no-amp", dest="amp", action="store_false")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--force-lock", "--force_lock", dest="force_lock",
                     action="store_true",
                     help="Break a stale run-directory lock. Only after confirming "
                          "the holding process is gone.")
    run.add_argument("--evaluate_test", "--evaluate-test", dest="evaluate_test",
                     action="store_true",
                     help="Open the test split after training. Deliberate by design: "
                          "the test partition is opened once, when every "
                          "validation-side decision is already made.")
    run.add_argument("--run_name", "--run-name", dest="run_name", default=None,
                     help="Subdirectory for this run; defaults to the profile name.")
    run.add_argument("--max_train_batches", "--max-train-batches",
                     dest="max_train_batches", type=int, default=None,
                     help="Sanity runs only.")
    run.add_argument("--max_eval_batches", "--max-eval-batches",
                     dest="max_eval_batches", type=int, default=None)
    return parser


def apply_yaml(parser: argparse.ArgumentParser, args: argparse.Namespace) -> argparse.Namespace:
    """Fold a YAML config in underneath the command line.

    Precedence is: dataclass defaults, then the profile, then the YAML, then
    explicit flags.  A flag the user typed always wins, which is checked by
    comparing against the parser's own defaults rather than by trusting order.
    """
    if not args.config:
        return args
    path = Path(args.config)
    if not path.exists():
        raise SystemExit(f"No config file at {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    flat = {}
    for key, value in payload.items():
        flat.update(value) if isinstance(value, dict) else flat.setdefault(key, value)
    for key, value in flat.items():
        if not hasattr(args, key):
            continue
        if getattr(args, key) == parser.get_default(key):
            setattr(args, key, value)
    return args


def configs_from_args(
    args: argparse.Namespace, profile: Profile,
) -> tuple[TrainerConfig, LossConfig, tuple[str, ...], dict]:
    """Turn parsed arguments into the objects the trainer takes."""
    experiment = DATASET_ALIASES[args.dataset]
    trainer = TrainerConfig(
        experiment=experiment,
        dataset=EXPERIMENTS[experiment].dataset,
        profile=args.run_name or profile.name,
        output_root=Path(args.checkpoint_dir) if args.checkpoint_dir else Path("results/hsen"),
        device=args.device,
        amp=args.amp,
        num_workers=args.num_workers,
        data_fraction=args.data_fraction,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        warmup_ratio=args.warmup_ratio,
        gradient_clip=args.gradient_clip,
        patience=args.patience,
        min_epochs=args.min_epochs,
        seed=args.seed,
        resume=args.resume,
        force_lock=args.force_lock,
        evaluate_test=args.evaluate_test,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
    )
    loss = LossConfig(
        lambda_cls=args.lambda_cls,
        lambda_valence=args.lambda_valence,
        lambda_arousal=args.lambda_arousal,
        focal_gamma=args.focal_gamma,
        class_weights=args.class_weights,
        label_smoothing=args.label_smoothing,
        regression_loss=args.regression_loss,
    )
    overrides = {
        "model_dim": args.model_dim,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "dropout": args.dropout,
        "attention_dropout": args.attention_dropout,
        "predict_valence": True,
        "predict_arousal": True,
    }
    return trainer, loss, MODALITY_SETS[args.modalities], overrides

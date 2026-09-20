"""Assemble a runnable HSEN experiment from manifests, caches and a config.

This is the join between everything built separately: the audited manifests, the
label adapters, the three feature caches, the architecture and the trainer.  Both
execution profiles come through :func:`build_experiment`, so a profile can only
change what :class:`~src.hsen.training.hsen_trainer.TrainerConfig` exposes.  It
cannot reach past this function to build a different model.

The data fraction applies to **training only**.  Validation and test are always
whole, because the point of the 25% profile is a fast estimate of what the full
run will do, and an estimate measured on a quarter of the validation set is
noisier for no gain in speed that matters -- validation is a fraction of the cost
of training.  It also keeps every run's numbers computed on the same evaluation
samples, which is what makes the fast profile's ranking of two configurations
mean anything for the slow one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.hsen.audit import SPLITS, assert_clean, audit_manifests, render
from src.hsen.data import HSENDataset, stratified_subset
from src.hsen.extract import cache_dir
from src.hsen.features.store import SequenceFeatureStore
from src.hsen.labels.base import label_set_from_manifest, resolve_label_adapter
from src.hsen.manifests import EXPERIMENTS, MODALITY_PAYLOADS, load_all
from src.hsen.models.hsen import DEFAULT_INPUT_DIMS, HSENConfig
from src.hsen.training.hsen_trainer import HSENTrainer, TrainerConfig
from src.hsen.training.losses import LossConfig

#: Which label protocol and dataset each experiment trains under.  Mirrors
#: :data:`src.hsen.manifests.EXPERIMENTS` deliberately -- an experiment that can
#: be trained must first be an experiment whose manifests exist.
EXPERIMENT_DATASETS = {name: config.dataset for name, config in EXPERIMENTS.items()}


class ExperimentError(RuntimeError):
    """Raised when an experiment cannot be assembled."""


@dataclass
class ExperimentBundle:
    """Everything a trainer needs, plus the provenance of how it was built."""

    experiment: str
    dataset: str
    datasets: dict[str, HSENDataset]
    manifests: dict[str, pd.DataFrame]
    class_names: tuple[str, ...]
    model_config: HSENConfig
    audit: dict = field(default_factory=dict)


def open_feature_sources(
    experiment: str, split: str, modalities: tuple[str, ...], feature_root: Path | None = None,
) -> dict[str, SequenceFeatureStore]:
    """Open one cache per modality for one split.

    A modality with no cache is reported as a missing extraction step rather
    than skipped: silently training audio+text under the name audio+video+text
    would put a wrong row in the ablation table.
    """
    sources: dict[str, SequenceFeatureStore] = {}
    missing: list[str] = []
    for modality in modalities:
        root = cache_dir(experiment, modality, split, feature_root)
        index = root / "index.json"
        if not index.exists():
            missing.append(f"{modality}/{split}")
            continue
        record = json.loads(index.read_text(encoding="utf-8"))
        sources[modality] = SequenceFeatureStore.open(
            root, record["fingerprint"], modality, int(record["feature_dim"]),
        )
    if missing:
        commands = "\n".join(
            f"    python -m src.hsen.extract --experiment {experiment} "
            f"--modality {item.split('/')[0]} --split {item.split('/')[1]}"
            for item in missing
        )
        raise ExperimentError(
            f"No feature cache for {missing}. Extract them first:\n{commands}"
        )
    return sources


def infer_input_dims(sources: dict[str, SequenceFeatureStore]) -> dict[str, int]:
    """Take each modality's width from its cache, not from a hardcoded default.

    CMU-MOSEI's provided features are 74-d audio and 35-d vision, nothing like
    IEMOCAP's 768-d emotion2vec frames, and the projection layer is what makes
    that irrelevant to the trunk.  Reading the width from the cache is what lets
    one model config serve both without a per-dataset special case.
    """
    return dict(DEFAULT_INPUT_DIMS) | {
        modality: int(source.feature_dim) for modality, source in sources.items()
    }


def build_experiment(
    experiment: str,
    modalities: tuple[str, ...] = ("audio", "video", "text"),
    data_fraction: float = 1.0,
    seed: int = 42,
    fusion: str = "husformer",
    model_overrides: dict | None = None,
    feature_root: Path | None = None,
    include_test: bool = True,
    strict_features: bool = True,
    subset_seed: int | None = None,
) -> ExperimentBundle:
    """Build the datasets and the architecture for one experiment.

    ``subset_seed`` seeds the stratified training-fraction draw and defaults to
    ``seed``.  A study that varies ``seed`` while holding ``subset_seed`` fixed
    is measuring initialisation and batch-order variance on one fixed training
    set; one that lets both move is also measuring which quarter of the data
    was drawn.
    """
    subset_seed = seed if subset_seed is None else subset_seed
    if experiment not in EXPERIMENT_DATASETS:
        raise ExperimentError(
            f"Unknown experiment {experiment!r}; expected one of "
            f"{sorted(EXPERIMENT_DATASETS)}"
        )
    dataset_name = EXPERIMENT_DATASETS[experiment]
    protocol = EXPERIMENTS[experiment].label_protocol
    adapter = resolve_label_adapter(protocol)
    space = adapter.label_space()

    manifests = load_all(experiment)
    wanted = [split for split in SPLITS if include_test or split != "test"]
    manifests = {split: manifests[split] for split in wanted}

    # The audit runs again here, over the manifests as they are on disk right
    # now, rather than trusting the audit.json written when they were built. A
    # manifest edited after the fact is exactly the case a stored verdict misses.
    report = audit_manifests(
        experiment=experiment,
        frames=manifests,
        class_names=space.classes,
        modalities=MODALITY_PAYLOADS.get(dataset_name),
        speaker_check_required=dataset_name == "IEMOCAP",
        cache_root=(Path(feature_root) if feature_root else Path("experiments/hsen"))
        / experiment / "features",
    )
    print(render(report))
    assert_clean(report)

    if data_fraction < 1.0:
        selected = stratified_subset(manifests["train"], data_fraction, subset_seed)
        manifests["train"] = manifests["train"].iloc[selected].reset_index(drop=True)
        print(
            f"Training fraction {data_fraction:.0%}: "
            f"{len(manifests['train']):,} of {len(selected):,} selected rows "
            f"(stratified, subset seed {subset_seed}). Validation and test are complete."
        )

    datasets: dict[str, HSENDataset] = {}
    input_dims: dict[str, int] = dict(DEFAULT_INPUT_DIMS)
    for split, frame in manifests.items():
        sources = open_feature_sources(experiment, split, modalities, feature_root)
        input_dims |= infer_input_dims(sources)
        datasets[split] = HSENDataset(
            manifest=frame,
            sources=sources,
            labels=label_set_from_manifest(frame, space, adapter.build(frame.head(0)).provenance),
            modalities=modalities,
            strict=strict_features,
        )

    config = HSENConfig(
        modalities=tuple(modalities),
        num_classes=space.num_classes,
        label_space=space.name,
        task="multi_label" if protocol == "mosei_emotion6" else "single_label",
        fusion=fusion,
        input_dims=input_dims,
        **(model_overrides or {}),
    )

    return ExperimentBundle(
        experiment=experiment,
        dataset=dataset_name,
        datasets=datasets,
        manifests=manifests,
        class_names=space.classes,
        model_config=config,
        audit=report.to_dict() | {"split_policy": EXPERIMENTS[experiment].split_policy},
    )


def run_experiment(
    trainer_config: TrainerConfig,
    modalities: tuple[str, ...] = ("audio", "video", "text"),
    fusion: str = "husformer",
    loss_config: LossConfig | None = None,
    model_overrides: dict | None = None,
    feature_root: Path | None = None,
) -> dict:
    """Build and train one experiment.  The single entry point both profiles use."""
    bundle = build_experiment(
        experiment=trainer_config.experiment,
        modalities=modalities,
        data_fraction=trainer_config.data_fraction,
        seed=trainer_config.seed,
        fusion=fusion,
        model_overrides=model_overrides,
        feature_root=feature_root,
        include_test=trainer_config.evaluate_test,
        subset_seed=trainer_config.subset_seed,
    )
    trainer_config.dataset = bundle.dataset
    trainer = HSENTrainer(
        config=trainer_config,
        model_config=bundle.model_config,
        loss_config=loss_config or LossConfig(),
        datasets=bundle.datasets,
        manifests=bundle.manifests,
        class_names=bundle.class_names,
        audit=bundle.audit,
    )
    return trainer.run()

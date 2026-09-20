"""Physiology baseline over the full WESAD window pool.

WESAD is the one modality pool that cannot join the common 7-class emotion
task: every WESAD record in the standardized metadata carries
``target_type == 'physiological_state'`` and ``canonical_emotion_valid ==
False``.  Rather than forcing invalid emotion labels, this baseline declares
the WESAD study-condition label space -- baseline / stress / amusement -- and
records that limitation in every artefact it writes (``task``,
``task_limitation``, ``class_order``).  The classifier therefore emits three
logits, not seven, and the confusion matrix is 3x3.

Two protections beyond the shared protocol matter here:

* **Subject-disjoint splits.**  Enforced upstream by the window builder and
  re-asserted here: the run summary records which subjects landed in which
  partition, and the runner refuses to train if any subject spans two.
* **Train-fitted feature standardisation.**  Per-feature mean and standard
  deviation come from the training windows only and are written to
  ``results/feature_normalization.json``; validation and test are transformed
  with those statistics, never with their own.

The module never runs itself; see ``src.training.run_physiology_experiment``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch.nn as nn

from src.data.dataloader import create_physiology_window_dataloader
from src.data.physiology_dataset import PhysiologyWindowDataset, fit_feature_scaling
from src.models.baselines import PhysiologyStateBaseline
from src.training.base_experiment import (
    DEBUG_OVERRIDES,
    BaseExperimentConfig,
    BaseExperimentRunner,
    save_json,
)


DESCRIPTION = (
    "Physiology-only WESAD study-condition baseline over subject-disjoint windows."
)


@dataclass
class PhysiologyExperimentConfig(BaseExperimentConfig):
    """Everything needed to reproduce one physiology training iteration."""

    experiment_name: str = "physiology_full"
    description: str = DESCRIPTION
    modality: str = "physiology"
    task: str = "wesad_state_3class"
    num_classes: int = 3
    batch_size: int = 64

    # model
    hidden_dim: int = 128

    # protocol guards
    require_subject_disjoint_splits: bool = True


def debug_config(config: PhysiologyExperimentConfig) -> PhysiologyExperimentConfig:
    """Return a tiny, CPU-fast variant that exercises the whole pipeline."""
    return replace(config, debug=True, **DEBUG_OVERRIDES)


class PhysiologyExperimentRunner(BaseExperimentRunner):
    """Execute one reproducible iteration of the physiology baseline."""

    INPUT_KEY = "physiology"
    LENGTHS_KEY = None
    LABEL_COLUMN = PhysiologyWindowDataset.LABEL_COLUMN

    config: PhysiologyExperimentConfig

    def __init__(self, config: PhysiologyExperimentConfig):
        super().__init__(config)
        self._scaling: tuple[list[float], list[float]] | None = None
        self._feature_dim: int | None = None
        self._split_subjects: dict[str, list[str]] = {}

    # ------------------------------------------------------------- loaders

    def build_loader(self, split: str, shuffle: bool, max_samples: int | None):
        mean, std = self._scaling if self._scaling is not None else (None, None)
        loader = create_physiology_window_dataloader(
            self.manifest_path(split),
            batch_size=self.config.batch_size,
            label_space=self.label_space,
            mean=mean,
            std=std,
            max_samples=max_samples,
            seed=self.config.effective_run_seed,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
        )
        dataset = loader.dataset
        self._feature_dim = dataset.feature_dim
        self._split_subjects[split] = dataset.subjects

        if split == "train" and self._scaling is None:
            # Fit standardisation on the training windows and apply it in place.
            # Validation and test loaders are built afterwards and inherit it.
            self._scaling = fit_feature_scaling(dataset)
            dataset.set_scaling(*self._scaling)
        return loader

    # -------------------------------------------------------------- hooks

    def prepare(self, train_loader, val_loader, logger) -> dict:
        self._assert_subject_disjoint(logger)
        mean, std = self._scaling if self._scaling else ([], [])
        record = {
            "feature_standardisation": {
                "computed_from_split": "train",
                "uses_validation_windows": False,
                "uses_test_windows": False,
                "feature_dim": self._feature_dim,
                "mean": mean,
                "std": std,
            },
            "subjects": dict(self._split_subjects),
        }
        save_json(
            self.layout.results_dir(self.iteration) / "feature_normalization.json",
            record["feature_standardisation"],
        )
        logger.info("Feature dimension  : %s", self._feature_dim)
        logger.info("Train subjects     : %s", ", ".join(self._split_subjects.get("train", [])))
        logger.info("Validation subjects: %s", ", ".join(self._split_subjects.get("validation", [])))
        return record

    def _assert_subject_disjoint(self, logger) -> None:
        train = set(self._split_subjects.get("train", []))
        validation = set(self._split_subjects.get("validation", []))
        overlap = sorted(train & validation)
        if overlap:
            message = (
                f"Subjects {overlap} appear in both the training and validation "
                f"partitions of {self.metadata_dir}. Windows of one subject are "
                f"highly autocorrelated, so this would measure memorisation. "
                f"Rebuild the experiment with "
                f"src.preprocessing.sampling.run_physiology_sampler."
            )
            if self.config.require_subject_disjoint_splits:
                raise ValueError(message)
            logger.warning(message)

    def extra_summary(self) -> dict:
        return {
            "subjects": dict(self._split_subjects),
            "subject_disjoint_splits": self._subject_disjoint(),
            "feature_dim": self._feature_dim,
        }

    def _subject_disjoint(self) -> bool:
        seen: dict[str, str] = {}
        for split, subjects in self._split_subjects.items():
            for subject in subjects:
                if seen.setdefault(subject, split) != split:
                    return False
        return True

    # -------------------------------------------------------------- model

    def build_model(self) -> nn.Module:
        if self._feature_dim is None:
            raise RuntimeError(
                "The training loader must be built before the model so the feature "
                "dimension is known."
            )
        return PhysiologyStateBaseline(
            feature_dim=self._feature_dim,
            hidden_dim=self.config.hidden_dim,
            num_classes=self.config.num_classes,
        )

    def model_record(self, model: nn.Module) -> dict:
        return {
            "feature_dim": self._feature_dim,
            "hidden_dim": self.config.hidden_dim,
            "input": "per-channel window statistics, standardised with train-fitted scaling",
        }

    def _hyperparameter_record(self) -> dict:
        record = super()._hyperparameter_record()
        record.update({
            "hidden_dim": self.config.hidden_dim,
            "feature_dim": self._feature_dim,
        })
        return record


def run_iteration(config: PhysiologyExperimentConfig) -> dict:
    """Execute a single iteration and return its run summary."""
    return PhysiologyExperimentRunner(config).run()

"""Shared lazy dataset base for modality-filtered classification experiments.

Every modality baseline reads the same kind of file -- an experiment split
Parquet produced by the sampler -- and applies the same three filters before a
single media file is opened:

1. the modality really is available for the record (``has_<modality>``, an
   accepted ``<modality>_source``, and a payload column that is actually
   populated), evaluated with the *same* rule the sampler used;
2. the canonical target is valid;
3. an optional deterministic bound on the number of samples, for debug runs.

Only the columns a modality needs are read, so a 250k-row manifest costs a few
megabytes rather than the whole standardized schema, and media is opened lazily
in ``__getitem__`` -- never preloaded.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset

from src.common.labels import EMOTION_7CLASS, LabelSpace
from src.preprocessing.sampling.rules import ModalityRule, get_modality_rule


class ModalityManifestDataset(Dataset):
    """Lazy dataset over one experiment split for a single modality."""

    #: Modality name; resolves the eligibility rule from the sampler's registry.
    MODALITY: str = ""
    #: Manifest column holding the integer target.
    LABEL_COLUMN: str = "canonical_emotion_id"
    #: Extra columns this modality needs beyond the rule's own requirements.
    EXTRA_COLUMNS: tuple[str, ...] = ()

    def __init__(
        self,
        manifest_path: str | Path,
        max_samples: int | None = None,
        seed: int = 42,
        columns: list[str] | tuple[str, ...] | None = None,
        label_space: LabelSpace = EMOTION_7CLASS,
    ):
        if not self.MODALITY:
            raise NotImplementedError(f"{type(self).__name__} must declare MODALITY")
        self.manifest_path = Path(manifest_path)
        self.label_space = label_space
        self.rule: ModalityRule = get_modality_rule(self.MODALITY)

        wanted = list(columns) if columns else list(self.required_columns())
        manifest = pd.read_parquet(self.manifest_path, columns=wanted)
        missing = set(self.required_columns()) - set(manifest.columns)
        if missing:
            raise ValueError(
                f"{self.MODALITY} manifest {self.manifest_path} is missing columns: "
                f"{sorted(missing)}"
            )

        self.manifest = manifest[self._eligible_mask(manifest)].copy()
        if self.manifest.empty:
            raise ValueError(
                f"No eligible {self.MODALITY} records with a valid "
                f"{self.label_space.name} target in {self.manifest_path}"
            )
        if max_samples is not None:
            if max_samples <= 0:
                raise ValueError("max_samples must be positive")
            self.manifest = self.manifest.sample(
                n=min(max_samples, len(self.manifest)), random_state=seed
            ).sort_index()
        self.manifest.reset_index(drop=True, inplace=True)

    # ------------------------------------------------------------- filtering

    @classmethod
    def required_columns(cls) -> tuple[str, ...]:
        rule = get_modality_rule(cls.MODALITY)
        columns = ["sample_id", "dataset", cls.LABEL_COLUMN, *rule.required_columns]
        columns.extend(cls.EXTRA_COLUMNS)
        return tuple(dict.fromkeys(columns))

    def _eligible_mask(self, manifest: pd.DataFrame) -> pd.Series:
        rule = self.rule
        available = manifest[rule.availability_column].fillna(False).astype(bool)
        source = manifest[rule.source_column]
        accepted = source.isin(list(rule.accepted_sources))

        payload = pd.Series(False, index=manifest.index)
        for accepted_source in rule.accepted_sources:
            required = rule.payload_columns(accepted_source)
            if not required:
                present = pd.Series(True, index=manifest.index)
            else:
                present = pd.Series(False, index=manifest.index)
                for column in required:
                    values = manifest[column]
                    column_present = values.notna()
                    if values.dtype == object:
                        column_present &= values.fillna("").astype(str).str.strip().ne("")
                    present |= column_present
            payload |= source.eq(accepted_source) & present

        labels = pd.to_numeric(manifest[self.LABEL_COLUMN], errors="coerce")
        valid_labels = labels.isin(list(self.label_space.valid_ids))
        return available & accepted & payload & valid_labels

    # ------------------------------------------------------------------ data

    def __len__(self) -> int:
        return len(self.manifest)

    def row(self, index: int):
        return self.manifest.iloc[index]

    def __getitem__(self, index: int) -> dict:  # pragma: no cover - abstract
        raise NotImplementedError

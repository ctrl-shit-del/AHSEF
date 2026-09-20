"""HSEN experiment manifests: one leakage-audited table per split, per protocol.

    python -m src.hsen.manifests --experiment iemocap_erc6
    python -m src.hsen.manifests --experiment mosei_sentiment
    python -m src.hsen.manifests --experiment iemocap_erc6 --audit-only

Where the records come from.  ``metadata/standardized/standardized.parquet`` is
the project's single source of record identity, labels, sessions and asset
paths, and it stays that way here: this module filters and partitions it, and
invents nothing.

The one exception, stated plainly.  IEMOCAP's adapter did not read transcripts
until this phase, so the standardized table on disk carries ``text = NULL`` for
all 10,039 IEMOCAP rows even though the gold transcripts sit in
``Session*/dialog/transcriptions/``.  Rather than regenerate 792,336 rows of
master metadata -- which would rescan AffectNet+'s 420,299 images to fix a
column on 10,039 IEMOCAP rows -- the transcript is joined in here, using the
same parser :mod:`src.preprocessing.adapters.iemocap` now uses, and only where
the standardized column is empty.  Once ``generate_metadata`` is next run the
column is populated at source and this join becomes a no-op it detects and
reports.  Nothing else about a record is ever sourced outside the standardized
table.

Splits are protocol splits, never random draws:

IEMOCAP
    Whole-session partitions.  Sessions 1-3 train, Session 4 validates,
    Session 5 tests.  IEMOCAP's ten actors appear in exactly one session each,
    so a session partition is a speaker partition, and the audit verifies that
    rather than assuming it.

CMU-MOSEI
    The official standard-partition ``train`` / ``valid`` / ``test`` split,
    carried through from the release's own ``mode`` column.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.paths import DATASETS_DIR
from src.hsen.audit import SPLITS, AuditReport, assert_clean, audit_manifests, render
from src.hsen.labels.base import MISSING_CLASS_ID, LabelSet, resolve_label_adapter

STANDARDIZED_PATH = Path("metadata/standardized/standardized.parquet")
MANIFEST_ROOT = Path("metadata/hsen")

#: Columns carried into every manifest.  Deliberately narrow: a manifest is
#: read once per epoch by the dataloader, and every column it does not need is
#: memory spent on all three splits for the whole run.
MANIFEST_COLUMNS: tuple[str, ...] = (
    "sample_id", "dataset", "split", "speaker_id", "gender",
    "source_emotion", "class_id", "sentiment_score",
    # Raw annotations pass through untouched; the ``_target`` columns are the
    # label adapter's output, already on the [-1, 1] scale the heads use. Both
    # are kept because collapsing them into one column is how a normalisation
    # gets applied twice -- once here and again when something downstream
    # re-derives targets from what it takes to be raw ratings.
    "valence", "arousal", "valence_target", "arousal_target",
    "audio_path", "video_path", "text",
    "has_audio", "has_video", "has_text",
    "duration", "segment_start", "segment_end",
    "feature_file", "feature_split", "feature_id",
)

#: Which column actually carries each modality's payload, per dataset.
#:
#: This is not cosmetic.  IEMOCAP's audio is a ``.wav`` on disk, so its payload
#: is a path; CMU-MOSEI ships no raw media at all -- its audio and video reach us
#: as rows inside ``Processed/aligned_50.pkl``, addressed by ``feature_id`` --
#: so checking for a populated ``audio_path`` there would report every one of
#: its 22,856 clips as a broken audio modality.  The standardizer already
#: records the distinction as ``audio_source='feature_container'``; this table is
#: the downstream half of it.
MODALITY_PAYLOADS: dict[str, dict[str, str]] = {
    "IEMOCAP": {"audio": "audio_path", "text": "text", "video": "video_path"},
    "CMU-MOSEI": {"audio": "feature_id", "text": "text", "video": "feature_id"},
}
DEFAULT_MODALITY_PAYLOADS = {"audio": "audio_path", "text": "text", "video": "video_path"}


class ManifestError(RuntimeError):
    """Raised when a manifest cannot be built safely."""


@dataclass
class ManifestConfig:
    """Everything that determines the contents of a manifest set."""

    experiment: str
    dataset: str
    label_protocol: str
    split_policy: str
    seed: int = 42
    #: IEMOCAP only.
    test_session: str = "Session5"
    validation_session: str = "Session4"
    source: str = str(STANDARDIZED_PATH)
    #: Set at build time; recorded so a manifest names the code that made it.
    git_revision: str | None = field(default=None)
    built_at: str | None = field(default=None)


EXPERIMENTS: dict[str, ManifestConfig] = {
    "iemocap_erc6": ManifestConfig(
        experiment="iemocap_erc6", dataset="IEMOCAP",
        label_protocol="iemocap_erc6", split_policy="iemocap_cross_session",
    ),
    "iemocap_ser4": ManifestConfig(
        experiment="iemocap_ser4", dataset="IEMOCAP",
        label_protocol="iemocap_ser4", split_policy="iemocap_cross_session",
    ),
    "mosei_sentiment": ManifestConfig(
        experiment="mosei_sentiment", dataset="CMU-MOSEI",
        label_protocol="mosei_sentiment", split_policy="official_standard_partition",
    ),
    "mosei_emotion6": ManifestConfig(
        experiment="mosei_emotion6", dataset="CMU-MOSEI",
        label_protocol="mosei_emotion6", split_policy="official_standard_partition",
    ),
}


def git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10,
        )
        return result.stdout.strip() or None if result.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


# ======================================================================
# IEMOCAP transcripts
# ======================================================================

def load_iemocap_transcripts(root: Path | None = None) -> dict[str, str]:
    """Utterance id to gold transcript, for every IEMOCAP dialog on disk."""
    from src.preprocessing.adapters.iemocap import parse_transcript_file

    root = Path(root) if root else DATASETS_DIR / "IEMOCAP"
    transcripts: dict[str, str] = {}
    for path in sorted(root.glob("Session*/dialog/transcriptions/*.txt")):
        transcripts.update(parse_transcript_file(path))
    return transcripts


def attach_iemocap_text(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Fill empty IEMOCAP transcripts from the dialog transcription files.

    Rows that already carry text keep it: once the metadata layer is
    regenerated with the transcript-aware adapter this function fills nothing,
    and the provenance it returns says so.
    """
    frame = frame.copy()
    existing = frame["text"].astype("string")
    already = existing.notna() & (existing.str.strip() != "")

    transcripts = load_iemocap_transcripts()
    if not transcripts:
        raise ManifestError(
            "No IEMOCAP transcription files found under "
            f"{DATASETS_DIR / 'IEMOCAP'}/Session*/dialog/transcriptions/.\n"
            "IEMOCAP's text modality cannot be built without them."
        )

    joined = frame["sample_id"].astype(str).map(transcripts).astype("string")
    frame["text"] = existing.where(already, joined)

    filled = frame["text"].astype("string")
    populated = filled.notna() & (filled.str.strip() != "")
    frame["has_text"] = populated

    return frame, {
        "already_in_metadata": int(already.sum()),
        "joined_from_transcription_files": int((populated & ~already).sum()),
        "still_missing": int((~populated).sum()),
        "transcript_files_read": len(transcripts),
        "note": (
            "joined_from_transcription_files drops to 0 once "
            "python -m src.preprocessing.generate_metadata is re-run with the "
            "transcript-aware IEMOCAP adapter"
        ),
    }


# ======================================================================
# Split policies
# ======================================================================

def iemocap_session_splits(frame: pd.DataFrame, config: ManifestConfig) -> pd.Series:
    """Assign whole IEMOCAP sessions to splits.

    ``evaluation_group`` is the standardizer's session column.  Nothing is
    shuffled and no seed is consumed: the partition is entirely determined by
    which session an utterance was recorded in, which is what makes it
    reproducible and speaker-disjoint at the same time.
    """
    sessions = frame["evaluation_group"].astype("string")
    known = {f"Session{index}" for index in range(1, 6)}
    unknown = set(sessions.dropna().unique()) - known
    if unknown:
        raise ManifestError(f"IEMOCAP rows carry unknown sessions: {sorted(unknown)}")
    if config.test_session == config.validation_session:
        raise ManifestError("IEMOCAP test and validation sessions must differ")

    assignment = pd.Series("train", index=frame.index, dtype="string")
    assignment[sessions == config.validation_session] = "validation"
    assignment[sessions == config.test_session] = "test"
    return assignment


def official_splits(frame: pd.DataFrame, config: ManifestConfig) -> pd.Series:
    """Carry the corpus's own train/validation/test partition through unchanged."""
    assignment = frame["training_split"].astype("string")
    unknown = set(assignment.dropna().unique()) - set(SPLITS)
    if unknown:
        raise ManifestError(
            f"{config.dataset} rows carry unknown official splits: {sorted(unknown)}"
        )
    if assignment.isna().any():
        raise ManifestError(
            f"{int(assignment.isna().sum())} {config.dataset} rows have no official split; "
            f"an official-split policy cannot invent one"
        )
    return assignment


SPLIT_POLICIES = {
    "iemocap_cross_session": iemocap_session_splits,
    "official_standard_partition": official_splits,
}


# ======================================================================
# Build
# ======================================================================

def speaker_identity(frame: pd.DataFrame, dataset: str) -> pd.Series:
    """A globally unique speaker id, or nulls when the corpus has none.

    IEMOCAP's ``speaker`` column holds only ``F`` or ``M`` -- the actor's role
    within a session, not an identity.  Two rows both reading ``F`` are the same
    person only if they share a session, so the identity is the pair.  Auditing
    the raw column instead would report speaker overlap between every split and
    be wrong in both directions.
    """
    if dataset == "IEMOCAP":
        return (
            frame["evaluation_group"].astype("string")
            + "_"
            + frame["speaker"].astype("string")
        )
    if "speaker" in frame.columns and frame["speaker"].notna().any():
        return frame["speaker"].astype("string")
    return pd.Series(pd.NA, index=frame.index, dtype="string")


def build(
    experiment: str,
    source: Path | None = None,
    output_root: Path | None = None,
    write: bool = True,
) -> tuple[dict[str, pd.DataFrame], AuditReport, dict]:
    """Build, audit and optionally write one experiment's manifest set."""
    if experiment not in EXPERIMENTS:
        raise ManifestError(
            f"Unknown experiment {experiment!r}; expected one of {sorted(EXPERIMENTS)}"
        )
    config = EXPERIMENTS[experiment]
    source = Path(source) if source else Path(config.source)
    if not source.exists():
        raise ManifestError(
            f"Standardized metadata not found at {source}. Build it first:\n"
            f"    python -m src.preprocessing.standardization.run_standardization"
        )
    output_root = Path(output_root) if output_root else MANIFEST_ROOT / experiment

    frame = pd.read_parquet(source)
    frame = frame[frame["dataset"] == config.dataset].reset_index(drop=True)
    if frame.empty:
        raise ManifestError(f"No {config.dataset} rows in {source}")
    total_records = len(frame)

    text_provenance = {}
    if config.dataset == "IEMOCAP":
        frame, text_provenance = attach_iemocap_text(frame)

    adapter = resolve_label_adapter(config.label_protocol)
    admitted = adapter.selects(frame).to_numpy(dtype=bool)
    frame = frame[admitted].reset_index(drop=True)
    if frame.empty:
        raise ManifestError(
            f"Protocol {config.label_protocol} admitted no {config.dataset} rows"
        )

    labels: LabelSet = adapter.build(frame)
    frame["class_id"] = labels.class_id if labels.class_id is not None else MISSING_CLASS_ID
    frame["source_emotion"] = frame.get("emotion", pd.Series(pd.NA, index=frame.index))
    frame["valence_target"] = labels.valence if labels.valence is not None else np.nan
    frame["arousal_target"] = labels.arousal if labels.arousal is not None else np.nan
    frame["speaker_id"] = speaker_identity(frame, config.dataset)

    assignment = SPLIT_POLICIES[config.split_policy](frame, config)
    frames = {
        split: frame[(assignment == split).to_numpy(dtype=bool)]
        .loc[:, [column for column in MANIFEST_COLUMNS if column in frame.columns]]
        .reset_index(drop=True)
        for split in SPLITS
    }
    empty = [split for split, part in frames.items() if part.empty]
    if empty:
        raise ManifestError(f"Split policy {config.split_policy} produced empty splits: {empty}")

    report = audit_manifests(
        experiment=experiment,
        frames=frames,
        class_names=adapter.label_space().classes,
        modalities=MODALITY_PAYLOADS.get(config.dataset, DEFAULT_MODALITY_PAYLOADS),
        speaker_check_required=config.dataset == "IEMOCAP",
        cache_root=Path("experiments/hsen") / experiment / "features",
    )

    summary = {
        "config": asdict(config) | {
            "git_revision": git_revision(),
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "source_records": int(total_records),
        "admitted_records": int(len(frame)),
        "excluded_by_protocol": int(total_records - len(frame)),
        "counts": {split: int(len(part)) for split, part in frames.items()},
        "label_space": adapter.label_space().to_dict(),
        "labels": labels.describe(),
        "text_provenance": text_provenance,
        "audit": report.to_dict(),
    }

    if write:
        output_root.mkdir(parents=True, exist_ok=True)
        for split, part in frames.items():
            part.to_parquet(output_root / f"{split}.parquet", index=False)
        (output_root / "manifest_summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )
        report.save(output_root / "audit.json")

    return frames, report, summary


def load(experiment: str, split: str, root: Path | None = None) -> pd.DataFrame:
    """Read one built manifest split."""
    root = Path(root) if root else MANIFEST_ROOT / experiment
    path = root / f"{split}.parquet"
    if not path.exists():
        raise ManifestError(
            f"No manifest at {path}. Build it first:\n"
            f"    python -m src.hsen.manifests --experiment {experiment}"
        )
    return pd.read_parquet(path)


def load_all(experiment: str, root: Path | None = None) -> dict[str, pd.DataFrame]:
    return {split: load(experiment, split, root) for split in SPLITS}


# ======================================================================
# CLI
# ======================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--experiment", required=True, choices=sorted(EXPERIMENTS))
    parser.add_argument("--source", default=None, help="Override the standardized metadata path.")
    parser.add_argument("--output-root", default=None)
    parser.add_argument(
        "--audit-only", action="store_true",
        help="Build in memory and print the audit without writing anything.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _, report, summary = build(
        experiment=args.experiment,
        source=args.source,
        output_root=args.output_root,
        write=not args.audit_only,
    )
    print(render(report))
    excluded = summary["excluded_by_protocol"]
    print(
        f"\nProtocol {summary['config']['label_protocol']} admitted "
        f"{summary['admitted_records']:,} of {summary['source_records']:,} records "
        f"({excluded:,} excluded)."
    )
    if summary["text_provenance"]:
        provenance = summary["text_provenance"]
        print(
            f"Transcripts: {provenance['already_in_metadata']:,} from metadata, "
            f"{provenance['joined_from_transcription_files']:,} joined from "
            f"transcription files, {provenance['still_missing']:,} missing."
        )
    if not args.audit_only:
        print(f"\nWritten to {MANIFEST_ROOT / args.experiment}")
    assert_clean(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())

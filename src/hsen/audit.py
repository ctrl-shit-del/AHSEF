"""The dataset audit that runs before any HSEN training starts.

Section 17 of the phase brief asks for five specific guarantees.  Each one is a
check here, each check returns a verdict, and :func:`assert_clean` refuses to
let training proceed on a failure rather than printing a warning nobody reads.

The checks are deliberately about *the manifests*, not about the code that wrote
them.  A leakage bug that produced a correct-looking builder and a contaminated
manifest would pass a unit test of the builder and fail here, which is the right
way round.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS: tuple[str, ...] = ("train", "validation", "test")


def split_pairs(present) -> list[tuple[str, str]]:
    """Every unordered pair of the splits present, in declared order.

    Ordered by position in :data:`SPLITS`, not alphabetically. A lexicographic
    comparison puts "test" before "train", so an overlap between the training
    and test partitions would be reported under the key ``test|train`` -- read
    right past by anyone scanning for ``train|test``, which is the exact failure
    this check exists to make impossible to miss.
    """
    order = [split for split in SPLITS if split in present]
    return [
        (order[i], order[j])
        for i in range(len(order))
        for j in range(i + 1, len(order))
    ]


class DatasetAuditError(RuntimeError):
    """Raised when a manifest set fails a leakage or integrity check."""


@dataclass
class Check:
    """One named verdict, with enough detail to act on a failure."""

    name: str
    passed: bool
    detail: str
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
            "evidence": self.evidence,
        }


@dataclass
class AuditReport:
    experiment: str
    checks: list[Check]
    counts: dict[str, int]
    class_counts: dict[str, dict[str, int]]
    modality_counts: dict[str, dict[str, int]]
    speakers: dict[str, list[str]]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def to_dict(self) -> dict:
        return {
            "experiment": self.experiment,
            "passed": self.passed,
            "counts": self.counts,
            "class_counts": self.class_counts,
            "modality_counts": self.modality_counts,
            "speakers": self.speakers,
            "checks": [check.to_dict() for check in self.checks],
        }

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


# ======================================================================
# Individual checks
# ======================================================================

def check_no_duplicate_ids(frames: dict[str, pd.DataFrame]) -> Check:
    offenders = {}
    for split, frame in frames.items():
        ids = frame["sample_id"].astype(str)
        duplicated = ids[ids.duplicated()].unique().tolist()
        if duplicated:
            offenders[split] = duplicated[:10]
    return Check(
        name="no_duplicate_sample_ids",
        passed=not offenders,
        detail="Every sample id appears at most once within its split"
        if not offenders else f"Duplicate ids within {sorted(offenders)}",
        evidence=offenders,
    )


def check_no_cross_split_ids(frames: dict[str, pd.DataFrame]) -> Check:
    sets = {split: set(frame["sample_id"].astype(str)) for split, frame in frames.items()}
    overlaps = {}
    for left, right in split_pairs(sets):
        shared = sets[left] & sets[right]
        if shared:
            overlaps[f"{left}|{right}"] = sorted(shared)[:10]
    return Check(
        name="no_cross_split_sample_ids",
        passed=not overlaps,
        detail="No sample id appears in more than one split"
        if not overlaps else f"Shared ids between {sorted(overlaps)}",
        evidence=overlaps,
    )


def check_speaker_independence(
    frames: dict[str, pd.DataFrame], column: str = "speaker_id", required: bool = True
) -> Check:
    """No speaker may appear in more than one split.

    ``required=False`` records the fact that a corpus does not identify its
    speakers instead of pretending the check passed.  CMU-MOSEI is that case:
    its official split is speaker-disjoint by construction, but the release does
    not ship speaker ids we could verify it with.
    """
    if not all(column in frame.columns for frame in frames.values()):
        return Check(
            name="speaker_independence",
            passed=not required,
            detail=f"No {column} column; speaker independence is asserted by the "
                   f"split protocol and cannot be verified from the manifest",
            evidence={"verifiable": False},
        )
    sets = {
        split: set(frame[column].dropna().astype(str)) for split, frame in frames.items()
    }
    if not any(sets.values()):
        # The column is present but empty for every row, which is not the same
        # thing as "no speaker appears twice". Reporting it as a pass would put
        # a green tick next to a check that never ran.
        return Check(
            name="speaker_independence",
            passed=not required,
            detail=f"{column} is present but unpopulated; speaker independence is "
                   f"asserted by the split protocol and cannot be verified from the manifest",
            evidence={"verifiable": False},
        )
    overlaps = {}
    for left, right in split_pairs(sets):
        shared = sets[left] & sets[right]
        if shared:
            overlaps[f"{left}|{right}"] = sorted(shared)
    return Check(
        name="speaker_independence",
        passed=not overlaps,
        detail="No speaker appears in more than one split"
        if not overlaps else f"Speaker overlap between {sorted(overlaps)}",
        evidence={"verifiable": True, "overlaps": overlaps,
                  "speakers_per_split": {k: sorted(v) for k, v in sets.items()}},
    )


def check_every_class_present(
    frames: dict[str, pd.DataFrame], class_names: tuple[str, ...], column: str = "class_id"
) -> Check:
    """Every protocol class must be represented in every split.

    A class absent from training gets a meaningless class weight; a class absent
    from validation makes its per-class F1 undefined and quietly drops it out of
    the macro average, which is exactly the way a weak class hides.
    """
    absent = {}
    for split, frame in frames.items():
        present = set(frame[column].astype(int).unique())
        missing = [name for index, name in enumerate(class_names) if index not in present]
        if missing:
            absent[split] = missing
    return Check(
        name="every_class_present_in_every_split",
        passed=not absent,
        detail="All protocol classes appear in train, validation and test"
        if not absent else f"Classes missing from a split: {absent}",
        evidence=absent,
    )


def check_no_unlabelled_rows(
    frames: dict[str, pd.DataFrame], column: str = "class_id", missing_value: int = -1
) -> Check:
    offenders = {
        split: int((frame[column].astype(int) == missing_value).sum())
        for split, frame in frames.items()
        if int((frame[column].astype(int) == missing_value).sum()) > 0
    }
    return Check(
        name="no_unlabelled_rows",
        passed=not offenders,
        detail="Every retained row carries a protocol class"
        if not offenders else f"Rows with no protocol class: {offenders}",
        evidence=offenders,
    )


def check_modality_payloads(frames: dict[str, pd.DataFrame], modalities: dict[str, str]) -> Check:
    """A declared modality must have a populated payload column.

    ``has_text=True`` with an empty transcript is worse than ``has_text=False``:
    the first trains a text branch on nothing while telling the ablation table
    that text was available.
    """
    offenders = {}
    for split, frame in frames.items():
        for modality, payload in modalities.items():
            flag = f"has_{modality}"
            if flag not in frame.columns or payload not in frame.columns:
                continue
            declared = frame[flag].fillna(False).astype(bool)
            empty = frame[payload].isna() | (frame[payload].astype("string").fillna("").str.strip() == "")
            broken = int((declared & empty).sum())
            if broken:
                offenders[f"{split}.{modality}"] = broken
    return Check(
        name="declared_modalities_have_payloads",
        passed=not offenders,
        detail="Every declared modality has a populated payload"
        if not offenders else f"Declared-but-empty modalities: {offenders}",
        evidence=offenders,
    )


def check_targets_in_range(
    frames: dict[str, pd.DataFrame],
    columns: tuple[str, ...] = ("valence_target", "arousal_target"),
    low: float = -1.0, high: float = 1.0,
) -> Check:
    """Regression targets must sit inside the range their transform declares.

    A value outside it means the raw annotation was not on the scale the
    transform assumed, which is a data problem worth stopping for and not a
    number to clip away.
    """
    offenders = {}
    for split, frame in frames.items():
        for column in columns:
            if column not in frame.columns:
                continue
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)
            finite = values[np.isfinite(values)]
            if finite.size and (finite.min() < low or finite.max() > high):
                offenders[f"{split}.{column}"] = [float(finite.min()), float(finite.max())]
    return Check(
        name="regression_targets_in_range",
        passed=not offenders,
        detail=f"All regression targets lie in [{low}, {high}]"
        if not offenders else f"Out-of-range targets: {offenders}",
        evidence=offenders,
    )


def check_feature_cache_split_isolation(
    frames: dict[str, pd.DataFrame], cache_root: Path | None
) -> Check:
    """No cached feature id may be claimed by two splits.

    The cache is written per split, so a sample that ended up in two shard
    indices is a sample two splits both believe they own -- the cheapest way to
    leak test data into training without any dataframe ever showing it.
    """
    if cache_root is None or not Path(cache_root).exists():
        return Check(
            name="feature_cache_split_isolation",
            passed=True,
            detail="No feature cache built yet; nothing to check",
            evidence={"cache_root": str(cache_root) if cache_root else None},
        )
    cached: dict[str, set[str]] = {}
    for split in SPLITS:
        index = Path(cache_root) / split / "index.json"
        if not index.exists():
            continue
        payload = json.loads(index.read_text(encoding="utf-8"))
        entries = payload.get("entries", payload)
        cached[split] = set(map(str, entries))
    overlaps = {}
    for left, right in split_pairs(cached):
        shared = cached[left] & cached[right]
        if shared:
            overlaps[f"{left}|{right}"] = sorted(shared)[:10]
    # A cached id that no manifest claims is stale, not leaked, but it means the
    # cache and the manifests have drifted apart and the run is not reproducible.
    declared = {split: set(frame["sample_id"].astype(str)) for split, frame in frames.items()}
    stray = {
        split: sorted(ids - declared.get(split, set()))[:10]
        for split, ids in cached.items()
        if ids - declared.get(split, set())
    }
    passed = not overlaps and not stray
    return Check(
        name="feature_cache_split_isolation",
        passed=passed,
        detail="Cached features are partitioned exactly as the manifests are"
        if passed else f"Cache overlap {sorted(overlaps)}, stray ids in {sorted(stray)}",
        evidence={"overlaps": overlaps, "stray": stray},
    )


# ======================================================================
# Driver
# ======================================================================

def audit_manifests(
    experiment: str,
    frames: dict[str, pd.DataFrame],
    class_names: tuple[str, ...],
    speaker_column: str = "speaker_id",
    speaker_check_required: bool = True,
    modalities: dict[str, str] | None = None,
    cache_root: Path | None = None,
) -> AuditReport:
    """Run every check over a manifest set and collect the verdicts."""
    modalities = modalities or {"audio": "audio_path", "text": "text", "video": "video_path"}
    checks = [
        check_no_duplicate_ids(frames),
        check_no_cross_split_ids(frames),
        check_speaker_independence(frames, speaker_column, speaker_check_required),
        check_no_unlabelled_rows(frames),
        check_every_class_present(frames, class_names),
        check_modality_payloads(frames, modalities),
        check_targets_in_range(frames),
        check_feature_cache_split_isolation(frames, cache_root),
    ]
    class_counts = {
        split: {
            name: int((frame["class_id"].astype(int) == index).sum())
            for index, name in enumerate(class_names)
        }
        for split, frame in frames.items()
    }
    modality_counts = {
        split: {
            modality: int(frame[f"has_{modality}"].fillna(False).astype(bool).sum())
            for modality in modalities
            if f"has_{modality}" in frame.columns
        }
        for split, frame in frames.items()
    }
    speakers = {
        split: sorted(frame[speaker_column].dropna().astype(str).unique())
        if speaker_column in frame.columns else []
        for split, frame in frames.items()
    }
    return AuditReport(
        experiment=experiment,
        checks=checks,
        counts={split: int(len(frame)) for split, frame in frames.items()},
        class_counts=class_counts,
        modality_counts=modality_counts,
        speakers=speakers,
    )


def render(report: AuditReport) -> str:
    """A human-readable audit, printed before every training run."""
    lines = [
        "=" * 74,
        f"DATASET AUDIT -- {report.experiment}",
        "=" * 74,
        "",
        "Split sizes",
    ]
    total = sum(report.counts.values())
    for split in SPLITS:
        if split in report.counts:
            count = report.counts[split]
            share = 100.0 * count / total if total else 0.0
            lines.append(f"  {split:<12} {count:>7,}  ({share:5.1f}%)")
    lines.append(f"  {'total':<12} {total:>7,}")

    lines.append("")
    lines.append("Class distribution (declared order, never sorted)")
    names = list(next(iter(report.class_counts.values())).keys()) if report.class_counts else []
    header = "  " + f"{'class':<14}" + "".join(f"{split:>12}" for split in SPLITS
                                               if split in report.class_counts)
    lines.append(header)
    for name in names:
        row = f"  {name:<14}"
        for split in SPLITS:
            if split in report.class_counts:
                row += f"{report.class_counts[split][name]:>12,}"
        lines.append(row)

    if report.modality_counts:
        lines.append("")
        lines.append("Modality availability")
        for split in SPLITS:
            if split in report.modality_counts:
                available = ", ".join(
                    f"{modality}={count:,}"
                    for modality, count in sorted(report.modality_counts[split].items())
                )
                lines.append(f"  {split:<12} {available}")

    if any(report.speakers.values()):
        lines.append("")
        lines.append("Speakers per split")
        for split in SPLITS:
            if report.speakers.get(split):
                lines.append(f"  {split:<12} {', '.join(report.speakers[split])}")

    lines.append("")
    lines.append("Leakage and integrity checks")
    for check in report.checks:
        mark = "PASS" if check.passed else "FAIL"
        lines.append(f"  [{mark}] {check.name}")
        lines.append(f"         {check.detail}")

    lines.append("")
    lines.append(f"AUDIT {'PASSED' if report.passed else 'FAILED'}")
    lines.append("=" * 74)
    return "\n".join(lines)


def assert_clean(report: AuditReport) -> AuditReport:
    """Refuse to continue on a failed audit."""
    if not report.passed:
        failures = [check.name for check in report.checks if not check.passed]
        raise DatasetAuditError(
            f"Dataset audit failed for {report.experiment}: {failures}.\n"
            f"Training is refused. Run the audit for details:\n"
            f"    python -m src.hsen.manifests --experiment {report.experiment} --audit-only"
        )
    return report

"""PHASE D/E -- fingerprints, the test lock, and the locked-artefact guard.

Three separate jobs live here, and they are separate because they fail in
different ways.

**Replay.**  Phase D/E must be re-runnable offline from stored predictions.  That
is only true if the record says exactly which stored predictions, which pool,
which features, which target, which estimator and which seed produced it, so
:func:`reproducibility_record` collects all of them into one block and hashes
the pieces that a silent edit could change.

**The test lock.**  ``test_partition_opened`` is not a comment; it is checked.
:func:`assert_validation_only` refuses any path whose name says ``test``, so the
lock is enforced where files are opened rather than asserted in prose at the end
of a report.  A guard that only prints ``false`` is a decoration.

**The locked artefacts.**  Stage 1, Stage 2, Stage 3 and audio_strong are inputs
to this phase and must come out of it byte-identical.  :func:`locked_artefact_digest`
hashes them, and the first run writes the digest as a baseline that every later
run is compared against.  This catches the failure that matters -- a helper that
"refreshes" a frozen export while loading it -- which no amount of care in the
driver can rule out by inspection.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Mapping, Sequence

#: Version of the Phase D/E protocol itself.  Bumping it is how a reader knows
#: that two records were produced by different procedures rather than by the
#: same procedure on different data.
PROTOCOL_VERSION = "ahsef.milestone.phase_de.v1"

#: Directories whose contents this phase reads and must not change.
LOCKED_ARTEFACT_ROOTS: tuple[Path, ...] = (
    Path("experiments") / "ahsef" / "stage1",
    Path("experiments") / "ahsef" / "stage2_llm",
    Path("experiments") / "ahsef" / "stage3_text_audio",
    Path("experiments") / "audio_strong",
)

#: Files whose names mark them as belonging to the locked evaluation partition.
TEST_MARKERS = ("__test.", "_test.", "test_predictions", "oracle_test", "decisions_test",
                "traces_test")


class EvaluationLockError(RuntimeError):
    """Raised when the locked test partition would be read or written."""


class LockedArtefactError(RuntimeError):
    """Raised when a frozen Stage 1/2/3 or audio_strong artefact has changed."""


def assert_validation_only(paths: Iterable[Path | str]) -> list[str]:
    """Refuse to open anything belonging to the locked test partition.

    Matching is on the file name rather than on a caller-supplied flag, because
    the mistake this prevents is a caller passing the wrong split, and a flag
    the caller sets cannot catch a caller that is wrong.
    """
    offending = []
    for item in paths:
        name = Path(item).name.lower()
        if any(marker in name for marker in TEST_MARKERS):
            offending.append(str(item))
    if offending:
        raise EvaluationLockError(
            f"Phase D/E is validation-only and the locked test partition stays shut "
            f"until the routing policy is frozen, but these paths belong to it: "
            f"{offending}"
        )
    return [str(item) for item in paths]


def file_digest(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def locked_artefact_digest(
    roots: Sequence[Path] = LOCKED_ARTEFACT_ROOTS,
) -> dict:
    """One content hash per locked directory, plus the file count behind it.

    Hashing the sorted ``(relative path, content hash)`` pairs means a renamed
    file, a deleted file and an edited file all move the digest, which a simple
    count or a directory mtime would not.
    """
    record = {}
    for root in roots:
        root = Path(root)
        if not root.exists():
            record[str(root)] = {"present": False}
            continue
        entries = []
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            entries.append((path.relative_to(root).as_posix(), file_digest(path)))
        combined = hashlib.sha256(
            json.dumps(entries, sort_keys=True).encode("utf-8")
        ).hexdigest()
        record[str(root)] = {
            "present": True,
            "files": len(entries),
            "sha256": combined,
        }
    return record


def verify_locked_artefacts(
    baseline_path: Path | str,
    roots: Sequence[Path] = LOCKED_ARTEFACT_ROOTS,
) -> dict:
    """Compare the locked directories against the stored baseline.

    The first run has nothing to compare against and says so rather than
    silently reporting success: an unverifiable check that prints "verified" is
    worse than one that admits it is establishing a baseline.
    """
    baseline_path = Path(baseline_path)
    current = locked_artefact_digest(roots)
    if not baseline_path.exists():
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(
            json.dumps(
                {
                    "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "protocol_version": PROTOCOL_VERSION,
                    "digests": current,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return {
            "verified": None,
            "baseline_established": True,
            "baseline_path": str(baseline_path),
            "digests": current,
            "note": (
                "No prior baseline existed, so this run recorded one. It establishes "
                "the reference; it does not prove that nothing changed before it."
            ),
        }

    stored = json.loads(baseline_path.read_text(encoding="utf-8")).get("digests") or {}
    changed = [
        name for name, entry in current.items()
        if (stored.get(name) or {}).get("sha256") != entry.get("sha256")
    ]
    return {
        "verified": not changed,
        "baseline_established": False,
        "baseline_path": str(baseline_path),
        "changed_roots": changed,
        "digests": current,
        "baseline_digests": stored,
        "note": (
            "Every locked Stage 1/2/3 and audio_strong artefact hashes identically "
            "to the recorded baseline."
            if not changed else
            f"CHANGED since the baseline: {changed}. Phase D/E must not modify these; "
            f"investigate before trusting any result in this record."
        ),
    }


def assert_locked_artefacts_unchanged(verification: Mapping) -> None:
    if verification.get("verified") is False:
        raise LockedArtefactError(
            f"Locked artefacts changed during Phase D/E: "
            f"{verification.get('changed_roots')}"
        )


# ============================================================
# Fingerprints
# ============================================================

def sequence_fingerprint(values: Sequence[str]) -> str:
    """Order-independent hash of a sample-id pool.

    Sorted before hashing so that two runs which enumerate the same pool in a
    different order agree, and so that a *different* pool cannot be disguised by
    reordering.
    """
    payload = json.dumps(sorted(str(value) for value in values), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def array_fingerprint(values) -> str:
    import numpy as np

    array = np.asarray(values, dtype=float)
    return hashlib.sha256(
        np.ascontiguousarray(array).tobytes() + str(array.shape).encode("utf-8")
    ).hexdigest()


def prediction_fingerprint(prediction_set) -> dict:
    """Identity of one stored prediction export, enough to replay against it."""
    frame = prediction_set.frame
    return {
        "modality": prediction_set.modality,
        "split": prediction_set.split,
        "samples": int(len(frame)),
        "pool_sha256": sequence_fingerprint(frame["sample_id"].astype(str).tolist()),
        "predicted_class_sha256": array_fingerprint(
            frame["predicted_class"].to_numpy()
        ),
        "true_class_sha256": array_fingerprint(frame["true_class"].to_numpy()),
        "experiment": prediction_set.meta.get("experiment"),
        "iteration": prediction_set.meta.get("iteration"),
        "checkpoint_sha256": prediction_set.meta.get("checkpoint_sha256"),
        "model": (prediction_set.meta.get("model") or {}).get("class"),
    }


def git_revision() -> dict:
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args], capture_output=True, text=True, timeout=10, check=True
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    commit = run("rev-parse", "HEAD")
    if commit is None:
        return {"available": False}
    status = run("status", "--porcelain")
    return {
        "available": True,
        "commit": commit,
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
    }


def environment_record() -> dict:
    import numpy as np
    import pandas as pd

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


def reproducibility_record(
    pool: Sequence[str],
    predictions: Mapping[str, object],
    alignment: Mapping,
    feature_configuration: Mapping,
    estimator_configuration: Mapping,
    target_fingerprints: Mapping[str, str],
    budget_grid: Sequence[float],
    cost_configuration: Mapping,
    seed: int,
    sources: Mapping[str, str],
) -> dict:
    """Everything needed to replay Phase D/E offline from stored predictions."""
    return {
        "protocol_version": PROTOCOL_VERSION,
        "seed": int(seed),
        "dataset_fingerprint": {
            "validation_pool_sha256": sequence_fingerprint(pool),
            "validation_pool_size": len(pool),
            "datasets": sorted({
                str(item).split("_", 1)[0] for item in pool
            }),
        },
        "alignment_fingerprint": dict(alignment),
        "model_fingerprints": {
            name: prediction_fingerprint(item) for name, item in predictions.items()
        },
        "feature_configuration": dict(feature_configuration),
        "estimator_configuration": dict(estimator_configuration),
        "target_fingerprints": dict(target_fingerprints),
        "budget_grid": [float(value) for value in budget_grid],
        "cost_configuration": dict(cost_configuration),
        "sources": dict(sources),
        "git": git_revision(),
        "environment": environment_record(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "replay": (
            "Every input is a stored artefact listed under 'sources'. Re-running "
            "the Phase D/E driver with the same seed reproduces every number in "
            "this record without re-scoring a model or calling the LLM."
        ),
    }

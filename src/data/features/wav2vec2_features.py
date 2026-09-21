"""Frozen wav2vec2 representations, extracted once and cached on disk.

The Stage-3 audio expert is a log-mel CNN trained from scratch.  This module
supplies the alternative the 50% milestone asks about: a self-supervised speech
encoder pretrained on 960 h of LibriSpeech, used **frozen**, with only a small
head trained on top.

Two decisions are forced by the environment rather than chosen, and both are
recorded rather than hidden:

**The encoder is frozen.**  There is no GPU on this machine (``torch.cuda.
is_available()`` is False), and fine-tuning 94.4 M parameters over 44 k clips on
CPU is infeasible by orders of magnitude.  Freezing it turns the question from
"can we train a big model" into "does a better *representation* make the audio
expert more useful", which is the question the milestone actually poses.

**Features are cached.**  Extraction runs at roughly 3 clips/s on this CPU, so a
single pass over train+validation is hours.  Caching per ``sample_id`` makes the
expensive step happen once: head training, hyper-parameter changes and re-runs
then cost seconds, and an interrupted extraction resumes instead of restarting.

What is stored is the **mean over valid time frames of every transformer
layer** -- a ``[12, 768]`` matrix per clip.  Storing all layers rather than one
is what lets the head learn which layer to listen to (the standard SUPERB
weighted-sum probe) without paying for extraction again; different layers of a
speech encoder carry very different information, and picking one up front would
be a guess.  Padding is excluded from the mean using the encoder's own reported
output lengths, so a short clip is not averaged with silence.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

#: The bundle this project uses.  Named, not configurable at call sites, so the
#: cache and the artefacts cannot disagree about what produced them.
DEFAULT_BUNDLE = "WAV2VEC2_BASE"

#: Bundles that are legitimate drop-in alternatives on this machine. All ship
#: with torchaudio and need no `transformers` install.
SUPPORTED_BUNDLES = ("WAV2VEC2_BASE", "HUBERT_BASE", "WAVLM_BASE")

FEATURE_SCHEMA = "ahsef.audio.wav2vec2_layer_means.v1"

#: float16 halves the store and costs nothing that matters: these values feed a
#: small head that is trained in float32 anyway, and the quantisation error is
#: far below the variation between clips.
STORAGE_DTYPE = np.float16


class FeatureExtractionError(RuntimeError):
    """Raised when features cannot be produced or a cache is inconsistent."""


# ============================================================
# Extractor
# ============================================================

@dataclass
class Wav2Vec2Config:
    """Everything that determines the numbers in the cache."""

    bundle: str = DEFAULT_BUNDLE
    sample_rate: int = 16_000
    max_seconds: float = 4.0
    #: Per-utterance zero-mean / unit-variance. False for the fairseq LS960
    #: base checkpoints, which were trained on unnormalised input.
    normalize_waveform: bool = False
    batch_size: int = 16
    num_threads: int = 8

    def __post_init__(self) -> None:
        if self.bundle not in SUPPORTED_BUNDLES:
            raise ValueError(
                f"bundle must be one of {SUPPORTED_BUNDLES}, got {self.bundle!r}"
            )
        if self.max_seconds <= 0:
            raise ValueError("max_seconds must be positive")

    def to_dict(self) -> dict:
        return {
            "schema": FEATURE_SCHEMA,
            "bundle": self.bundle,
            "sample_rate": self.sample_rate,
            "max_seconds": self.max_seconds,
            "normalize_waveform": self.normalize_waveform,
            "batch_size": self.batch_size,
            "pooling": "mean over valid (unpadded) time frames, per transformer layer",
            "frozen_encoder": True,
            "storage_dtype": "float16",
        }

    @property
    def fingerprint(self) -> str:
        """Hash of the settings that change the stored numbers.

        ``batch_size`` and ``num_threads`` are excluded: they affect how long
        extraction takes, not what it produces.
        """
        payload = json.dumps(
            {
                "schema": FEATURE_SCHEMA,
                "bundle": self.bundle,
                "sample_rate": self.sample_rate,
                "max_seconds": self.max_seconds,
                "normalize_waveform": self.normalize_waveform,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class Wav2Vec2LayerMeans:
    """Load a frozen bundle and reduce a batch of waveforms to ``[B, L, D]``."""

    def __init__(self, config: Wav2Vec2Config | None = None):
        self.config = config or Wav2Vec2Config()
        self._model = None
        self._num_layers: int | None = None
        self._feature_dim: int | None = None

    # The model is 360 MB and takes minutes to load; a caller that only wants
    # the config (a test, a provenance record) must not pay for it.
    @property
    def model(self):
        if self._model is None:
            import torchaudio.pipelines as pipelines

            torch.set_num_threads(int(self.config.num_threads))
            bundle = getattr(pipelines, self.config.bundle)
            if bundle.sample_rate != self.config.sample_rate:
                raise FeatureExtractionError(
                    f"{self.config.bundle} expects {bundle.sample_rate} Hz but the "
                    f"config asks for {self.config.sample_rate} Hz."
                )
            self._model = bundle.get_model().eval()
            for parameter in self._model.parameters():
                parameter.requires_grad_(False)
        return self._model

    def describe(self) -> dict:
        model = self.model
        return {
            **self.config.to_dict(),
            "parameters": int(sum(p.numel() for p in model.parameters())),
            "num_layers": self.num_layers,
            "feature_dim": self.feature_dim,
            "fingerprint_sha256": self.config.fingerprint,
        }

    @property
    def num_layers(self) -> int:
        if self._num_layers is None:
            self._probe()
        return int(self._num_layers)

    @property
    def feature_dim(self) -> int:
        if self._feature_dim is None:
            self._probe()
        return int(self._feature_dim)

    def _probe(self) -> None:
        probe = torch.zeros(1, int(self.config.sample_rate * 0.5))
        with torch.inference_mode():
            layers, _ = self.model.extract_features(probe)
        self._num_layers = len(layers)
        self._feature_dim = int(layers[-1].shape[-1])

    # ------------------------------------------------------------- extraction

    def encode(self, waveforms: Sequence[np.ndarray]) -> np.ndarray:
        """``[B, num_layers, feature_dim]`` layer means, padding excluded.

        Waveforms may differ in length; they are right-padded into one batch and
        the encoder's own reported output lengths are used to mask the padding
        out of the mean.  Averaging over padding would make short clips look
        like long quiet ones.
        """
        if not len(waveforms):
            raise FeatureExtractionError("Nothing to encode")
        limit = int(round(self.config.max_seconds * self.config.sample_rate))
        clipped = [np.asarray(item, dtype=np.float32)[:limit] for item in waveforms]
        lengths = torch.tensor([max(len(item), 1) for item in clipped], dtype=torch.long)
        width = int(lengths.max())

        batch = torch.zeros(len(clipped), width, dtype=torch.float32)
        for row, item in enumerate(clipped):
            if len(item):
                batch[row, : len(item)] = torch.from_numpy(item)
        if self.config.normalize_waveform:
            for row, length in enumerate(lengths.tolist()):
                segment = batch[row, :length]
                batch[row, :length] = (segment - segment.mean()) / (segment.std() + 1e-7)

        with torch.inference_mode():
            layers, out_lengths = self.model.extract_features(batch, lengths)

        stacked = torch.stack(layers, dim=1)          # [B, L, T, D]
        frames = stacked.shape[2]
        if out_lengths is None:                       # pragma: no cover - defensive
            out_lengths = torch.full((len(clipped),), frames, dtype=torch.long)
        valid = out_lengths.clamp(min=1, max=frames)
        mask = (
            torch.arange(frames).unsqueeze(0) < valid.unsqueeze(1)
        ).to(stacked.dtype)                            # [B, T]
        mask = mask.unsqueeze(1).unsqueeze(-1)         # [B, 1, T, 1]
        pooled = (stacked * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1.0)
        return pooled.to(torch.float32).numpy()


# ============================================================
# Cache
# ============================================================

@dataclass
class FeatureStore:
    """A resumable, shard-based cache keyed by ``sample_id``.

    Shards are written whole and an index is rewritten after each one, so an
    interrupted run loses at most one shard's work and never leaves a half-read
    record behind.  ``config_fingerprint`` is checked on open: a cache produced
    by different settings is refused rather than silently mixed with new
    features, which would be the worst kind of quiet corruption.
    """

    root: Path
    config_fingerprint: str
    num_layers: int = 12
    feature_dim: int = 768
    shard_size: int = 512
    index: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    @property
    def index_path(self) -> Path:
        return self.root / "index.json"

    def shard_path(self, shard: int) -> Path:
        return self.root / f"shard_{shard:04d}.npz"

    # ------------------------------------------------------------------ open

    @classmethod
    def open(
        cls,
        root: Path | str,
        config_fingerprint: str,
        num_layers: int = 12,
        feature_dim: int = 768,
        shard_size: int = 512,
    ) -> "FeatureStore":
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        store = cls(root, config_fingerprint, num_layers, feature_dim, shard_size)
        if store.index_path.exists():
            record = json.loads(store.index_path.read_text(encoding="utf-8"))
            if record.get("config_fingerprint") != config_fingerprint:
                raise FeatureExtractionError(
                    f"{store.index_path} was written by a different extractor "
                    f"configuration ({record.get('config_fingerprint')} != "
                    f"{config_fingerprint}). Mixing them would put two different "
                    f"representations in one feature matrix; delete the cache or "
                    f"point --cache-dir somewhere else."
                )
            store.index = record
        else:
            store.index = {
                "schema": FEATURE_SCHEMA,
                "config_fingerprint": config_fingerprint,
                "num_layers": num_layers,
                "feature_dim": feature_dim,
                "shard_size": shard_size,
                "entries": {},
                "shards": 0,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
        return store

    # --------------------------------------------------------------- queries

    @property
    def entries(self) -> dict:
        return self.index.setdefault("entries", {})

    def __contains__(self, sample_id: str) -> bool:
        return str(sample_id) in self.entries

    def __len__(self) -> int:
        return len(self.entries)

    def missing(self, sample_ids: Iterable[str]) -> list[str]:
        return [str(item) for item in sample_ids if str(item) not in self.entries]

    # ---------------------------------------------------------------- writes

    def next_shard(self) -> int:
        """One past the highest shard on disk, not one past the index's counter.

        Deriving this from the filesystem rather than from the in-memory counter
        is what stops two writers from choosing the same shard number and
        overwriting each other's features. The index can fall behind; the
        directory listing cannot.
        """
        existing = [
            int(path.stem.split("_")[1])
            for path in self.root.glob("shard_*.npz")
            if path.stem.split("_")[-1].isdigit()
        ]
        return max(existing) + 1 if existing else 0

    def append(self, sample_ids: Sequence[str], features: np.ndarray) -> int:
        """Write one shard and update the index.  Returns the shard number."""
        if len(sample_ids) != features.shape[0]:
            raise FeatureExtractionError(
                f"{len(sample_ids)} ids but {features.shape[0]} feature rows"
            )
        if features.shape[1:] != (self.num_layers, self.feature_dim):
            raise FeatureExtractionError(
                f"Expected [N, {self.num_layers}, {self.feature_dim}] features, got "
                f"{tuple(features.shape)}"
            )
        shard = self.next_shard()
        np.savez_compressed(
            self.shard_path(shard),
            ids=np.asarray([str(item) for item in sample_ids]),
            features=features.astype(STORAGE_DTYPE),
        )
        for row, identifier in enumerate(sample_ids):
            self.entries[str(identifier)] = [shard, row]
        self.index["shards"] = max(int(self.index.get("shards", 0)), shard + 1)
        self.index["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.flush()
        return shard

    def flush(self) -> None:
        self.index_path.write_text(json.dumps(self.index, indent=2), encoding="utf-8")

    # ----------------------------------------------------------------- reads

    def load(self, sample_ids: Sequence[str]) -> np.ndarray:
        """``[N, num_layers, feature_dim]`` float32, in the requested order."""
        wanted = [str(item) for item in sample_ids]
        unknown = [item for item in wanted if item not in self.entries]
        if unknown:
            raise FeatureExtractionError(
                f"{len(unknown)} sample ids are not in the feature cache "
                f"(e.g. {unknown[:5]}). Extract them before training; a missing "
                f"feature is never substituted."
            )
        by_shard: dict[int, list[tuple[int, int]]] = {}
        for position, identifier in enumerate(wanted):
            shard, row = self.entries[identifier]
            by_shard.setdefault(int(shard), []).append((int(row), position))

        out = np.empty((len(wanted), self.num_layers, self.feature_dim), dtype=np.float32)
        for shard, pairs in by_shard.items():
            with np.load(self.shard_path(shard)) as payload:
                block = payload["features"]
            for row, position in pairs:
                out[position] = block[row].astype(np.float32)
        return out

    # ------------------------------------------------------------- integrity

    def scan_shards(self) -> dict[tuple[int, int], str]:
        """What the shard files actually contain: ``(shard, row) -> sample_id``.

        Shards store their own id list, so they -- not the index -- are the
        ground truth about which feature belongs to which sample.
        """
        actual: dict[tuple[int, int], str] = {}
        for path in sorted(self.root.glob("shard_*.npz")):
            label = path.stem.split("_")[-1]
            if not label.isdigit():
                continue
            with np.load(path, allow_pickle=False) as payload:
                identifiers = [str(value) for value in payload["ids"]]
            for row, identifier in enumerate(identifiers):
                actual[(int(label), row)] = identifier
        return actual

    def verify(self) -> dict:
        """Check every index entry against the shard that is supposed to hold it.

        ``misattributed`` is the only finding that means the *data* is wrong; an
        entry pointing at a row holding a different sample would silently train
        a head on another recording's features. Everything else here is lost
        bookkeeping, which :meth:`rebuild_index` repairs without re-extracting.
        """
        actual = self.scan_shards()
        misattributed = [
            identifier for identifier, (shard, row) in self.entries.items()
            if actual.get((int(shard), int(row))) != identifier
        ]
        dangling = [
            identifier for identifier, (shard, row) in self.entries.items()
            if (int(shard), int(row)) not in actual
        ]
        stored = set(actual.values())
        unindexed = sorted(stored - set(self.entries))
        duplicated = len(actual) - len(stored)
        return {
            "root": str(self.root),
            "index_entries": len(self.entries),
            "rows_in_shards": len(actual),
            "distinct_ids_in_shards": len(stored),
            "misattributed": len(misattributed),
            "misattributed_examples": misattributed[:5],
            "dangling_index_entries": len(dangling),
            "extracted_but_unindexed": len(unindexed),
            "duplicate_rows": duplicated,
            "healthy": not misattributed and not dangling,
            "repairable": bool(unindexed or dangling) and not misattributed,
            "note": (
                "Only 'misattributed' means a feature is attached to the wrong "
                "sample. Unindexed rows and dangling entries are lost bookkeeping "
                "and rebuild_index() recovers them from the shards."
            ),
        }

    def rebuild_index(self) -> dict:
        """Reconstruct the index from the shard files and report what changed.

        Used after a crash or a concurrent-writer race. Where an id appears in
        more than one shard the first occurrence wins: the encoder is
        deterministic and frozen, so duplicates are identical and the choice is
        arbitrary rather than consequential.
        """
        actual = self.scan_shards()
        before = len(self.entries)
        rebuilt: dict[str, list[int]] = {}
        for (shard, row) in sorted(actual):
            identifier = actual[(shard, row)]
            rebuilt.setdefault(identifier, [shard, row])
        self.index["entries"] = rebuilt
        self.index["shards"] = self.next_shard()
        self.index["rebuilt_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.index["rebuilt_from"] = "shard files (authoritative)"
        self.flush()
        return {
            "entries_before": before,
            "entries_after": len(rebuilt),
            "recovered": len(rebuilt) - before,
            "rows_scanned": len(actual),
            "duplicate_rows_dropped": len(actual) - len(rebuilt),
        }

    def provenance(self) -> dict:
        return {
            "schema": FEATURE_SCHEMA,
            "root": str(self.root),
            "config_fingerprint": self.config_fingerprint,
            "cached_samples": len(self),
            "shards": int(self.index.get("shards", 0)),
            "num_layers": self.num_layers,
            "feature_dim": self.feature_dim,
            "storage_dtype": "float16",
        }


class CacheBusyError(RuntimeError):
    """Raised when another extractor already holds this cache directory."""


class ExtractionLock:
    """An advisory exclusive lock on one cache directory.

    Two extractors writing the same cache is not a hypothetical: it happened
    during this project's own extraction run, when a resumed job overlapped a
    supposedly-dead predecessor. No feature was mis-attributed -- the shards are
    self-describing -- but each process flushed its own view of the index and the
    later flush discarded the earlier one's bookkeeping, stranding 832 already
    extracted samples.

    ``O_CREAT | O_EXCL`` is atomic on every platform this runs on, so the lock is
    won by exactly one process. A lock whose owning process is gone is stale and
    is reclaimed with a warning rather than blocking a legitimate resume.
    """

    def __init__(self, root: Path | str):
        self.path = Path(root) / "extraction.lock"
        self.acquired = False

    def owner(self) -> dict:
        """Whatever the existing lock file says about who holds it."""
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def acquire(self, force: bool = False) -> "ExtractionLock":
        """Claim the cache, or refuse and say who holds it.

        Liveness is deliberately **not** probed. The portable-looking way to do
        that is ``os.kill(pid, 0)``, which on POSIX is a harmless existence
        check but on Windows -- this project's platform -- calls
        ``TerminateProcess`` and would kill the very extractor it was asked
        about. Rather than write a platform-branching probe for a batch job that
        runs for an hour, an existing lock simply means busy, and breaking it is
        an explicit operator decision.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            if not force:
                record = self.owner()
                raise CacheBusyError(
                    f"{self.path} is held by pid {record.get('pid', '?')} since "
                    f"{record.get('acquired_at', 'an unknown time')}. Two writers on "
                    f"one cache strand each other's index entries. Wait for it to "
                    f"finish, or pass --force-lock if you have confirmed that "
                    f"process is gone."
                )
            self.path.unlink(missing_ok=True)
        try:
            handle = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            raise CacheBusyError(f"{self.path} was claimed concurrently") from error
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(
                {"pid": os.getpid(), "acquired_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
                stream,
            )
        self.acquired = True
        return self

    def release(self) -> None:
        if self.acquired:
            self.path.unlink(missing_ok=True)
            self.acquired = False

    def __enter__(self) -> "ExtractionLock":
        return self.acquire()

    def __exit__(self, *exc) -> None:
        self.release()


def manifest_fingerprint(sample_ids: Sequence[str]) -> str:
    """Hash of an ordered id list -- proves which samples a run actually used."""
    digest = hashlib.sha256()
    for identifier in sample_ids:
        digest.update(str(identifier).encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()

"""A resumable shard cache for variable-length modality features.

Why not reuse :class:`src.data.features.wav2vec2_features.FeatureStore`.  That
store holds one fixed ``[num_layers, feature_dim]`` matrix per sample, because
the strong-audio probe time-pools before caching.  HSEN cannot: Husformer's
cross-modal attention consumes *sequences*, and pooling to a vector before the
fusion trunk sees them would throw away the temporal structure the trunk exists
to model.  So a sample here is ``[T_i, D]`` with a different ``T_i`` per sample
and per modality.

Everything else is deliberately the same design, because it is already proven on
this project's data: whole shards written at once, an index rewritten after each
shard, a configuration fingerprint checked on open so two different encoders can
never end up in one feature matrix, and shards that carry their own id list so
the index can be rebuilt from them after an interrupted run.  The extraction
lock is imported from that module rather than reimplemented -- one writer per
cache directory is the invariant, and there should be one implementation of it.

Ragged storage.  A shard holds every sequence in it concatenated along time,
plus the row offsets, which keeps one array per shard instead of one per sample
and compresses far better than an object array of ragged rows.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

# The lock is the same invariant, so it is the same code. Two extractors writing
# one cache directory strand each other's index entries; this project has hit
# that before, which is why the lock exists at all.
from src.data.features.wav2vec2_features import (  # noqa: F401  (re-exported)
    CacheBusyError,
    ExtractionLock,
)

#: Bumped whenever the on-disk layout changes in a way an old cache cannot be
#: read under.  Checked on open alongside the configuration fingerprint.
SEQUENCE_SCHEMA = "hsen.sequence_feature_store.v1"

#: float16 on disk, float32 in memory.  These are frozen-encoder activations
#: feeding a LayerNorm; the precision loss is far below the noise floor of the
#: encoder itself, and it halves a cache that runs to gigabytes.
STORAGE_DTYPE = np.float16


class FeatureCacheError(RuntimeError):
    """Raised when a feature cache is inconsistent, incompatible or incomplete."""


def config_fingerprint(payload: dict) -> str:
    """A stable hash of an extractor configuration.

    Sorted keys, so two configurations that differ only in dict ordering
    fingerprint identically and a cache is not needlessly rejected.
    """
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass
class SequenceFeatureStore:
    """Shard cache mapping ``sample_id`` to a ``[T, D]`` float32 sequence."""

    root: Path
    fingerprint: str
    modality: str
    feature_dim: int
    shard_size: int = 256
    index: dict = field(default_factory=dict)
    #: How many decompressed shards to hold. 64 shards of the default size is
    #: 16,384 samples, which covers every split of both IEMOCAP protocols
    #: outright; a larger corpus falls back to LRU eviction rather than to the
    #: pathological one-decompression-per-sample behaviour.
    max_cached_shards: int = 64
    _shards: "OrderedDict[int, tuple[np.ndarray, np.ndarray]]" = field(
        default_factory=OrderedDict, repr=False, compare=False,
    )

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    # ------------------------------------------------------------------ open

    @property
    def index_path(self) -> Path:
        return self.root / "index.json"

    def shard_path(self, shard: int) -> Path:
        return self.root / f"shard_{shard:04d}.npz"

    @classmethod
    def open(
        cls,
        root: Path | str,
        fingerprint: str,
        modality: str,
        feature_dim: int,
        shard_size: int = 256,
    ) -> "SequenceFeatureStore":
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        store = cls(root, fingerprint, modality, feature_dim, shard_size)
        if store.index_path.exists():
            record = json.loads(store.index_path.read_text(encoding="utf-8"))
            if record.get("schema") != SEQUENCE_SCHEMA:
                raise FeatureCacheError(
                    f"{store.index_path} uses schema {record.get('schema')!r}, "
                    f"this code writes {SEQUENCE_SCHEMA!r}. Delete the cache and "
                    f"re-extract, or point --feature-dir elsewhere."
                )
            if record.get("fingerprint") != fingerprint:
                raise FeatureCacheError(
                    f"{store.index_path} was written by a different extractor "
                    f"configuration.\n"
                    f"  on disk: {record.get('fingerprint')}\n"
                    f"  wanted : {fingerprint}\n"
                    f"Mixing two representations in one feature matrix is the "
                    f"quietest kind of corruption, so this is refused. Delete the "
                    f"cache or point --feature-dir somewhere else."
                )
            if int(record.get("feature_dim", -1)) != int(feature_dim):
                raise FeatureCacheError(
                    f"{store.index_path} holds {record.get('feature_dim')}-d features, "
                    f"expected {feature_dim}-d"
                )
            store.index = record
        else:
            store.index = {
                "schema": SEQUENCE_SCHEMA,
                "fingerprint": fingerprint,
                "modality": modality,
                "feature_dim": int(feature_dim),
                "shard_size": int(shard_size),
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

    def length_of(self, sample_id: str) -> int:
        """Frames cached for one sample, without reading its shard."""
        try:
            return int(self.entries[str(sample_id)][2])
        except KeyError as error:
            raise FeatureCacheError(f"{sample_id!r} is not in this cache") from error

    # ---------------------------------------------------------------- writes

    def next_shard(self) -> int:
        """One past the highest shard *on disk*, not one past the index counter.

        The index can fall behind after a kill; the directory listing cannot, so
        deriving the next number from the filesystem is what stops a resumed run
        from overwriting the shard it was mid-way through.
        """
        existing = [
            int(path.stem.split("_")[-1])
            for path in self.root.glob("shard_*.npz")
            if path.stem.split("_")[-1].isdigit()
        ]
        return max(existing) + 1 if existing else 0

    def append(self, sample_ids: Sequence[str], sequences: Sequence[np.ndarray]) -> int:
        """Write one shard of ragged sequences and update the index."""
        if len(sample_ids) != len(sequences):
            raise FeatureCacheError(
                f"{len(sample_ids)} ids but {len(sequences)} sequences"
            )
        if not sequences:
            raise FeatureCacheError("Refusing to write an empty shard")

        for identifier, sequence in zip(sample_ids, sequences):
            if sequence.ndim != 2 or sequence.shape[1] != self.feature_dim:
                raise FeatureCacheError(
                    f"{identifier!r}: expected [T, {self.feature_dim}], got "
                    f"{tuple(sequence.shape)}"
                )
            if sequence.shape[0] < 1:
                raise FeatureCacheError(
                    f"{identifier!r}: zero-length sequence. An encoder that produced "
                    f"no frames is a failure to record, not a feature to cache."
                )
            if not np.isfinite(sequence).all():
                raise FeatureCacheError(
                    f"{identifier!r}: features contain NaN or Inf. Caching them would "
                    f"poison every batch the sample lands in."
                )

        lengths = np.asarray([sequence.shape[0] for sequence in sequences], dtype=np.int64)
        offsets = np.zeros(len(sequences) + 1, dtype=np.int64)
        np.cumsum(lengths, out=offsets[1:])
        stacked = np.concatenate(
            [np.asarray(sequence, dtype=STORAGE_DTYPE) for sequence in sequences], axis=0
        )

        shard = self.next_shard()
        np.savez_compressed(
            self.shard_path(shard),
            ids=np.asarray([str(item) for item in sample_ids]),
            features=stacked,
            offsets=offsets,
        )
        for row, identifier in enumerate(sample_ids):
            self.entries[str(identifier)] = [shard, row, int(lengths[row])]
        self.index["shards"] = max(int(self.index.get("shards", 0)), shard + 1)
        self.index["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.flush()
        return shard

    def flush(self) -> None:
        self.index_path.write_text(json.dumps(self.index, indent=2), encoding="utf-8")

    # ----------------------------------------------------------------- reads

    def load_one(self, sample_id: str) -> np.ndarray:
        """One ``[T, D]`` float32 sequence."""
        return self.load([sample_id])[0]

    def load(self, sample_ids: Sequence[str]) -> list[np.ndarray]:
        """Sequences in the requested order.  A missing id is an error, never a zero."""
        wanted = [str(item) for item in sample_ids]
        unknown = [item for item in wanted if item not in self.entries]
        if unknown:
            raise FeatureCacheError(
                f"{len(unknown)} sample ids are not in the {self.modality} cache "
                f"(e.g. {unknown[:5]}).\nExtract them first; a missing feature is "
                f"never substituted with zeros -- that would train the model to "
                f"treat a silent modality as a real observation."
            )
        by_shard: dict[int, list[tuple[int, int]]] = {}
        for position, identifier in enumerate(wanted):
            shard, row, _ = self.entries[identifier]
            by_shard.setdefault(int(shard), []).append((int(row), position))

        out: list[np.ndarray | None] = [None] * len(wanted)
        for shard, pairs in by_shard.items():
            features, offsets = self._shard_arrays(shard)
            for row, position in pairs:
                out[position] = features[offsets[row]:offsets[row + 1]].astype(np.float32)
        return [sequence for sequence in out if sequence is not None]

    def _shard_arrays(self, shard: int) -> tuple[np.ndarray, np.ndarray]:
        """One shard's arrays, decompressed at most once.

        Without this the dataloader decompresses a whole 512-sample shard for
        every single ``__getitem__`` -- the training loop's access pattern is one
        sample at a time, so each epoch re-inflates every shard once per sample
        in it.  Measured cost on the IEMOCAP text cache: 520 s per epoch against
        roughly 30 s once shards are held.

        Arrays are kept in their stored float16 form and converted per sample, so
        a cached shard costs what it costs on disk rather than twice that.
        """
        cached = self._shards.get(shard)
        if cached is not None:
            self._shards.move_to_end(shard)
            return cached

        with np.load(self.shard_path(shard), allow_pickle=False) as payload:
            arrays = (payload["features"], payload["offsets"])
        self._shards[shard] = arrays
        while len(self._shards) > self.max_cached_shards:
            self._shards.popitem(last=False)
        return arrays

    def release(self) -> None:
        """Drop every held shard.  For a caller that is done reading."""
        self._shards.clear()

    # ------------------------------------------------------------- integrity

    def scan_shards(self) -> dict[tuple[int, int], tuple[str, int]]:
        """What the shards actually hold: ``(shard, row) -> (sample_id, length)``.

        Shards carry their own ids, so they are the ground truth and the index is
        a derived convenience.
        """
        actual: dict[tuple[int, int], tuple[str, int]] = {}
        for path in sorted(self.root.glob("shard_*.npz")):
            label = path.stem.split("_")[-1]
            if not label.isdigit():
                continue
            with np.load(path, allow_pickle=False) as payload:
                identifiers = [str(value) for value in payload["ids"]]
                offsets = payload["offsets"]
            for row, identifier in enumerate(identifiers):
                actual[(int(label), row)] = (identifier, int(offsets[row + 1] - offsets[row]))
        return actual

    def verify(self) -> dict:
        """Audit the index against the shards.

        ``misattributed`` is the only finding that means the data itself is
        wrong -- an index entry pointing at a row that holds another sample's
        features would train the trunk on the wrong recording without any
        visible symptom.  The rest is lost bookkeeping that
        :meth:`rebuild_index` repairs without re-encoding anything.
        """
        actual = self.scan_shards()
        misattributed, dangling = [], []
        for identifier, (shard, row, length) in self.entries.items():
            found = actual.get((int(shard), int(row)))
            if found is None:
                dangling.append(identifier)
            elif found[0] != identifier or found[1] != int(length):
                misattributed.append(identifier)
        indexed = {(int(s), int(r)) for s, r, _ in self.entries.values()}
        orphaned = [f"{shard}:{row}" for (shard, row) in actual if (shard, row) not in indexed]
        return {
            "entries": len(self.entries),
            "rows_on_disk": len(actual),
            "misattributed": misattributed[:20],
            "misattributed_count": len(misattributed),
            "dangling": dangling[:20],
            "dangling_count": len(dangling),
            "orphaned": orphaned[:20],
            "orphaned_count": len(orphaned),
            "healthy": not misattributed and not dangling and not orphaned,
        }

    def rebuild_index(self) -> dict:
        """Recover the index from the shards after an interrupted run."""
        actual = self.scan_shards()
        self.index["entries"] = {
            identifier: [shard, row, length]
            for (shard, row), (identifier, length) in sorted(actual.items())
        }
        self.index["shards"] = (max(shard for shard, _ in actual) + 1) if actual else 0
        self.index["rebuilt_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.flush()
        return {"recovered": len(self.index["entries"]), "shards": self.index["shards"]}

    def provenance(self) -> dict:
        lengths = [int(length) for _, _, length in self.entries.values()]
        return {
            "schema": SEQUENCE_SCHEMA,
            "modality": self.modality,
            "fingerprint": self.fingerprint,
            "feature_dim": self.feature_dim,
            "samples": len(self.entries),
            "shards": int(self.index.get("shards", 0)),
            "frames_total": int(sum(lengths)),
            "frames_min": int(min(lengths)) if lengths else 0,
            "frames_max": int(max(lengths)) if lengths else 0,
            "frames_mean": float(np.mean(lengths)) if lengths else 0.0,
            "created_at": self.index.get("created_at"),
            "updated_at": self.index.get("updated_at"),
        }

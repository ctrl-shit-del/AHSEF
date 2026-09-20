"""Run each frozen encoder once and write its features to disk.

    # always smoke-test before a long pass
    python -m src.hsen.extract --experiment iemocap_erc6 --modality text  --split validation --limit 32

    # the real passes
    python -m src.hsen.extract --experiment iemocap_erc6 --modality text  --split train
    python -m src.hsen.extract --experiment iemocap_erc6 --modality audio --split train

    # audit an existing cache without extracting anything
    python -m src.hsen.extract --experiment iemocap_erc6 --modality audio --split train --verify-only

Resumable.  Every sample already in the cache is skipped, so an interrupted run
costs at most the shard it was mid-way through, and re-running a finished split
reports "nothing to do".

On extracting the test split.  This project's earlier audio extractor demands an
explicit ``--i-am-running-the-locked-evaluation`` flag before it will touch the
test partition, and that rule is right in its context -- a locked evaluation.
It is not required here, and the reason is worth stating rather than assuming:
these encoders read waveforms, frames and transcripts, and never labels, so
extracting test features reveals nothing about test *answers*.  The guard that
matters is on evaluation, not extraction, and it stays: the trainer opens the
test split only when explicitly asked, and the audit checks that no cached test
id appears in a training index.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.paths import DATASETS_DIR
from src.hsen.features.store import (
    CacheBusyError,
    ExtractionLock,
    FeatureCacheError,
    SequenceFeatureStore,
)
from src.hsen.manifests import EXPERIMENTS, load

FEATURE_ROOT = Path("experiments/hsen")
MODALITIES = ("text", "audio", "video")


def cache_dir(experiment: str, modality: str, split: str, root: Path | None = None) -> Path:
    """Where one (experiment, modality, split) cache lives.

    Split is the last path component, and that is deliberate: the caches are
    physically separate directories with separate indices, so a train loader
    cannot reach a test feature even by accident.
    """
    return (Path(root) if root else FEATURE_ROOT) / experiment / "features" / modality / split


def build_encoder(modality: str, args: argparse.Namespace):
    """Construct the frozen encoder for one modality."""
    if modality == "text":
        from src.hsen.features.text_roberta import RoBERTaTextEncoder

        return RoBERTaTextEncoder(
            model_name=args.text_model, max_tokens=args.max_tokens,
            device=args.device, batch_size=args.batch_size,
        )
    if modality == "audio":
        from src.hsen.features.audio_emotion2vec import Emotion2VecEncoder

        return Emotion2VecEncoder(
            repo=args.audio_model, model_dir=args.audio_model_dir,
            max_seconds=args.max_seconds, device=args.device,
            num_threads=args.num_threads,
        )
    if modality == "video":
        from src.hsen.features.video_face import FaceVideoEncoder

        return FaceVideoEncoder(
            backbone=args.video_model, max_frames=args.max_frames, device=args.device,
        )
    raise ValueError(f"Unknown modality {modality!r}; expected one of {list(MODALITIES)}")


def payloads_for(modality: str, frame: pd.DataFrame) -> list:
    """Turn manifest rows into whatever the encoder consumes."""
    if modality == "text":
        return frame["text"].astype(str).tolist()
    column = {"audio": "audio_path", "video": "video_path"}[modality]
    if column not in frame.columns:
        raise FeatureCacheError(
            f"The manifest has no {column}; {modality} cannot be extracted from it"
        )
    return [DATASETS_DIR / str(path) for path in frame[column]]


def eligible_rows(modality: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Rows that declare the modality.

    Nothing is extracted for a row that does not have the modality: a fabricated
    feature for an absent one is exactly the failure the availability mask exists
    to prevent.
    """
    flag = f"has_{modality}"
    if flag not in frame.columns:
        return frame
    return frame[frame[flag].fillna(False).astype(bool)].reset_index(drop=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--experiment", required=True, choices=sorted(EXPERIMENTS))
    parser.add_argument("--modality", required=True, choices=list(MODALITIES))
    parser.add_argument("--split", required=True, choices=["train", "validation", "test"])
    parser.add_argument("--feature-dir", default=None, help="Override the cache root.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None,
                        help="Extract at most this many pending samples (smoke tests).")
    parser.add_argument("--progress-every", type=int, default=128)
    parser.add_argument("--num-threads", type=int, default=8,
                        help="Torch intra-op threads for CPU extraction. Doubles "
                             "emotion2vec throughput on this machine; affects speed "
                             "only, never the features.")

    parser.add_argument("--text-model", default="roberta-base")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--audio-model", default="emotion2vec/emotion2vec_base")
    parser.add_argument("--audio-model-dir", default=None)
    parser.add_argument("--max-seconds", type=float, default=10.0)
    parser.add_argument("--video-model", default="mobilenet_v3_small")
    parser.add_argument("--max-frames", type=int, default=16)

    parser.add_argument("--verify-only", action="store_true",
                        help="Audit the cache against its shards and exit.")
    parser.add_argument("--rebuild-index", action="store_true",
                        help="Rebuild the index from the shard files, then continue.")
    parser.add_argument("--force-lock", action="store_true",
                        help="Break a stale extraction lock. Only after confirming "
                             "the holder is gone.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    frame = load(args.experiment, args.split)
    frame = eligible_rows(args.modality, frame)
    if frame.empty:
        print(f"No {args.split} rows declare {args.modality}; nothing to do.")
        return 0

    encoder = build_encoder(args.modality, args)
    spec = encoder.spec
    root = cache_dir(args.experiment, args.modality, args.split, args.feature_dir)

    print(f"[cfg ] {args.experiment} | {args.modality} | {args.split}")
    print(f"[cfg ] {spec.model} | max_frames={spec.max_frames} | dim={spec.feature_dim}")
    print(f"[cfg ] fingerprint={spec.fingerprint[:16]} | cache={root}")

    store = SequenceFeatureStore.open(
        root, spec.fingerprint, args.modality, spec.feature_dim, args.shard_size,
    )

    if args.rebuild_index:
        print(f"[idx ] rebuilt: {store.rebuild_index()}")
    if args.verify_only:
        report = store.verify()
        print(json.dumps(report, indent=2))
        return 0 if report["healthy"] else 1

    sample_ids = frame["sample_id"].astype(str).tolist()
    payloads = payloads_for(args.modality, frame)
    pending = [
        (sample_id, payload)
        for sample_id, payload in zip(sample_ids, payloads)
        if sample_id not in store
    ]
    if args.limit:
        pending = pending[:args.limit]

    print(f"[plan] {len(sample_ids):,} eligible | {len(store):,} cached | "
          f"{len(pending):,} to extract")
    if not pending:
        print("Nothing to do.")
        return 0

    try:
        lock = ExtractionLock(root).acquire(force=args.force_lock)
    except CacheBusyError as error:
        raise SystemExit(str(error))

    started = time.time()
    done = 0
    try:
        for start in range(0, len(pending), args.shard_size):
            chunk = pending[start:start + args.shard_size]
            features: list[np.ndarray] = []
            identifiers: list[str] = []
            for offset in range(0, len(chunk), args.batch_size):
                batch = chunk[offset:offset + args.batch_size]
                encoded = encoder.encode([payload for _, payload in batch])
                # Truncation to max_frames happens here rather than inside each
                # encoder, so one rule governs every modality and the cap in the
                # spec is the cap that was actually applied.
                for (sample_id, _), sequence in zip(batch, encoded):
                    identifiers.append(sample_id)
                    features.append(sequence[:spec.max_frames])
                done += len(batch)
                if args.progress_every and done % args.progress_every < args.batch_size:
                    rate = done / max(1e-9, time.time() - started)
                    remaining = (len(pending) - done) / max(1e-9, rate)
                    print(f"[work] {done:>6,}/{len(pending):,}  "
                          f"{rate:5.1f}/s  eta {remaining / 60:5.1f} min")
            shard = store.append(identifiers, features)
            print(f"[save] shard {shard:04d}: {len(identifiers)} samples "
                  f"({len(store):,} cached)")
    finally:
        lock.release()

    provenance = store.provenance() | {
        "encoder": encoder.describe(),
        "experiment": args.experiment,
        "split": args.split,
        "eligible_samples": len(sample_ids),
        "extraction_seconds": round(time.time() - started, 1),
    }
    (root / "extraction_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True, default=str), encoding="utf-8",
    )

    elapsed = time.time() - started
    print(f"\nExtracted {done:,} samples in {elapsed / 60:.1f} min "
          f"({done / max(1e-9, elapsed):.1f}/s)")
    print(f"Cache now holds {len(store):,} samples; "
          f"frames {provenance['frames_min']}-{provenance['frames_max']} "
          f"(mean {provenance['frames_mean']:.1f})")
    report = store.verify()
    print(f"Integrity: {'healthy' if report['healthy'] else report}")
    return 0 if report["healthy"] else 1


if __name__ == "__main__":
    sys.exit(main())

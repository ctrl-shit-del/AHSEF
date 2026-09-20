"""Extract and cache frozen wav2vec2 features for an audio experiment split.

    # smoke test first -- never start a multi-hour run unverified
    python -m src.training.extract_audio_features --split validation --limit 64

    # the real passes
    python -m src.training.extract_audio_features --split train
    python -m src.training.extract_audio_features --split validation

Resumable: every sample already in the cache is skipped, so an interrupted run
costs only the shard it was mid-way through.  Re-running a finished split is a
no-op that reports "nothing to do".

``--split test`` exists but refuses to run without ``--i-am-running-the-locked-
evaluation``.  Extracting test features is harmless in itself -- it reads audio,
not labels -- but the flag makes opening the locked split a deliberate act that
shows up in the shell history rather than something that happens by habit.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.experiment_layout import ExperimentLayout
from src.common.paths import DATASETS_DIR
from src.data.features.wav2vec2_features import (
    DEFAULT_BUNDLE,
    SUPPORTED_BUNDLES,
    CacheBusyError,
    ExtractionLock,
    FeatureStore,
    Wav2Vec2Config,
    Wav2Vec2LayerMeans,
)
from src.data.loaders.audio import AudioLoader
from src.training.audio_strong_manifest import STRONG_EXPERIMENT

DEFAULT_CACHE = Path("experiments") / "audio_strong" / "features"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", required=True, choices=["train", "validation", "test"])
    parser.add_argument("--experiment", default=STRONG_EXPERIMENT)
    parser.add_argument("--root", default="experiments")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE, choices=list(SUPPORTED_BUNDLES))
    parser.add_argument("--max-seconds", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None,
                        help="Only extract this many pending samples (smoke tests).")
    parser.add_argument("--progress-every", type=int, default=256)
    parser.add_argument(
        "--i-am-running-the-locked-evaluation", action="store_true",
        help="Required to extract the test split. Makes opening it deliberate.",
    )
    parser.add_argument(
        "--force-lock", action="store_true",
        help="Break a stale extraction lock. Only after confirming the holder is gone.",
    )
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Audit the cache against its shards and exit without extracting.",
    )
    parser.add_argument(
        "--rebuild-index", action="store_true",
        help="Rebuild the index from the shard files, then continue.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.split == "test" and not args.i_am_running_the_locked_evaluation:
        raise SystemExit(
            "Refusing to extract the test split without "
            "--i-am-running-the-locked-evaluation. The locked split is opened once, "
            "deliberately, after every validation-side decision is frozen."
        )

    layout = ExperimentLayout.from_name(args.experiment, Path(args.root))
    manifest_path = layout.split_path(args.split)
    if not manifest_path.exists():
        raise SystemExit(
            f"No manifest at {manifest_path}. Build it first:\n"
            f"  python -c \"from src.training.audio_strong_manifest import "
            f"build_manifests; build_manifests()\""
        )
    frame = pd.read_parquet(
        manifest_path, columns=["sample_id", "dataset", "audio_path", "canonical_emotion_id"]
    )
    frame["sample_id"] = frame["sample_id"].astype(str)

    config = Wav2Vec2Config(
        bundle=args.bundle, max_seconds=args.max_seconds,
        batch_size=args.batch_size, num_threads=args.num_threads,
    )
    cache_dir = Path(args.cache_dir) if args.cache_dir else DEFAULT_CACHE / args.split
    encoder = Wav2Vec2LayerMeans(config)

    print(f"[cfg ] {args.bundle} | max_seconds={args.max_seconds} "
          f"| batch={args.batch_size} | threads={args.num_threads}")
    print(f"[cfg ] fingerprint={config.fingerprint[:16]} | cache={cache_dir}")

    # One writer per cache. Two extractors on the same directory strand each
    # other's index entries -- this happened during the project's own run.
    try:
        lock = ExtractionLock(cache_dir).acquire(force=args.force_lock)
    except CacheBusyError as error:
        raise SystemExit(str(error))

    try:
        return _run(args, frame, config, cache_dir, encoder, manifest_path)
    finally:
        lock.release()


def _run(args, frame, config, cache_dir, encoder, manifest_path) -> int:
    store = FeatureStore.open(
        cache_dir, config.fingerprint,
        num_layers=encoder.num_layers, feature_dim=encoder.feature_dim,
        shard_size=args.shard_size,
    )
    print(f"[enc ] loaded: {encoder.num_layers} layers x {encoder.feature_dim} dims "
          f"| {sum(p.numel() for p in encoder.model.parameters())/1e6:.1f}M params (frozen)")

    health = store.verify()
    print(f"[chk ] index={health['index_entries']} rows_in_shards="
          f"{health['rows_in_shards']} misattributed={health['misattributed']} "
          f"unindexed={health['extracted_but_unindexed']}")
    if health["misattributed"]:
        raise SystemExit(
            f"{health['misattributed']} cached entries point at a shard row holding a "
            f"DIFFERENT sample (e.g. {health['misattributed_examples']}). That is "
            f"corrupted data, not lost bookkeeping; delete this cache and re-extract."
        )
    if args.rebuild_index or health["extracted_but_unindexed"]:
        repair = store.rebuild_index()
        print(f"[fix ] index rebuilt from shards: {repair['entries_before']} -> "
              f"{repair['entries_after']} entries "
              f"(+{repair['recovered']} recovered, "
              f"{repair['duplicate_rows_dropped']} duplicate rows dropped)")
    if args.verify_only:
        print("[done] --verify-only: no extraction performed")
        return 0

    pending = store.missing(frame["sample_id"])
    print(f"[plan] {args.split}: {len(frame)} samples, {len(store)} cached, "
          f"{len(pending)} pending")
    if args.limit is not None:
        pending = pending[: args.limit]
        print(f"[plan] --limit {args.limit}: extracting {len(pending)} this run")
    if not pending:
        print("[done] nothing to do; cache already covers this split")
        _write_provenance(cache_dir, store, encoder, args, frame, 0, 0.0)
        return 0

    paths = dict(zip(frame["sample_id"], frame["audio_path"]))
    loader = AudioLoader(DATASETS_DIR)

    started = time.perf_counter()
    done = failures = 0
    batch_ids: list[str] = []
    batch_waveforms: list[np.ndarray] = []
    shard_ids: list[str] = []
    shard_features: list[np.ndarray] = []
    failed: list[dict] = []

    def flush_shard() -> None:
        nonlocal shard_ids, shard_features
        if not shard_ids:
            return
        store.append(shard_ids, np.concatenate(shard_features, axis=0))
        shard_ids, shard_features = [], []

    for identifier in pending:
        try:
            waveform = loader.load_waveform(
                paths[identifier],
                target_rate=config.sample_rate,
                max_seconds=config.max_seconds,
            )
        except Exception as error:
            # A decode failure is a real, reportable gap. It is counted and the
            # sample is left out of the cache -- never replaced with silence,
            # which would be a fabricated recording.
            failures += 1
            failed.append({"sample_id": identifier, "error": str(error)[:200]})
            continue
        batch_ids.append(identifier)
        batch_waveforms.append(waveform)

        if len(batch_ids) >= config.batch_size:
            shard_features.append(encoder.encode(batch_waveforms))
            shard_ids.extend(batch_ids)
            done += len(batch_ids)
            batch_ids, batch_waveforms = [], []
            if len(shard_ids) >= args.shard_size:
                flush_shard()
            if done % max(args.progress_every, 1) < config.batch_size:
                _progress(done, len(pending), started, failures)

    if batch_ids:
        shard_features.append(encoder.encode(batch_waveforms))
        shard_ids.extend(batch_ids)
        done += len(batch_ids)
    flush_shard()

    elapsed = time.perf_counter() - started
    _progress(done, len(pending), started, failures)
    print(f"[done] {done} extracted in {elapsed/60:.1f} min "
          f"({done/max(elapsed,1e-9):.2f}/s) | {failures} decode failures "
          f"| cache now holds {len(store)}")
    if failed:
        path = Path(cache_dir) / "decode_failures.json"
        path.write_text(json.dumps(failed, indent=2), encoding="utf-8")
        print(f"[warn] decode failures listed in {path}")

    _write_provenance(cache_dir, store, encoder, args, frame, done, elapsed)
    return 0


def _progress(done: int, total: int, started: float, failures: int) -> None:
    elapsed = time.perf_counter() - started
    rate = done / max(elapsed, 1e-9)
    remaining = (total - done) / max(rate, 1e-9) / 60
    print(f"       {done}/{total}  {rate:.2f}/s  eta {remaining:.1f} min  "
          f"failures={failures}", flush=True)


def _write_provenance(cache_dir, store, encoder, args, frame, extracted, elapsed) -> None:
    record = {
        "split": args.split,
        "experiment": args.experiment,
        "manifest_samples": int(len(frame)),
        "extractor": encoder.describe(),
        "store": store.provenance(),
        "this_invocation": {
            "extracted": int(extracted),
            "elapsed_seconds": float(elapsed),
            "rate_per_second": float(extracted / elapsed) if elapsed > 0 else None,
            "num_threads": args.num_threads,
            "batch_size": args.batch_size,
        },
        "coverage": len(store) / len(frame) if len(frame) else None,
        "device": "cpu",
        "encoder_trained_by_this_project": False,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": sys.version.split()[0],
    }
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    (Path(cache_dir) / "extraction_provenance.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    raise SystemExit(main())

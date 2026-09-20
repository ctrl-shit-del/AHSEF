"""Convert CMU-MOSEI's provided feature container into the standard HSEN cache.

    python -m src.hsen.mosei_features --experiment mosei_sentiment
    python -m src.hsen.mosei_features --experiment mosei_sentiment --container unaligned_50.pkl

Why CMU-MOSEI does not go through ``src.hsen.extract``.  It ships no usable raw
media: ``datasets/CMU-MOSEI/`` holds a 27 GB unextracted ``Raw.zip`` and a
``Processed/`` directory of precomputed features.  emotion2vec and MobileNetV3
have nothing to read.

Using the provided features is also the *right* choice rather than a fallback.
LDDU, CARAT and TAILOR -- the papers whose 0.496 accuracy / 0.587 micro-F1 the
survey names as CMU-MOSEI reference points -- all report against exactly these
COVAREP and FACET features.  Substituting our own encoders would improve the
inputs and destroy the comparability at the same time, and there would be no way
to tell which of the two moved a number.

    audio   COVAREP     74-d
    vision  FACET       35-d
    text    BERT       768-d   (the container's own text_bert stream)

The projection layer is what makes the width difference irrelevant downstream:
74-d and 768-d both become 256-d before the trunk sees them, so one model config
serves both corpora.

Text, and a choice worth stating.  The container carries a BERT text stream, but
CMU-MOSEI's gold transcripts are also in ``label.csv`` and already in the
manifest -- so ``--text-source roberta`` runs the same frozen RoBERTa-base used
for IEMOCAP, which keeps the text branch identical across the two corpora.
``--text-source container`` uses the packaged BERT stream instead, which keeps
the whole input identical to the published baselines.  Neither is strictly
better; they answer different questions, so both are available and the choice is
recorded in the cache provenance.

Memory.  ``aligned_50.pkl`` is 4.6 GB and unpickles whole -- there is no partial
read of a pickle.  Expect a peak of roughly 6-8 GB, and do not run this
alongside a training job.
"""

from __future__ import annotations

import argparse
import gc
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from src.common.paths import DATASETS_DIR
from src.hsen.extract import cache_dir
from src.hsen.features.base import EncoderSpec
from src.hsen.features.store import SequenceFeatureStore
from src.hsen.manifests import load

DEFAULT_CONTAINER = "CMU-MOSEI/Processed/aligned_50.pkl"

#: Container split names against ours.  The release calls the middle one
#: ``valid``; this project calls it ``validation`` everywhere.
SPLIT_NAMES = {"train": "train", "validation": "valid", "test": "test"}

#: Our modality name against the container's key and expected width.
STREAMS = {
    "audio": ("audio", 74, "COVAREP"),
    "video": ("vision", 35, "FACET"),
    "text": ("text_bert", 768, "BERT"),
}


class ContainerError(RuntimeError):
    """Raised when the feature container is missing or shaped unexpectedly."""


def container_spec(modality: str, container: str, dimension: int, source: str) -> EncoderSpec:
    """A spec for provided features, so they fingerprint like extracted ones.

    The container is not an encoder we run, but the cache still needs a
    fingerprint that changes when the source does -- otherwise aligned and
    unaligned features could be mixed in one cache without complaint.
    """
    return EncoderSpec(
        modality=modality,
        model=f"cmu_mosei_container:{source}",
        feature_dim=dimension,
        max_frames=50,          # the container is fixed at 50 steps per clip
        layer=-1,
        options={"container": container, "provided_by": "CMU-MOSEI release",
                 "extracted_by_this_project": False},
    )


def trim_padding(sequence: np.ndarray) -> np.ndarray:
    """Drop the container's zero padding, keeping at least one frame.

    The container pads every clip to 50 steps with zero rows.  Caching those
    would tell the trunk that a three-word clip was observed for the same
    duration as a fifty-word one, and the attention mask -- which is built from
    the cached length -- would have nothing to mask.  An all-zero clip keeps one
    frame rather than none, because a zero-length sequence is not something the
    store or the trunk can represent.
    """
    if sequence.ndim != 2:
        raise ContainerError(f"Expected [T, D], got {sequence.shape}")
    non_zero = np.flatnonzero(np.abs(sequence).sum(axis=1) > 0)
    if non_zero.size == 0:
        return sequence[:1]
    return sequence[: int(non_zero[-1]) + 1]


def convert(
    experiment: str,
    container_path: Path,
    modalities: tuple[str, ...],
    splits: tuple[str, ...],
    feature_root: Path | None = None,
    shard_size: int = 512,
) -> dict:
    """Read the container once and write every requested cache."""
    if not container_path.exists():
        raise ContainerError(
            f"No CMU-MOSEI feature container at {container_path}.\n"
            f"The release ships it as Processed/aligned_50.pkl."
        )

    print(f"Loading {container_path} ({container_path.stat().st_size / 1e9:.1f} GB). "
          f"This unpickles whole and peaks around 6-8 GB.")
    started = time.time()
    with container_path.open("rb") as handle:
        data = pickle.load(handle)
    print(f"Loaded in {time.time() - started:.0f}s. Splits: {sorted(data)}")

    report: dict[str, dict] = {}
    try:
        for split in splits:
            key = SPLIT_NAMES[split]
            if key not in data:
                raise ContainerError(
                    f"Container has no {key!r} split; found {sorted(data)}"
                )
            block = data[key]
            # The container's own id list is the join key, and it is the same
            # ``<video_id>$_$<clip_id>`` form the standardizer recorded as
            # feature_id -- so the join uses the release's identifiers rather
            # than a second, independently parsed set of them.
            identifiers = [
                "$_$".join(str(part) for part in row) if isinstance(row, (list, tuple, np.ndarray))
                else str(row)
                for row in block["id"]
            ]
            wanted = set(load(experiment, split)["feature_id"].astype(str))
            print(f"\n[{split}] container holds {len(identifiers):,}; "
                  f"manifest wants {len(wanted):,}")

            for modality in modalities:
                stream, dimension, source = STREAMS[modality]
                if stream not in block:
                    raise ContainerError(
                        f"Container split {key!r} has no {stream!r}; "
                        f"found {sorted(block)}"
                    )
                matrix = np.asarray(block[stream], dtype=np.float32)
                if matrix.shape[-1] != dimension:
                    raise ContainerError(
                        f"{stream} is {matrix.shape[-1]}-d, expected {dimension}-d. "
                        f"Update STREAMS rather than reshaping the features."
                    )

                spec = container_spec(modality, container_path.name, dimension, source)
                root = cache_dir(experiment, modality, split, feature_root)
                store = SequenceFeatureStore.open(
                    root, spec.fingerprint, modality, dimension, shard_size,
                )

                pending_ids, pending_features = [], []
                written = 0
                for index, identifier in enumerate(identifiers):
                    if identifier not in wanted or identifier in store:
                        continue
                    pending_ids.append(identifier)
                    pending_features.append(trim_padding(matrix[index]))
                    if len(pending_ids) >= shard_size:
                        store.append(pending_ids, pending_features)
                        written += len(pending_ids)
                        pending_ids, pending_features = [], []
                if pending_ids:
                    store.append(pending_ids, pending_features)
                    written += len(pending_ids)

                missing = wanted - set(store.entries)
                provenance = store.provenance() | {
                    "container": str(container_path),
                    "stream": stream,
                    "source": source,
                    "experiment": experiment,
                    "split": split,
                    "extracted_by_this_project": False,
                    "padding": "container zero-padding trimmed; true lengths cached",
                    "manifest_rows": len(wanted),
                    "unmatched_manifest_rows": len(missing),
                }
                (root / "extraction_provenance.json").write_text(
                    json.dumps(provenance, indent=2, sort_keys=True, default=str),
                    encoding="utf-8",
                )
                report[f"{split}/{modality}"] = {
                    "written": written, "cached": len(store), "missing": len(missing),
                    "frames_mean": provenance["frames_mean"],
                }
                status = "OK" if not missing else f"{len(missing):,} MISSING"
                print(f"  {modality:<6} {source:<8} {dimension:>4}-d  "
                      f"+{written:>6,} written  {len(store):>6,} cached  "
                      f"mean {provenance['frames_mean']:5.1f} frames  {status}")
                del matrix
                gc.collect()
    finally:
        del data
        gc.collect()
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--experiment", default="mosei_sentiment",
                        choices=["mosei_sentiment", "mosei_emotion6"])
    parser.add_argument("--container", default=DEFAULT_CONTAINER,
                        help="Path under datasets/, or an absolute path.")
    parser.add_argument("--modalities", default="audio,video,text")
    parser.add_argument("--splits", default="train,validation,test")
    parser.add_argument("--feature-dir", default=None)
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument(
        "--text-source", default="container", choices=["container", "roberta"],
        help="'container' uses the packaged BERT stream, matching the published "
             "baselines. 'roberta' skips text here so src.hsen.extract can run "
             "the same frozen RoBERTa-base used for IEMOCAP.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    container = Path(args.container)
    if not container.is_absolute():
        container = DATASETS_DIR / args.container

    modalities = tuple(m.strip() for m in args.modalities.split(",") if m.strip())
    unknown = set(modalities) - set(STREAMS)
    if unknown:
        raise SystemExit(f"Unknown modalities {sorted(unknown)}; expected {sorted(STREAMS)}")
    if args.text_source == "roberta" and "text" in modalities:
        modalities = tuple(m for m in modalities if m != "text")
        print("Text will come from RoBERTa, not the container. Run:\n"
              f"    python -m src.hsen.extract --experiment {args.experiment} "
              f"--modality text --split <split>")

    report = convert(
        experiment=args.experiment,
        container_path=container,
        modalities=modalities,
        splits=tuple(s.strip() for s in args.splits.split(",") if s.strip()),
        feature_root=Path(args.feature_dir) if args.feature_dir else None,
        shard_size=args.shard_size,
    )
    unmatched = {key: value["missing"] for key, value in report.items() if value["missing"]}
    if unmatched:
        print(f"\nWARNING: manifest rows with no container features: {unmatched}")
        print("Training will refuse these samples rather than zero-fill them.")
        return 1
    print("\nEvery manifest row has features in every requested modality.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

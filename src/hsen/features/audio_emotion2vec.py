"""The frozen audio branch: emotion2vec, cached as 768-d frame features.

The survey's strongest reuse recommendation (REUSE, 9.5/10), and the reason is
specific: a *linear probe* on frozen emotion2vec reaches 71.79 weighted accuracy
on IEMOCAP, matching a fully fine-tuned WavLM-large specialist while training
0.20 M parameters.  For a project whose whole thesis is about spending compute
only where it buys something, an encoder that gives away that much for free and
stays frozen is exactly the right shape.  Its features also separate along a
continuous arousal gradient (emotion2vec Fig. 3), which matters here because
HSEN regresses arousal rather than bucketing it.

Where the weights come from.  ``modelscope.cn`` is unreachable from this
machine, so the checkpoint is pulled from the HuggingFace mirror of the same
release (``emotion2vec/emotion2vec_base``) and FunASR is pointed at the local
directory.  Same weights, reachable host.

Frame granularity, not utterance.  FunASR will happily return one pooled 768-d
vector per clip, and that would make this module trivial -- but it would also
pool away the temporal structure before the fusion trunk sees it, which is the
one thing Husformer's cross-modal attention exists to use.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from src.hsen.features.base import EncoderSpec

DEFAULT_REPO = "emotion2vec/emotion2vec_base"
DEFAULT_FEATURE_DIM = 768
SAMPLE_RATE = 16_000

#: emotion2vec emits one frame per 20 ms, so 500 frames is 10 seconds.  That cap
#: truncates 6.5% of IEMOCAP utterances; the 4-second cap this project's earlier
#: strong-audio probe used truncated 45% of them, which is a lot of speech to
#: throw away when accuracy is the stated priority.
DEFAULT_MAX_SECONDS = 10.0
FRAMES_PER_SECOND = 50


class AudioEncodingError(RuntimeError):
    """Raised when audio cannot be read or encoded."""


def load_waveform(path: Path | str, max_seconds: float | None = None) -> np.ndarray:
    """Read one clip as 16 kHz mono float32, optionally head-trimmed.

    Trimming takes the head rather than a centre crop or a random one: a random
    crop would make the cache non-reproducible, and a centre crop would move the
    window depending on a length the model never sees.  The head is boring,
    deterministic, and identical on every re-run.
    """
    import soundfile as sound_file

    path = Path(path)
    if not path.exists():
        raise AudioEncodingError(f"Audio file not found: {path}")

    frames = int(max_seconds * SAMPLE_RATE) if max_seconds else None
    try:
        waveform, rate = sound_file.read(str(path), frames=frames or -1, dtype="float32")
    except Exception as error:
        raise AudioEncodingError(f"Could not read {path}: {error}") from error

    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if rate != SAMPLE_RATE:
        import librosa

        waveform = librosa.resample(waveform, orig_sr=rate, target_sr=SAMPLE_RATE)
    if waveform.size == 0:
        raise AudioEncodingError(f"{path} contains no samples")
    return np.ascontiguousarray(waveform, dtype=np.float32)


class Emotion2VecEncoder:
    """Frozen emotion2vec, cached as ``[T, 768]`` frame features."""

    def __init__(
        self,
        repo: str = DEFAULT_REPO,
        model_dir: Path | str | None = None,
        max_seconds: float = DEFAULT_MAX_SECONDS,
        device: str = "cpu",
        num_threads: int | None = None,
    ):
        self.repo = repo
        self.model_dir = Path(model_dir) if model_dir else None
        self.max_seconds = float(max_seconds)
        self.device = device
        # Measured on this machine: 0.5 clips/s at torch's default thread count
        # against 1.0/s at eight. The extraction pass is hours long either way,
        # so this is the single most valuable knob on the whole stage -- and it
        # changes throughput only, never the features, which is why it is not
        # part of the spec fingerprint.
        self.num_threads = num_threads
        self._model = None
        self.truncated = 0
        self.spec = EncoderSpec(
            modality="audio",
            model=repo,
            feature_dim=DEFAULT_FEATURE_DIM,
            max_frames=int(self.max_seconds * FRAMES_PER_SECOND),
            layer=-1,
            options={
                "granularity": "frame",
                "sample_rate": SAMPLE_RATE,
                "max_seconds": self.max_seconds,
                "trim": "head",
            },
        )

    # ------------------------------------------------------------------ load

    def resolve_checkpoint(self) -> Path:
        """The local directory holding the emotion2vec checkpoint.

        Downloads it on first use.  Kept separate from :meth:`_load` so an
        extraction run can fetch weights up front and fail on a network problem
        before it starts reading audio.
        """
        if self.model_dir and self.model_dir.exists():
            return self.model_dir
        try:
            from huggingface_hub import snapshot_download
        except ImportError as error:  # pragma: no cover - environment dependent
            raise AudioEncodingError(
                "huggingface_hub is required to fetch the emotion2vec checkpoint"
            ) from error
        self.model_dir = Path(
            snapshot_download(self.repo, allow_patterns=["*.pt", "*.yaml", "*.json"])
        )
        return self.model_dir

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from funasr import AutoModel
        except ImportError as error:  # pragma: no cover - environment dependent
            raise AudioEncodingError(
                "funasr is required for the emotion2vec audio branch.\n"
                "    pip install funasr"
            ) from error

        if self.num_threads:
            import torch

            torch.set_num_threads(int(self.num_threads))

        directory = self.resolve_checkpoint()
        # A local path keeps FunASR away from ModelScope entirely; passing the
        # repo id would send it back to the host that cannot be reached here.
        self._model = AutoModel(
            model=str(directory), device=self.device, disable_update=True, hub="hf",
        )

    # ---------------------------------------------------------------- encode

    def encode(self, payloads: Sequence[Path | str | np.ndarray]) -> list[np.ndarray]:
        """Clips to ``[T, 768]`` float32 frame features, one per clip.

        Payloads are either paths or already-loaded 16 kHz mono waveforms; the
        extractor passes paths, and tests pass arrays so they need no audio on
        disk.
        """
        self._load()
        limit = int(self.max_seconds * SAMPLE_RATE)

        waveforms: list[np.ndarray] = []
        for payload in payloads:
            waveform = (
                np.ascontiguousarray(payload, dtype=np.float32)
                if isinstance(payload, np.ndarray)
                else load_waveform(payload, max_seconds=None)
            )
            if waveform.shape[0] > limit:
                waveform, self.truncated = waveform[:limit], self.truncated + 1
            waveforms.append(waveform)

        # FunASR accepts a list and returns one record per input, in order.
        # Worth about 20% over looping, on top of the thread-count win.
        results = self._model.generate(
            waveforms,
            granularity="frame",
            extract_embedding=True,
            disable_pbar=True,
            batch_size=len(waveforms),
        )
        if len(results) != len(waveforms):
            raise AudioEncodingError(
                f"emotion2vec returned {len(results)} results for "
                f"{len(waveforms)} clips; the batch would be misaligned, so the "
                f"features would be attributed to the wrong samples"
            )

        out: list[np.ndarray] = []
        for record in results:
            features = self._features_of(record)
            if features.ndim != 2 or features.shape[1] != DEFAULT_FEATURE_DIM:
                raise AudioEncodingError(
                    f"emotion2vec returned {features.shape}; expected "
                    f"[T, {DEFAULT_FEATURE_DIM}]. Check that granularity='frame' "
                    f"is still honoured by this FunASR version."
                )
            out.append(np.ascontiguousarray(features, dtype=np.float32))
        return out

    @staticmethod
    def _features_of(record) -> np.ndarray:
        """Pull the frame features out of one FunASR result record."""
        if record is None:
            raise AudioEncodingError("emotion2vec returned no result for a clip")
        if isinstance(record, (list, tuple)):
            record = record[0]
        for key in ("feats", "feature", "embedding"):
            if key in record:
                value = record[key]
                return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
        raise AudioEncodingError(
            f"emotion2vec result has no feature field; keys were {sorted(record)}"
        )

    def describe(self) -> dict:
        return self.spec.describe() | {
            "class": "Emotion2VecEncoder",
            "checkpoint": str(self.model_dir) if self.model_dir else self.repo,
            "granularity": "frame (50 Hz)",
            "pooling": "none -- per-frame features are cached",
            "truncated_samples": int(self.truncated),
            "device": self.device,
            "frozen": True,
            "trained_by_this_project": False,
        }

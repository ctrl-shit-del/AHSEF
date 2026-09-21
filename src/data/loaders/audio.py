from pathlib import Path

import numpy as np


def _require_soundfile():
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "soundfile is required for audio loading. "
            "Install it with: pip install soundfile"
        ) from exc
    return sf


def resample(waveform: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Rate-convert a mono waveform deterministically.

    Uses polyphase resampling from SciPy, which is exact for the integer rate
    ratios these corpora use (48 kHz RAVDESS to 16 kHz, and so on).  Failing
    loudly is deliberate: silently training on mismatched sample rates would
    change what the mel front end measures.
    """
    if source_rate == target_rate:
        return waveform
    try:
        from math import gcd

        from scipy.signal import resample_poly
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            f"Resampling {source_rate} Hz audio to {target_rate} Hz needs SciPy. "
            f"Install it with: pip install scipy"
        ) from exc
    divisor = gcd(int(source_rate), int(target_rate))
    return resample_poly(waveform, int(target_rate) // divisor, int(source_rate) // divisor)


class AudioLoader:
    """
    Load an audio file.

    Uses soundfile rather than performing preprocessing here.
    """

    def __init__(self, datasets_root: Path):
        self.datasets_root = datasets_root

    def resolve(self, relative_path: str) -> Path:
        if not relative_path:
            raise ValueError(
                "Audio path is empty."
            )

        path = self.datasets_root / relative_path

        if not path.exists():
            raise FileNotFoundError(
                f"Audio not found: {path}"
            )

        return path

    def load(self, relative_path: str):

        path = self.resolve(relative_path)
        sf = _require_soundfile()

        waveform, sample_rate = sf.read(path)

        return {
            "waveform": np.asarray(waveform),
            "sample_rate": sample_rate,
            "path": str(path),
        }

    def load_waveform(
        self,
        relative_path: str,
        target_rate: int = 16_000,
        max_seconds: float | None = None,
    ) -> np.ndarray:
        """Read a bounded, mono, rate-normalised waveform as ``float32``.

        Only the first ``max_seconds`` are decoded, so a long MSP-Podcast clip
        costs the same memory as a short RAVDESS one and no file is ever fully
        materialised beyond that bound.
        """
        path = self.resolve(relative_path)
        sf = _require_soundfile()

        with sf.SoundFile(str(path)) as handle:
            source_rate = int(handle.samplerate)
            frames = -1
            if max_seconds is not None:
                if max_seconds <= 0:
                    raise ValueError("max_seconds must be positive")
                frames = int(round(max_seconds * source_rate))
            waveform = handle.read(frames=frames, dtype="float32", always_2d=True)

        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        waveform = resample(waveform, source_rate, target_rate)
        return np.ascontiguousarray(waveform, dtype=np.float32)

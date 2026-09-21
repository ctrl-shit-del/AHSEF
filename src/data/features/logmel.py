"""Self-contained log-mel spectrogram features for the audio baseline.

Implemented on top of ``torch.stft`` and a hand-built HTK mel filterbank so the
audio baseline depends on nothing beyond core PyTorch.  That matters for
reproducibility: the same waveform yields bit-identical features on any machine
that can import torch, with no optional audio backend deciding the outcome and
no model downloads.

The extractor is deliberately compact -- 64 mel bands at a 10 ms hop over a
bounded clip -- because the baseline phase compares protocols, not front ends.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def hz_to_mel(frequency: np.ndarray | float) -> np.ndarray | float:
    """HTK mel scale."""
    return 2595.0 * np.log10(1.0 + np.asarray(frequency, dtype=np.float64) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float = 20.0,
    f_max: float | None = None,
) -> torch.Tensor:
    """Return a ``[n_mels, n_fft // 2 + 1]`` triangular mel filterbank."""
    if n_mels < 1:
        raise ValueError("n_mels must be at least 1")
    if n_fft < 2:
        raise ValueError("n_fft must be at least 2")
    f_max = float(f_max if f_max is not None else sample_rate / 2)
    if not 0.0 <= f_min < f_max <= sample_rate / 2:
        raise ValueError(
            f"require 0 <= f_min < f_max <= sample_rate/2; got {f_min}, {f_max}, {sample_rate}"
        )

    bins = np.linspace(0.0, sample_rate / 2, n_fft // 2 + 1, dtype=np.float64)
    edges = mel_to_hz(np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2))

    filters = np.zeros((n_mels, bins.size), dtype=np.float64)
    for index in range(n_mels):
        left, centre, right = edges[index], edges[index + 1], edges[index + 2]
        if centre <= left or right <= centre:  # pragma: no cover - degenerate config
            continue
        rising = (bins - left) / (centre - left)
        falling = (right - bins) / (right - centre)
        filters[index] = np.clip(np.minimum(rising, falling), 0.0, None)
    # Slaney-style area normalisation keeps band energies comparable.
    widths = edges[2:] - edges[:-2]
    filters *= np.where(widths > 0, 2.0 / widths, 0.0)[:, None]
    return torch.from_numpy(filters.astype(np.float32))


@dataclass(frozen=True)
class LogMelConfig:
    """Front-end geometry; recorded verbatim in every experiment artefact."""

    sample_rate: int = 16_000
    n_fft: int = 400            # 25 ms
    hop_length: int = 160       # 10 ms
    n_mels: int = 64
    f_min: float = 20.0
    f_max: float | None = None
    log_offset: float = 1e-6

    def frames_for(self, seconds: float) -> int:
        """Number of frames a clip of ``seconds`` produces (centred STFT)."""
        samples = int(round(seconds * self.sample_rate))
        return samples // self.hop_length + 1

    def to_dict(self) -> dict:
        return {
            "sample_rate": self.sample_rate,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "n_mels": self.n_mels,
            "f_min": self.f_min,
            "f_max": self.f_max,
            "log_offset": self.log_offset,
        }


class LogMelExtractor:
    """Turn a mono waveform into a ``[frames, n_mels]`` log-mel tensor."""

    def __init__(self, config: LogMelConfig | None = None):
        self.config = config or LogMelConfig()
        self._window = torch.hann_window(self.config.n_fft)
        self._filters = mel_filterbank(
            sample_rate=self.config.sample_rate,
            n_fft=self.config.n_fft,
            n_mels=self.config.n_mels,
            f_min=self.config.f_min,
            f_max=self.config.f_max,
        )

    @property
    def n_mels(self) -> int:
        return self.config.n_mels

    def __call__(self, waveform) -> torch.Tensor:
        config = self.config
        if not torch.is_tensor(waveform):
            waveform = torch.from_numpy(np.ascontiguousarray(waveform))
        waveform = waveform.to(torch.float32)
        if waveform.ndim == 2:
            # soundfile returns [frames, channels]; downmix to mono.
            waveform = waveform.mean(dim=1)
        elif waveform.ndim > 2:
            raise ValueError(f"Expected a mono or [frames, channels] waveform, got {tuple(waveform.shape)}")
        if waveform.numel() == 0:
            return torch.zeros(1, config.n_mels, dtype=torch.float32)
        # ``torch.stft`` reflection-pads by n_fft // 2; a shorter clip cannot be
        # reflected, so zero-pad it up to that minimum first.
        minimum = config.n_fft
        if waveform.numel() < minimum:
            waveform = torch.nn.functional.pad(waveform, (0, minimum - waveform.numel()))

        spectrum = torch.stft(
            waveform,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            win_length=config.n_fft,
            window=self._window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        power = spectrum.abs().pow(2)                      # [freq, frames]
        mel = self._filters @ power                        # [n_mels, frames]
        return torch.log(mel + config.log_offset).transpose(0, 1).contiguous()

    def fixed_length(self, waveform, frames: int) -> tuple[torch.Tensor, int]:
        """Return ``([frames, n_mels], valid_frames)`` by truncation or padding.

        A fixed frame count keeps every batch tensor the same shape, so the
        default collate applies and peak memory is predictable; the returned
        valid length lets the model pool over real frames only.
        """
        if frames < 1:
            raise ValueError("frames must be at least 1")
        features = self(waveform)
        valid = min(int(features.shape[0]), frames)
        if features.shape[0] >= frames:
            return features[:frames].contiguous(), valid
        padded = torch.zeros(frames, self.config.n_mels, dtype=torch.float32)
        padded[: features.shape[0]] = features
        return padded, valid

"""Deliberately small per-modality classification baselines.

These are baselines, not candidate final architectures.  Each one is the
smallest model that can honestly represent its modality on CPU, so that the
five reported numbers differ because of the modality and the protocol rather
than because one modality happened to get a bigger network.  Every classifier
emits exactly ``num_classes`` logits in the declared class order.

Shared shape contract for the sequence modalities::

    forward(features, lengths=None)
        features : [B, T, ...]
        lengths  : [B] long, the number of real (non-padded) steps

``lengths`` is what keeps padding out of the pooled representation.  Passing
``None`` pools over the full padded length, which is correct only when nothing
was padded.
"""

from __future__ import annotations

import torch
import torch.nn as nn


TEMPORAL_MODES = ("mean", "gru")


def sequence_mask(lengths: torch.Tensor, max_length: int) -> torch.Tensor:
    """Return a ``[B, T, 1]`` float mask that is 1 on real steps."""
    positions = torch.arange(max_length, device=lengths.device).unsqueeze(0)
    return (positions < lengths.clamp(min=1).unsqueeze(1)).unsqueeze(-1).to(torch.float32)


def masked_mean(features: torch.Tensor, lengths: torch.Tensor | None) -> torch.Tensor:
    """Mean over the time axis, ignoring padded steps."""
    if lengths is None:
        return features.mean(dim=1)
    mask = sequence_mask(lengths, features.shape[1])
    total = (features * mask).sum(dim=1)
    return total / mask.sum(dim=1).clamp(min=1.0)


def masked_mean_std(features: torch.Tensor, lengths: torch.Tensor | None) -> torch.Tensor:
    """Concatenated masked mean and standard deviation over the time axis."""
    mean = masked_mean(features, lengths)
    if lengths is None:
        variance = features.var(dim=1, unbiased=False)
    else:
        mask = sequence_mask(lengths, features.shape[1])
        centred = (features - mean.unsqueeze(1)) * mask
        variance = centred.pow(2).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    return torch.cat([mean, torch.sqrt(variance + 1e-8)], dim=-1)


class _Head(nn.Sequential):
    """Projection, normalisation, non-linearity -- the shared classifier body."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )


class AudioEmotionBaseline(nn.Module):
    """Log-mel statistics pooled over time, then a small MLP classifier.

    Mean-and-standard-deviation pooling is the classic lightweight utterance
    representation for speech emotion: it keeps the spectral profile and its
    variability while collapsing a variable-length clip to a fixed vector.
    """

    def __init__(self, n_mels: int = 64, hidden_dim: int = 256, num_classes: int = 7):
        super().__init__()
        if n_mels < 1:
            raise ValueError("n_mels must be at least 1")
        self.n_mels, self.num_classes = n_mels, num_classes
        self.encoder = _Head(2 * n_mels, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        pooled = masked_mean_std(features, lengths)
        return self.classifier(self.encoder(pooled))


class TextEmotionBaseline(nn.Module):
    """Hashed-token embeddings, masked mean pooling, then a small MLP.

    No pretrained weights and no downloads: the embedding table is learned from
    the training partition of this experiment alone.
    """

    def __init__(
        self,
        vocab_size: int = 32_768,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_classes: int = 7,
        padding_index: int = 0,
    ):
        super().__init__()
        if vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")
        self.vocab_size, self.embedding_dim = vocab_size, embedding_dim
        self.num_classes = num_classes
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_index)
        self.encoder = _Head(embedding_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        pooled = masked_mean(self.embedding(tokens), lengths)
        return self.classifier(self.encoder(pooled))


class VideoEmotionBaseline(nn.Module):
    """Per-frame projection followed by temporal pooling, then a classifier.

    ``temporal='mean'`` is the CPU-cheap default and keeps the video baseline
    directly comparable with the image baseline (the same per-frame encoder,
    averaged over frames).  ``temporal='gru'`` swaps in a single-layer GRU when
    the ordering of frames is worth paying for.
    """

    def __init__(
        self,
        frame_size: int = 48,
        hidden_dim: int = 256,
        num_classes: int = 7,
        temporal: str = "mean",
        channels: int = 3,
    ):
        super().__init__()
        if frame_size < 1:
            raise ValueError("frame_size must be at least 1")
        if temporal not in TEMPORAL_MODES:
            raise ValueError(f"temporal must be one of {TEMPORAL_MODES}, got {temporal!r}")
        self.frame_size, self.num_classes, self.temporal = frame_size, num_classes, temporal
        self.frame_dim = channels * frame_size * frame_size
        self.frame_encoder = _Head(self.frame_dim, hidden_dim)
        self.recurrent = (
            nn.GRU(hidden_dim, hidden_dim, batch_first=True) if temporal == "gru" else None
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, frames: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        batch, steps = frames.shape[0], frames.shape[1]
        encoded = self.frame_encoder(frames.reshape(batch * steps, -1)).reshape(batch, steps, -1)
        if self.recurrent is not None:
            encoded, _ = self.recurrent(encoded)
        return self.classifier(masked_mean(encoded, lengths))


class PhysiologyStateBaseline(nn.Module):
    """Standardised physiological window statistics into a small MLP."""

    def __init__(self, feature_dim: int, hidden_dim: int = 128, num_classes: int = 3):
        super().__init__()
        if feature_dim < 1:
            raise ValueError("feature_dim must be at least 1")
        self.feature_dim, self.num_classes = feature_dim, num_classes
        self.encoder = _Head(feature_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        return self.classifier(self.encoder(features))

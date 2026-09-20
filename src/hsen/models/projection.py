"""The Conv1D + positional-encoding front end that brings modalities to d=256.

Husformer's front end, and the reason the trunk needs no CTC alignment: three
streams arrive at different widths (768-d emotion2vec frames, 768-d RoBERTa
tokens, 576-d MobileNetV3 frames) and different rates (50 Hz audio, one token
per word-piece, a handful of video frames), and a per-modality temporal
convolution maps all of them to a common ``d`` while letting each keep its own
length.  From that point on the fusion trunk does not care which stream is
which, or that they were never sampled together.

Why a convolution rather than a linear layer.  A ``Linear`` would project each
frame independently; a ``Conv1d`` with kernel > 1 gives every projected frame a
small local context first, which is what lets a 50 Hz audio stream and a
one-per-word text stream meet at a comparable level of abstraction rather than
at a comparable dimensionality only.

Why d = 256.  ESED Table 7 measures it: 128 / 256 / 512 give 67.13 / 69.54 /
71.07 with a small encoder and 71.25 / **72.66** / 70.95 with a large one, so
256 is the accuracy-optimal choice and not merely the affordable one.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal position signal, added after projection.

    Parameter-free on purpose.  A learned table would have to declare a maximum
    length up front and would be one more thing that differs between a
    10-second audio stream and a 12-token transcript; the sinusoidal form
    extends to any length the cache happens to hold, and costs nothing.
    """

    def __init__(self, dim: int, max_length: int = 4096):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"Positional encoding needs an even dimension, got {dim}")
        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        scale = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10_000.0) / dim)
        )
        table = torch.zeros(max_length, dim)
        table[:, 0::2] = torch.sin(position * scale)
        table[:, 1::2] = torch.cos(position * scale)
        self.register_buffer("table", table, persistent=False)
        self.dim, self.max_length = dim, max_length

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """``[B, L, d]`` -> ``[B, L, d]`` with position added."""
        length = sequence.shape[1]
        if length > self.max_length:
            raise ValueError(
                f"Sequence of length {length} exceeds the positional table "
                f"({self.max_length}); raise max_length or lower the encoder's "
                f"max_frames"
            )
        return sequence + self.table[:length].unsqueeze(0)


class ModalityProjection(nn.Module):
    """Project one modality's cached features into the shared ``d``-space."""

    def __init__(
        self,
        input_dim: int,
        model_dim: int = 256,
        kernel_size: int = 3,
        dropout: float = 0.1,
        max_length: int = 4096,
    ):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError(f"kernel_size must be odd for 'same' padding, got {kernel_size}")
        self.input_dim, self.model_dim, self.kernel_size = input_dim, model_dim, kernel_size

        # Cached features are raw frozen-encoder activations whose scale varies
        # a lot between modalities -- RoBERTa hidden states and emotion2vec
        # frames are not on remotely the same magnitude. Normalising here stops
        # one branch dominating the fusion for reasons of scale rather than
        # usefulness, and it is a per-sample statistic, so it leaks nothing
        # between splits the way a fitted feature mean would.
        self.norm = nn.LayerNorm(input_dim)
        self.conv = nn.Conv1d(
            input_dim, model_dim, kernel_size=kernel_size, padding=kernel_size // 2, bias=False,
        )
        self.positional = SinusoidalPositionalEncoding(model_dim, max_length=max_length)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """``[B, L, input_dim]`` -> ``[B, L, model_dim]``.

        ``mask`` is ``True`` at valid positions.  Padding is zeroed before the
        convolution as well as after, because a kernel wider than one frame
        would otherwise smear whatever happens to sit in the padded tail back
        into the last real frame.
        """
        if features.dim() != 3:
            raise ValueError(f"Expected [B, L, D], got {tuple(features.shape)}")
        if features.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim}-d features, got {features.shape[-1]}"
            )
        if mask is not None:
            features = features * mask.unsqueeze(-1).to(features.dtype)

        projected = self.conv(self.norm(features).transpose(1, 2)).transpose(1, 2)
        projected = self.positional(projected)
        if mask is not None:
            projected = projected * mask.unsqueeze(-1).to(projected.dtype)
        return self.dropout(projected)

    def describe(self) -> dict:
        return {
            "class": "ModalityProjection",
            "input_dim": self.input_dim,
            "model_dim": self.model_dim,
            "kernel_size": self.kernel_size,
            "parameters": int(sum(p.numel() for p in self.parameters())),
        }

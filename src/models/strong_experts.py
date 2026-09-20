"""Probe heads over frozen pretrained encoder representations.

The five Layer-3 baselines are trained from scratch with no pretrained weights.
These heads answer a different question: given a *better representation*, how
much more can the same task extract?  The encoder is frozen and lives outside
this module (see :mod:`src.data.features.wav2vec2_features`); what is trained
here is only the small model that reads its output.

Keeping the two apart is what makes the comparison clean.  The representation is
identical across every head experiment, so a difference between two runs is a
difference between heads, not between two accidentally different feature passes.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LayerWeightedProbe(nn.Module):
    """Learned softmax weighting over encoder layers, then an MLP classifier.

    This is the standard SUPERB probing head, and the layer weighting is the
    part that matters.  A self-supervised speech encoder distributes information
    very unevenly across depth -- lower layers carry speaker and channel
    characteristics, middle layers carry phonetic content, upper layers drift
    toward the pretraining objective -- and which of those helps emotion
    recognition is an empirical question.  Fixing a layer in advance would be a
    guess; learning ``softmax(w)`` over all of them lets validation answer it,
    and the fitted weights are readable afterwards as a description of where the
    useful signal was.

    Input is ``[batch, num_layers, feature_dim]``: the per-layer time-means the
    extractor cached.  Time pooling has already happened, so this module never
    sees padding and needs no mask.
    """

    def __init__(
        self,
        num_layers: int = 12,
        feature_dim: int = 768,
        hidden_dim: int = 256,
        num_classes: int = 7,
        dropout: float = 0.2,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if feature_dim < 1:
            raise ValueError("feature_dim must be at least 1")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.num_layers = num_layers
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout = dropout

        # Initialised uniform, so before training every layer contributes
        # equally and the fitted weights are a finding rather than a prior.
        self.layer_logits = nn.Parameter(torch.zeros(num_layers))
        # The cached features are raw encoder activations whose scale varies a
        # lot by layer; normalising before the MLP stops one layer dominating
        # for reasons of magnitude rather than usefulness.
        self.norm = nn.LayerNorm(feature_dim)
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def layer_weights(self) -> torch.Tensor:
        """The fitted, normalised contribution of each encoder layer."""
        return torch.softmax(self.layer_logits, dim=0)

    def forward(
        self, features: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``[batch, num_layers, feature_dim]`` -> ``[batch, num_classes]``.

        ``lengths`` is accepted and ignored: the shared trainer passes it for
        sequence modalities, and the cached features are already time-pooled.
        """
        if features.dim() != 3:
            raise ValueError(
                f"Expected [batch, num_layers, feature_dim], got {tuple(features.shape)}"
            )
        if features.shape[1] != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} encoder layers, got {features.shape[1]}"
            )
        weights = self.layer_weights().view(1, -1, 1)
        pooled = (features * weights).sum(dim=1)
        return self.classifier(self.encoder(self.norm(pooled)))

    def describe(self) -> dict:
        return {
            "class": "LayerWeightedProbe",
            "num_layers": self.num_layers,
            "feature_dim": self.feature_dim,
            "hidden_dim": self.hidden_dim,
            "num_classes": self.num_classes,
            "dropout": self.dropout,
            "parameters": int(sum(p.numel() for p in self.parameters())),
            "trainable_parameters": int(
                sum(p.numel() for p in self.parameters() if p.requires_grad)
            ),
            "pooling": "learned softmax weighting over encoder layers",
            "encoder_frozen": True,
            "encoder_trained_by_this_project": False,
        }

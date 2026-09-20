"""Controlled fusion ablations, for section 13 of the phase brief.

Three variants behind one interface, so an ablation changes a config string and
nothing else.  Everything outside the fusion trunk -- the cached features, the
projections, the heads, the loss, the metrics, the split -- is byte-identical
across the three, which is what makes the comparison an ablation rather than
three separate experiments.

``concat``
    Husformer's ``HusFuse`` minus even the self-attention: mean-pool each
    modality, concatenate, project.  No cross-modal interaction at all. The
    floor.

``self_attention``
    Concatenate the streams along time and run the same self-attention stack
    Husformer ends with, but with no fusion-to-modality cross-attention.  This
    is the variant that isolates the paper's actual claim: Husformer's own
    Table V puts it 9.9 points below full Husformer on WESAD, and if the
    topology transfers to audio/video/text we should see the same ordering.

``husformer``
    The full topology.  The default, and the one the survey recommends.

Note what these ablations deliberately hold constant: layer count and head
count.  A concatenation baseline with more capacity elsewhere would confound
"the mechanism helps" with "these parameters help".
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.hsen.models.husformer import HusformerFusion, SelfAttentionLayer, masked_mean


class ConcatFusion(nn.Module):
    """Mean-pool each modality, concatenate, project back to ``d``.

    The absent-modality rule matters more here than anywhere else: a missing
    branch contributes a zero block to the concatenation, which a linear layer
    can and does learn to read as a distinct signal.  The availability mask is
    passed to the projection as explicit input instead, so "absent" is a stated
    fact rather than something the model has to infer from a suspicious run of
    zeros.
    """

    def __init__(self, modalities: tuple[str, ...], model_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.modalities = tuple(modalities)
        self.model_dim = model_dim
        width = model_dim * len(self.modalities) + len(self.modalities)
        self.project = nn.Sequential(
            nn.Linear(width, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim),
        )
        self.output_norm = nn.LayerNorm(model_dim)

    def forward(self, projected, masks, available) -> dict[str, torch.Tensor]:
        present = [m for m in self.modalities if m in projected]
        effective = {m: masks[m] & available[m].unsqueeze(1) for m in present}
        vectors = {m: masked_mean(projected[m], effective[m]) for m in present}
        flags = torch.stack([available[m].float() for m in present], dim=1)
        pooled = torch.cat([vectors[m] for m in present] + [flags], dim=-1)
        fused = self.output_norm(self.project(pooled))
        return {
            "fused": fused,
            "fused_sequence": fused.unsqueeze(1),
            "fused_mask": torch.ones(fused.shape[0], 1, dtype=torch.bool, device=fused.device),
            "modality_sequences": {m: projected[m] for m in present},
            "modality_vectors": vectors,
            "effective_masks": effective,
        }

    def describe(self) -> dict:
        return {
            "class": "ConcatFusion",
            "topology": "mean-pool per modality, concatenate, MLP",
            "modalities": list(self.modalities),
            "model_dim": self.model_dim,
            "cross_modal_attention": False,
            "parameters": int(sum(p.numel() for p in self.parameters())),
        }


class SelfAttentionFusion(nn.Module):
    """Concatenate the streams along time and self-attend.  No cross-modal stage."""

    def __init__(
        self,
        modalities: tuple[str, ...],
        model_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        ffn_multiplier: int = 4,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
    ):
        super().__init__()
        self.modalities = tuple(modalities)
        self.model_dim, self.num_layers, self.num_heads = model_dim, num_layers, num_heads
        self.layers = nn.ModuleList([
            SelfAttentionLayer(
                model_dim, num_heads, model_dim * ffn_multiplier, dropout, attention_dropout,
            )
            for _ in range(num_layers)
        ])
        self.output_norm = nn.LayerNorm(model_dim)

    def forward(self, projected, masks, available) -> dict[str, torch.Tensor]:
        present = [m for m in self.modalities if m in projected]
        effective = {m: masks[m] & available[m].unsqueeze(1) for m in present}
        combined = torch.cat([projected[m] for m in present], dim=1)
        combined_mask = torch.cat([effective[m] for m in present], dim=1)
        for layer in self.layers:
            combined = layer(combined, ~combined_mask)
        combined = self.output_norm(combined)

        # Slice the attended stream back apart so per-modality representations
        # stay available under this variant too; the uncertainty work reads them
        # regardless of which fusion produced them.
        sequences, offset = {}, 0
        for modality in present:
            length = projected[modality].shape[1]
            sequences[modality] = combined[:, offset:offset + length]
            offset += length
        return {
            "fused": masked_mean(combined, combined_mask),
            "fused_sequence": combined,
            "fused_mask": combined_mask,
            "modality_sequences": sequences,
            "modality_vectors": {
                m: masked_mean(sequences[m], effective[m]) for m in present
            },
            "effective_masks": effective,
        }

    def describe(self) -> dict:
        return {
            "class": "SelfAttentionFusion",
            "topology": "concatenate along time, self-attention only",
            "modalities": list(self.modalities),
            "model_dim": self.model_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "cross_modal_attention": False,
            "parameters": int(sum(p.numel() for p in self.parameters())),
        }


FUSION_VARIANTS = ("husformer", "self_attention", "concat")


def build_fusion(
    variant: str,
    modalities: tuple[str, ...],
    model_dim: int = 256,
    num_layers: int = 2,
    num_heads: int = 4,
    ffn_multiplier: int = 4,
    dropout: float = 0.1,
    attention_dropout: float = 0.1,
) -> nn.Module:
    """Build one fusion trunk by name."""
    if variant == "husformer":
        return HusformerFusion(
            modalities, model_dim, num_layers, num_heads, ffn_multiplier,
            dropout, attention_dropout,
        )
    if variant == "self_attention":
        return SelfAttentionFusion(
            modalities, model_dim, num_layers, num_heads, ffn_multiplier,
            dropout, attention_dropout,
        )
    if variant == "concat":
        return ConcatFusion(modalities, model_dim, dropout)
    raise ValueError(
        f"Unknown fusion variant {variant!r}; expected one of {list(FUSION_VARIANTS)}"
    )

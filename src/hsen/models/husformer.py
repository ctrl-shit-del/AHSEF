"""Husformer fusion: every modality attends to one concatenated fusion stream.

The topology, from Husformer Figure 1:

    Y_F = concat(Y_1 .. Y_M)                    along time -- low-level fusion
    Z_m = CrossModalTransformer(target=Y_m, source=Y_F)     for each modality m
    Z_F = SelfAttentionTransformer(concat(Z_1 .. Z_M))

The contribution is the *source*.  Each modality attends to a single
concatenated fusion representation instead of to every other modality pairwise,
so cost grows linearly in the modality count rather than quadratically -- and
the paper's own ablation says that is not merely cheaper but better: 78.68
against 73.57 accuracy on WESAD, at 0.71 M parameters against HusPair's 3.90 M
and 3,084 MB against 5,417 MB.  Removing cross-modal attention altogether
(concatenation plus self-attention, ``HusFuse``) costs 9.9 points, so the
mechanism is doing the work rather than the depth.

Unaligned streams, no CTC.  Nothing here requires the three sequences to have
equal length or to correspond frame-for-frame; they are concatenated along time
and attended over. LDDU reports the same conclusion independently -- its
unaligned CMU-MOSEI numbers beat its aligned ones.

Missing modalities.  A modality absent for a sample contributes no valid
positions to ``Y_F`` and its ``Z_m`` is excluded from pooling, rather than being
represented by a zero vector.  A zero vector is a *statement* -- it says "this
modality was observed and was silent" -- and the whole point of the later
routing work is to distinguish that from "this modality was never acquired".
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CrossModalAttentionLayer(nn.Module):
    """One pre-norm cross-attention block: target queries, source keys/values."""

    def __init__(self, model_dim: int, num_heads: int, ffn_dim: int, dropout: float,
                 attention_dropout: float):
        super().__init__()
        self.target_norm = nn.LayerNorm(model_dim)
        self.source_norm = nn.LayerNorm(model_dim)
        self.attention = nn.MultiheadAttention(
            model_dim, num_heads, dropout=attention_dropout, batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, model_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, target: torch.Tensor, source: torch.Tensor, source_padding: torch.Tensor | None,
    ) -> torch.Tensor:
        normed_target = self.target_norm(target)
        normed_source = self.source_norm(source)
        attended, _ = self.attention(
            normed_target, normed_source, normed_source,
            key_padding_mask=source_padding, need_weights=False,
        )
        target = target + self.dropout(attended)
        return target + self.ffn(self.ffn_norm(target))


class SelfAttentionLayer(nn.Module):
    """One pre-norm self-attention block over the concatenated modality outputs."""

    def __init__(self, model_dim: int, num_heads: int, ffn_dim: int, dropout: float,
                 attention_dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(model_dim)
        self.attention = nn.MultiheadAttention(
            model_dim, num_heads, dropout=attention_dropout, batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, model_dim),
            nn.Dropout(dropout),
        )

    def forward(self, sequence: torch.Tensor, padding: torch.Tensor | None) -> torch.Tensor:
        normed = self.norm(sequence)
        attended, _ = self.attention(
            normed, normed, normed, key_padding_mask=padding, need_weights=False,
        )
        sequence = sequence + self.dropout(attended)
        return sequence + self.ffn(self.ffn_norm(sequence))


def masked_mean(sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over valid positions only.

    Rows with no valid position -- an absent modality -- return zeros rather
    than dividing by zero.  Those rows are separately excluded from the fused
    representation by the availability mask, so the zero is never read as a
    measurement.
    """
    valid = mask.unsqueeze(-1).to(sequence.dtype)
    total = (sequence * valid).sum(dim=1)
    count = valid.sum(dim=1).clamp(min=1.0)
    return total / count


class HusformerFusion(nn.Module):
    """Fusion-to-modality cross-attention followed by one self-attention stage.

    Returns per-modality representations as well as the fused one.  That is a
    deliberate interface decision, not spare output: ESED's per-modality
    evidence heads, per-modality vacuity, and HSIG's expected-gain features all
    read exactly these, and exposing them now means the uncertainty phase
    attaches to HSEN rather than rewriting it.
    """

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
        if not modalities:
            raise ValueError("HusformerFusion needs at least one modality")
        if model_dim % num_heads != 0:
            raise ValueError(
                f"model_dim {model_dim} must divide evenly into {num_heads} heads"
            )
        self.modalities = tuple(modalities)
        self.model_dim, self.num_layers, self.num_heads = model_dim, num_layers, num_heads
        ffn_dim = model_dim * ffn_multiplier

        # One cross-modal stack per modality: each learns how to read the shared
        # fusion stream from its own point of view.
        self.cross = nn.ModuleDict({
            modality: nn.ModuleList([
                CrossModalAttentionLayer(model_dim, num_heads, ffn_dim, dropout, attention_dropout)
                for _ in range(num_layers)
            ])
            for modality in self.modalities
        })
        self.cross_norm = nn.ModuleDict({
            modality: nn.LayerNorm(model_dim) for modality in self.modalities
        })
        self.self_attention = nn.ModuleList([
            SelfAttentionLayer(model_dim, num_heads, ffn_dim, dropout, attention_dropout)
            for _ in range(num_layers)
        ])
        self.output_norm = nn.LayerNorm(model_dim)

    def forward(
        self,
        projected: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
        available: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Fuse the projected streams.

        ``projected[m]``  ``[B, L_m, d]`` -- output of the modality projection.
        ``masks[m]``      ``[B, L_m]`` bool, ``True`` at real frames.
        ``available[m]``  ``[B]`` bool, whether the sample has this modality at all.
        """
        present = [m for m in self.modalities if m in projected]
        if not present:
            raise ValueError("No modality streams were supplied to the fusion trunk")

        # A frame counts only if it is a real frame *and* the sample actually has
        # the modality. Folding availability into the frame mask here means one
        # rule governs both padding and absence, so the two can never disagree.
        effective = {
            modality: masks[modality] & available[modality].unsqueeze(1)
            for modality in present
        }

        if not bool((torch.stack([effective[m].any(dim=1) for m in present], dim=1)).any(dim=1).all()):
            raise ValueError(
                "At least one sample in this batch has no available modality at all. "
                "A sample with nothing observed has nothing to predict from and "
                "should have been excluded by the manifest, not zero-filled here."
            )

        # ---- low-level fusion: one concatenated stream over time -------------
        fusion = torch.cat([projected[modality] for modality in present], dim=1)
        fusion_mask = torch.cat([effective[modality] for modality in present], dim=1)
        # MultiheadAttention's key_padding_mask marks positions to *ignore*.
        fusion_padding = ~fusion_mask

        # ---- fusion-to-modality cross-attention ------------------------------
        modality_sequences: dict[str, torch.Tensor] = {}
        for modality in present:
            sequence = projected[modality]
            for layer in self.cross[modality]:
                sequence = layer(sequence, fusion, fusion_padding)
            modality_sequences[modality] = self.cross_norm[modality](sequence)

        # ---- single self-attention stage over the concatenated outputs -------
        combined = torch.cat([modality_sequences[modality] for modality in present], dim=1)
        combined_mask = torch.cat([effective[modality] for modality in present], dim=1)
        for layer in self.self_attention:
            combined = layer(combined, ~combined_mask)
        combined = self.output_norm(combined)

        modality_vectors = {
            modality: masked_mean(modality_sequences[modality], effective[modality])
            for modality in present
        }
        return {
            "fused": masked_mean(combined, combined_mask),
            "fused_sequence": combined,
            "fused_mask": combined_mask,
            "modality_sequences": modality_sequences,
            "modality_vectors": modality_vectors,
            "effective_masks": effective,
        }

    def describe(self) -> dict:
        return {
            "class": "HusformerFusion",
            "topology": "fusion-to-modality cross-attention + one self-attention stage",
            "modalities": list(self.modalities),
            "model_dim": self.model_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "ctc_alignment": False,
            "parameters": int(sum(p.numel() for p in self.parameters())),
        }

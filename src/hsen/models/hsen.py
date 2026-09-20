"""HSEN: the shared multimodal trunk and its three heads.

    cached features -> projection (d=256) -> fusion trunk -> {emotion, valence, arousal}

The state representation this phase targets is ``[valence, arousal, emotion]``.
Uncertainty is the next phase's fourth channel, and the forward pass is built so
that it can be attached rather than retrofitted: :meth:`HSEN.forward` returns
every intermediate the ESED/LDDU work and the routing work will need --
projected features, per-modality sequences, per-modality pooled vectors, the
fused representation, and the head outputs -- under stable keys.

What is deliberately *not* here: no evidential Dirichlet head, no vacuity, no
ordinality calibration, no HSIG, no UGAPR.  Section 24 asks that they be
attachable later, not present now, and a half-built uncertainty head would be
the kind of thing that quietly changes the baseline it is supposed to be
measured against.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn

from src.common.labels import LabelSpace, get_label_space
from src.hsen.models.fusion_variants import FUSION_VARIANTS, build_fusion
from src.hsen.models.projection import ModalityProjection

#: Feature width each frozen encoder writes into the cache.  Declared here so
#: the model can be rebuilt from its config alone, without opening a cache.
DEFAULT_INPUT_DIMS: dict[str, int] = {
    "audio": 768,   # emotion2vec base
    "text": 768,    # RoBERTa-base
    "video": 576,   # MobileNetV3-Small final feature map, pooled
}

MODALITY_ORDER: tuple[str, ...] = ("audio", "video", "text")


@dataclass
class HSENConfig:
    """Everything that determines the architecture.

    Saved beside every checkpoint, and the *only* thing needed to rebuild the
    model.  Section 23 requires the CPU and CUDA profiles to instantiate an
    identical architecture; the enforcement is that both build from this object
    and neither profile is allowed to touch a field of it.
    """

    modalities: tuple[str, ...] = MODALITY_ORDER
    num_classes: int = 6
    label_space: str = "iemocap_erc6"
    task: str = "single_label"          # or "multi_label"
    model_dim: int = 256
    num_layers: int = 2
    num_heads: int = 4
    ffn_multiplier: int = 4
    conv_kernel: int = 3
    dropout: float = 0.1
    attention_dropout: float = 0.1
    fusion: str = "husformer"
    #: Per-modality auxiliary classifier heads.  Off by default: this phase's
    #: baseline is the architecture the survey specifies, and an auxiliary loss
    #: nobody asked for would make it something else. ESED's per-modality
    #: evidence heads attach here when the uncertainty phase starts.
    modality_heads: bool = False
    predict_valence: bool = True
    predict_arousal: bool = True
    input_dims: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_INPUT_DIMS))
    max_length: int = 4096

    def __post_init__(self) -> None:
        self.modalities = tuple(self.modalities)
        if not self.modalities:
            raise ValueError("HSEN needs at least one modality")
        unknown = set(self.modalities) - set(DEFAULT_INPUT_DIMS)
        if unknown:
            raise ValueError(f"Unknown modalities {sorted(unknown)}")
        if self.fusion not in FUSION_VARIANTS:
            raise ValueError(
                f"Unknown fusion {self.fusion!r}; expected one of {list(FUSION_VARIANTS)}"
            )
        if self.task not in ("single_label", "multi_label"):
            raise ValueError(f"Unknown task {self.task!r}")
        if self.model_dim % self.num_heads:
            raise ValueError(
                f"model_dim {self.model_dim} must divide evenly into "
                f"{self.num_heads} heads"
            )
        missing = [m for m in self.modalities if m not in self.input_dims]
        if missing:
            raise ValueError(f"No input dimension declared for {missing}")

    def resolved_label_space(self) -> LabelSpace | None:
        try:
            return get_label_space(self.label_space)
        except ValueError:
            return None

    def to_dict(self) -> dict:
        return asdict(self) | {"modalities": list(self.modalities)}


class RegressionHead(nn.Module):
    """A small MLP onto one bounded affect dimension.

    ``tanh`` because both targets are defined on ``[-1, 1]`` by construction --
    IEMOCAP's 1-5 ratings and CMU-MOSEI's -3..3 sentiment are both mapped there
    by a fixed transform -- so an unbounded head would spend capacity learning a
    range the label adapter already guarantees.
    """

    def __init__(self, model_dim: int, hidden_dim: int | None = None, dropout: float = 0.1):
        super().__init__()
        hidden_dim = hidden_dim or model_dim // 2
        self.net = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class HSEN(nn.Module):
    """The always-on multimodal Human State Estimation Network."""

    def __init__(self, config: HSENConfig | None = None):
        super().__init__()
        self.config = config or HSENConfig()

        self.projections = nn.ModuleDict({
            modality: ModalityProjection(
                input_dim=self.config.input_dims[modality],
                model_dim=self.config.model_dim,
                kernel_size=self.config.conv_kernel,
                dropout=self.config.dropout,
                max_length=self.config.max_length,
            )
            for modality in self.config.modalities
        })
        self.fusion = build_fusion(
            variant=self.config.fusion,
            modalities=self.config.modalities,
            model_dim=self.config.model_dim,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            ffn_multiplier=self.config.ffn_multiplier,
            dropout=self.config.dropout,
            attention_dropout=self.config.attention_dropout,
        )

        self.trunk_dropout = nn.Dropout(self.config.dropout)
        self.emotion_head = nn.Linear(self.config.model_dim, self.config.num_classes)
        self.valence_head = (
            RegressionHead(self.config.model_dim, dropout=self.config.dropout)
            if self.config.predict_valence else None
        )
        self.arousal_head = (
            RegressionHead(self.config.model_dim, dropout=self.config.dropout)
            if self.config.predict_arousal else None
        )
        self.modality_classifiers = nn.ModuleDict({
            modality: nn.Linear(self.config.model_dim, self.config.num_classes)
            for modality in self.config.modalities
        }) if self.config.modality_heads else None

    # ---------------------------------------------------------------- forward

    def forward(
        self,
        features: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
        available: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Run the trunk and every head.

        ``features[m]``   ``[B, L_m, D_m]`` cached frozen-encoder features.
        ``masks[m]``      ``[B, L_m]`` bool, ``True`` at real frames.
        ``available[m]``  ``[B]`` bool, whether the sample carries the modality.

        Returns a dict rather than a tensor.  Section 24 needs the intermediates
        and a later phase needs them under names that do not move, so the return
        shape is part of the interface, not an implementation detail.
        """
        streams = [m for m in self.config.modalities if m in features]
        if not streams:
            raise ValueError(
                f"No modality features supplied; the model expects "
                f"{list(self.config.modalities)}"
            )

        projected = {
            modality: self.projections[modality](
                features[modality], masks[modality] & available[modality].unsqueeze(1)
            )
            for modality in streams
        }
        fused = self.fusion(projected, masks, available)

        trunk = self.trunk_dropout(fused["fused"])
        outputs: dict[str, torch.Tensor] = {
            "logits": self.emotion_head(trunk),
            "fused": fused["fused"],
            "fused_sequence": fused["fused_sequence"],
            "fused_mask": fused["fused_mask"],
            "projected": projected,
            "modality_sequences": fused["modality_sequences"],
            "modality_vectors": fused["modality_vectors"],
            "effective_masks": fused["effective_masks"],
        }
        if self.valence_head is not None:
            outputs["valence"] = self.valence_head(trunk)
        if self.arousal_head is not None:
            outputs["arousal"] = self.arousal_head(trunk)
        if self.modality_classifiers is not None:
            outputs["modality_logits"] = {
                modality: self.modality_classifiers[modality](fused["modality_vectors"][modality])
                for modality in streams
            }
        return outputs

    @torch.inference_mode()
    def predict_state(
        self,
        features: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
        available: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """The AHSEF state vector: ``[valence, arousal, emotion]``.

        The fourth channel -- uncertainty -- is added by the next phase, which
        is why this returns a dict and not a fixed-width tensor.
        """
        outputs = self.forward(features, masks, available)
        probabilities = (
            torch.softmax(outputs["logits"], dim=-1)
            if self.config.task == "single_label"
            else torch.sigmoid(outputs["logits"])
        )
        state = {
            "emotion": probabilities.argmax(dim=-1)
            if self.config.task == "single_label" else (probabilities > 0.5).long(),
            "emotion_probabilities": probabilities,
        }
        for name in ("valence", "arousal"):
            if name in outputs:
                state[name] = outputs[name]
        return state

    # ------------------------------------------------------------- provenance

    def parameter_counts(self) -> dict[str, int]:
        def count(module: nn.Module | None) -> int:
            return int(sum(p.numel() for p in module.parameters())) if module else 0

        return {
            "total": int(sum(p.numel() for p in self.parameters())),
            "trainable": int(sum(p.numel() for p in self.parameters() if p.requires_grad)),
            "projections": count(self.projections),
            "fusion": count(self.fusion),
            "emotion_head": count(self.emotion_head),
            "valence_head": count(self.valence_head),
            "arousal_head": count(self.arousal_head),
            "modality_classifiers": count(self.modality_classifiers),
        }

    def describe(self) -> dict:
        return {
            "class": "HSEN",
            "config": self.config.to_dict(),
            "projections": {
                modality: module.describe()
                for modality, module in self.projections.items()
            },
            "fusion": self.fusion.describe(),
            "parameters": self.parameter_counts(),
            "encoders_frozen": True,
            "encoders_trained_by_this_project": False,
            "state_representation": ["valence", "arousal", "emotion"],
            "uncertainty_implemented": False,
        }


def build_hsen(config: HSENConfig | None = None) -> HSEN:
    """Single construction path, used by both execution profiles.

    Both ``train_hsen_cuda.py`` and ``train_hsen_cpu25.py`` reach the model
    through here.  There is deliberately no second constructor a "lighter" CPU
    variant could be slipped into.
    """
    return HSEN(config or HSENConfig())

"""What every frozen HSEN encoder must look like from the outside.

Three encoders, three completely different runtimes -- a HuggingFace
transformer, a FunASR speech model, a torchvision CNN behind a face detector --
and the extraction driver should not know which is which.  So they meet at one
narrow interface: hand it a list of payloads, get back one ``[T, D]`` float32
array per payload, in order.

The spec is the other half.  Its fingerprint goes into the cache index, so a
cache built with 4-second audio crops can never be silently extended with
8-second ones; changing any field that affects the numbers changes the
fingerprint and the store refuses the mismatch on open.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np

from src.hsen.features.store import config_fingerprint


@dataclass(frozen=True)
class EncoderSpec:
    """Everything about an encoder that changes the features it produces.

    Everything that does *not* change the numbers -- batch size, thread count,
    progress interval -- is deliberately absent, because putting it here would
    invalidate a perfectly good cache the moment someone ran the extractor with
    a different batch size.
    """

    modality: str
    #: The checkpoint identifier, e.g. ``roberta-base``.
    model: str
    #: Width of one frame of output.
    feature_dim: int
    #: Hard cap on sequence length.  Bounds both the cache and the attention
    #: cost of the fusion trunk, and is a real modelling choice, so it is part
    #: of the fingerprint.
    max_frames: int
    #: Which layer's activations are taken; ``-1`` is the final hidden state.
    layer: int = -1
    #: Encoder-specific settings that affect the output.
    options: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        return config_fingerprint(self.to_dict())

    def describe(self) -> dict:
        return self.to_dict() | {"fingerprint": self.fingerprint}


@runtime_checkable
class FrozenEncoder(Protocol):
    """A frozen encoder that turns raw payloads into cacheable sequences."""

    spec: EncoderSpec

    def encode(self, payloads: Sequence[Any]) -> list[np.ndarray]:
        """One ``[T, D]`` float32 array per payload, in the order given.

        Implementations raise on a payload they cannot encode rather than
        returning a zero array.  A zero-filled feature is indistinguishable
        downstream from a genuinely silent recording, and the difference matters
        to every modality-availability number this phase reports.
        """

    def describe(self) -> dict:
        """Provenance recorded beside the cache."""


def assert_frozen(module) -> None:
    """Fail loudly if any parameter of an encoder is still trainable.

    Cheap to call and worth calling: a single ``requires_grad=True`` left on an
    encoder turns a frozen-feature experiment into a partially fine-tuned one,
    and the only visible symptom is that the numbers are better than they should
    be.
    """
    trainable = [name for name, parameter in module.named_parameters() if parameter.requires_grad]
    if trainable:
        raise RuntimeError(
            f"{type(module).__name__} has {len(trainable)} trainable parameters "
            f"(e.g. {trainable[:3]}); HSEN encoders are frozen in this phase"
        )

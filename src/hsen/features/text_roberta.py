"""The frozen text branch: RoBERTa-base over gold transcripts.

The survey's choice, on two independent pieces of evidence: it is ESED's text
encoder, and ESED Table 6 plus MPLMM Table 1 both put text well ahead of the
other two modalities (69.74 weighted F1 text-only against 53.62 audio and 50.13
video on IEMOCAP).  That makes the text branch the one most worth getting right,
and the cheapest to run.

Gold transcripts, never ASR.  IEMOCAP ships human transcriptions per utterance
and CMU-MOSEI ships them in ``label.csv``; both are read straight from the
manifest's ``text`` column.  Running ASR here would add a second error source
to a branch whose whole value is that it is clean, and would do it once per
epoch for no benefit.

Token sequences, not a pooled vector.  ``[CLS]`` alone would collapse the
sequence before the fusion trunk ever sees it, and cross-modal attention over a
one-token "sequence" is just a linear layer.  The cache therefore holds the full
per-token hidden states, capped at :attr:`~src.hsen.features.base.EncoderSpec.max_frames`.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from src.hsen.features.base import EncoderSpec, assert_frozen

DEFAULT_MODEL = "roberta-base"
DEFAULT_FEATURE_DIM = 768

#: 64 word-piece tokens.  IEMOCAP utterances are short -- the median is around
#: 12 tokens and the 99th percentile is under 60 -- so this truncates almost
#: nothing while keeping the cached sequences small.  CMU-MOSEI's are longer and
#: the extractor reports how many samples it truncated so the choice stays
#: visible rather than silently discarding half a transcript.
DEFAULT_MAX_TOKENS = 64


class TextEncodingError(RuntimeError):
    """Raised when a transcript cannot be encoded."""


class RoBERTaTextEncoder:
    """Frozen RoBERTa-base, cached as per-token hidden states."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        layer: int = -1,
        device: str = "cpu",
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.max_tokens = int(max_tokens)
        self.layer = int(layer)
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self._model = None
        self._tokenizer = None
        self.truncated = 0
        self.spec = EncoderSpec(
            modality="text",
            model=model_name,
            feature_dim=DEFAULT_FEATURE_DIM,
            max_frames=self.max_tokens,
            layer=self.layer,
            options={"tokens": "word-piece hidden states, special tokens kept"},
        )

    # ------------------------------------------------------------------ load

    def _load(self) -> None:
        """Load on first use, so constructing an extractor stays cheap."""
        if self._model is not None:
            return
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as error:  # pragma: no cover - environment dependent
            raise TextEncodingError(
                "transformers is required for the RoBERTa text branch.\n"
                "    pip install transformers"
            ) from error

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name, add_pooling_layer=False)
        model.eval().to(self.device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        assert_frozen(model)
        self._model = model

        hidden = int(model.config.hidden_size)
        if hidden != self.spec.feature_dim:
            raise TextEncodingError(
                f"{self.model_name} produces {hidden}-d hidden states but the "
                f"encoder spec declares {self.spec.feature_dim}. Update the spec "
                f"rather than reshaping the features."
            )

    # ---------------------------------------------------------------- encode

    @torch.inference_mode()
    def encode(self, payloads: Sequence[str]) -> list[np.ndarray]:
        """Transcripts to ``[T, 768]`` float32 arrays, one per transcript."""
        self._load()

        texts = []
        for index, payload in enumerate(payloads):
            text = "" if payload is None else str(payload).strip()
            if not text:
                raise TextEncodingError(
                    f"Payload {index} is an empty transcript. The manifest audit "
                    f"declares text available for every retained row, so an empty "
                    f"one means the manifest and the data disagree."
                )
            texts.append(text)

        out: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start:start + self.batch_size]
            encoded = self._tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=self.max_tokens,
                return_tensors="pt",
            )
            # Count truncation before moving to the device, so the report is
            # about the transcripts rather than about the padded batch.
            lengths = [
                len(self._tokenizer(text, truncation=False)["input_ids"]) for text in chunk
            ]
            self.truncated += sum(1 for length in lengths if length > self.max_tokens)

            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            hidden_states = self._model(**encoded, output_hidden_states=True).hidden_states
            states = hidden_states[self.layer]

            mask = encoded["attention_mask"].bool()
            for row in range(states.shape[0]):
                # Padding is dropped here rather than carried into the cache:
                # the store holds true lengths, and the trunk builds its own
                # attention mask from them at batch time.
                sequence = states[row][mask[row]].float().cpu().numpy()
                if sequence.shape[0] < 1:
                    raise TextEncodingError(f"{chunk[row]!r} tokenised to nothing")
                out.append(sequence)
        return out

    def describe(self) -> dict:
        return self.spec.describe() | {
            "class": "RoBERTaTextEncoder",
            "transcript_source": "gold transcripts from the manifest; no ASR",
            "pooling": "none -- per-token hidden states are cached",
            "truncated_samples": int(self.truncated),
            "device": str(self.device),
            "frozen": True,
            "trained_by_this_project": False,
        }

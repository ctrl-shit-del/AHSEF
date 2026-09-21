"""Deterministic, offline hashing tokenizer for the text baseline.

Two constraints shaped this: the experiment must not need internet access or a
downloaded language model, and the text representation must not leak anything
from validation or test into training.

A hashing tokenizer satisfies both by construction.  There is no vocabulary to
fit, so no partition can influence the feature space; the token to bucket map
is a pure function of the token string (BLAKE2b, not Python's per-process
randomised ``hash``), so it is identical in every process and on every machine.

Index 0 is reserved for padding and never produced by hashing.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import torch


PAD_INDEX = 0

#: Words, contractions, and standalone punctuation that carries affect ("!").
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z]+)*|[!?]")


def tokenize(text: str) -> list[str]:
    """Lowercase, unicode-normalising word tokenizer.

    Deliberately simple and dependency-free; MELD transcripts arrive with
    smart quotes and mixed casing, which this folds away consistently.
    """
    if not text:
        return []
    return _TOKEN_PATTERN.findall(str(text).lower())


def token_index(token: str, vocab_size: int) -> int:
    """Map a token to ``1..vocab_size - 1`` deterministically across processes."""
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (vocab_size - 1) + 1


@dataclass(frozen=True)
class TokenizerConfig:
    """Text front-end geometry; recorded verbatim in every experiment artefact."""

    vocab_size: int = 32_768
    max_tokens: int = 64

    def __post_init__(self) -> None:
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be at least 2 (index 0 is padding)")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")

    def to_dict(self) -> dict:
        return {
            "kind": "hashing",
            "vocab_size": self.vocab_size,
            "max_tokens": self.max_tokens,
            "pad_index": PAD_INDEX,
            "hash": "blake2b-64",
            "fitted_on": None,
            "note": "Hashing needs no fitted vocabulary, so no partition can influence it.",
        }


class HashingTokenizer:
    """Encode text into a fixed-width, padded tensor of bucket indices."""

    def __init__(self, config: TokenizerConfig | None = None):
        self.config = config or TokenizerConfig()

    @property
    def vocab_size(self) -> int:
        return self.config.vocab_size

    @property
    def max_tokens(self) -> int:
        return self.config.max_tokens

    def encode(self, text: str) -> tuple[torch.Tensor, int]:
        """Return ``([max_tokens] long, valid_token_count)``.

        Empty text yields an all-padding row with length 1 rather than 0: a
        zero-length sequence would make masked pooling divide by zero, and an
        empty transcript is a real record whose label still counts.
        """
        config = self.config
        tokens = tokenize(text)[: config.max_tokens]
        indices = torch.zeros(config.max_tokens, dtype=torch.long)
        for position, token in enumerate(tokens):
            indices[position] = token_index(token, config.vocab_size)
        return indices, max(len(tokens), 1)

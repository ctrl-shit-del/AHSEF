"""Tests for the modality front ends: log-mel, hashing tokenizer, frame picking.

All three must be deterministic across processes, because an experiment that
cannot be reproduced cannot be compared.  Synthetic signals only; no dataset is
touched.
"""

import subprocess
import sys

import numpy as np
import pytest
import torch

from src.data.features.logmel import (
    LogMelConfig,
    LogMelExtractor,
    hz_to_mel,
    mel_filterbank,
    mel_to_hz,
)
from src.data.features.text_tokenizer import (
    PAD_INDEX,
    HashingTokenizer,
    TokenizerConfig,
    token_index,
    tokenize,
)
from src.data.loaders.video import uniform_indices


def _tone(seconds: float = 1.0, rate: int = 16_000, frequency: float = 220.0) -> np.ndarray:
    time = np.arange(int(seconds * rate), dtype=np.float32) / rate
    return np.sin(2 * np.pi * frequency * time).astype(np.float32)


# ============================================================
# Log-mel
# ============================================================

def test_mel_scale_round_trips():
    values = np.array([0.0, 100.0, 1000.0, 8000.0])
    assert np.allclose(mel_to_hz(hz_to_mel(values)), values, atol=1e-6)


def test_filterbank_shape_and_positivity():
    filters = mel_filterbank(16_000, 400, 64)
    assert filters.shape == (64, 201)
    assert bool((filters >= 0).all())
    # Every band must actually pass something, otherwise a mel channel is dead.
    assert bool((filters.sum(dim=1) > 0).all())


def test_filterbank_rejects_impossible_geometry():
    with pytest.raises(ValueError):
        mel_filterbank(16_000, 400, 64, f_min=9000.0)
    with pytest.raises(ValueError):
        mel_filterbank(16_000, 400, 0)


def test_extractor_shape_matches_declared_frame_count():
    config = LogMelConfig()
    extractor = LogMelExtractor(config)
    features = extractor(_tone(1.0))
    assert features.shape == (config.frames_for(1.0), config.n_mels)
    assert features.dtype == torch.float32
    assert bool(torch.isfinite(features).all())


def test_extractor_is_deterministic_and_content_sensitive():
    extractor = LogMelExtractor()
    first = extractor(_tone(0.5, frequency=220.0))
    again = extractor(_tone(0.5, frequency=220.0))
    other = extractor(_tone(0.5, frequency=1100.0))
    assert torch.equal(first, again)
    assert not torch.equal(first, other)


def test_fixed_length_pads_short_clips_and_reports_valid_frames():
    extractor = LogMelExtractor()
    features, valid = extractor.fixed_length(_tone(0.1), frames=200)
    assert features.shape == (200, 64)
    assert 0 < valid < 200
    assert bool((features[valid:] == 0).all())


def test_fixed_length_truncates_long_clips():
    extractor = LogMelExtractor()
    features, valid = extractor.fixed_length(_tone(5.0), frames=50)
    assert features.shape == (50, 64)
    assert valid == 50


def test_stereo_is_downmixed_and_very_short_clips_survive():
    extractor = LogMelExtractor()
    mono = _tone(0.05)
    stereo = np.stack([mono, mono], axis=1)
    assert torch.allclose(extractor(stereo), extractor(mono), atol=1e-5)
    assert extractor(np.zeros(3, dtype=np.float32)).shape[1] == 64


# ============================================================
# Hashing tokenizer
# ============================================================

def test_tokenizer_lowercases_and_keeps_affective_punctuation():
    assert tokenize("Well, I DON'T know!") == ["well", "i", "don't", "know", "!"]
    assert tokenize("") == []
    assert tokenize(None) == []


def test_encoding_is_fixed_width_padded_and_reports_length():
    tokenizer = HashingTokenizer(TokenizerConfig(vocab_size=64, max_tokens=8))
    tokens, length = tokenizer.encode("one two three")
    assert tokens.shape == (8,)
    assert length == 3
    assert bool((tokens[3:] == PAD_INDEX).all())
    assert bool((tokens[:3] != PAD_INDEX).all())


def test_empty_text_reports_length_one_so_pooling_never_divides_by_zero():
    tokenizer = HashingTokenizer(TokenizerConfig(vocab_size=64, max_tokens=8))
    tokens, length = tokenizer.encode("")
    assert length == 1
    assert bool((tokens == PAD_INDEX).all())


def test_long_text_is_truncated_to_max_tokens():
    tokenizer = HashingTokenizer(TokenizerConfig(vocab_size=64, max_tokens=4))
    _, length = tokenizer.encode("a b c d e f g")
    assert length == 4


def test_token_index_is_stable_across_processes():
    # Python randomises str.__hash__ per process; the tokenizer must not.
    script = (
        "from src.data.features.text_tokenizer import token_index;"
        "print(token_index('sadness', 4096))"
    )
    outputs = {
        subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=True
        ).stdout.strip()
        for _ in range(2)
    }
    assert len(outputs) == 1
    assert outputs == {str(token_index("sadness", 4096))}


def test_tokenizer_config_rejects_degenerate_geometry():
    with pytest.raises(ValueError):
        TokenizerConfig(vocab_size=1)
    with pytest.raises(ValueError):
        TokenizerConfig(max_tokens=0)


# ============================================================
# Frame selection
# ============================================================

def test_uniform_indices_are_spread_sorted_and_in_range():
    picked = uniform_indices(100, 8)
    assert len(picked) == 8
    assert picked == sorted(picked)
    assert all(0 <= index < 100 for index in picked)
    assert len(set(picked)) == 8


def test_uniform_indices_degenerate_cases():
    assert uniform_indices(3, 8) == [0, 1, 2]
    assert uniform_indices(0, 8) == []
    with pytest.raises(ValueError):
        uniform_indices(10, 0)

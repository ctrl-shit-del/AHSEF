"""Frozen encoders and the cache they write into.

The encoders in this package are never trained.  They run exactly once per
sample, in a separate extraction pass, and write ``[T, D]`` sequences to disk;
HSEN training then reads only those.  That is what keeps a 25%-data CPU run and
a full-data CUDA run comparable -- both consume byte-identical features -- and
what makes iterating on the fusion trunk cost minutes rather than hours.
"""

from src.hsen.features.base import EncoderSpec, FrozenEncoder
from src.hsen.features.store import (
    FeatureCacheError,
    SequenceFeatureStore,
    config_fingerprint,
)

__all__ = [
    "EncoderSpec",
    "FrozenEncoder",
    "FeatureCacheError",
    "SequenceFeatureStore",
    "config_fingerprint",
]

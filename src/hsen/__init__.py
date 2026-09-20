"""AHSEF HSEN -- the always-on multimodal Human State Estimation Network.

This package is the literature-backed baseline the adaptive layer is measured
against, and it is deliberately separate from :mod:`src.ahsef`.  ``src.ahsef``
holds the frozen Stage 1-3 routing work, which reads five frozen unimodal
baselines and never trains anything end-to-end; ``src.hsen`` trains one
multimodal trunk over cached frozen-encoder features.  Nothing here writes into
``experiments/<modality>/``, ``experiments/ahsef/`` or ``experiments/audio_strong/``.

The architecture is fixed by the AHSEF literature survey and is not a free
parameter of this phase:

    audio  emotion2vec, frozen, 768-d          -+
    video  face crop + MobileNetV3, frozen     -+-> Conv1D -> d=256 -> Husformer
    text   RoBERTa-base, frozen, 768-d         -+    2 layers, 4 heads, no CTC
                                                        |
                          categorical emotion + valence + arousal heads

Every encoder is frozen and every feature is cached to disk once, so training
never runs an encoder.  That is what makes both execution profiles -- full data
on CUDA and 25% of the data on CPU -- run the *same* model.
"""

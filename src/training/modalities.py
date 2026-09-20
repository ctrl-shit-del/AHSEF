"""Registry tying each modality to its experiment, CLI, and model builder.

One place answers three questions that would otherwise be answered
inconsistently in five: which config and runner belong to a modality, which
extra flags its CLI exposes, and how to rebuild its architecture from a
recorded run summary (which is what lets ``verify_artifacts`` reload any
modality's checkpoint without knowing the modality in advance).
"""

from __future__ import annotations

from typing import Callable, Mapping

import torch
import torch.nn as nn

from src.models.baselines import (
    AudioEmotionBaseline,
    PhysiologyStateBaseline,
    TextEmotionBaseline,
    VideoEmotionBaseline,
)
from src.models.multimodal import ImageEmotionBaseline
from src.models.strong_experts import LayerWeightedProbe
from src.training import (
    audio_experiment,
    audio_strong_experiment,
    image_experiment,
    physiology_experiment,
    text_experiment,
    video_experiment,
)
from src.training.experiment_cli import ExperimentSpec


IMAGE_SPEC = ExperimentSpec(
    modality="image",
    default_experiment="image_25pct",
    description=image_experiment.DESCRIPTION,
    config_class=image_experiment.ImageExperimentConfig,
    run_iteration=image_experiment.run_iteration,
    help="Train the image-only 7-class categorical emotion baseline.",
    extra_arguments=(
        ("--image-size", {"type": int}),
        ("--hidden-dim", {"type": int}),
    ),
)

AUDIO_SPEC = ExperimentSpec(
    modality="audio",
    default_experiment="audio_25pct",
    description=audio_experiment.DESCRIPTION,
    config_class=audio_experiment.AudioExperimentConfig,
    run_iteration=audio_experiment.run_iteration,
    help="Train the audio-only 7-class categorical emotion baseline.",
    extra_arguments=(
        ("--max-seconds", {"type": float}),
        ("--sample-rate", {"type": int}),
        ("--n-fft", {"type": int}),
        ("--hop-length", {"type": int}),
        ("--n-mels", {"type": int}),
        ("--hidden-dim", {"type": int}),
    ),
)

TEXT_SPEC = ExperimentSpec(
    modality="text",
    default_experiment="text_full",
    description=text_experiment.DESCRIPTION,
    config_class=text_experiment.TextExperimentConfig,
    run_iteration=text_experiment.run_iteration,
    help="Train the text-only 7-class categorical emotion baseline.",
    extra_arguments=(
        ("--vocab-size", {"type": int}),
        ("--max-tokens", {"type": int}),
        ("--embedding-dim", {"type": int}),
        ("--hidden-dim", {"type": int}),
    ),
)

VIDEO_SPEC = ExperimentSpec(
    modality="video",
    default_experiment="video_25pct",
    description=video_experiment.DESCRIPTION,
    config_class=video_experiment.VideoExperimentConfig,
    run_iteration=video_experiment.run_iteration,
    help="Train the video-only 7-class categorical emotion baseline.",
    extra_arguments=(
        ("--num-frames", {"type": int}),
        ("--frame-size", {"type": int}),
        ("--hidden-dim", {"type": int}),
        ("--temporal", {"choices": ["mean", "gru"]}),
    ),
)

PHYSIOLOGY_SPEC = ExperimentSpec(
    modality="physiology",
    default_experiment="physiology_full",
    description=physiology_experiment.DESCRIPTION,
    config_class=physiology_experiment.PhysiologyExperimentConfig,
    run_iteration=physiology_experiment.run_iteration,
    help=(
        "Train the physiology-only WESAD study-condition baseline. WESAD carries no "
        "canonical emotion target, so this baseline reports the WESAD label space."
    ),
    extra_arguments=(
        ("--hidden-dim", {"type": int}),
    ),
)


#: A separate expert, not a replacement. It registers under its own modality
#: name so nothing it does can reach ``experiments/audio/`` or the frozen
#: ``audio_25pct`` artefacts the locked AHSEF stages depend on.
AUDIO_STRONG_SPEC = ExperimentSpec(
    modality="audio_strong",
    default_experiment=audio_strong_experiment.STRONG_EXPERIMENT,
    description=audio_strong_experiment.DESCRIPTION,
    config_class=audio_strong_experiment.AudioStrongExperimentConfig,
    run_iteration=audio_strong_experiment.run_iteration,
    help=(
        "Train the audio 7-class expert over FROZEN wav2vec2 features. Requires "
        "the feature cache; see src.training.extract_audio_features."
    ),
    extra_arguments=(
        ("--hidden-dim", {"type": int}),
        ("--dropout", {"type": float}),
        ("--cache-root", {"type": str}),
    ),
)


EXPERIMENT_SPECS: Mapping[str, ExperimentSpec] = {
    spec.modality: spec
    for spec in (
        IMAGE_SPEC, AUDIO_SPEC, TEXT_SPEC, VIDEO_SPEC, PHYSIOLOGY_SPEC,
        AUDIO_STRONG_SPEC,
    )
}


#: The runner class behind each modality's ``run_iteration``.  Training reaches
#: a runner through the spec, but anything that wants to rebuild a modality's
#: *dataloader* without training -- post-hoc inference, for instance -- needs
#: the class itself, and reconstructing it by hand in five places is how the
#: five loaders would drift apart.
RUNNER_CLASSES: Mapping[str, type] = {
    "image": image_experiment.ImageExperimentRunner,
    "audio": audio_experiment.AudioExperimentRunner,
    "text": text_experiment.TextExperimentRunner,
    "video": video_experiment.VideoExperimentRunner,
    "physiology": physiology_experiment.PhysiologyExperimentRunner,
    "audio_strong": audio_strong_experiment.AudioStrongExperimentRunner,
}


def get_runner_class(modality: str) -> type:
    try:
        return RUNNER_CLASSES[modality]
    except KeyError as error:
        raise ValueError(
            f"Unknown modality {modality!r}; expected one of {sorted(RUNNER_CLASSES)}"
        ) from error


def get_spec(modality: str) -> ExperimentSpec:
    try:
        return EXPERIMENT_SPECS[modality]
    except KeyError as error:
        raise ValueError(
            f"Unknown modality {modality!r}; expected one of {sorted(EXPERIMENT_SPECS)}"
        ) from error


# ============================================================
# Model reconstruction
# ============================================================
#
# ``run_summary['model']`` records the class name and every geometry field
# needed to rebuild the architecture.  Rebuilding from that record -- rather
# than from a live import of whatever the code happens to define today -- is
# what makes checkpoint verification meaningful.

def _image_model(record: Mapping, config: Mapping) -> nn.Module:
    return ImageEmotionBaseline(
        image_size=int(record.get("image_size", config.get("image_size", 48))),
        hidden_dim=int(record.get("hidden_dim", config.get("hidden_dim", 256))),
        num_classes=int(record.get("num_classes", config.get("num_classes", 7))),
    )


def _audio_model(record: Mapping, config: Mapping) -> nn.Module:
    return AudioEmotionBaseline(
        n_mels=int(record.get("n_mels", config.get("n_mels", 64))),
        hidden_dim=int(record.get("hidden_dim", config.get("hidden_dim", 256))),
        num_classes=int(record.get("num_classes", config.get("num_classes", 7))),
    )


def _text_model(record: Mapping, config: Mapping) -> nn.Module:
    return TextEmotionBaseline(
        vocab_size=int(record.get("vocab_size", config.get("vocab_size", 32_768))),
        embedding_dim=int(record.get("embedding_dim", config.get("embedding_dim", 128))),
        hidden_dim=int(record.get("hidden_dim", config.get("hidden_dim", 256))),
        num_classes=int(record.get("num_classes", config.get("num_classes", 7))),
    )


def _video_model(record: Mapping, config: Mapping) -> nn.Module:
    return VideoEmotionBaseline(
        frame_size=int(record.get("frame_size", config.get("frame_size", 48))),
        hidden_dim=int(record.get("hidden_dim", config.get("hidden_dim", 256))),
        num_classes=int(record.get("num_classes", config.get("num_classes", 7))),
        temporal=str(record.get("temporal", config.get("temporal", "mean"))),
    )


def _layer_weighted_probe(record: Mapping, config: Mapping) -> nn.Module:
    """Rebuild the frozen-encoder probe head from its recorded geometry.

    Only the head is rebuilt. The encoder is not part of the checkpoint: it is a
    frozen, externally-versioned artefact identified by the cache fingerprint in
    ``model.front_end``, and reconstructing it here would imply this project
    trained it.
    """
    return LayerWeightedProbe(
        num_layers=int(record.get("num_layers", config.get("num_layers", 12))),
        feature_dim=int(record.get("feature_dim", config.get("feature_dim", 768))),
        hidden_dim=int(record.get("hidden_dim", config.get("hidden_dim", 256))),
        num_classes=int(record.get("num_classes", config.get("num_classes", 7))),
        dropout=float(record.get("dropout", config.get("dropout", 0.2))),
    )


def _physiology_model(record: Mapping, config: Mapping) -> nn.Module:
    feature_dim = record.get("feature_dim")
    if feature_dim is None:
        raise ValueError(
            "The physiology run summary does not record 'model.feature_dim'; the "
            "architecture cannot be rebuilt without it."
        )
    return PhysiologyStateBaseline(
        feature_dim=int(feature_dim),
        hidden_dim=int(record.get("hidden_dim", config.get("hidden_dim", 128))),
        num_classes=int(record.get("num_classes", config.get("num_classes", 3))),
    )


MODEL_BUILDERS: Mapping[str, Callable[[Mapping, Mapping], nn.Module]] = {
    "ImageEmotionBaseline": _image_model,
    "AudioEmotionBaseline": _audio_model,
    "TextEmotionBaseline": _text_model,
    "VideoEmotionBaseline": _video_model,
    "PhysiologyStateBaseline": _physiology_model,
    "LayerWeightedProbe": _layer_weighted_probe,
}

#: Probe geometry per architecture, used to prove that a reloaded model emits
#: exactly ``num_classes`` logits.  Each entry yields
#: ``(input_shape, lengths_shape_or_None, input_dtype)``; the text model indexes
#: an embedding table, so its probe must be integral rather than float.
FORWARD_PROBES: Mapping[str, Callable[[Mapping, Mapping], tuple]] = {
    "ImageEmotionBaseline": lambda record, config: (
        (2, 3, int(record.get("image_size", 48)), int(record.get("image_size", 48))),
        None, torch.float32,
    ),
    "AudioEmotionBaseline": lambda record, config: (
        (2, 4, int(record.get("n_mels", 64))), (2,), torch.float32,
    ),
    "TextEmotionBaseline": lambda record, config: ((2, 4), (2,), torch.long),
    "VideoEmotionBaseline": lambda record, config: (
        (2, 2, 3, int(record.get("frame_size", 48)), int(record.get("frame_size", 48))),
        (2,), torch.float32,
    ),
    "PhysiologyStateBaseline": lambda record, config: (
        (2, int(record.get("feature_dim") or 1)), None, torch.float32,
    ),
    "LayerWeightedProbe": lambda record, config: (
        (2, int(record.get("num_layers", 12)), int(record.get("feature_dim", 768))),
        None, torch.float32,
    ),
}


def build_model_from_record(run_summary: Mapping) -> nn.Module:
    """Rebuild the architecture an iteration recorded in its run summary."""
    record = run_summary.get("model") or {}
    config = run_summary.get("config") or {}
    class_name = record.get("class")
    builder = MODEL_BUILDERS.get(class_name)
    if builder is None:
        raise ValueError(
            f"No model builder registered for {class_name!r}; known architectures "
            f"are {sorted(MODEL_BUILDERS)}"
        )
    return builder(record, config)

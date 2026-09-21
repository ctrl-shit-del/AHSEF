import pytest

from src.data.resolver import SampleResolver


@pytest.mark.integration
def test_resolver():

    resolver = SampleResolver()

    print("=" * 70)
    print("DATA RESOLVER TEST")
    print("=" * 70)

    # ---------------------------------------------------------
    # CMU-MOSEI
    # ---------------------------------------------------------

    print("\nCMU-MOSEI")

    sample = resolver.resolve(
        "CMU-MOSEI_-3g5yACwYnA_10"
    )

    print("ID:", sample.sample_id)
    print("Dataset:", sample.dataset)
    print("Split:", sample.split)
    print("Emotion:", sample.emotion)
    print("Emotion ID:", sample.emotion_id)
    print("Sentiment:", sample.sentiment)

    print(
        "Audio shape:",
        sample.audio.shape,
    )

    print(
        "Video shape:",
        sample.video.shape,
    )

    print(
        "Text:",
        sample.text[:100],
    )

    print(
        "Text feature shape:",
        sample.extras[
            "mosei_features"
        ]["text_features"].shape,
    )

    # ---------------------------------------------------------
    # AffectNet+
    # ---------------------------------------------------------

    print("\nAffectNet+")

    affect = resolver.df[
        resolver.df["dataset"]
        == "AffectNet+"
    ].iloc[0]

    sample = resolver.resolve(
        affect["sample_id"]
    )

    print(
        "ID:",
        sample.sample_id,
    )

    print(
        "Emotion:",
        sample.emotion,
    )

    print(
        "Image:",
        sample.image.size,
    )

    # ---------------------------------------------------------
    # MELD text-only edge case
    # ---------------------------------------------------------

    print("\nMELD edge case")

    sample = resolver.resolve(
        "validation_dia110_utt7"
    )

    print(
        "ID:",
        sample.sample_id,
    )

    print(
        "Emotion:",
        sample.emotion,
    )

    print(
        "Text:",
        sample.text,
    )

    print(
        "Video:",
        sample.video,
    )

    assert sample.sample_id == "validation_dia110_utt7"
    assert sample.text is not None
    assert sample.video is None

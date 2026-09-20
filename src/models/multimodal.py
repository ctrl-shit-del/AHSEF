import torch
import torch.nn as nn

from src.models.encoders.simple import (
    TemporalMeanEncoder,
)


class MultimodalSentimentBaseline(nn.Module):
    """
    Minimal CMU-MOSEI multimodal sentiment baseline.

    Audio:
        [B, 50, 74]

    Vision:
        [B, 50, 35]

    Text:
        [B, 50, 768]

    Output:
        [B]
    """

    def __init__(
        self,
        audio_dim: int = 74,
        vision_dim: int = 35,
        text_dim: int = 768,
        hidden_dim: int = 128,
        fusion_dim: int = 256,
    ):
        super().__init__()

        self.audio_encoder = TemporalMeanEncoder(
            input_dim=audio_dim,
            output_dim=hidden_dim,
        )

        self.vision_encoder = TemporalMeanEncoder(
            input_dim=vision_dim,
            output_dim=hidden_dim,
        )

        self.text_encoder = TemporalMeanEncoder(
            input_dim=text_dim,
            output_dim=hidden_dim,
        )

        self.fusion = nn.Sequential(
            nn.Linear(
                hidden_dim * 3,
                fusion_dim,
            ),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),

            nn.Linear(
                fusion_dim,
                hidden_dim,
            ),
            nn.GELU(),
        )

        self.sentiment_head = nn.Linear(
            hidden_dim,
            1,
        )

    def forward(
        self,
        audio: torch.Tensor,
        vision: torch.Tensor,
        text: torch.Tensor,
    ) -> torch.Tensor:

        audio_embedding = self.audio_encoder(
            audio
        )

        vision_embedding = self.vision_encoder(
            vision
        )

        text_embedding = self.text_encoder(
            text
        )

        fused = torch.cat(
            [
                audio_embedding,
                vision_embedding,
                text_embedding,
            ],
            dim=-1,
        )

        fused = self.fusion(fused)

        output = self.sentiment_head(
            fused
        )

        # [B, 1] -> [B]
        return output.squeeze(-1)


class ImageEmotionBaseline(nn.Module):
    """Deliberately small image-only seven-class categorical baseline."""

    def __init__(self, image_size: int = 32, hidden_dim: int = 128, num_classes: int = 7):
        super().__init__()
        self.image_size, self.num_classes = image_size, num_classes
        self.encoder = nn.Sequential(
            nn.Flatten(), nn.Linear(3 * image_size * image_size, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(image))

import torch
import torch.nn as nn


class TemporalMeanEncoder(nn.Module):
    """
    Simple baseline encoder.

    Input:
        [B, T, D]

    Output:
        [B, output_dim]
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 128,
    ):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        # [B, T, D] -> [B, D]
        x = x.mean(dim=1)

        # [B, D] -> [B, output_dim]
        x = self.projection(x)

        return x
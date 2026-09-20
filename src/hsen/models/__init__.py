"""The HSEN architecture: projection, fusion trunk, heads.

Fixed by the AHSEF literature survey and not a free parameter of this phase --
d = 256, 2 layers, 4 heads, fusion-to-modality cross-attention, no CTC.  The
fusion variants exist for the section-13 ablation, which is the one place the
topology is allowed to change.
"""

from src.hsen.models.fusion_variants import FUSION_VARIANTS, build_fusion
from src.hsen.models.hsen import HSEN, DEFAULT_INPUT_DIMS, HSENConfig, build_hsen
from src.hsen.models.husformer import HusformerFusion
from src.hsen.models.projection import ModalityProjection

__all__ = [
    "HSEN", "HSENConfig", "build_hsen", "DEFAULT_INPUT_DIMS",
    "HusformerFusion", "ModalityProjection", "build_fusion", "FUSION_VARIANTS",
]

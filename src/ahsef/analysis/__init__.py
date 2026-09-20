"""Cross-modality analysis that reads frozen artefacts and writes only reports.

Nothing in this package trains, retrains, or modifies a model. It consumes the
prediction exports Stage 1 already wrote and answers questions about *modality
capability* -- which is a different question from "which model is best", and one
that has to be asked carefully because the five baselines were evaluated on
disjoint corpora.
"""

from __future__ import annotations

ANALYSIS_EXPERIMENT_ID = "classwise_modality_analysis_v1"

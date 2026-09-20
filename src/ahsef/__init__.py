"""AHSEF -- Adaptive Hierarchical Sensor/Modality Evidence Fusion.

This package sits *beside* the frozen unimodal baselines rather than inside
them.  Nothing here trains a model, rewrites a checkpoint, or touches an
existing ``experiments/<modality>/`` artefact: every AHSEF run writes into its
own ``experiments/ahsef/<run>/`` tree.

Stage 1 (the 50% milestone groundwork) provides:

* :mod:`src.ahsef.inference`        -- one prediction interface for all five baselines
* :mod:`src.ahsef.uncertainty`      -- confidence, predictive entropy, normalised entropy, margin
* :mod:`src.ahsef.calibration`      -- ECE, reliability bins, validation-only temperature scaling
* :mod:`src.ahsef.identity`         -- cross-modality sample identity and co-split safety
* :mod:`src.ahsef.fusion`           -- probability-level fusion of aligned prediction sets
* :mod:`src.ahsef.information_gain` -- per-sample empirical dU(m | A)
* :mod:`src.ahsef.costs`            -- measured latency / compute cost, normalised per candidate set
* :mod:`src.ahsef.evaluation`       -- metrics over prediction sets, modality x emotion tables
* :mod:`src.ahsef.routing_log`      -- the auditable per-sample routing trace format

HSIG (:mod:`hsig`), UGAPR (:mod:`ugapr`), and the dynamic router are stage 2
and are deliberately absent until stage 1 is verified.
"""

from src.ahsef.layout import AhsefLayout

__all__ = ["AhsefLayout"]

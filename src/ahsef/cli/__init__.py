"""Command-line entry points for the AHSEF stage-1 pipeline.

Run them in order; each reads only what the previous one wrote::

    python -m src.ahsef.cli.export_predictions    --run stage1
    python -m src.ahsef.cli.run_alignment         --run stage1 --anchor audio
    python -m src.ahsef.cli.run_calibration       --run stage1
    python -m src.ahsef.cli.run_pairwise_fusion   --run stage1 --anchor audio
    python -m src.ahsef.cli.run_gain_diagnostics  --run stage1 --split validation

Every command writes under ``experiments/ahsef/<run>/`` and nowhere else.
"""

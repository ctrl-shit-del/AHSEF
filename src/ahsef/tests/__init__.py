"""Unit tests for the AHSEF stage-1 components.

Everything here runs on synthetic tensors in milliseconds; nothing opens a
checkpoint, a dataset, or an experiment directory.  The tests that need real
artefacts are marked ``integration`` and live in
``test_inference_integration.py``.
"""

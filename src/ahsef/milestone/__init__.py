"""The 50%-milestone layer: stronger experts and a class-aware routing policy.

Stage 3 built the first working Text->Audio router and measured, honestly, that
its advantage over random acquisition at the same budget did not survive the
locked test. Two causes were identified there and both are addressed here:

* the gain estimator reduced to the uncertainty signal, so the router had no
  more information than Stage 2's gate;
* the routing target counted a *correction* as the unit of value, while the
  reported metric was macro-F1 -- and on this corpus those pull in different
  directions, because most corrections land in the majority classes.

Nothing in this package modifies Stage 1, Stage 2 or Stage 3. Their artefacts,
frozen configurations and locked results are inputs.
"""

from __future__ import annotations

MILESTONE_EXPERIMENT_ID = "ahsef_50pct_milestone_v1"

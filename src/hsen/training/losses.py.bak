"""The combined HSEN objective: focal classification plus two affect regressors.

    L_total = lambda_cls * L_cls + lambda_val * L_valence + lambda_aro * L_arousal

Every weight is configurable and none has a default the code pretends is
optimal.  The starting point is ``(1.0, 0.0, 0.0)`` -- classification only --
because the categorical head is what the primary metric measures and because a
regression term added before the classifier trains at all is a confound rather
than a contribution.  Turning the affect heads on is a deliberate, recorded
config change, and each term is logged separately so their effect is visible.

Focal loss, from Husformer.  IEMOCAP under the six-way protocol runs from 387
``happy`` to 1,066 ``neutral`` training samples, and CMU-MOSEI's sentiment
buckets are worse -- 207 ``highly_positive`` against 6,798 ``neutral``.  Plain
cross-entropy on that spends its capacity on the majority bucket and the minority
classes disappear into an aggregate accuracy that looks fine.

Masking is the other half of the design.  A target of ``MISSING_CLASS_ID`` or
``NaN`` contributes nothing -- no gradient, no denominator.  CMU-MOSEI does not
annotate arousal, so its arousal head sees no supervision at all, and the
correct behaviour is for that term to be exactly zero rather than for the head
to be trained toward a fabricated value.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as functional

from src.hsen.labels.base import MISSING_CLASS_ID

REGRESSION_LOSSES = ("mse", "huber", "ccc")


@dataclass
class LossConfig:
    """Weights and settings for the combined objective."""

    lambda_cls: float = 1.0
    #: Off by default. The affect heads exist and are exercised by the sanity
    #: test, but switching them into the objective changes what the model
    #: optimises and is therefore an experiment, not a default.
    lambda_valence: float = 0.0
    lambda_arousal: float = 0.0
    #: Husformer's focusing parameter. 2.0 is the value the focal-loss
    #: literature settles on; it is a hyperparameter of this project, not a
    #: constant, and section 14 lists it as one of the few worth searching.
    focal_gamma: float = 2.0
    #: ``balanced`` derives alpha from the training distribution; ``none``
    #: disables it; a list sets it explicitly, in declared class order.
    class_weights: str | list[float] = "balanced"
    label_smoothing: float = 0.0
    regression_loss: str = "mse"
    huber_delta: float = 1.0
    #: Auxiliary per-modality classification, ESED's ``L_spec``. Zero here: the
    #: heads that would consume it are off in this phase's baseline.
    lambda_modality: float = 0.0

    def __post_init__(self) -> None:
        if self.regression_loss not in REGRESSION_LOSSES:
            raise ValueError(
                f"Unknown regression loss {self.regression_loss!r}; "
                f"expected one of {list(REGRESSION_LOSSES)}"
            )
        if self.focal_gamma < 0:
            raise ValueError("focal_gamma must be non-negative")
        if isinstance(self.class_weights, str) and self.class_weights not in ("balanced", "none"):
            raise ValueError(
                f"class_weights must be 'balanced', 'none' or an explicit list, "
                f"got {self.class_weights!r}"
            )

    def to_dict(self) -> dict:
        return asdict(self)


def balanced_class_weights(class_ids: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Inverse-frequency alpha, normalised to mean 1.

    Computed from the *training* labels only -- the trainer passes the training
    split and nothing else -- because a weight vector derived from validation
    counts is a small, real leak of the validation distribution into training.

    Normalising to mean 1 keeps the loss on the same scale as the unweighted one,
    so ``lambda_cls`` means the same thing whether or not weighting is on.
    A class absent from the training split gets weight 1 rather than infinity:
    the model cannot learn it either way, and an infinite weight would make the
    normalisation meaningless for every other class.
    """
    valid = class_ids[class_ids != MISSING_CLASS_ID]
    counts = torch.bincount(valid, minlength=num_classes).float()
    weights = torch.where(counts > 0, counts.sum() / (num_classes * counts.clamp(min=1)),
                          torch.ones_like(counts))
    return weights / weights.mean().clamp(min=1e-8)


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Multi-class focal loss with per-class alpha, masked on ``MISSING_CLASS_ID``.

    Reduces to weighted cross-entropy at ``gamma = 0``, which is what makes a
    gamma sweep an honest ablation rather than a change of loss family.
    """
    valid = targets != MISSING_CLASS_ID
    if not bool(valid.any()):
        return logits.sum() * 0.0

    logits, targets = logits[valid], targets[valid]
    log_probabilities = functional.log_softmax(logits, dim=-1)
    true_log_probability = log_probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)

    # Smoothing is applied here rather than through ``cross_entropy`` because
    # the focal modulator below needs the log-probabilities anyway, and because
    # ``nll_loss`` -- which is what takes the per-class weight -- has no
    # smoothing argument.
    cross_entropy = -true_log_probability
    if label_smoothing:
        uniform = -log_probabilities.mean(dim=-1)
        cross_entropy = (1.0 - label_smoothing) * cross_entropy + label_smoothing * uniform
    if alpha is not None:
        # Weighted mean of per-sample losses, not torch's weighted-mean-with-
        # weighted-denominator: alpha is normalised to mean 1 by
        # ``balanced_class_weights``, so the loss keeps the same scale as the
        # unweighted form and ``lambda_cls`` means one thing throughout.
        cross_entropy = cross_entropy * alpha.to(cross_entropy.device)[targets]
    # The modulating factor uses the *unweighted* probability of the true class:
    # alpha handles class frequency, gamma handles per-sample difficulty, and
    # mixing them would make the two knobs interact in a way neither paper's
    # reported values would transfer through.
    true_probability = true_log_probability.exp()
    return ((1.0 - true_probability).clamp(min=0.0) ** gamma * cross_entropy).mean()


def multilabel_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: torch.Tensor | None = None,
) -> torch.Tensor:
    """Focal binary cross-entropy over non-exclusive labels.

    CMU-MOSEI's six-way protocol needs this rather than the single-label form:
    a clip can be both ``sad`` and ``angry``, and a softmax would force the two
    to compete for the same probability mass.  Rows with any non-finite target
    are dropped whole, since a partially annotated row cannot say whether an
    unmarked emotion is absent or simply unrecorded.
    """
    valid = torch.isfinite(targets).all(dim=-1)
    if not bool(valid.any()):
        return logits.sum() * 0.0

    logits, targets = logits[valid], targets[valid]
    probabilities = torch.sigmoid(logits)
    bce = functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    modulator = (targets * (1 - probabilities) + (1 - targets) * probabilities).clamp(min=0.0)
    loss = (modulator ** gamma) * bce
    if alpha is not None:
        loss = loss * alpha.view(1, -1)
    return loss.mean()


def concordance_correlation_loss(
    prediction: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-8
) -> torch.Tensor:
    """``1 - CCC``, the loss form of the metric this phase reports for affect.

    CCC penalises a prediction that correlates well but is systematically
    shifted or scaled, which is the failure mode a plain MSE run on a
    narrow-range target falls into: predict the mean everywhere and MSE looks
    respectable while CCC sits near zero.

    Needs at least two valid samples to have a variance at all; a batch with
    fewer contributes nothing rather than a degenerate value.
    """
    if prediction.numel() < 2:
        return prediction.sum() * 0.0
    prediction_mean, target_mean = prediction.mean(), target.mean()
    prediction_var = prediction.var(unbiased=False)
    target_var = target.var(unbiased=False)
    covariance = ((prediction - prediction_mean) * (target - target_mean)).mean()
    ccc = (2 * covariance) / (
        prediction_var + target_var + (prediction_mean - target_mean) ** 2 + epsilon
    )
    return 1.0 - ccc


def regression_loss(
    prediction: torch.Tensor, target: torch.Tensor, kind: str = "mse", huber_delta: float = 1.0,
) -> torch.Tensor:
    """One affect-dimension loss, masked on ``NaN`` targets."""
    valid = torch.isfinite(target)
    if not bool(valid.any()):
        return prediction.sum() * 0.0
    prediction, target = prediction[valid], target[valid]
    if kind == "mse":
        return functional.mse_loss(prediction, target)
    if kind == "huber":
        return functional.huber_loss(prediction, target, delta=huber_delta)
    if kind == "ccc":
        return concordance_correlation_loss(prediction, target)
    raise ValueError(f"Unknown regression loss {kind!r}")


class HSENLoss(nn.Module):
    """The combined objective, reporting every term it sums."""

    def __init__(
        self,
        config: LossConfig | None = None,
        num_classes: int = 6,
        task: str = "single_label",
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.config = config or LossConfig()
        self.num_classes = num_classes
        self.task = task
        # Registered as a buffer so it moves with the module and is saved into
        # the checkpoint: the weight vector is part of what produced the numbers,
        # and reconstructing it later from a re-derived count would not be the
        # same object.
        self.register_buffer(
            "alpha", class_weights if class_weights is not None else torch.empty(0),
        )

    @property
    def _alpha(self) -> torch.Tensor | None:
        return self.alpha if self.alpha.numel() else None

    def forward(self, outputs: dict, targets: dict) -> dict[str, torch.Tensor]:
        """Total loss plus each component, all as tensors on the model's device."""
        settings = self.config
        components: dict[str, torch.Tensor] = {}

        if self.task == "multi_label":
            classification = multilabel_focal_loss(
                outputs["logits"], targets["multilabel"], settings.focal_gamma, self._alpha,
            )
        else:
            classification = focal_loss(
                outputs["logits"], targets["class_id"], settings.focal_gamma,
                self._alpha, settings.label_smoothing,
            )
        components["cls"] = classification
        total = settings.lambda_cls * classification

        for name, weight in (("valence", settings.lambda_valence),
                             ("arousal", settings.lambda_arousal)):
            if name not in outputs or name not in targets:
                continue
            term = regression_loss(
                outputs[name], targets[name], settings.regression_loss, settings.huber_delta,
            )
            components[name] = term
            # The term is computed and logged even at weight zero: it is a free
            # read on whether the affect heads are learning anything, and it
            # contributes no gradient while the weight stays at zero.
            if weight:
                total = total + weight * term

        if settings.lambda_modality and "modality_logits" in outputs:
            per_modality = [
                focal_loss(logits, targets["class_id"], settings.focal_gamma,
                           self._alpha, settings.label_smoothing)
                for logits in outputs["modality_logits"].values()
            ]
            if per_modality:
                term = torch.stack(per_modality).mean()
                components["modality"] = term
                total = total + settings.lambda_modality * term

        components["total"] = total
        return components

    def describe(self) -> dict:
        return {
            "class": "HSENLoss",
            "task": self.task,
            "config": self.config.to_dict(),
            "class_weights": self.alpha.tolist() if self.alpha.numel() else None,
        }

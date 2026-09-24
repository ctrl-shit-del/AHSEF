"""Per-sample predictions for every modality subset, from one trained model.

    python -m src.hsen.export_subsets \
        --run results/hsen/mosei_sentiment/cp2_evid_no_kl_seed42 \
        --split validation

Writes ``subset_predictions.parquet`` into the run directory: one row per
(sample, subset), carrying the predicted class, whether it was correct, the
raw logits, and the full uncertainty decomposition.

WHY THIS EXISTS.  HSIG has to estimate, without paying for the candidate, how
much acquiring modality ``m`` would improve the decision.  Its target, fixed by
stage 1 after uncertainty reduction was measured and rejected, is::

    gain(m | A) = P(correct | A u {m}) - P(correct | A)

Both terms are per-sample quantities of the *same* model under two different
availability masks.  One trained model and one forward pass per subset produces
every term for every sample -- seven passes over the split for three
modalities, no retraining, no second model to confound the comparison.

The same file is what UGAPR's budget curve reads: it carries, for every sample,
what each routing decision would have cost and what it would have bought.

READ THE MASKING WARNING.  Unless the model was trained with
``--modality_dropout`` above zero, masking a modality at inference is an input
it has never seen, and the numbers here describe off-distribution behaviour
rather than that modality's value.  This module refuses to run silently in that
case: it reads the run's own record and says so.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.hsen.counterfactual import subset_availability
from src.hsen.data import build_dataloader
from src.hsen.evidential import alpha_from_logits, uncertainty_decomposition
from src.hsen.experiment import build_experiment
from src.hsen.labels.base import MISSING_CLASS_ID
from src.hsen.models.hsen import build_hsen


class ExportError(RuntimeError):
    """Raised when the run cannot be reloaded, or must not be trusted."""


def subsets_of(modalities: tuple[str, ...]) -> list[tuple[str, ...]]:
    """Every non-empty subset, smallest first.

    The empty set is excluded: a sample with nothing observed is not a routing
    state, and the Husformer fusion raises on it by design.
    """
    return [
        combination
        for size in range(1, len(modalities) + 1)
        for combination in itertools.combinations(modalities, size)
    ]


def load_run(run_dir: Path) -> dict:
    path = run_dir / "run_summary.json"
    if not path.exists():
        raise ExportError(f"No run_summary.json in {run_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def restore_model(run_dir: Path, summary: dict, bundle, device: torch.device):
    """Rebuild the architecture from its own record and load the best weights.

    Rebuilt from ``model_config`` in the summary rather than from the CLI
    defaults, because a run that used a non-default width or fusion would
    otherwise load into a differently shaped model and fail -- or worse, not
    fail, if only the head differed.
    """
    model = build_hsen(bundle.model_config).to(device)

    candidates = [run_dir / "checkpoints" / "best.pt", run_dir / "best.pt"]
    candidates += sorted(run_dir.rglob("best*.pt"))
    for path in candidates:
        if path.exists():
            payload = torch.load(path, map_location=device, weights_only=False)
            state = payload.get("model_state_dict", payload)
            model.load_state_dict(state)
            model.eval()
            return model, path
    raise ExportError(
        f"No best checkpoint under {run_dir}. Looked for checkpoints/best.pt, "
        f"best.pt and any best*.pt."
    )


def masking_was_trained(summary: dict) -> float:
    """The dropout rate the run actually used, 0.0 if it used none."""
    trainer = summary.get("trainer_config", {})
    return float(trainer.get("modality_dropout", 0.0) or 0.0)


@torch.no_grad()
def export(
    run_dir: Path,
    split: str = "validation",
    batch_size: int = 128,
    num_workers: int = 2,
    device_name: str = "cuda",
    allow_untrained_masking: bool = False,
) -> Path:
    """One row per (sample, subset), written beside the run."""
    summary = load_run(run_dir)
    trainer = summary.get("trainer_config", {})
    experiment = trainer.get("experiment") or summary.get("experiment")
    if not experiment:
        raise ExportError(f"{run_dir}/run_summary.json names no experiment.")

    rate = masking_was_trained(summary)
    if rate <= 0.0 and not allow_untrained_masking:
        raise ExportError(
            f"{run_dir.name} was trained with modality_dropout = 0.0, so every "
            f"modality was present in every training batch.\n"
            f"Masking one at inference is then an input this model has never "
            f"seen, and the gains measured from it would describe its "
            f"off-distribution behaviour rather than the modality's value.\n"
            f"Retrain with --modality_dropout 0.3, or pass "
            f"--allow-untrained-masking if you specifically want the "
            f"off-distribution numbers and will label them as such."
        )

    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    model_config = summary.get("model_config", {})
    modalities = tuple(model_config.get("modalities")
                       or trainer.get("modalities", "audio+text").split("+"))

    bundle = build_experiment(
        experiment=experiment,
        modalities=modalities,
        data_fraction=float(trainer.get("data_fraction", 1.0)),
        seed=int(trainer.get("seed", 42)),
        fusion=model_config.get("fusion", "concat"),
        model_overrides={
            key: model_config[key]
            for key in ("model_dim", "num_layers", "num_heads", "dropout",
                        "attention_dropout", "predict_valence", "predict_arousal")
            if key in model_config
        },
        include_test=(split == "test"),
    )
    if split not in bundle.datasets:
        raise ExportError(f"Split {split!r} not built; have {sorted(bundle.datasets)}")

    model, checkpoint = restore_model(run_dir, summary, bundle, device)
    evidential = (summary.get("loss_config", {}).get("classification_loss")
                  == "evidential")

    loader = build_dataloader(bundle.datasets[split], batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              seed=int(trainer.get("seed", 42)))
    wanted = subsets_of(modalities)
    print(f"{run_dir.name}  split={split}  modalities={'+'.join(modalities)}")
    print(f"  checkpoint      {checkpoint}")
    print(f"  head            {'evidential' if evidential else 'softmax'}")
    print(f"  trained masking {rate}")
    print(f"  subsets         {len(wanted)}\n")

    rows: list[dict] = []
    for subset in wanted:
        label = "+".join(subset)
        correct_total = seen = 0
        for batch in loader:
            features = {k: v.to(device) for k, v in batch["features"].items()}
            masks = {k: v.to(device) for k, v in batch["masks"].items()}
            present = {k: v.to(device) for k, v in batch["available"].items()}
            truth = batch["targets"]["class_id"]

            # Intersected with what the sample actually has, so a subset naming
            # a modality this corpus lacks degrades instead of fabricating it.
            availability = subset_availability(present, subset)
            if not bool(torch.stack(list(availability.values()), dim=1).any(dim=1).all()):
                # Every modality in this subset is absent for some sample. Those
                # rows have nothing to predict from; skip the subset for them
                # rather than feed the fusion an empty observation.
                keep = torch.stack(list(availability.values()), dim=1).any(dim=1).cpu()
            else:
                keep = torch.ones(len(truth), dtype=torch.bool)

            logits = model(features, masks, availability)["logits"].float().cpu()
            predicted = logits.argmax(dim=-1)
            num_classes = logits.shape[-1]
            # The protocol's neutral point, for an ORDINAL label space. MOSEI's
            # classes run highly_negative -> highly_positive, so the middle
            # index is neutral and distance from it is polarity strength. On an
            # unordered space (IEMOCAP's emotions) the two derived columns below
            # are meaningless and only the one-hot indicators should be used.
            neutral_index = (num_classes - 1) // 2
            valid = (truth != MISSING_CLASS_ID) & keep

            alpha = alpha_from_logits(logits.double())
            pieces = uncertainty_decomposition(alpha)
            probabilities = torch.softmax(logits.double(), dim=-1)
            entropy_softmax = -(probabilities * (probabilities + 1e-12).log()).sum(-1)
            entropy_softmax = entropy_softmax / np.log(logits.shape[-1])

            for row in range(len(truth)):
                if not bool(valid[row]):
                    continue
                rows.append({
                    "sample_id": batch["sample_ids"][row],
                    "subset": label,
                    "n_modalities": len(subset),
                    "true_id": int(truth[row]),
                    "predicted_id": int(predicted[row]),
                    # The ANCHOR's prediction, known before the candidate is
                    # paid for -- not the fused one, which would be leakage.
                    **{f"predicted_class_{index}": int(int(predicted[row]) == index)
                       for index in range(num_classes)},
                    "predicted_is_neutral": int(int(predicted[row]) == neutral_index),
                    "predicted_polarity": (abs(int(predicted[row]) - neutral_index)
                                           / max(1, neutral_index)),
                    "correct": int(predicted[row] == truth[row]),
                    "max_probability": float(probabilities[row].max()),
                    "softmax_entropy": float(entropy_softmax[row]),
                    "vacuity": float(pieces["vacuity"][row]),
                    "dissonance": float(pieces["dissonance"][row]),
                    "composite": float(pieces["composite"][row]),
                    "dirichlet_entropy": float(pieces["dirichlet_entropy"][row]),
                    "evidence_total": float(pieces["evidence_total"][row]),
                    **{f"logit_{index}": float(value)
                       for index, value in enumerate(logits[row])},
                })
                correct_total += int(predicted[row] == truth[row])
                seen += 1

        accuracy = correct_total / max(1, seen)
        print(f"  {label:<20} n={seen:<6} accuracy {accuracy:.4f}")

    frame = pd.DataFrame(rows)
    out = run_dir / f"subset_predictions_{split}.parquet"
    frame.to_parquet(out, index=False)

    (run_dir / f"subset_predictions_{split}.meta.json").write_text(json.dumps({
        "run": run_dir.name,
        "experiment": experiment,
        "split": split,
        "modalities": list(modalities),
        "subsets": ["+".join(s) for s in wanted],
        "head": "evidential" if evidential else "softmax",
        "modality_dropout_in_training": rate,
        "allow_untrained_masking": bool(allow_untrained_masking),
        "checkpoint": str(checkpoint),
        "rows": len(frame),
        "samples": int(frame["sample_id"].nunique()) if len(frame) else 0,
    }, indent=2), encoding="utf-8")

    print(f"\nwrote {out}  ({len(frame):,} rows, "
          f"{frame['sample_id'].nunique() if len(frame) else 0:,} samples)")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="A finished run directory.")
    parser.add_argument("--split", default="validation",
                        choices=["train", "validation", "test"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-untrained-masking", action="store_true",
                        help="Export from a model trained without modality "
                             "dropout. The numbers then describe off-"
                             "distribution behaviour; label them as such.")
    args = parser.parse_args(argv)

    try:
        export(Path(args.run), split=args.split, batch_size=args.batch_size,
               num_workers=args.num_workers, device_name=args.device,
               allow_untrained_masking=args.allow_untrained_masking)
    except ExportError as exc:
        print(f"\n{exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
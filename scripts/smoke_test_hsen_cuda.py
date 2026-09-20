#!/usr/bin/env python
"""CUDA smoke test for the Phase-16 full-data IEMOCAP run.  One batch, measured.

    python scripts/smoke_test_hsen_cuda.py --dataset iemocap --data_fraction 1.0
    python scripts/smoke_test_hsen_cuda.py --batch_sizes 32,24,16,8
    python scripts/smoke_test_hsen_cuda.py --device cpu          # dry run, no VRAM figures

Resolves the configuration through exactly the code path
``scripts/train_hsen_cuda.py`` uses -- same parser, same profile, same YAML
precedence -- refuses anything other than audio+text / concat / iemocap_erc6 /
full data, builds the experiment from the real caches (train and validation
only; the test split is never opened), and then runs one real training step
per candidate batch size on the *worst-case* batch: the B longest audio
sequences the training split holds.  A batch that fits on average and OOMs on
the longest utterances is a run that dies at epoch 7, so the figure reported
here is the one that matters.

For each batch size it records, under AMP and again in FP32: whether forward,
loss, backward and the optimizer step completed; that nothing went non-finite;
that the parameters actually moved; and CUDA allocated / reserved / peak
allocated / peak reserved memory.  Parameter counts are reported by module so
the fusion trunk's size is never confused with the model's.

Nothing here trains.  The run directory is a temporary one and is discarded.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.hsen.cli import CUDA_PROFILE, apply_yaml, build_parser, configs_from_args
from src.hsen.data import collate_hsen
from src.hsen.experiment import build_experiment
from src.hsen.training.hsen_trainer import HSENTrainer

EXPECTED = {
    "experiment": "iemocap_erc6",
    "modalities": ("audio", "text"),
    "fusion": "concat",
    "data_fraction": 1.0,
}
#: A batch is "safe" when its peak reserved memory leaves this much of the
#: memory that was free at start untouched.  Fragmentation, the validation pass
#: (eval batch is 2x train) and the CUDA context all live in that margin.
SAFETY_FRACTION = 0.80


def gib(value: int | float) -> float:
    return float(value) / 1024 ** 3


def cuda_memory() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    return {
        "allocated_gb": gib(torch.cuda.memory_allocated()),
        "reserved_gb": gib(torch.cuda.memory_reserved()),
        "peak_allocated_gb": gib(torch.cuda.max_memory_allocated()),
        "peak_reserved_gb": gib(torch.cuda.max_memory_reserved()),
    }


def reset_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def longest_audio_batch(dataset, batch_size: int) -> tuple[dict, dict]:
    """The B training samples with the longest audio sequences, collated."""
    store = dataset.sources["audio"]
    lengths = [(store.length_of(sid), index) for index, sid in enumerate(dataset.sample_ids)
               if sid in store]
    lengths.sort(reverse=True)
    chosen = [index for _, index in lengths[:batch_size]]
    samples = [dataset[index] for index in chosen]
    batch = collate_hsen(samples, dataset.modalities, dataset.feature_dims)
    shape = {m: list(batch["features"][m].shape) for m in dataset.modalities}
    return batch, shape


def first_real_batch(trainer: HSENTrainer) -> tuple[dict, dict]:
    """The first batch the real seeded, shuffled training loader would yield."""
    batch = next(iter(trainer.loader("train", shuffle=True)))
    shape = {m: list(batch["features"][m].shape) for m in trainer.datasets["train"].modalities}
    return batch, shape


def finite_state(model: torch.nn.Module) -> dict[str, bool]:
    params = all(torch.isfinite(p).all().item() for p in model.parameters())
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    return {
        "parameters_finite": bool(params),
        "gradients_present": bool(grads),
        "gradients_finite": bool(all(torch.isfinite(g).all().item() for g in grads)),
    }


def run_steps(trainer: HSENTrainer, batch: dict, steps: int) -> dict:
    """``steps`` real training steps on one batch through the trainer's own loop.

    ``train_one_epoch`` is handed a list, so this exercises the exact code the
    full run will: autocast, GradScaler, unscale, clip, step, scheduler.  Under
    AMP the scaler may legitimately skip a step while it finds its scale; the
    parameter snapshot says whether at least one step was applied.
    """
    before = [p.detach().clone() for p in trainer.model.parameters()]
    scale_before = trainer.scaler.get_scale() if trainer.scaler else None
    losses = []
    started = time.perf_counter()
    for step in range(steps):
        totals = trainer.train_one_epoch([batch], epoch=step)
        losses.append(float(totals["train_total"]))
    if trainer.device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    moved = any(not torch.equal(a, b.detach()) for a, b in zip(before, trainer.model.parameters()))

    with torch.no_grad(), torch.autocast("cuda", enabled=trainer.amp):
        outputs = trainer.model(batch["features"], batch["masks"], batch["available"])
    return {
        "steps": steps,
        "losses": losses,
        "losses_finite": all(torch.isfinite(torch.tensor(losses)).tolist()),
        "logits_finite": bool(torch.isfinite(outputs["logits"]).all().item()),
        "parameters_moved": bool(moved),
        "grad_scale_before": scale_before,
        "grad_scale_after": trainer.scaler.get_scale() if trainer.scaler else None,
        "seconds_per_step": elapsed / steps,
        **finite_state(trainer.model),
    }


def measure(bundle, trainer_config, loss_config, batch_size: int, amp: bool, steps: int) -> dict:
    """Fresh model + optimizer at one batch size and precision; one worst-case
    batch and one real loader batch; memory around each."""
    reset_cuda()
    config = trainer_config.__class__(**{**trainer_config.to_dict(), "batch_size": batch_size,
                                         "eval_batch_size": None, "amp": amp})
    trainer = HSENTrainer(
        config=config, model_config=bundle.model_config, loss_config=loss_config,
        datasets=bundle.datasets, manifests=bundle.manifests,
        class_names=bundle.class_names, audit=bundle.audit,
    )
    result: dict = {
        "batch_size": batch_size, "amp_requested": amp, "amp_active": trainer.amp,
        "baseline_memory": cuda_memory(), "status": "ok",
    }
    try:
        worst, worst_shape = longest_audio_batch(bundle.datasets["train"], batch_size)
        reset_cuda()
        result["worst_case"] = {"shapes": worst_shape,
                                **run_steps(trainer, trainer._to_device(worst), steps)}
        result["worst_case"]["memory"] = cuda_memory()
        del worst

        real, real_shape = first_real_batch(trainer)
        reset_cuda()
        result["real_batch"] = {"shapes": real_shape,
                                **run_steps(trainer, trainer._to_device(real), 1)}
        result["real_batch"]["memory"] = cuda_memory()
        del real
    except torch.cuda.OutOfMemoryError as error:
        result["status"] = "oom"
        result["error"] = str(error).splitlines()[0]
    finally:
        del trainer
        reset_cuda()
    return result


def verdict(result: dict, free_gb: float | None) -> tuple[bool, str]:
    if result["status"] != "ok":
        return False, result.get("error", result["status"])
    checks = []
    for key in ("worst_case", "real_batch"):
        block = result[key]
        checks += [block["losses_finite"], block["logits_finite"], block["parameters_finite"],
                   block["gradients_finite"], block["parameters_moved"]]
    if not all(checks):
        return False, "a finiteness or parameter-update check failed"
    if free_gb is None:
        return True, "completed (CPU dry run: no VRAM measurement)"
    peak = result["worst_case"]["memory"]["peak_reserved_gb"]
    budget = SAFETY_FRACTION * free_gb
    if peak > budget:
        return False, f"peak reserved {peak:.2f} GB exceeds {SAFETY_FRACTION:.0%} of free ({budget:.2f} GB)"
    return True, f"peak reserved {peak:.2f} GB within {SAFETY_FRACTION:.0%} of free ({budget:.2f} GB)"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser(CUDA_PROFILE, __doc__.splitlines()[0])
    smoke = parser.add_argument_group("smoke test")
    smoke.add_argument("--batch_sizes", "--batch-sizes", dest="batch_sizes", default=None,
                       help="Comma-separated physical batch sizes to try, largest first. "
                            "Default: the profile's batch size, then 24, 16, 8.")
    smoke.add_argument("--steps", type=int, default=3,
                       help="Training steps on the worst-case batch (AMP needs a couple "
                            "to settle its loss scale).")
    smoke.add_argument("--report", default=None,
                       help="Where to write the JSON report. Default: "
                            "results/hsen/<experiment>/cuda_smoke_test.json")
    args = apply_yaml(parser, parser.parse_args(argv))
    trainer_config, loss_config, modalities, overrides = configs_from_args(args, CUDA_PROFILE)

    # ---- 1. resolution: refuse to smoke-test anything but the Phase-16 configuration
    resolved = {
        "experiment": trainer_config.experiment, "modalities": tuple(modalities),
        "fusion": args.fusion, "data_fraction": trainer_config.data_fraction,
    }
    mismatch = {k: (resolved[k], v) for k, v in EXPECTED.items() if resolved[k] != v}
    if mismatch:
        raise SystemExit(f"Not the Phase-16 configuration: {mismatch} (got, expected)")
    print(f"resolved: {resolved} | device {args.device} | amp {args.amp} | "
          f"profile batch size {args.batch_size}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda but torch reports no CUDA device.")
    free_gb = total_gb = None
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        free_gb, total_gb = gib(free), gib(total)
        props = torch.cuda.get_device_properties(device)
        print(f"CUDA {torch.version.cuda} | {props.name} | {total_gb:.2f} GB total, "
              f"{free_gb:.2f} GB free at start | capability {props.major}.{props.minor} | "
              f"torch {torch.__version__}")
    else:
        print(f"[dry run] device={device}: the step is exercised, VRAM is not measured.")

    # ---- 2. the real experiment: audit, manifests, caches (train + validation only)
    bundle = build_experiment(
        experiment=trainer_config.experiment, modalities=tuple(modalities),
        data_fraction=trainer_config.data_fraction, seed=trainer_config.seed,
        fusion=args.fusion, model_overrides=overrides,
        feature_root=Path(args.feature_dir) if args.feature_dir else None,
        include_test=False,
    )
    opened = {split: sorted(ds.sources) for split, ds in bundle.datasets.items()}
    assert all(v == ["audio", "text"] for v in opened.values()), opened
    assert "test" not in bundle.datasets
    print(f"feature caches opened: {opened}")

    # ---- 3. parameter accounting, from one throwaway model
    tmp = Path(tempfile.mkdtemp(prefix="hsen_smoke_"))
    trainer_config.output_root = tmp
    trainer_config.profile = "cuda_smoke"
    trainer_config.device = str(device)
    trainer_config.num_workers = 0
    probe = HSENTrainer(
        config=trainer_config, model_config=bundle.model_config, loss_config=loss_config,
        datasets=bundle.datasets, manifests=bundle.manifests,
        class_names=bundle.class_names, audit=bundle.audit,
    )
    counts = probe.model.parameter_counts()
    print(
        f"parameters: total {counts['total']:,} | trainable {counts['trainable']:,} | "
        f"fusion ({bundle.model_config.fusion}) {counts['fusion']:,} | "
        f"projections {counts['projections']:,} | emotion head {counts['emotion_head']:,} | "
        f"valence head {counts['valence_head']:,} | arousal head {counts['arousal_head']:,}"
    )
    del probe
    reset_cuda()

    # ---- 4. the measured steps
    if args.batch_sizes:
        candidates = [int(x) for x in args.batch_sizes.split(",")]
    else:
        candidates = sorted({args.batch_size, 24, 16, 8}, reverse=True)
        candidates = [b for b in candidates if b <= args.batch_size]
    results = []
    for batch_size in candidates:
        for amp in ((True, False) if device.type == "cuda" else (False,)):
            label = f"batch {batch_size:>3}  {'AMP ' if amp else 'FP32'}"
            result = measure(bundle, trainer_config, loss_config, batch_size, amp, args.steps)
            ok, reason = verdict(result, free_gb)
            result["safe"], result["verdict"] = ok, reason
            results.append(result)
            if result["status"] == "ok":
                w = result["worst_case"]
                mem = w.get("memory", {})
                print(
                    f"  {label}  worst-case audio {w['shapes']['audio']} "
                    f"text {w['shapes']['text']}  loss {w['losses'][-1]:.4f}  "
                    f"{w['seconds_per_step']*1000:.0f} ms/step  "
                    + (f"peak alloc {mem['peak_allocated_gb']:.2f} GB  "
                       f"peak reserved {mem['peak_reserved_gb']:.2f} GB  " if mem else "")
                    + f"-> {'SAFE' if ok else 'NOT SAFE'}: {reason}"
                )
            else:
                print(f"  {label}  {result['status'].upper()}: {reason}")

    safe_amp = [r["batch_size"] for r in results if r["safe"] and r["amp_active"]]
    safe_any = [r["batch_size"] for r in results if r["safe"]]
    recommended = max(safe_amp or safe_any) if (safe_amp or safe_any) else None
    amp_worked = any(r["status"] == "ok" and r["amp_active"]
                     and r["worst_case"]["parameters_moved"] for r in results)

    report = {
        "resolved_configuration": {**resolved, "modalities": list(modalities),
                                   "device": str(device), "amp": bool(args.amp),
                                   "profile_batch_size": args.batch_size,
                                   "eval_batch_size": trainer_config.eval_batch_size},
        "feature_caches_opened": opened,
        "cuda": {"torch": torch.__version__, "cuda": torch.version.cuda,
                 "total_gb": total_gb, "free_at_start_gb": free_gb,
                 "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
        "parameter_counts": counts,
        "train_samples": len(bundle.datasets["train"]),
        "validation_samples": len(bundle.datasets["validation"]),
        "results": results,
        "amp_worked": amp_worked,
        "recommended_batch_size": recommended,
        "safety_fraction": SAFETY_FRACTION,
    }
    out = Path(args.report) if args.report else (
        Path("results/hsen") / trainer_config.experiment / "cuda_smoke_test.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("-" * 74)
    print(f"AMP worked: {amp_worked}" if device.type == "cuda" else "AMP: not exercised (CPU)")
    print(f"recommended physical batch size: {recommended}"
          + ("" if device.type == "cuda" else "  (CPU dry run: functional only, not a VRAM verdict)"))
    print(f"report: {out}")
    return 0 if recommended is not None else 1


if __name__ == "__main__":
    sys.exit(main())

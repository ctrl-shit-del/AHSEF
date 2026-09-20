"""CPU-friendly, reproducible image-only seven-class emotion baseline."""

from __future__ import annotations

import json
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.data.dataloader import create_emotion_image_dataloader
from src.models.multimodal import ImageEmotionBaseline
from src.preprocessing.standardization.targets import CANONICAL_EMOTIONS
from src.training.metrics import class_weights, classification_metrics


@dataclass(frozen=True)
class EmotionImageConfig:
    name: str = "emotion_image_baseline"
    manifest_dir: str = "metadata/experiments/emotion_7class"
    results_dir: str = "results/emotion_image_baseline"
    modality: str = "image"
    image_size: int = 32
    hidden_dim: int = 128
    num_classes: int = 7
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 5
    seed: int = 42
    max_train_samples: int | None = None
    max_validation_samples: int | None = None
    max_test_samples: int | None = None


def _save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def _evaluate(model, loader, criterion, device) -> dict:
    model.eval(); losses=[]; predictions=[]; targets=[]
    with torch.no_grad():
        for batch in loader:
            image, label = batch["image"].to(device), batch["label"].to(device)
            logits = model(image); loss = criterion(logits, label)
            losses.append(float(loss.item()) * label.numel()); predictions.append(logits.argmax(1).cpu()); targets.append(label.cpu())
    metrics = classification_metrics(torch.cat(predictions), torch.cat(targets))
    metrics["loss"] = sum(losses) / len(loader.dataset)
    metrics["samples"] = len(loader.dataset)
    return metrics


def run(config: EmotionImageConfig) -> dict:
    """Train exclusively from train metadata, select by validation macro-F1, test once."""
    _seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifests = Path(config.manifest_dir)
    result_dir = Path(config.results_dir); result_dir.mkdir(parents=True, exist_ok=True)
    train = create_emotion_image_dataloader(manifests / "train.parquet", config.batch_size, config.image_size, config.max_train_samples, config.seed, True)
    validation = create_emotion_image_dataloader(manifests / "validation.parquet", config.batch_size, config.image_size, config.max_validation_samples, config.seed, False)
    test = create_emotion_image_dataloader(manifests / "test.parquet", config.batch_size, config.image_size, config.max_test_samples, config.seed, False)
    labels = torch.tensor(train.dataset.manifest["canonical_emotion_id"].tolist(), dtype=torch.long)
    weights = class_weights(labels, config.num_classes)
    _save_json(result_dir / "config.json", {**asdict(config), "class_mapping": CANONICAL_EMOTIONS, "device": str(device), "python": platform.python_version(), "torch": torch.__version__})
    _save_json(result_dir / "class_weights.json", {"weights": weights.tolist(), "training_distribution": torch.bincount(labels, minlength=7).tolist()})
    model = ImageEmotionBaseline(config.image_size, config.hidden_dim, config.num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device)); optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    history=[]; best=-1.0; started=time.monotonic()
    for epoch in range(1, config.epochs + 1):
        model.train(); total=0.0
        for batch in train:
            image, label = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True); loss=criterion(model(image), label)
            if not torch.isfinite(loss): raise RuntimeError("Non-finite classification loss")
            loss.backward(); optimizer.step(); total += float(loss.item()) * label.numel()
        validation_metrics = _evaluate(model, validation, criterion, device)
        history.append({"epoch": epoch, "train_loss": total / len(train.dataset), "validation": validation_metrics})
        state={"epoch":epoch,"model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict(),"validation":validation_metrics,"config":asdict(config)}
        torch.save(state, result_dir / "last.pt")
        if validation_metrics["macro_f1"] > best:
            best=validation_metrics["macro_f1"]; torch.save(state, result_dir / "best.pt")
    best_state=torch.load(result_dir / "best.pt", map_location=device); model.load_state_dict(best_state["model_state_dict"])
    validation_metrics=_evaluate(model, validation, criterion, device); test_metrics=_evaluate(model, test, criterion, device)
    _save_json(result_dir / "training_history.json", {"history": history})
    _save_json(result_dir / "validation_metrics.json", validation_metrics); _save_json(result_dir / "test_metrics.json", test_metrics)
    _save_json(result_dir / "confusion_matrix.json", {"validation": validation_metrics["confusion_matrix"], "test": test_metrics["confusion_matrix"]})
    summary={"duration_seconds": time.monotonic()-started, "best_epoch":best_state["epoch"], "validation":validation_metrics, "test":test_metrics, "samples":{"train":len(train.dataset),"validation":len(validation.dataset),"test":len(test.dataset)}}
    _save_json(result_dir / "run_summary.json", summary)
    return summary


if __name__ == "__main__":
    run(EmotionImageConfig())

"""Flexible vision training utility used by the AutoML Pro demos."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Literal, Optional

import pandas as pd
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ART_DIR = REPO_ROOT / "artifacts" / "vision"
REPORTS_DIR = REPO_ROOT / "reports"
DATA_ROOT = REPO_ROOT / "data"

MODEL_CHOICES = ("resnet18", "mobilenetv3_small", "cnn")
AUGMENT_CHOICES = ("none", "basic", "strong")

PRESETS = {
    "baseline": {
        "model": "resnet18",
        "epochs": 8,
        "batch_size": 128,
        "learning_rate": 1e-3,
        "optimizer": "adam",
        "augment": "basic",
        "finetune": True,
    },
    "efficient": {
        "model": "mobilenetv3_small",
        "epochs": 12,
        "batch_size": 160,
        "learning_rate": 5e-4,
        "optimizer": "adamw",
        "augment": "strong",
        "finetune": True,
    },
    "demo": {
        "model": "resnet18",
        "epochs": 3,
        "batch_size": 96,
        "learning_rate": 1e-3,
        "optimizer": "adam",
        "augment": "basic",
        "finetune": False,
    },
}


def apply_preset(args: argparse.Namespace) -> Dict[str, object]:
    resolved = {
        "model": args.model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "optimizer": args.optimizer,
        "augment": args.augment,
        "finetune": args.finetune,
    }
    if args.preset:
        preset = PRESETS[args.preset]
        for key, value in preset.items():
            if key == "finetune":
                if not args.finetune:
                    resolved["finetune"] = value
                continue
            if resolved.get(key) is None:
                resolved[key] = value
    defaults = {
        "model": "resnet18",
        "epochs": 5,
        "batch_size": 128,
        "learning_rate": 1e-3,
        "optimizer": "adam",
        "augment": "basic",
        "finetune": False,
    }
    for key, value in defaults.items():
        if resolved.get(key) is None:
            resolved[key] = value
    resolved["preset"] = args.preset or "custom"
    resolved["finetune"] = bool(resolved["finetune"])
    return resolved


def build_transforms(transforms, level: str, image_size: int, train: bool):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ops = [transforms.Resize((image_size, image_size))]
    if train:
        if level in {"basic", "strong"}:
            ops.extend([transforms.RandomHorizontalFlip(), transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0))])
        if level == "strong":
            ops.extend([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), transforms.RandomErasing(p=0.25)])
    ops.append(transforms.ToTensor())
    ops.append(normalize)
    return transforms.Compose(ops)


def build_dataloaders(
    datasets,
    transforms,
    data_path: Optional[str],
    augment_level: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
):
    import torch

    if data_path and Path(data_path).expanduser().exists():
        train_tfms = build_transforms(transforms, augment_level, image_size, train=True)
        test_tfms = build_transforms(transforms, "none", image_size, train=False)
        root_path = Path(data_path).expanduser()
        train_dataset = datasets.ImageFolder(root_path, transform=train_tfms)
        test_dataset = datasets.ImageFolder(root_path, transform=test_tfms)
        dataset_name = root_path.name
        class_names = train_dataset.classes
    else:
        train_tfms = build_transforms(transforms, augment_level, image_size, train=True)
        test_tfms = build_transforms(transforms, "none", image_size, train=False)
        train_dataset = datasets.CIFAR10(root=str(DATA_ROOT), train=True, download=True, transform=train_tfms)
        test_dataset = datasets.CIFAR10(root=str(DATA_ROOT), train=False, download=True, transform=test_tfms)
        class_names = train_dataset.classes
        dataset_name = "CIFAR10"

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=max(64, batch_size),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader, class_names, dataset_name


def build_model(models, name: str, num_classes: int, finetune: bool, image_size: int, device: str):
    import torch.nn as nn  # type: ignore

    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        if not finetune:
            for param in model.parameters():
                param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "mobilenetv3_small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        if not finetune:
            for param in model.features.parameters():
                param.requires_grad = False
        last_dim = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(last_dim, num_classes)
    else:
        class SmallCNN(nn.Module):
            def __init__(self, classes: int):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Flatten(),
                    nn.Linear(64 * (image_size // 4) * (image_size // 4), 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, classes),
                )

            def forward(self, x):  # type: ignore[override]
                return self.net(x)

        model = SmallCNN(num_classes)
    model.to(device)
    return model


def build_optimizer(model, kind: str, lr: float, weight_decay: float):
    import torch.optim as optim  # type: ignore

    params = [p for p in model.parameters() if p.requires_grad]
    if kind == "sgd":
        return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    if kind == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    return optim.Adam(params, lr=lr, weight_decay=weight_decay)


def train_one_epoch(model, loader, criterion, optimizer, device: str) -> float:
    import torch

    model.train()
    running_loss = 0.0
    count = 0
    for batch, labels in loader:
        batch = batch.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * labels.size(0)
        count += labels.size(0)
    return running_loss / max(1, count)


def evaluate(model, loader, device: str, return_preds: bool = False):
    import torch
    import torch.nn.functional as F  # type: ignore

    model.eval()
    correct = 0
    total = 0
    preds_all = []
    labels_all = []
    probs_all = []
    with torch.no_grad():
        for batch, labels in loader:
            batch = batch.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(batch)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            if return_preds:
                preds_all.append(preds.cpu())
                labels_all.append(labels.cpu())
                probs_all.append(F.softmax(logits, dim=1).cpu())
    accuracy = correct / max(1, total)
    if return_preds:
        preds_cat = torch.cat(preds_all).tolist()
        labels_cat = torch.cat(labels_all).tolist()
        probs_cat = torch.cat(probs_all).tolist()
        return accuracy, {"preds": preds_cat, "labels": labels_cat, "probs": probs_cat}
    return accuracy, None


def build_sample_predictions(pred_payload: Dict[str, list], class_names, max_samples: int):
    samples = []
    num_classes = len(class_names)
    for actual, pred in zip(pred_payload["labels"], pred_payload["preds"]):
        actual_idx = int(actual)
        pred_idx = int(pred)
        if not (0 <= actual_idx < num_classes and 0 <= pred_idx < num_classes):
            continue
        samples.append({"actual": class_names[actual_idx], "pred": class_names[pred_idx]})
        if len(samples) >= max_samples:
            break
    return samples


def update_leaderboard(dataset: str, model_name: str, accuracy: float, config: Dict[str, object], path: Path) -> None:
    row = {
        "dataset": dataset,
        "framework": f"Vision_{model_name}",
        "accuracy": accuracy,
        "epochs": config["epochs"],
        "preset": config["preset"],
        "augment": config["augment"],
    }
    df_new = pd.DataFrame([row])
    if path.exists():
        existing = pd.read_csv(path)
        # remove duplicates for same dataset+framework to keep latest run
        existing = existing[~((existing["dataset"] == dataset) & (existing["framework"] == row["framework"]))]
        df_new = pd.concat([existing, df_new], ignore_index=True)
    df_new.to_csv(path, index=False)


def save_metrics(dataset: str, config: Dict[str, object], accuracy: float, history, cm, samples, class_names):
    payload = {
        "dataset": dataset,
        "model": config["model"],
        "preset": config["preset"],
        "epochs": config["epochs"],
        "final_accuracy": accuracy,
        "class_names": class_names,
        "history": history,
        "confusion_matrix": cm,
        "timestamp": time.time(),
    }
    (REPORTS_DIR / "vision_metrics.json").write_text(json.dumps(payload, indent=2))
    (REPORTS_DIR / "vision_samples.json").write_text(json.dumps(samples, indent=2))
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vision training helper for AutoML demos.")
    parser.add_argument("--vision-data-path", type=str, default=None, help="Optional folder with class-subdirectories. If omitted, CIFAR10 is used.")
    parser.add_argument("--dataset-name", type=str, default=None, help="Override dataset name for logging.")
    parser.add_argument("--model", type=str, choices=MODEL_CHOICES, default=None, help="Backbone architecture.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate.")
    parser.add_argument("--optimizer", type=str, choices=["adam", "adamw", "sgd"], default=None, help="Optimizer type.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--preset", choices=list(PRESETS.keys()), default=None, help="Preset covering model + hyperparameters.")
    parser.add_argument("--augment", choices=AUGMENT_CHOICES, default=None, help="Augmentation strength.")
    parser.add_argument("--finetune", action="store_true", help="Fine-tune entire backbone (otherwise only classifier head).")
    parser.add_argument("--image-size", type=int, default=224, help="Resize images to this square size.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers.")
    parser.add_argument("--max-samples", type=int, default=12, help="Sample predictions to log.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ART_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    try:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore
        from torchvision import datasets, models, transforms  # type: ignore
        from sklearn.metrics import confusion_matrix  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"[Torch] Not available → skipping vision demo: {exc}")
        return

    resolved = apply_preset(args)
    device = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")

    train_loader, test_loader, class_names, dataset_name = build_dataloaders(
        datasets, transforms, args.vision_data_path, resolved["augment"], args.image_size, resolved["batch_size"], args.num_workers
    )
    if args.dataset_name:
        dataset_name = args.dataset_name

    model = build_model(models, resolved["model"], len(class_names), resolved["finetune"], args.image_size, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, resolved["optimizer"], resolved["learning_rate"], args.weight_decay)

    history = []
    for epoch in range(resolved["epochs"]):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_acc, _ = evaluate(model, test_loader, device)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_accuracy": val_acc})
        print(f"[Vision] epoch={epoch+1}/{resolved['epochs']} train_loss={train_loss:.4f} val_acc={val_acc:.4f} ({time.time() - start:.1f}s)")

    final_acc, preds_labels = evaluate(model, test_loader, device, return_preds=True)
    labels_range = list(range(len(class_names)))
    cm = confusion_matrix(preds_labels["labels"], preds_labels["preds"], labels=labels_range).tolist()
    sample_payload = build_sample_predictions(preds_labels, class_names, args.max_samples)

    scripted = torch.jit.script(model.cpu())
    scripted.save(str(ART_DIR / f"{dataset_name}_{resolved['model']}.pt"))

    update_leaderboard(dataset_name, resolved["model"], final_acc, resolved, REPORTS_DIR / "leaderboard_vision.csv")
    save_metrics(dataset_name, resolved, final_acc, history, cm, sample_payload, class_names)

    print(f"[Vision] Training complete — dataset={dataset_name} model={resolved['model']} accuracy={final_acc:.4f}")


if __name__ == "__main__":
    main()

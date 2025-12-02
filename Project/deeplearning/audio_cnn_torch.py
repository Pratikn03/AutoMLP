"""Compact 1D CNN trainer for synthetic audio datasets."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import wave
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset, Subset

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = REPO_ROOT / "reports"
ART_DIR = REPO_ROOT / "artifacts" / "audio"
DATA_ROOT = REPO_ROOT / "src" / "data" / "datasets" / "audio"


def _load_wave(path: Path, target_len: int, sample_rate: int) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        frames = wf.getnframes()
        samples = np.frombuffer(wf.readframes(frames), dtype=np.int16).astype(np.float32)
    if sample_rate != sr and sr > 0:
        # naive resample via interpolation
        t_old = np.linspace(0, 1, samples.size, endpoint=False)
        t_new = np.linspace(0, 1, int(samples.size * sample_rate / sr), endpoint=False)
        samples = np.interp(t_new, t_old, samples)
    samples = samples / 32768.0
    if samples.size < target_len:
        pad = np.zeros(target_len - samples.size, dtype=np.float32)
        samples = np.concatenate([samples, pad])
    else:
        samples = samples[:target_len]
    return samples


class AudioFolderDataset(Dataset):
    def __init__(self, root: Path, sample_rate: int, duration: float):
        self.root = root
        self.sample_rate = sample_rate
        self.target_len = int(sample_rate * duration)
        self.items: List[Tuple[Path, str]] = []
        self.classes: List[str] = []
        if root.exists():
            for class_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
                self.classes.append(class_dir.name)
                for wav in class_dir.glob("*.wav"):
                    self.items.append((wav, class_dir.name))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        wav_path, label = self.items[idx]
        waveform = _load_wave(wav_path, self.target_len, self.sample_rate)
        tensor = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
        return tensor, self.class_to_idx[label]


class AudioCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=9, stride=2, padding=4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(32),
            nn.Flatten(),
            nn.Linear(64 * 32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):  # type: ignore[override]
        return self.net(x)


def build_loaders(dataset: AudioFolderDataset, batch_size: int, seed: int) -> Tuple[DataLoader, DataLoader]:
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    split = max(1, int(0.8 * len(indices)))
    train_indices = indices[:split]
    val_indices = indices[split:] or indices[: max(1, len(indices) // 5)]
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def train_one_epoch(model, loader, criterion, optimizer, device: str) -> float:
    model.train()
    running = 0.0
    count = 0
    for batch, labels in loader:
        batch = batch.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running += loss.item() * labels.size(0)
        count += labels.size(0)
    return running / max(1, count)


def evaluate(model, loader, device: str) -> Tuple[float, List[int], List[int]]:
    model.eval()
    correct = 0
    total = 0
    preds_all: List[int] = []
    labels_all: List[int] = []
    with torch.no_grad():
        for batch, labels in loader:
            batch = batch.to(device)
            labels = labels.to(device)
            logits = model(batch)
            preds = logits.argmax(dim=1)
            preds_all.extend(preds.cpu().tolist())
            labels_all.extend(labels.cpu().tolist())
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / max(1, total)
    return accuracy, preds_all, labels_all


def save_metrics(dataset: str, config: Dict[str, object], accuracy: float, history: List[Dict[str, float]], cm, samples):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset,
        "preset": config,
        "accuracy": accuracy,
        "history": history,
        "confusion_matrix": cm,
        "samples": samples,
        "timestamp": time.time(),
    }
    (REPORTS_DIR / "audio_metrics.json").write_text(json.dumps(payload, indent=2))


def update_leaderboard(dataset: str, accuracy: float, epochs: int):
    row = {
        "dataset": dataset,
        "framework": "Audio_CNN",
        "accuracy": accuracy,
        "epochs": epochs,
    }
    dest = REPORTS_DIR / "leaderboard_audio.csv"
    if dest.exists():
        df = pd.read_csv(dest)
        df = df[df["dataset"] != dataset]
    else:
        df = pd.DataFrame()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(dest, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny audio CNN on the staged dataset.")
    parser.add_argument("--dataset", default="fsdd", help="Name of the dataset folder under src/data/datasets/audio.")
    parser.add_argument("--data-root", default=str(DATA_ROOT), help="Base folder containing audio datasets.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--duration", type=float, default=1.0, help="Clip duration (seconds).")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.data_root) / args.dataset
    dataset = AudioFolderDataset(dataset_root, sample_rate=args.sample_rate, duration=args.duration)
    if len(dataset) == 0:
        raise RuntimeError(f"No audio samples found in {dataset_root}. Run stage_datasets.py --audio first.")
    train_loader, val_loader = build_loaders(dataset, args.batch_size, args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AudioCNN(len(dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    history: List[Dict[str, float]] = []
    for epoch in range(args.epochs):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        acc, _, _ = evaluate(model, val_loader, device)
        history.append({"epoch": epoch + 1, "loss": loss, "val_accuracy": acc})
        print(f"[audio] epoch={epoch+1}/{args.epochs} loss={loss:.4f} val_acc={acc:.4f}")

    final_acc, preds, labels = evaluate(model, val_loader, device)
    cm = confusion_matrix(labels, preds, labels=list(range(len(dataset.classes)))).tolist()
    samples = [
        {"actual": dataset.classes[label], "pred": dataset.classes[pred]}
        for label, pred in zip(labels[:10], preds[:10])
    ]
    ART_DIR.mkdir(parents=True, exist_ok=True)
    scripted = torch.jit.script(model.cpu())
    scripted.save(str(ART_DIR / f"{args.dataset}_cnn.pt"))
    save_metrics(args.dataset, vars(args), final_acc, history, cm, samples)
    update_leaderboard(args.dataset, final_acc, args.epochs)
    print(f"[audio] Training complete — dataset={args.dataset} accuracy={final_acc:.4f}")


if __name__ == "__main__":
    main()

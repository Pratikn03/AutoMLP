"""Fine-tune a lightweight ResNet18 on CIFAR-10 for demo purposes."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ART_DIR = REPO_ROOT / "artifacts" / "vision"
REPORTS_DIR = REPO_ROOT / "reports"
DATA_ROOT = REPO_ROOT / "data"


def main() -> None:
    ART_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    try:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore
        import torch.optim as optim  # type: ignore
        from torchvision import datasets, models, transforms  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"[Torch] Not available → skipping vision demo: {exc}")
        return

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    transform_train = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    transform_test = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.CIFAR10(root=str(DATA_ROOT), train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root=str(DATA_ROOT), train=False, download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = int(os.getenv("VISION_EPOCHS", "2"))
    for _ in range(epochs):
        model.train()
        for batch, labels in train_loader:
            batch = batch.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch), labels)
            loss.backward()
            optimizer.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch, labels in test_loader:
            batch = batch.to(device)
            labels = labels.to(device)
            predictions = model(batch).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / max(1, total)

    scripted = torch.jit.script(model.cpu())
    scripted.save(str(ART_DIR / "model.pt"))
    (REPORTS_DIR / "leaderboard_vision.csv").write_text(
        "framework,top1_acc\nResNet18_CIFAR10,{:.4f}\n".format(accuracy)
    )
    print(f"[Torch] Vision demo complete. Accuracy={accuracy:.4f}")


if __name__ == "__main__":
    main()

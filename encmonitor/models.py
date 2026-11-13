from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def require_torch():
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:
        raise RuntimeError("PyTorch is required for model code") from exc
    return torch, nn


def build_model(name: str, n_classes: int = 10):
    torch, nn = require_torch()
    name = name.lower()

    class Normalize(nn.Module):
        def __init__(self, mean, std):
            super().__init__()
            self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
            self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

        def forward(self, x):
            return (x - self.mean) / self.std

    class LeNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.normalize = Normalize([0.1307], [0.3081])
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            )
            self.embedding = nn.Sequential(nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU())
            self.classifier = nn.Linear(128, n_classes)

        def forward_features(self, x):
            x = self.normalize(x)
            return self.embedding(self.features(x))

        def forward(self, x):
            return self.classifier(self.forward_features(x))

    if name in {"lenet", "mnist_lenet"}:
        return LeNet()

    if name in {"resnet18_cifar", "resnet18"}:
        from torchvision.models import resnet18

        backbone = resnet18(weights=None, num_classes=n_classes)
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()

        class CIFARResNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.normalize = Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
                self.backbone = backbone

            def forward_features(self, x):
                x = self.normalize(x)
                b = self.backbone
                x = b.conv1(x)
                x = b.bn1(x)
                x = b.relu(x)
                x = b.maxpool(x)
                x = b.layer1(x)
                x = b.layer2(x)
                x = b.layer3(x)
                x = b.layer4(x)
                x = b.avgpool(x)
                return torch.flatten(x, 1)

            def forward(self, x):
                return self.backbone.fc(self.forward_features(x))

        return CIFARResNet()
    raise ValueError(f"Unsupported model={name!r}")

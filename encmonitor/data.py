from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .shifts import ShiftSpec, apply_tensor_shift


def load_dataset(name: str, split: str, root: str, download: bool = False):
    try:
        from torchvision import datasets, transforms
    except Exception as exc:
        raise RuntimeError("torchvision is required") from exc
    tfm = transforms.ToTensor()
    name = name.lower()
    train = split.lower() in {"train", "training"}
    if name == "mnist":
        return datasets.MNIST(root=root, train=train, download=download, transform=tfm)
    if name in {"cifar10", "cifar-10"}:
        return datasets.CIFAR10(root=root, train=train, download=download, transform=tfm)
    raise ValueError(f"Unsupported dataset={name!r}")


def labels_array(dataset):
    import numpy as np

    labels = getattr(dataset, "targets", None)
    if labels is None:
        labels = [dataset[i][1] for i in range(len(dataset))]
    return np.asarray(labels, dtype=np.int64)


class IndexedShiftDataset:
    """Dataset view over explicit indices with a deterministic image shift."""

    def __init__(self, base, indices, shift: ShiftSpec | None = None, seed: int = 0):
        import numpy as np

        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)
        self.shift = shift
        self.seed = int(seed)

    def __len__(self):
        return int(self.indices.size)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        x, y = self.base[idx]
        if self.shift is not None and self.shift.name != "none":
            x = apply_tensor_shift(x, self.shift, seed=self.seed + idx)
        return x, y, idx

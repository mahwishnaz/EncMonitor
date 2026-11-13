from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class EmbeddingStandardizer:
    mean: np.ndarray
    scale: np.ndarray
    clip_radius: float | None = None

    @classmethod
    def fit(cls, x: np.ndarray, clip_radius: float | None = None, eps: float = 1e-6) -> "EmbeddingStandardizer":
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError("x must have shape [n,d]")
        mean = arr.mean(axis=0)
        scale = arr.std(axis=0)
        scale = np.where(scale < eps, 1.0, scale)
        return cls(mean, scale, clip_radius)

    def transform(self, x: np.ndarray) -> np.ndarray:
        arr = (np.asarray(x, dtype=np.float64) - self.mean) / self.scale
        if self.clip_radius is not None:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            arr = arr * np.minimum(1.0, self.clip_radius / np.maximum(norms, 1e-12))
        return arr

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            mean=self.mean,
            scale=self.scale,
            clip_radius=np.array([np.nan if self.clip_radius is None else self.clip_radius]),
        )

    @classmethod
    def load(cls, path: str | Path) -> "EmbeddingStandardizer":
        with np.load(path) as z:
            value = float(np.asarray(z["clip_radius"]).reshape(-1)[0])
            return cls(
                np.asarray(z["mean"], dtype=np.float64),
                np.asarray(z["scale"], dtype=np.float64),
                None if np.isnan(value) else value,
            )


def extract_with_hook(model, loader, layer_name: str, device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    """Extract flattened activations from a named PyTorch module."""
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("PyTorch is required") from exc

    modules = dict(model.named_modules())
    if layer_name not in modules:
        raise KeyError(f"Unknown layer {layer_name!r}; examples={list(modules)[:12]}")
    cache: list[torch.Tensor] = []

    def hook(_module, _inputs, output):
        value = output[0] if isinstance(output, (tuple, list)) else output
        cache.append(value.detach().flatten(start_dim=1).cpu())

    handle = modules[layer_name].register_forward_hook(hook)
    ys: list[torch.Tensor] = []
    model.eval().to(device)
    try:
        with torch.no_grad():
            for batch in loader:
                x, y = batch[0].to(device), batch[1]
                _ = model(x)
                ys.append(y.detach().cpu())
    finally:
        handle.remove()
    return torch.cat(cache).numpy(), torch.cat(ys).numpy()

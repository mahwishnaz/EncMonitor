from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


def _as_2d_float(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2-D array [n,d], got shape={arr.shape}")
    if arr.shape[0] < 1 or arr.shape[1] < 1:
        raise ValueError("Input must have at least one sample and one feature")
    if not np.isfinite(arr).all():
        raise ValueError("Input contains NaN or infinity")
    return arr


def l2_clip_rows(x: np.ndarray, radius: float) -> np.ndarray:
    """Clip each row to an L2 ball without changing zero vectors."""
    arr = _as_2d_float(x)
    if radius <= 0:
        raise ValueError("radius must be positive")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    scales = np.minimum(1.0, radius / np.maximum(norms, 1e-12))
    return arr * scales


def median_bandwidth(
    x: np.ndarray,
    *,
    max_samples: int = 2_000,
    seed: int = 0,
    min_value: float = 1e-6,
) -> float:
    """Median pairwise Euclidean distance on a deterministic subsample.

    The returned value is the RBF sigma in exp(-||x-y||^2/(2 sigma^2)).
    """
    arr = _as_2d_float(x)
    rng = np.random.default_rng(seed)
    if arr.shape[0] > max_samples:
        arr = arr[rng.choice(arr.shape[0], size=max_samples, replace=False)]

    # Upper-triangle distances without a scipy dependency.
    sq = np.sum(arr * arr, axis=1, keepdims=True)
    d2 = np.maximum(sq + sq.T - 2.0 * (arr @ arr.T), 0.0)
    tri = d2[np.triu_indices(arr.shape[0], k=1)]
    tri = tri[tri > 0]
    if tri.size == 0:
        return float(min_value)
    return float(max(np.sqrt(np.median(tri)), min_value))


@dataclass(frozen=True)
class RFFMap:
    """Public random Fourier feature map for an RBF kernel.

    z(x) = sqrt(2/m) cos(x W + b),
    W_ij ~ N(0, 1/sigma^2), b_j ~ Uniform(0, 2pi).
    """

    omega: np.ndarray  # [d,m]
    phase: np.ndarray  # [m]
    bandwidth: float
    seed: int

    @property
    def input_dim(self) -> int:
        return int(self.omega.shape[0])

    @property
    def output_dim(self) -> int:
        return int(self.omega.shape[1])

    @classmethod
    def fit(cls, input_dim: int, output_dim: int, bandwidth: float, seed: int) -> "RFFMap":
        if input_dim < 1 or output_dim < 1:
            raise ValueError("input_dim and output_dim must be positive")
        if bandwidth <= 0:
            raise ValueError("bandwidth must be positive")
        rng = np.random.default_rng(seed)
        omega = rng.normal(0.0, 1.0 / float(bandwidth), size=(input_dim, output_dim))
        phase = rng.uniform(0.0, 2.0 * np.pi, size=(output_dim,))
        return cls(omega=omega, phase=phase, bandwidth=float(bandwidth), seed=int(seed))

    def transform(self, x: np.ndarray, batch_size: int = 8192) -> np.ndarray:
        arr = _as_2d_float(x)
        if arr.shape[1] != self.input_dim:
            raise ValueError(f"Expected d={self.input_dim}, got d={arr.shape[1]}")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        out = np.empty((arr.shape[0], self.output_dim), dtype=np.float64)
        scale = np.sqrt(2.0 / self.output_dim)
        for start in range(0, arr.shape[0], batch_size):
            stop = min(start + batch_size, arr.shape[0])
            out[start:stop] = scale * np.cos(arr[start:stop] @ self.omega + self.phase)
        return out

    def mean_sketch(self, x: np.ndarray) -> np.ndarray:
        return self.transform(x).mean(axis=0)

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            omega=self.omega,
            phase=self.phase,
            bandwidth=np.array([self.bandwidth], dtype=np.float64),
            seed=np.array([self.seed], dtype=np.int64),
        )

    @classmethod
    def load(cls, path: str) -> "RFFMap":
        with np.load(path) as z:
            return cls(
                omega=np.asarray(z["omega"], dtype=np.float64),
                phase=np.asarray(z["phase"], dtype=np.float64),
                bandwidth=float(np.asarray(z["bandwidth"]).reshape(-1)[0]),
                seed=int(np.asarray(z["seed"]).reshape(-1)[0]),
            )

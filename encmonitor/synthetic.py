from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Geometry = Literal["clean", "single", "coherent", "opposing", "mixed"]


@dataclass(frozen=True)
class SyntheticCase:
    residuals: np.ndarray
    shifted_mask: np.ndarray
    weights: np.ndarray
    direction: np.ndarray


def make_case(
    n_sites: int,
    dim: int,
    geometry: Geometry,
    severity: float,
    seed: int,
    noise_std: float = 0.01,
) -> SyntheticCase:
    """Generate controlled residual geometry for unit and cancellation tests."""
    if n_sites < 2 or dim < 1:
        raise ValueError("n_sites >= 2 and dim >= 1 required")
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim)
    v /= max(np.linalg.norm(v), 1e-12)
    delta = rng.normal(0.0, noise_std, size=(n_sites, dim))
    shifted = np.zeros(n_sites, dtype=bool)

    if geometry == "clean":
        pass
    elif geometry == "single":
        delta[0] += severity * v
        shifted[0] = True
    elif geometry == "coherent":
        k = max(2, n_sites // 2)
        delta[:k] += severity * v
        shifted[:k] = True
    elif geometry == "opposing":
        delta[0] += severity * v
        delta[1] -= severity * v
        shifted[:2] = True
    elif geometry == "mixed":
        k = max(3, n_sites // 2)
        signs = np.where(np.arange(k) % 2 == 0, 1.0, -0.55)
        delta[:k] += severity * signs[:, None] * v[None, :]
        shifted[:k] = True
    else:
        raise ValueError(f"Unknown geometry={geometry}")

    weights = np.ones(n_sites, dtype=np.float64) / n_sites
    return SyntheticCase(delta, shifted, weights, v)

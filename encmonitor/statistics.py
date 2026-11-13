from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class DualStatistics:
    d_mix: float
    d_site: float
    heterogeneity: float
    contributions: np.ndarray
    directional: np.ndarray
    mean_residual: np.ndarray

    def assert_identities(self, atol: float = 1e-10) -> None:
        if self.d_mix < -atol or self.d_site < -atol or self.heterogeneity < -atol:
            raise AssertionError("Expected nonnegative dual statistics")
        if not np.isclose(self.contributions.sum(), self.d_site, atol=atol, rtol=0):
            raise AssertionError("sum(A_s) != D_site")
        if not np.isclose(self.directional.sum(), self.d_mix, atol=atol, rtol=0):
            raise AssertionError("sum(G_s) != D_mix")
        if not np.isclose(self.d_site - self.d_mix, self.heterogeneity, atol=atol, rtol=0):
            raise AssertionError("D_site - D_mix != H")


def normalize_weights(weights: Sequence[float], n_sites: int) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.size != n_sites:
        raise ValueError(f"Expected {n_sites} weights, got {w.size}")
    if not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError("Weights must be finite and nonnegative")
    total = float(w.sum())
    if total <= 0:
        raise ValueError("At least one weight must be positive")
    return w / total


def compute_dual_statistics(
    residuals: np.ndarray,
    weights: Sequence[float],
    *,
    validate: bool = True,
) -> DualStatistics:
    """Compute cancellation-aware global, site-energy, and attribution scores.

    Args:
        residuals: [S,m] site-specific current-minus-baseline sketch residuals.
        weights: S nonnegative policy weights; normalized internally.
    """
    delta = np.asarray(residuals, dtype=np.float64)
    if delta.ndim != 2 or delta.shape[0] < 1 or delta.shape[1] < 1:
        raise ValueError(f"Expected residuals [S,m], got {delta.shape}")
    if not np.isfinite(delta).all():
        raise ValueError("Residuals contain NaN or infinity")
    alpha = normalize_weights(weights, delta.shape[0])

    mean = np.einsum("s,sm->m", alpha, delta)
    sq_norm = np.einsum("sm,sm->s", delta, delta)
    d_mix = float(mean @ mean)
    contributions = alpha * sq_norm
    d_site = float(contributions.sum())
    heterogeneity = float(max(d_site - d_mix, 0.0))
    directional = alpha * (delta @ mean)

    out = DualStatistics(
        d_mix=d_mix,
        d_site=d_site,
        heterogeneity=heterogeneity,
        contributions=contributions,
        directional=directional,
        mean_residual=mean,
    )
    if validate:
        out.assert_identities(atol=1e-8)
    return out


def residual_matrix(
    current: Mapping[str, np.ndarray], baseline: Mapping[str, np.ndarray], site_order: Sequence[str]
) -> np.ndarray:
    rows = []
    for site in site_order:
        if site not in current or site not in baseline:
            raise KeyError(f"Missing site={site!r} in current or baseline")
        a = np.asarray(current[site], dtype=np.float64).reshape(-1)
        b = np.asarray(baseline[site], dtype=np.float64).reshape(-1)
        if a.shape != b.shape:
            raise ValueError(f"Sketch shape mismatch for {site}: {a.shape} vs {b.shape}")
        rows.append(a - b)
    return np.stack(rows, axis=0)

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ..statistics import normalize_weights
from .backend import ArithmeticBackend


@dataclass(frozen=True)
class EncryptedDualStatistics:
    d_mix: Any
    d_site: Any
    heterogeneity: Any
    contributions: tuple[Any, ...]
    directional: tuple[Any, ...]
    mean_residual: Any


def power_of_two_width(width: int) -> int:
    if width < 1:
        raise ValueError("width must be positive")
    return 1 << (int(width) - 1).bit_length()


def rotation_steps(width: int) -> list[int]:
    """Powers of two required by the binary slot-reduction tree."""
    width = power_of_two_width(width)
    out: list[int] = []
    k = 1
    while k < width:
        out.append(k)
        k <<= 1
    return out


def slot_sum(backend: ArithmeticBackend, ct: Any, width: int) -> Any:
    """Sum a power-of-two packed vector and replicate the scalar in all slots.

    The CKKS context must use batch_size == power_of_two_width(width), and the
    input must be zero-padded to that same width.
    """
    total = ct
    for step in rotation_steps(width):
        total = backend.add(total, backend.rotate(total, step))
    return total


def _weighted_sum(backend: ArithmeticBackend, cts: Sequence[Any], weights: np.ndarray) -> Any:
    if not cts:
        raise ValueError("At least one ciphertext is required")
    total = backend.mul_scalar(cts[0], float(weights[0]))
    for ct, weight in zip(cts[1:], weights[1:]):
        total = backend.add(total, backend.mul_scalar(ct, float(weight)))
    return total


def evaluate_dual_statistics(
    backend: ArithmeticBackend,
    residual_cts: Sequence[Any],
    weights: Sequence[float],
    *,
    width: int,
    disclose_contributions: bool = True,
    disclose_directional: bool = False,
) -> EncryptedDualStatistics:
    """Evaluate EncMonitor's one-level dual-statistic circuit.

    Each residual ciphertext holds one site's padded delta_s vector. Public
    policy weights are normalized before use.
    """
    alpha = normalize_weights(weights, len(residual_cts))
    packed_width = power_of_two_width(width)
    mean = _weighted_sum(backend, residual_cts, alpha)
    d_mix = slot_sum(backend, backend.mul(mean, mean), packed_width)

    squares = [backend.mul(ct, ct) for ct in residual_cts]
    site_energy_vec = _weighted_sum(backend, squares, alpha)
    d_site = slot_sum(backend, site_energy_vec, packed_width)
    heterogeneity = backend.sub(d_site, d_mix)

    contributions: tuple[Any, ...] = ()
    if disclose_contributions:
        contributions = tuple(
            slot_sum(backend, backend.mul_scalar(sq, float(w)), packed_width)
            for sq, w in zip(squares, alpha)
        )

    directional: tuple[Any, ...] = ()
    if disclose_directional:
        directional = tuple(
            slot_sum(
                backend,
                backend.mul_scalar(backend.mul(ct, mean), float(w)),
                packed_width,
            )
            for ct, w in zip(residual_cts, alpha)
        )

    return EncryptedDualStatistics(
        d_mix=d_mix,
        d_site=d_site,
        heterogeneity=heterogeneity,
        contributions=contributions,
        directional=directional,
        mean_residual=mean,
    )

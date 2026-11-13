from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def topk_recall(contributions: Sequence[float], shifted_mask: Sequence[bool], k: int | None = None) -> float:
    a = np.asarray(contributions, dtype=np.float64)
    truth = np.asarray(shifted_mask, dtype=bool)
    if a.ndim != 1 or truth.shape != a.shape:
        raise ValueError("contributions and shifted_mask must be aligned vectors")
    n_true = int(truth.sum())
    if n_true == 0:
        return float("nan")
    k = n_true if k is None else int(k)
    k = max(1, min(k, a.size))
    pred = np.argsort(-a)[:k]
    return float(truth[pred].sum() / n_true)


def top1_accuracy(contributions: Sequence[float], shifted_mask: Sequence[bool]) -> float:
    a = np.asarray(contributions, dtype=np.float64)
    truth = np.asarray(shifted_mask, dtype=bool)
    if not truth.any():
        return float("nan")
    return float(truth[int(np.argmax(a))])


def relative_error(actual: float, expected: float, floor: float = 1e-12) -> float:
    return float(abs(actual - expected) / max(abs(expected), floor))

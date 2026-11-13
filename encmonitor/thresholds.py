from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class Thresholds:
    d_mix: float
    d_site: float
    heterogeneity: float
    quantile: float


def calibrate_thresholds(clean_scores: Sequence[dict], quantile: float = 0.95) -> Thresholds:
    if not 0.5 < quantile < 1.0:
        raise ValueError("quantile should be in (0.5,1)")
    if len(clean_scores) < 20:
        raise ValueError("Use at least 20 clean calibration windows")

    def q(key: str) -> float:
        vals = np.asarray([float(x[key]) for x in clean_scores], dtype=np.float64)
        if not np.isfinite(vals).all():
            raise ValueError(f"Non-finite clean score for {key}")
        return float(np.quantile(vals, quantile, method="higher"))

    return Thresholds(q("d_mix"), q("d_site"), q("heterogeneity"), float(quantile))


def classify(score: dict, thresholds: Thresholds) -> dict:
    pooled = float(score["d_mix"]) > thresholds.d_mix
    site = float(score["d_site"]) > thresholds.d_site
    hetero = float(score["heterogeneity"]) > thresholds.heterogeneity
    return {
        "global_alert": bool(pooled or site),
        "pooled_alert": bool(pooled),
        "site_energy_alert": bool(site),
        "heterogeneity_alert": bool(hetero),
        "hidden_cancellation": bool((site or hetero) and not pooled),
    }

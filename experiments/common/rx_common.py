"""Shared helpers for the re-analyses of the saved per-window records.

Reads only the saved per-window records; calibration is identical to scripts 16, 25 and 26:
per dataset and partition seed, 95th percentile ("higher") of each statistic over the 500 clean
calibration windows.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RECORD_DIRS = {
    "mnist": ROOT / "outputs/main/mnist",
    "cifar10": ROOT / "outputs/main/cifar10",
    "camelyon17": ROOT / "outputs/camelyon/main/camelyon17",
}
NATURAL_EXCLUDE = {"vector"}
OPP_MIX = ("opposing", "mixed")


def qh(v, q):
    return float(np.quantile(np.asarray(v, dtype=float), q, method="higher"))


def load(ds: str) -> pd.DataFrame:
    frames = [pd.read_json(f, lines=True) for f in sorted(RECORD_DIRS[ds].glob("seed_*/records.jsonl"))]
    if not frames:
        raise RuntimeError(f"No records under {RECORD_DIRS[ds]}")
    df = pd.concat(frames, ignore_index=True)
    df["dataset"] = ds
    return df


def score_seed(g: pd.DataFrame, q: float = 0.95) -> tuple[pd.DataFrame, dict]:
    """Score all non-calibration windows of one dataset/seed. Returns (scored windows, calibration info)."""
    cal = g[g.condition == "clean_calibration"]
    S = len(cal.site_energy.iloc[0])
    t_mix, t_site, t_h = qh(cal.d_mix, q), qh(cal.d_site, q), qh(cal.heterogeneity, q)
    E_cal = np.stack(cal.site_energy.to_numpy())
    tau_s = np.array([qh(E_cal[:, s], q) for s in range(S)])
    lm_cal = (E_cal / tau_s).max(axis=1)
    tau_lm = qh(lm_cal, q)
    enc_cal = np.maximum(cal.d_mix / t_mix, cal.d_site / t_site).to_numpy()
    c_joint = qh(enc_cal, q)
    tau_exact = qh(cal.exact_mmd, q)

    out = g[g.condition != "clean_calibration"].copy()
    E = np.stack(out.site_energy.to_numpy())
    out["encmon_score"] = np.maximum(out.d_mix / t_mix, out.d_site / t_site)
    out["localmax_score"] = (E / tau_s).max(axis=1)
    out["pooled"] = out.d_mix > t_mix
    out["site"] = out.d_site > t_site
    out["H"] = out.heterogeneity > t_h
    out["encmon_or"] = out.encmon_score > 1.0
    out["encmon_joint"] = out.encmon_score > c_joint
    out["localany"] = out.localmax_score > tau_lm
    out["exact"] = out.exact_mmd > tau_exact
    out["majority"] = (E > tau_s).sum(axis=1) >= int(np.ceil((S + 1) / 2))
    out["hidden"] = (out.site | out.H) & ~out.pooled
    info = {
        "S": S, "t_mix": t_mix, "t_site": t_site, "t_h": t_h, "tau_s": tau_s, "tau_lm": tau_lm,
        "c_joint": c_joint, "tau_exact": tau_exact,
        # calibration scores, for re-thresholding at other quantiles
        "cal_scores": {"d_mix": cal.d_mix.to_numpy(float), "encmon_score": enc_cal,
                       "localmax_score": lm_cal, "exact_mmd": cal.exact_mmd.to_numpy(float)},
        # clean calibration means, for noise correction
        "m0_mix": float(cal.d_mix.mean()), "m0_site": float(cal.d_site.mean()), "e0_s": E_cal.mean(axis=0),
    }
    return out, info


def load_scored(ds: str, q: float = 0.95) -> tuple[pd.DataFrame, dict]:
    df = load(ds)
    parts, infos = [], {}
    for seed, g in df.groupby("partition_seed"):
        s, info = score_seed(g, q)
        parts.append(s)
        infos[int(seed)] = info
    return pd.concat(parts, ignore_index=True), infos


def strat_boot(x: np.ndarray, strata: np.ndarray, reps: int, rng) -> tuple[float, float]:
    """95% bootstrap over windows stratified by seed (the interval of scripts/16_analyze_screening.py)."""
    groups = [x[strata == s] for s in np.unique(strata)]
    stats = [np.mean(np.concatenate([gr[rng.integers(0, gr.size, gr.size)] for gr in groups])) for _ in range(reps)]
    return float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))


def cluster_boot(per_seed: np.ndarray, reps: int, rng, stat=np.mean) -> tuple[float, float]:
    """95% bootstrap over partition seeds. per_seed: array whose first axis is the seed; `stat`
    reduces the resampled array (default: mean over everything, i.e. equal weight per seed)."""
    k = per_seed.shape[0]
    stats = [stat(per_seed[rng.integers(0, k, k)]) for _ in range(reps)]
    return float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))


def aggregate_masks(cond: pd.DataFrame) -> dict[str, pd.Series]:
    """Pre-specified condition aggregates."""
    nat = ~cond.family.isin(NATURAL_EXCLUDE)
    om = cond.geometry.isin(OPP_MIX)
    return {
        "A1_natural_opposing_mixed": nat & om,
        "A2_all_opposing_mixed": om,
        "A3_all_shifted": cond.geometry.notna(),
        "A4_single": cond.geometry == "single",
        "A5_coherent": cond.geometry == "coherent",
    }

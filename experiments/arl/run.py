#!/usr/bin/env python3
"""False alarms over time (ARL0) and abrupt-shift delay at matched ARL0.

Re-analysis of saved records only. Writes into this folder.
Usage (bundle root): python experiments/arl/run.py > experiments/arl/run_log.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
from rx_common import RECORD_DIRS, aggregate_masks, load_scored, qh  # noqa: E402

DEFAULT_METHODS = ["pooled", "encmon_or", "encmon_joint", "localany", "exact"]
SCORE_OF = {"pooled": "d_mix", "encmon": "encmon_score", "localany": "localmax_score", "exact": "exact_mmd"}
TARGETS = [100, 500]
HORIZON = 10
STREAMS, CAP = 2000, 20000
BEST = Path(__file__).resolve().parents[1] / "equal_fpr" / "best_rows.csv"


# ---------------- ARL of the 2-of-3 rule for i.i.d. Bernoulli(p) alerts (exact Markov chain)
def arl_2of3(p: float) -> float:
    """Expected index (1-based) of the first window at which >= 2 of the last 3 windows alerted,
    starting with no alerts before the stream (as in script 26). Transient states: last two bits
    (0,0), (0,1), (1,0)."""
    if p <= 0:
        return np.inf
    q = 1 - p
    # Q[i, j]: transition among transient states 0=(0,0), 1=(0,1), 2=(1,0)
    Q = np.array([[q, p, 0.0],      # (0,0) -0-> (0,0); -1-> (0,1)
                  [0.0, 0.0, q],    # (0,1) -0-> (1,0); -1-> absorbed (0+1+1)
                  [q, 0.0, 0.0]])   # (1,0) -0-> (0,0); -1-> absorbed (1+0+1)
    return float(np.linalg.solve(np.eye(3) - Q, np.ones(3))[0])


def p_for_arl(target: float) -> float:
    lo, hi = 1e-6, 0.5
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if arl_2of3(mid) > target else (lo, mid)
    return (lo + hi) / 2


def empirical_arl(bits: np.ndarray, rng) -> float:
    """Resample a seed's held-out alert bits into random streams; mean first 2-of-3 confirmation."""
    firsts = np.full(STREAMS, np.nan)
    pending = np.arange(STREAMS)
    offset, prev = 0, np.zeros((STREAMS, 2), dtype=int)
    chunk = 2000
    while pending.size and offset < CAP:
        a = rng.choice(bits.astype(int), size=(pending.size, chunk))
        full = np.concatenate([prev[pending], a], axis=1)
        s3 = full[:, :-2] + full[:, 1:-1] + full[:, 2:]
        hit = s3 >= 2
        found = hit.any(axis=1)
        firsts[pending[found]] = offset + np.argmax(hit[found], axis=1) + 1
        prev[pending] = full[:, -2:]
        pending = pending[~found]
        offset += chunk
    firsts[pending] = CAP
    return float(np.mean(firsts))


def first_confirm(a: np.ndarray) -> int:
    c = np.convolve(a.astype(int), np.ones(3, dtype=int))[: a.size] >= 2
    return int(np.argmax(c)) + 1 if c.any() else 0


def abrupt_delay(a: np.ndarray) -> tuple[float, float]:
    d = np.array([first_confirm(a[k:k + HORIZON]) for k in range(0, a.size - HORIZON + 1)])
    hit = d > 0
    return (float(d[hit].mean()) if hit.any() else np.nan, float(hit.mean()))


def main() -> int:
    rng = np.random.default_rng(20261022)
    p_star = {t: p_for_arl(t) for t in TARGETS}
    default_rows, matched_rows, delay_rows = [], [], []

    for ds in RECORD_DIRS:
        sc, infos = load_scored(ds)
        seeds = sorted(infos)
        clean = sc[sc.condition == "clean_heldout"].sort_values(["partition_seed", "window"])
        shifted = sc[~sc.condition.str.startswith("clean")].sort_values(["partition_seed", "condition", "window"])

        # ---- ARL0 at default thresholds
        for m in DEFAULT_METHODS:
            ps = np.array([clean.loc[clean.partition_seed == s, m].mean() for s in seeds])
            emp = [empirical_arl(clean.loc[clean.partition_seed == s, m].to_numpy(), rng) for s in seeds]
            p = float(clean[m].mean())
            default_rows.append({"dataset": ds, "method": m, "fpr": p, "arl0_single": 1 / p if p > 0 else np.inf,
                                 "arl0_2of3": arl_2of3(p), "arl0_2of3_seed_min": min(arl_2of3(x) for x in ps),
                                 "arl0_2of3_seed_max": max(arl_2of3(x) for x in ps),
                                 "arl0_2of3_markov_mean_over_seeds": float(np.mean([arl_2of3(x) for x in ps])),
                                 "arl0_2of3_empirical_mean_over_seeds": float(np.mean(emp)),
                                 "false_confirmations_per_1000_windows": 1000 / arl_2of3(p)})

        # ---- matched ARL0: thresholds at the (1 - p*) calibration quantile of each score
        for t in TARGETS:
            for k, col in SCORE_OF.items():
                thr = {s: qh(infos[s]["cal_scores"][col], 1 - p_star[t]) for s in seeds}
                ca = clean[col].to_numpy() > clean.partition_seed.map(thr).to_numpy()
                p = float(ca.mean())
                matched_rows.append({"dataset": ds, "target_arl0": t, "p_star": p_star[t], "method": k,
                                     "heldout_fpr": p, "achieved_arl0_2of3": arl_2of3(p)})
                al = shifted[col].to_numpy() > shifted.partition_seed.map(thr).to_numpy()
                tmp = shifted[["condition", "family", "severity", "geometry", "partition_seed"]].assign(alert=al)
                for cond, g in tmp.groupby("condition", sort=False):
                    d, f = zip(*[abrupt_delay(gs.alert.to_numpy()) for _, gs in g.groupby("partition_seed")])
                    delay_rows.append({"dataset": ds, "target_arl0": t, "method": k, "condition": cond,
                                       "family": g.family.iloc[0], "severity": g.severity.iloc[0], "geometry": g.geometry.iloc[0],
                                       "tpr": float(g.alert.mean()),
                                       "delay": float(np.nanmean(d)) if np.isfinite(d).any() else np.nan,
                                       "confirmed": float(np.mean(f))})

    default = pd.DataFrame(default_rows)
    matched = pd.DataFrame(matched_rows)
    delay = pd.DataFrame(delay_rows)

    # summaries over the pre-specified aggregates and best-case rows
    best = pd.read_csv(BEST)[["dataset", "condition"]].assign(best=True)
    delay = delay.merge(best, on=["dataset", "condition"], how="left").fillna({"best": False})
    summ = []
    for (ds, t, k), g in delay.groupby(["dataset", "target_arl0", "method"]):
        masks = aggregate_masks(g) | {"best_case_rows": g.best.astype(bool)}
        for name, mk in masks.items():
            gg = g[mk.to_numpy()]
            if len(gg):
                summ.append({"dataset": ds, "target_arl0": t, "method": k, "aggregate": name, "conditions": len(gg),
                             "tpr": gg.tpr.mean(), "confirmed": gg.confirmed.mean(), "delay_mean": gg.delay.mean(),
                             "delay_min": gg.delay.min(), "delay_max": gg.delay.max(),
                             "confirmed_min": gg.confirmed.min()})
    summ = pd.DataFrame(summ)

    default.to_csv(HERE / "arl0_default.csv", index=False)
    matched.to_csv(HERE / "arl0_matched.csv", index=False)
    delay.to_csv(HERE / "delay_matched_by_condition.csv", index=False)
    summ.to_csv(HERE / "delay_matched_summary.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    f = lambda x: f"{x:.3f}"
    print(f"p* for target ARL0 (2-of-3): " + ", ".join(f"{t}: {p_star[t]:.4f}" for t in TARGETS))
    print("check arl_2of3(0.05) =", round(arl_2of3(0.05), 2))
    print("\n== ARL0 at default thresholds (windows until first false alarm) ==")
    print(default.to_string(index=False, float_format=lambda x: f"{x:.1f}" if x > 2 else f"{x:.3f}"))
    print("\n== Matched ARL0: achieved on held-out clean windows ==")
    print(matched.to_string(index=False, float_format=f))
    print("\n== Abrupt-shift delay at matched ARL0 (2-of-3, horizon 10) ==")
    keep = summ["aggregate"].isin(["A1_natural_opposing_mixed", "A4_single", "A5_coherent", "best_case_rows"])
    print(summ[keep].sort_values(["dataset", "aggregate", "target_arl0", "method"]).to_string(index=False, float_format=f))
    print(f"\nWrote arl0_default.csv, arl0_matched.csv, delay_matched_by_condition.csv, delay_matched_summary.csv to {HERE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

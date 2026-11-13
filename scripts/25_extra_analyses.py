#!/usr/bin/env python3
"""Extra analyses from the saved screening records (no new windows are generated).

A. Detection delay with 2-of-3 confirmation and the clean false-confirmation rate.
B. Joint calibration of the combined global rule: one threshold c on the
   EncMonitor score max(D_mix/tau_mix, D_site/tau_site), set at the q-quantile
   of the calibration windows, so the combined rule has a nominal level q.
F. Attribution ranking quality: NDCG and Spearman rank correlation.

Thresholds are calibrated per dataset and partition seed on the 500 clean
calibration windows, exactly as in 16_analyze_screening.py.

Usage (bundle root):
  python scripts/25_extra_analyses.py --cache-dirs outputs/main/mnist outputs/main/cifar10 \
      --out-dir outputs/main/analysis/extra_q095
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load(cache_dir: Path) -> pd.DataFrame:
    frames = [pd.read_json(f, lines=True) for f in sorted(cache_dir.glob("seed_*/records.jsonl"))]
    if not frames:
        raise RuntimeError(f"No records under {cache_dir}")
    df = pd.concat(frames, ignore_index=True)
    df["dataset"] = cache_dir.name
    return df


def qh(v, q):
    return float(np.quantile(np.asarray(v, dtype=float), q, method="higher"))


def score_seed(g: pd.DataFrame, q: float) -> pd.DataFrame:
    cal = g[g.condition == "clean_calibration"]
    t_mix, t_site, t_h = qh(cal.d_mix, q), qh(cal.d_site, q), qh(cal.heterogeneity, q)
    s_cal = np.maximum(cal.d_mix / t_mix, cal.d_site / t_site)
    c_joint = qh(s_cal, q)
    out = g[g.condition != "clean_calibration"].copy()
    out["score"] = np.maximum(out.d_mix / t_mix, out.d_site / t_site)
    out["pooled"] = out.d_mix > t_mix
    out["encmon"] = out.score > 1.0                      # OR rule, as in the paper
    out["joint"] = out.score > c_joint                   # jointly calibrated OR rule
    out["dual"] = (out.d_site > t_site) | (out.heterogeneity > t_h)
    out["hidden"] = out.dual & ~out.pooled
    out["c_joint"] = c_joint
    return out


def confirm_2of3(alerts: np.ndarray) -> np.ndarray:
    """confirmed[t] is True when at least 2 of windows t-2, t-1, t alerted."""
    a = alerts.astype(int)
    c = np.convolve(a, np.ones(3, dtype=int))[: a.size]
    return c >= 2


def delay_stats(alerts: np.ndarray, horizon: int) -> tuple[float, float]:
    """Start the stream at every possible window k; delay = number of shifted windows
    observed until the first 2-of-3 confirmation (windows before k count as no alert).
    Returns (mean delay over confirmed starts, fraction confirmed within `horizon`)."""
    delays, hits = [], 0
    starts = range(0, alerts.size - horizon + 1)
    for k in starts:
        conf = confirm_2of3(alerts[k : k + horizon])
        if conf.any():
            hits += 1
            delays.append(int(np.argmax(conf)) + 1)
    n = len(starts)
    return (float(np.mean(delays)) if delays else float("nan"), hits / n if n else float("nan"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Delay, joint calibration and ranking metrics from the main-experiment records")
    ap.add_argument("--cache-dirs", nargs="+", required=True)
    ap.add_argument("--quantile", type=float, default=0.95)
    ap.add_argument("--horizon", type=int, default=10, help="windows allowed for confirmation")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    scored = []
    for cd in args.cache_dirs:
        df = load(Path(cd))
        for _, g in df.groupby("partition_seed"):
            scored.append(score_seed(g, args.quantile))
    sc = pd.concat(scored, ignore_index=True).sort_values(["dataset", "partition_seed", "condition", "window"])

    # ---- clean held-out windows: single-window FPR and 2-of-3 false-confirmation rate
    clean_rows = []
    for ds, g in sc[sc.condition == "clean_heldout"].groupby("dataset"):
        row = {"dataset": ds, "windows": len(g)}
        for m in ("pooled", "encmon", "joint"):
            row[f"fpr_{m}"] = float(g[m].mean())
            conf = [confirm_2of3(gs[m].to_numpy())[2:] for _, gs in g.groupby("partition_seed")]
            row[f"false_confirm_{m}"] = float(np.concatenate(conf).mean())
        row["c_joint_mean"] = float(g.c_joint.mean())
        clean_rows.append(row)
    clean = pd.DataFrame(clean_rows)

    # ---- shifted conditions
    rows = []
    for (ds, cond), g in sc[~sc.condition.str.startswith("clean")].groupby(["dataset", "condition"]):
        row = {"dataset": ds, "condition": cond, "family": g.family.iloc[0], "severity": g.severity.iloc[0],
               "geometry": g.geometry.iloc[0], "windows": len(g),
               "tpr_pooled": float(g.pooled.mean()), "tpr_encmon": float(g.encmon.mean()),
               "tpr_joint": float(g.joint.mean()), "tpr_dual": float(g.dual.mean()),
               "tpr_hidden": float(g.hidden.mean())}
        for m in ("pooled", "encmon", "joint"):
            d, f = zip(*[delay_stats(gs[m].to_numpy(), args.horizon) for _, gs in g.groupby("partition_seed")])
            row[f"delay_{m}"] = float(np.nanmean(d)) if np.isfinite(d).any() else float("nan")
            row[f"confirmed_{m}"] = float(np.mean(f))
        for k in ("top1", "topk_recall", "ndcg", "spearman"):
            if k in g:
                row[k] = float(pd.to_numeric(g[k], errors="coerce").mean())
        rows.append(row)
    cond = pd.DataFrame(rows)

    # best-case rows: per dataset and family, the opposing/mixed condition with the highest hidden-drift rate
    om = cond[cond.geometry.isin(["opposing", "mixed"])]
    t9 = om.loc[om.groupby(["dataset", "family"]).tpr_hidden.idxmax()].sort_values(["dataset", "family"])

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    clean.to_csv(out / "clean_confirmation.csv", index=False)
    cond.to_csv(out / "condition_extra.csv", index=False)
    t9.to_csv(out / "table9_extra.csv", index=False)
    rank = cond.groupby(["dataset", "geometry"])[["top1", "topk_recall", "ndcg", "spearman"]].mean()
    rank.to_csv(out / "ranking_by_geometry.csv")
    (out / "settings.json").write_text(json.dumps({"quantile": args.quantile, "horizon": args.horizon}, indent=2))

    pd.set_option("display.width", 200)
    f = lambda x: f"{x:.3f}"
    print(f"== Clean held-out windows (q={args.quantile}) ==")
    print(clean.to_string(index=False, float_format=f))
    print("\n== Table 9 conditions: detection, joint calibration, 2-of-3 delay (windows), ranking ==")
    cols = ["dataset", "condition", "tpr_pooled", "tpr_encmon", "tpr_joint", "delay_pooled", "confirmed_pooled",
            "delay_encmon", "confirmed_encmon", "delay_joint", "confirmed_joint", "ndcg", "spearman"]
    print(t9[cols].to_string(index=False, float_format=f))
    print("\n== Attribution ranking by geometry (mean over families and severities) ==")
    print(rank.to_string(float_format=f))
    print(f"\nWrote clean_confirmation.csv, condition_extra.csv, table9_extra.csv, ranking_by_geometry.csv to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

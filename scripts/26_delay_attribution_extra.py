#!/usr/bin/env python3
"""Additional analyses from saved records only (no new windows are generated).

A. Detection delay across shift severity (2-of-3 confirmation)
   abrupt shift: delay for every shifted condition (all severities, all geometries);
   gradual shift: a stream whose severity rises step by step (sev1 -> sev2 -> sev3 -> sev4,
   `--steps` windows per step) for every family/geometry with four severities.
B. False accusation rate of the attribution and a per-geometry site-flag table:
   after a global alert, a site is flagged when its A_s exceeds the 95th percentile of its own
   A_s on the clean calibration windows. Reported: flag rate of shifted sites (true flags) and of
   unshifted sites (false accusations).
C. Number of sites: the S = 10, m = 64 rows of the ablation core grid (S x m x n was run as a full
   factorial), read from ablation_summary.csv, to show whether longer windows recover the dual power.

Thresholds are calibrated per dataset and partition seed exactly as in 16_analyze_screening.py.

Usage (from the repository root):
  python scripts/26_delay_attribution_extra.py --cache-dirs outputs/main/mnist outputs/main/cifar10 \
      --ablation-summary outputs/ablation/analysis/q095/ablation_summary.csv --out-dir outputs/main/analysis/extra2_q095
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
    t_mix, t_site = qh(cal.d_mix, q), qh(cal.d_site, q)
    c_joint = qh(np.maximum(cal.d_mix / t_mix, cal.d_site / t_site), q)
    A_cal = np.stack(cal.contributions.to_numpy())
    tau_A = np.array([qh(A_cal[:, s], q) for s in range(A_cal.shape[1])])
    out = g[g.condition != "clean_calibration"].copy()
    score = np.maximum(out.d_mix / t_mix, out.d_site / t_site)
    out["pooled"] = out.d_mix > t_mix
    out["encmon"] = score > 1.0
    out["joint"] = score > c_joint
    A = np.stack(out.contributions.to_numpy())
    out["site_flags"] = list(A > tau_A)
    return out


def confirm_2of3(a: np.ndarray) -> np.ndarray:
    return np.convolve(a.astype(int), np.ones(3, dtype=int))[: a.size] >= 2


def first_confirm(a: np.ndarray) -> int:
    """1-based index of the first confirmed window, 0 if never confirmed."""
    c = confirm_2of3(a)
    return int(np.argmax(c)) + 1 if c.any() else 0


def abrupt_delay(a: np.ndarray, horizon: int) -> tuple[float, float]:
    d = [first_confirm(a[k:k + horizon]) for k in range(0, a.size - horizon + 1)]
    d = np.array(d)
    hit = d > 0
    return (float(d[hit].mean()) if hit.any() else float("nan"), float(hit.mean()))


def main() -> int:
    ap = argparse.ArgumentParser(description="Delay across severity, gradual drift, false accusation, S=10 lookup")
    ap.add_argument("--cache-dirs", nargs="+", required=True)
    ap.add_argument("--ablation-summary", default="outputs/ablation/analysis/q095/ablation_summary.csv")
    ap.add_argument("--quantile", type=float, default=0.95)
    ap.add_argument("--horizon", type=int, default=10, help="abrupt shift: windows allowed for confirmation")
    ap.add_argument("--steps", type=int, default=5, help="gradual shift: windows per severity step")
    ap.add_argument("--streams", type=int, default=200, help="gradual shift: random streams per seed")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(20261012)

    sc = []
    for cd in args.cache_dirs:
        df = load(Path(cd))
        for _, g in df.groupby("partition_seed"):
            sc.append(score_seed(g, args.quantile))
    sc = pd.concat(sc, ignore_index=True).sort_values(["dataset", "partition_seed", "condition", "window"])
    shifted = sc[~sc.condition.str.startswith("clean")]
    methods = ("pooled", "encmon", "joint")

    # ---------------- abrupt shift: delay for every condition
    rows = []
    for (ds, cond), g in shifted.groupby(["dataset", "condition"]):
        r = {"dataset": ds, "condition": cond, "family": g.family.iloc[0], "severity": g.severity.iloc[0],
             "geometry": g.geometry.iloc[0]}
        for m in methods:
            d, f = zip(*[abrupt_delay(gs[m].to_numpy(), args.horizon) for _, gs in g.groupby("partition_seed")])
            r[f"tpr_{m}"] = float(g[m].mean())
            r[f"delay_{m}"] = float(np.nanmean(d)) if np.isfinite(d).any() else float("nan")
            r[f"confirmed_{m}"] = float(np.mean(f))
        rows.append(r)
    abrupt = pd.DataFrame(rows)

    # ---------------- gradual shift: severity rises step by step
    rows = []
    for (ds, fam, geo), g in shifted.groupby(["dataset", "family", "geometry"]):
        sevs = sorted(g.severity.unique())
        if len(sevs) < 2:
            continue
        res = {m: [] for m in methods}
        reached = {m: [] for m in methods}
        for _, gs in g.groupby("partition_seed"):
            pools = [gs[gs.severity == s] for s in sevs]
            for _ in range(args.streams):
                parts = [p.iloc[rng.choice(len(p), args.steps, replace=False)] for p in pools]
                stream = pd.concat(parts)
                for m in methods:
                    k = first_confirm(stream[m].to_numpy())
                    res[m].append(k)
                    reached[m].append(sevs[(k - 1) // args.steps] if k > 0 else np.nan)
        r = {"dataset": ds, "family": fam, "geometry": geo, "severities": ",".join(f"{s:g}" for s in sevs),
             "stream_windows": args.steps * len(sevs)}
        for m in methods:
            k = np.array(res[m]); hit = k > 0
            r[f"delay_{m}"] = float(k[hit].mean()) if hit.any() else float("nan")
            r[f"confirmed_{m}"] = float(hit.mean())
            r[f"median_severity_at_confirm_{m}"] = float(np.nanmedian(reached[m])) if hit.any() else float("nan")
        rows.append(r)
    gradual = pd.DataFrame(rows)

    # ---------------- B false accusation after a global alert
    rows = []
    for (ds, geo), g in shifted[shifted.encmon].groupby(["dataset", "geometry"]):
        F = np.stack(g.site_flags.to_numpy()); M = np.stack(g.shifted_mask.to_numpy()).astype(bool)
        rows.append({"dataset": ds, "geometry": geo, "alerted_windows": len(g),
                     "true_flag_rate": float(F[M].mean()) if M.any() else float("nan"),
                     "false_accusation_rate": float(F[~M].mean()) if (~M).any() else float("nan"),
                     "windows_with_any_false_accusation": float((F & ~M).any(axis=1).mean())})
    for ds, g in sc[sc.condition == "clean_heldout"].groupby("dataset"):
        F = np.stack(g.site_flags.to_numpy())
        rows.append({"dataset": ds, "geometry": "clean (all windows)", "alerted_windows": len(g),
                     "true_flag_rate": float("nan"), "false_accusation_rate": float(F.mean()),
                     "windows_with_any_false_accusation": float(F.any(axis=1).mean())})
    accuse = pd.DataFrame(rows)

    # ---------------- C S = 10 rows of the ablation core grid
    s10 = pd.DataFrame()
    p = Path(args.ablation_summary)
    if p.exists():
        ab = pd.read_csv(p)
        keep = (ab.sites == 10) & (ab.rff_dim == 64) & (ab.layer == "penultimate") & np.isclose(ab.bandwidth_multiplier.astype(float), 1.0) \
            & (ab.baseline == "site") & (ab.weights == "uniform")
        cols = [c for c in ("dataset", "condition", "sites", "rff_dim", "window_size", "tpr_pooled", "tpr_dual",
                            "tpr_hidden", "fpr_encmon") if c in ab.columns]
        s10 = ab[keep][cols].sort_values([c for c in ("dataset", "condition", "rff_dim", "window_size") if c in cols])

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    abrupt.to_csv(out / "delay_abrupt_by_condition.csv", index=False)
    gradual.to_csv(out / "delay_gradual.csv", index=False)
    accuse.to_csv(out / "false_accusation.csv", index=False)
    if len(s10):
        s10.to_csv(out / "sites10_core_grid.csv", index=False)
    (out / "settings.json").write_text(json.dumps(vars(args), indent=2))

    pd.set_option("display.width", 220)
    f3 = lambda x: f"{x:.3f}"
    om = abrupt[abrupt.geometry.isin(["opposing", "mixed"])]
    print("== Abrupt shift, opposing/mixed: mean over severities (delay in windows; confirmed = share of streams) ==")
    print(om.groupby(["dataset", "family", "severity"])[[f"{k}_{m}" for m in methods for k in ("tpr", "delay", "confirmed")]]
          .mean().round(3).to_string())
    print("\n== Gradual shift (severity rises every", args.steps, "windows) ==")
    print(gradual.to_string(index=False, float_format=f3))
    print("\n== B Site flags after a global alert (A_s above its clean 95th percentile) ==")
    print(accuse.to_string(index=False, float_format=f3))
    if len(s10):
        print("\n== C S = 10, base layer/bandwidth/reference/weights, ablation core grid ==")
        print(s10.to_string(index=False, float_format=f3))
    print(f"\nWrote delay_abrupt_by_condition.csv, delay_gradual.csv, false_accusation.csv, sites10_core_grid.csv to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

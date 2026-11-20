#!/usr/bin/env python3
"""Equal-FPR comparison, full condition tables, seed-level uncertainty, pool sizes.

Re-analysis of saved records only. Writes into this folder.
Usage (from the repository root): python experiments/equal_fpr/run.py > experiments/equal_fpr/run_log.txt
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
from rx_common import (ROOT, RECORD_DIRS, aggregate_masks, cluster_boot, load_scored, qh,  # noqa: E402
                       strat_boot)

METHODS = ["pooled", "site", "encmon_or", "encmon_joint", "localany", "exact", "majority"]
POWER = ["pooled", "site", "encmon_or", "encmon_joint", "localany", "exact"]
SCORES = {"pooled": "d_mix", "site": "d_site", "encmon": "encmon_score", "localany": "localmax_score",
          "exact": "exact_mmd"}
CI_METHODS = ["pooled", "encmon_or", "encmon_joint", "localany"]
SAVED_SUMMARY = {
    "mnist": ROOT / "outputs/main/analysis/q095_all/condition_summary.csv",
    "cifar10": ROOT / "outputs/main/analysis/q095_all/condition_summary.csv",
    "camelyon17": ROOT / "outputs/camelyon/analysis/condition_summary.csv",
}
STRAT_REPS, CLUSTER_REPS = 500, 2000


def per_seed_rates(g: pd.DataFrame, col: str, seeds: list[int]) -> np.ndarray:
    return np.array([g.loc[g.partition_seed == s, col].mean() for s in seeds])


def main() -> int:
    rng = np.random.default_rng(20261020)
    fpr_rows, cond_rows, agg_rows, best_rows, pool_rows, checks = [], [], [], [], [], {}

    for ds in RECORD_DIRS:
        sc, infos = load_scored(ds)
        seeds = sorted(infos)
        clean = sc[sc.condition == "clean_heldout"]
        shifted = sc[~sc.condition.str.startswith("clean")]

        # ---- 1. clean FPR
        for m in METHODS:
            r = per_seed_rates(clean, m, seeds)
            lo, hi = cluster_boot(r, CLUSTER_REPS, rng)
            fpr_rows.append({"dataset": ds, "method": m, "windows": len(clean), "fpr": float(clean[m].mean()),
                             "seed_min": float(r.min()), "seed_max": float(r.max()),
                             "per_seed": ";".join(f"{x:.3f}" for x in r), "cluster_ci_low": lo, "cluster_ci_high": hi})

        # ---- 2. matched held-out FPR thresholds (oracle, sensitivity only)
        mthr = {}
        for s in seeds:
            cs = clean[clean.partition_seed == s]
            for k, col in SCORES.items():
                t = qh(cs[col], 0.95)
                mthr[(s, k)] = t
                fpr_rows.append({"dataset": ds, "method": f"matched_{k}", "windows": len(cs), "fpr": float((cs[col] > t).mean()),
                                 "seed_min": np.nan, "seed_max": np.nan, "per_seed": f"seed {s}",
                                 "cluster_ci_low": np.nan, "cluster_ci_high": np.nan})
        shifted = shifted.copy()
        for k, col in SCORES.items():
            t = shifted.partition_seed.map(lambda s: mthr[(s, k)])
            shifted[f"matched_{k}"] = shifted[col] > t

        # ---- 3. every condition
        rate_mats = {m: [] for m in POWER + [f"matched_{k}" for k in SCORES] + ["hidden"]}
        conds = []
        for cond, g in shifted.groupby("condition", sort=False):
            row = {"dataset": ds, "condition": cond, "family": g.family.iloc[0], "severity": g.severity.iloc[0],
                   "geometry": g.geometry.iloc[0], "windows": len(g)}
            strata = g.partition_seed.to_numpy()
            for m in rate_mats:
                r = per_seed_rates(g, m, seeds)
                rate_mats[m].append(r)
                row[f"tpr_{m}"] = float(g[m].mean())
                if m in CI_METHODS:
                    row[f"tpr_{m}_strat_lo"], row[f"tpr_{m}_strat_hi"] = strat_boot(g[m].to_numpy(float), strata, STRAT_REPS, rng)
                    row[f"tpr_{m}_clus_lo"], row[f"tpr_{m}_clus_hi"] = cluster_boot(r, CLUSTER_REPS, rng)
            for k, col in SCORES.items():
                y = np.r_[np.zeros(len(clean)), np.ones(len(g))]
                row[f"auroc_{k}"] = float(roc_auc_score(y, np.r_[clean[col].to_numpy(float), g[col].to_numpy(float)]))
            cond_rows.append(row)
            conds.append(row)
        cond = pd.DataFrame(conds)
        R = {m: np.stack(v, axis=1) for m, v in rate_mats.items()}   # (seeds, conditions)

        # sanity check against the saved condition summary of script 16 (same thresholds => identical rates)
        saved = pd.read_csv(SAVED_SUMMARY[ds])
        saved = saved[saved.dataset == ds] if "dataset" in saved else saved
        mg = cond.merge(saved[["condition", "tpr_pooled", "tpr_encmon", "tpr_localmax"]], on="condition", suffixes=("", "_saved"))
        checks[ds] = {"conditions_matched": int(len(mg)), "conditions": int(len(cond)),
                      "max_diff_pooled": float((mg.tpr_pooled - mg.tpr_pooled_saved).abs().max()),
                      "max_diff_encmon_or": float((mg.tpr_encmon_or - mg.tpr_encmon).abs().max()),
                      "max_diff_localany": float((mg.tpr_localany - mg.tpr_localmax).abs().max())}

        # ---- 4. pre-specified aggregates
        for name, mask in aggregate_masks(cond).items():
            idx = np.flatnonzero(mask.to_numpy())
            if idx.size == 0:
                continue
            row = {"dataset": ds, "aggregate": name, "conditions": int(idx.size)}
            for m in R:
                sub = R[m][:, idx]
                row[m] = float(sub.mean())
                row[f"{m}_lo"], row[f"{m}_hi"] = cluster_boot(sub, CLUSTER_REPS, rng)
            for a, b in (("encmon_joint", "localany"), ("encmon_joint", "pooled"), ("matched_encmon", "matched_localany")):
                d = R[a][:, idx] - R[b][:, idx]
                row[f"diff_{a}_minus_{b}"] = float(d.mean())
                row[f"diff_{a}_minus_{b}_lo"], row[f"diff_{a}_minus_{b}_hi"] = cluster_boot(d, CLUSTER_REPS, rng)
                row[f"share_conditions_{a}_below_{b}_by_0.05"] = float((d.mean(axis=0) < -0.05).mean())
            agg_rows.append(row)

        # ---- 5. best-case rows: the opposing/mixed condition with the highest hidden-drift rate per family (selection rule of script 16)
        geos = ["opposing"] if ds == "camelyon17" else ["opposing", "mixed"]
        cand = cond[cond.geometry.isin(geos) & (cond.family != "blur")]
        # literal selection code of script 16 (pass criterion), so that ties resolve as in script 16
        best = pd.concat([g.sort_values("tpr_hidden", ascending=False).head(1) for _, g in cand.groupby("family")])
        for _, r in best.iterrows():
            best_rows.append({k: r[k] for k in r.index if not k.startswith("auroc")} | {"auroc_pooled": r.auroc_pooled,
                             "auroc_encmon": r.auroc_encmon, "auroc_localany": r.auroc_localany, "label": "best case"})

        # ---- 6. pool sizes
        for s in seeds:
            man = np.load(RECORD_DIRS[ds] / f"seed_{s}" / "manifest.npz")
            for site in range(infos[s]["S"]):
                nb, nm = man[f"baseline_site_{site:03d}"].size, man[f"monitor_site_{site:03d}"].size
                pool_rows.append({"dataset": ds, "partition_seed": s, "site": site, "reference_pool": nb,
                                  "monitoring_pool": nm, "window_over_pool": 250 / nm})

    fpr = pd.DataFrame(fpr_rows)
    cond_all = pd.DataFrame(cond_rows)
    agg = pd.DataFrame(agg_rows)
    best = pd.DataFrame(best_rows)
    pools = pd.DataFrame(pool_rows)
    fpr.to_csv(HERE / "clean_fpr.csv", index=False)
    cond_all.to_csv(HERE / "condition_full.csv", index=False)
    cond_all[["dataset", "condition", "family", "severity", "geometry"] + [c for c in cond_all if c.startswith("tpr_matched")]
             + [c for c in cond_all if c.startswith("auroc")]].to_csv(HERE / "matched_fpr.csv", index=False)
    agg.to_csv(HERE / "aggregate.csv", index=False)
    best.to_csv(HERE / "best_rows.csv", index=False)
    pools.to_csv(HERE / "pool_sizes.csv", index=False)
    (HERE / "checks.json").write_text(json.dumps(checks, indent=2))

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    f3 = lambda x: f"{x:.3f}"
    print("== Sanity check: rates recomputed vs saved condition summary of script 16 ==")
    print(json.dumps(checks, indent=2))
    print("\n== 1. Held-out clean FPR (default thresholds) ==")
    print(fpr[~fpr.method.str.startswith("matched")].drop(columns="per_seed").to_string(index=False, float_format=f3))
    print("\n   per-seed EncMonitor-OR / joint / Local-any FPR:")
    print(fpr[fpr.method.isin(["encmon_or", "encmon_joint", "localany"])][["dataset", "method", "per_seed"]].to_string(index=False))
    mf = fpr[fpr.method.str.startswith("matched")]
    print("\n   matched-FPR thresholds, achieved held-out FPR (range over seeds):")
    print(mf.groupby(["dataset", "method"]).fpr.agg(["min", "max"]).to_string(float_format=f3))
    print("\n== 4. Pre-specified aggregates (mean of per-condition rates; 95% cluster bootstrap over seeds) ==")
    for _, r in agg.iterrows():
        print(f"{r['dataset']:10s} {r['aggregate']:28s} n={int(r['conditions']):3d}  "
              + "  ".join(f"{m}={r[m]:.3f}[{r[m + '_lo']:.3f},{r[m + '_hi']:.3f}]"
                          for m in ("pooled", "encmon_or", "encmon_joint", "localany", "exact")))
        print(" " * 44 + f"matched: pooled={r.matched_pooled:.3f} encmon={r.matched_encmon:.3f} localany={r.matched_localany:.3f} "
              f"exact={r.matched_exact:.3f} | joint-local={r.diff_encmon_joint_minus_localany:+.3f}"
              f"[{r.diff_encmon_joint_minus_localany_lo:+.3f},{r.diff_encmon_joint_minus_localany_hi:+.3f}] "
              f"matched enc-local={r.diff_matched_encmon_minus_matched_localany:+.3f}"
              f"[{r.diff_matched_encmon_minus_matched_localany_lo:+.3f},{r.diff_matched_encmon_minus_matched_localany_hi:+.3f}] "
              f"joint-pooled={r.diff_encmon_joint_minus_pooled:+.3f}")
    print("\n== 5. Best-case rows (highest hidden-drift rate per family), with joint calibration and Local-any ==")
    cols = ["dataset", "condition", "tpr_pooled", "tpr_encmon_or", "tpr_encmon_joint", "tpr_localany", "tpr_exact",
            "tpr_matched_pooled", "tpr_matched_encmon", "tpr_matched_localany",
            "tpr_pooled_strat_lo", "tpr_pooled_strat_hi", "tpr_pooled_clus_lo", "tpr_pooled_clus_hi",
            "tpr_encmon_joint_strat_lo", "tpr_encmon_joint_strat_hi", "tpr_encmon_joint_clus_lo", "tpr_encmon_joint_clus_hi"]
    print(best[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("\n== 6. Pool sizes (per site, over seeds) ==")
    print(pools.groupby("dataset")[["reference_pool", "monitoring_pool", "window_over_pool"]].agg(["min", "median", "max"])
          .to_string(float_format=lambda x: f"{x:.2f}"))
    print(f"\nWrote clean_fpr.csv, condition_full.csv, matched_fpr.csv, aggregate.csv, best_rows.csv, pool_sizes.csv, checks.json to {HERE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

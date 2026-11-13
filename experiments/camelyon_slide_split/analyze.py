#!/usr/bin/env python3
"""Analysis (CPU): slide-level split vs patch-level split.

Usage (bundle root): python experiments/camelyon_slide_split/analyze.py > experiments/camelyon_slide_split/analysis_log.txt
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
import rx_common as rc  # noqa: E402

RULES = ["pooled", "site", "encmon_or", "encmon_joint", "localany", "exact"]
SPLITS = {"patch_level (paper)": rc.RECORD_DIRS["camelyon17"], "slide_level (E5)": HERE / "cache" / "camelyon17"}


def main() -> int:
    rng = np.random.default_rng(20261021)
    best = pd.read_csv(HERE.parent / "equal_fpr" / "best_rows.csv")
    best = set(best[best.dataset == "camelyon17"].condition)
    fpr_rows, agg_rows, cond_rows = [], [], []
    for split, d in SPLITS.items():
        rc.RECORD_DIRS["camelyon17"] = d
        sc, info = rc.load_scored("camelyon17")
        held = sc[sc.condition == "clean_heldout"]
        for k in RULES:
            per = held.groupby("partition_seed")[k].mean()
            fpr_rows.append({"split": split, "rule": k, "fpr": float(held[k].mean()), "seed_min": float(per.min()),
                             "seed_max": float(per.max()), "per_seed": ";".join(f"{v:.3f}" for v in per)})
        cond = sc[sc.condition != "clean_heldout"].groupby(["condition", "family", "geometry", "severity"])[RULES + ["hidden"]].mean().reset_index()
        cond["split"] = split
        cond_rows.append(cond)
        masks = rc.aggregate_masks(cond)
        masks["best_case_rows_of_paper"] = cond.condition.isin(best)
        for agg, mk in masks.items():
            sub = sc[sc.condition.isin(set(cond.condition[mk]))]
            ps = sub.groupby("partition_seed")[["pooled", "encmon_joint", "localany", "exact", "encmon_or"]].mean()
            r = {"split": split, "aggregate": agg, "conditions": int(mk.sum())}
            for k in ps.columns:
                r[k] = float(ps[k].mean())
                lo, hi = rc.cluster_boot(ps[k].to_numpy(), 2000, rng)
                r[f"{k}_lo"], r[f"{k}_hi"] = lo, hi
            agg_rows.append(r)
    fpr = pd.DataFrame(fpr_rows)
    agg = pd.DataFrame(agg_rows)
    cond = pd.concat(cond_rows, ignore_index=True)
    fpr.to_csv(HERE / "clean_fpr.csv", index=False)
    agg.to_csv(HERE / "aggregate.csv", index=False)
    cond.to_csv(HERE / "condition_rates.csv", index=False)

    # tumor fraction of reference vs monitoring pools per hospital and seed
    gaps = []
    for f in sorted((HERE / "cache" / "camelyon17").glob("seed_*/cache_meta.json")):
        meta = json.load(open(f))
        for s, v in meta["slide_split"].items():
            gaps.append({"partition_seed": meta["partition_seed"], "hospital": int(s), **{k: (str(x) if isinstance(x, list) else x) for k, x in v.items()},
                         "abs_gap": abs(v["tumor_reference"] - v["tumor_monitoring"]),
                         "baseline_size": meta["sizes"]["baseline"][int(s)], "monitor_size": meta["sizes"]["monitor"][int(s)]})
    gaps = pd.DataFrame(gaps)
    # explanation check: clean site energy (calibration windows) against the reference/monitoring tumor gap
    rc.RECORD_DIRS["camelyon17"] = SPLITS["slide_level (E5)"]
    rec = rc.load("camelyon17")
    cal = rec[rec.condition == "clean_calibration"]
    e = {(int(p), s): float(np.stack(g.site_energy.to_numpy())[:, s].mean()) for p, g in cal.groupby("partition_seed") for s in range(5)}
    gaps["clean_mean_site_energy"] = [e[(int(p), int(h))] for p, h in zip(gaps.partition_seed, gaps.hospital)]
    rc.RECORD_DIRS["camelyon17"] = SPLITS["patch_level (paper)"]
    rec0 = rc.load("camelyon17")
    cal0 = rec0[rec0.condition == "clean_calibration"]
    e0 = {(int(p), s): float(np.stack(g.site_energy.to_numpy())[:, s].mean()) for p, g in cal0.groupby("partition_seed") for s in range(5)}
    gaps["clean_mean_site_energy_patch_level"] = [e0[(int(p), int(h))] for p, h in zip(gaps.partition_seed, gaps.hospital)]
    from scipy.stats import spearmanr
    rho = spearmanr(gaps.abs_gap, gaps.clean_mean_site_energy)
    gaps.to_csv(HERE / "slide_split_tumor_gap.csv", index=False)
    (HERE / "energy_vs_gap.json").write_text(json.dumps({
        "spearman_abs_tumor_gap_vs_clean_site_energy": float(rho.statistic), "p_value": float(rho.pvalue),
        "median_ratio_clean_energy_slide_over_patch": float(np.median(gaps.clean_mean_site_energy / gaps.clean_mean_site_energy_patch_level)),
        "range_ratio": [float((gaps.clean_mean_site_energy / gaps.clean_mean_site_energy_patch_level).min()),
                        float((gaps.clean_mean_site_energy / gaps.clean_mean_site_energy_patch_level).max())],
        "abs_gap_median": float(gaps.abs_gap.median()), "abs_gap_range": [float(gaps.abs_gap.min()), float(gaps.abs_gap.max())]}, indent=2))

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    f = lambda x: f"{x:.3f}"
    print("== Clean held-out FPR (1,000 windows) ==")
    print(fpr.to_string(index=False, float_format=f))
    print("\n== Aggregates (mean over seeds, 95% cluster bootstrap) ==")
    print(agg.to_string(index=False, float_format=f))
    print("\n== Best-case rows of the paper, both splits ==")
    print(cond[cond.condition.isin(best)].pivot_table(index="condition", columns="split", values=["pooled", "encmon_joint", "localany"]).to_string(float_format=f))
    print("\n== Tumor fraction, reference vs monitoring slides ==")
    print(gaps.to_string(index=False, float_format=f))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

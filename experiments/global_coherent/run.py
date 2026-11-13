#!/usr/bin/env python3
"""Weak coherent shift at all sites vs two sites vs one site (vector and label-prior shifts on
cached representations, CPU).

Usage (bundle root): python experiments/global_coherent/run.py > experiments/global_coherent/run_log.txt
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
from rx_common import load, qh, score_seed  # noqa: E402
from rx_windows import SeedContext, stats_from_gram  # noqa: E402

DATASETS = ["mnist", "cifar10", "camelyon17"]
FAMILIES = {"vector": [0.1, 0.25, 0.5], "label_prior": [0.05, 0.1, 0.2]}
GEOMETRIES = {"single": [1, 0, 0, 0, 0], "coherent": [1, 1, 0, 0, 0], "all": [1, 1, 1, 1, 1]}
K = {"single": 1, "coherent": 2, "all": 5}
WINDOWS = 50
METHODS = ["pooled", "site", "encmon_or", "encmon_joint", "localany"]
SCORES = {"pooled": "d_mix", "site": "d_site", "encmon": "encmon_score", "localany": "localmax_score"}


def main() -> int:
    rows, theory, check = [], [], {}
    for ds in DATASETS:
        rec = load(ds)
        check[ds] = 0.0
        for seed, saved in rec.groupby("partition_seed"):
            ctx = SeedContext(ds, seed)
            new = []
            for fam, sevs in FAMILIES.items():
                for sev in sevs:
                    for geo, dirs in GEOMETRIES.items():
                        c = {"condition": f"{fam}{sev:g}__{geo}", "family": fam, "kind": fam, "severity": float(sev),
                             "geometry": geo, "directions": dirs}
                        st = stats_from_gram(ctx.run(ctx.condition_feats(c), WINDOWS, seed + 2000)["gram"], ctx.alpha)
                        for w in range(WINDOWS):
                            new.append({"condition": c["condition"], "family": fam, "severity": float(sev), "geometry": geo,
                                        "window": w, "d_mix": st["d_mix"][w], "d_site": st["d_site"][w],
                                        "heterogeneity": st["heterogeneity"][w], "site_energy": st["site_energy"][w].tolist(),
                                        "contributions": st["contributions"][w].tolist(), "exact_mmd": np.nan,
                                        "partition_seed": seed})
                        # conditions that also exist in the main experiment must reproduce the saved records
                        old = saved[saved.condition == c["condition"]]
                        if len(old):
                            check[ds] = max(check[ds], float(np.abs(old.d_mix.to_numpy() - st["d_mix"]).max()))
            base = saved[saved.condition.isin(["clean_calibration", "clean_heldout"])]
            df = pd.concat([base, pd.DataFrame(new)], ignore_index=True)
            sc, info = score_seed(df, 0.95)
            clean = sc[sc.condition == "clean_heldout"]
            shifted = sc[sc.condition != "clean_heldout"].copy()
            for k, col in SCORES.items():
                shifted[f"matched_{k}"] = shifted[col] > qh(clean[col], 0.95)
            for cond, g in shifted.groupby("condition", sort=False):
                r = {"dataset": ds, "partition_seed": seed, "condition": cond, "family": g.family.iloc[0],
                     "severity": g.severity.iloc[0], "geometry": g.geometry.iloc[0]}
                for m in METHODS + [f"matched_{k}" for k in SCORES]:
                    r[m] = float(g[m].mean())
                for col in ("d_mix", "d_site"):
                    r[f"z_{col}"] = float((g[col].mean() - clean[col].mean()) / clean[col].std(ddof=1))
                rows.append(r)

    per_seed = pd.DataFrame(rows)
    rate_cols = METHODS + [f"matched_{k}" for k in SCORES]
    cond = per_seed.groupby(["dataset", "family", "severity", "geometry"], sort=False)[rate_cols + ["z_d_mix", "z_d_site"]].mean().reset_index()
    lo = per_seed.groupby(["dataset", "family", "severity", "geometry"], sort=False)[["pooled", "localany", "encmon_joint"]].min().add_suffix("_seed_min").reset_index()
    hi = per_seed.groupby(["dataset", "family", "severity", "geometry"], sort=False)[["pooled", "localany", "encmon_joint"]].max().add_suffix("_seed_max").reset_index()
    cond = cond.merge(lo).merge(hi)
    cond["k"] = cond.geometry.map(K)
    cond["z_ratio_mix_over_site"] = cond.z_d_mix / cond.z_d_site
    cond["predicted_ratio_k_over_sqrtS"] = cond.k / np.sqrt(5)
    cond["pooled_minus_localany_matched"] = cond.matched_pooled - cond.matched_localany
    cond["encmon_minus_localany_matched"] = cond.matched_encmon - cond.matched_localany
    per_seed.to_csv(HERE / "condition_rates_per_seed.csv", index=False)
    cond.to_csv(HERE / "condition_rates.csv", index=False)
    th = cond[cond.z_d_site > 1.0].groupby(["dataset", "geometry"])[["z_ratio_mix_over_site", "predicted_ratio_k_over_sqrtS"]].median().reset_index()
    th.to_csv(HERE / "theory_check.csv", index=False)
    (HERE / "reproduction_check.json").write_text(json.dumps(
        {"max_abs_diff_d_mix_vs_saved_main_records_for_overlapping_conditions": check}, indent=2))

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    f = lambda x: f"{x:.3f}"
    print("Reproduction check (overlapping conditions vs saved records):", json.dumps(check))
    print("\n== Detection rates (mean over 5 seeds; default thresholds and held-out-matched FPR) ==")
    cols = ["dataset", "family", "severity", "geometry", "pooled", "site", "encmon_or", "encmon_joint", "localany",
            "matched_pooled", "matched_site", "matched_encmon", "matched_localany", "pooled_minus_localany_matched",
            "encmon_minus_localany_matched", "z_ratio_mix_over_site"]
    print(cond[cols].to_string(index=False, float_format=f))
    print("\n== Theory check: median z(D_mix)/z(D_site) over conditions with z(D_site) > 1, vs k/sqrt(S) ==")
    print(th.to_string(index=False, float_format=f))
    print(f"\nWrote condition_rates.csv, condition_rates_per_seed.csv, theory_check.csv, reproduction_check.json to {HERE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

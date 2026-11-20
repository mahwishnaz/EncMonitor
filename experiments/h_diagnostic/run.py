#!/usr/bin/env python3
"""Can the released values tell a cancelling shift (opposing/mixed) from a non-cancelling one
(single/coherent) after an alert? Scores rH, rH_corr (L2) and C, C_corr (L3).

Re-analysis of saved records only. Writes into this folder.
Usage (from the repository root): python experiments/h_diagnostic/run.py > experiments/h_diagnostic/run_log.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
from rx_common import RECORD_DIRS, load_scored  # noqa: E402

SCORES = ["rH", "rH_corr", "1-C", "1-C_corr"]
CLUSTER_REPS = 2000
RULE_THRESHOLD = 0.5      # fixed before the analysis: C_corr < 0.5 => cancellation


def add_scores(sc: pd.DataFrame, infos: dict) -> pd.DataFrame:
    sc = sc.copy()
    S = infos[next(iter(infos))]["S"]
    alpha = 1.0 / S
    A = np.stack(sc.contributions.to_numpy())                     # alpha_s ||delta_s||^2
    E = np.stack(sc.site_energy.to_numpy())                       # ||delta_s||^2
    m0_mix = sc.partition_seed.map(lambda s: infos[s]["m0_mix"]).to_numpy()
    m0_site = sc.partition_seed.map(lambda s: infos[s]["m0_site"]).to_numpy()
    e0 = np.stack(sc.partition_seed.map(lambda s: infos[s]["e0_s"]).to_numpy())
    d_mix, d_site = sc.d_mix.to_numpy(), sc.d_site.to_numpy()

    sc["rH"] = sc.heterogeneity / sc.d_site
    ex_mix, ex_site = d_mix - m0_mix, d_site - m0_site
    with np.errstate(divide="ignore", invalid="ignore"):
        r = 1.0 - np.where(ex_site > 0, ex_mix / ex_site, np.nan)
    sc["rH_corr"] = np.clip(np.nan_to_num(r, nan=1.0), 0.0, 1.0)

    C = d_mix / np.sqrt(alpha * A).sum(axis=1) ** 2
    sc["1-C"] = 1.0 - np.clip(C, 0.0, 1.0)
    e = np.maximum(E - e0, 0.0)
    den = (alpha * np.sqrt(e)).sum(axis=1) ** 2
    num = np.maximum(d_mix - m0_mix, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        Cc = np.where(den > 0, num / den, 1.0)
    sc["C_corr"] = np.clip(Cc, 0.0, 1.0)
    sc["1-C_corr"] = 1.0 - sc["C_corr"]
    return sc


def auc_cluster(g: pd.DataFrame, score: str, pos: pd.Series, rng) -> tuple[float, float, float]:
    y = pos.to_numpy().astype(int)
    if y.min() == y.max():
        return np.nan, np.nan, np.nan
    auc = roc_auc_score(y, g[score].to_numpy())
    seeds = g.partition_seed.to_numpy()
    useeds = np.unique(seeds)
    idx_by = {s: np.flatnonzero(seeds == s) for s in useeds}
    stats = []
    for _ in range(CLUSTER_REPS):
        pick = np.concatenate([idx_by[s] for s in rng.choice(useeds, useeds.size)])
        yy = y[pick]
        if yy.min() != yy.max():
            stats.append(roc_auc_score(yy, g[score].to_numpy()[pick]))
    return float(auc), float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))


def main() -> int:
    rng = np.random.default_rng(20261021)
    auc_rows, geo_rows, fam_rows, rule_rows = [], [], [], []
    for ds in RECORD_DIRS:
        sc, infos = load_scored(ds)
        S = infos[next(iter(infos))]["S"]
        sc = add_scores(sc, infos)
        shifted = sc[~sc.condition.str.startswith("clean")]
        clean = sc[sc.condition == "clean_heldout"]
        fam_geos = shifted.groupby("family").geometry.agg(lambda x: set(x))
        full_fams = [f for f, gs in fam_geos.items() if {"single", "coherent", "opposing", "mixed"} <= gs]

        for alert in ("encmon_joint", "encmon_or"):
            al = shifted[shifted[alert]]
            pops = {"all_families": al, "families_with_4_geometries": al[al.family.isin(full_fams)]}
            for pname, g in pops.items():
                tasks = {
                    "opp+mixed_vs_single+coherent": (g, g.geometry.isin(["opposing", "mixed"])),
                    "opposing_vs_coherent": (g[g.geometry.isin(["opposing", "coherent"])],
                                             g[g.geometry.isin(["opposing", "coherent"])].geometry == "opposing"),
                    "opp+mixed_vs_single": (g[g.geometry != "coherent"], g[g.geometry != "coherent"].geometry.isin(["opposing", "mixed"])),
                }
                for tname, (gg, pos) in tasks.items():
                    row = {"dataset": ds, "alert": alert, "population": pname, "task": tname, "windows": len(gg),
                           "positives": int(pos.sum())}
                    for s in SCORES:
                        row[s], row[f"{s}_lo"], row[f"{s}_hi"] = auc_cluster(gg, s, pos, rng)
                    auc_rows.append(row)
            if alert != "encmon_joint":
                continue
            # by geometry (alerted windows) and clean held-out windows
            k_by_geo = {"single": 1, "coherent": 2}
            for geo, gg in list(al.groupby("geometry")) + [("clean (held-out, all)", clean)]:
                row = {"dataset": ds, "geometry": geo, "windows": len(gg),
                       "rH_noise_free_expectation": (1 - k_by_geo[geo] / S) if geo in k_by_geo else (1.0 if geo == "opposing" else np.nan)}
                for s in ["rH", "rH_corr", "C_corr"]:
                    q = gg[s].quantile([0.25, 0.5, 0.75]).to_numpy()
                    row[f"{s}_median"], row[f"{s}_q25"], row[f"{s}_q75"] = q[1], q[0], q[2]
                row["C"] = float((1 - gg["1-C"]).median())
                geo_rows.append(row)
            # by family, primary task
            for fam, gg in al.groupby("family"):
                pos = gg.geometry.isin(["opposing", "mixed"])
                if pos.all() or not pos.any():
                    continue
                row = {"dataset": ds, "family": fam, "windows": len(gg), "positives": int(pos.sum())}
                for s in SCORES:
                    row[s] = float(roc_auc_score(pos.astype(int), gg[s]))
                fam_rows.append(row)
            # fixed rule C_corr < 0.5
            pred = al.C_corr < RULE_THRESHOLD
            truth = al.geometry.isin(["opposing", "mixed"])
            for geo, gg in al.groupby("geometry"):
                rule_rows.append({"dataset": ds, "geometry": geo, "windows": len(gg),
                                  "share_called_cancellation": float((gg.C_corr < RULE_THRESHOLD).mean())})
            tpr, tnr = float(pred[truth].mean()), float((~pred[~truth]).mean())
            rule_rows.append({"dataset": ds, "geometry": "BALANCED ACCURACY", "windows": len(al),
                              "share_called_cancellation": (tpr + tnr) / 2})

    auc = pd.DataFrame(auc_rows)
    geo = pd.DataFrame(geo_rows)
    fam = pd.DataFrame(fam_rows)
    rule = pd.DataFrame(rule_rows)
    auc.to_csv(HERE / "auroc.csv", index=False)
    geo.to_csv(HERE / "by_geometry.csv", index=False)
    fam.to_csv(HERE / "by_family.csv", index=False)
    rule.to_csv(HERE / "fixed_rule.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    f = lambda x: f"{x:.3f}"
    print("== AUROC for telling cancelling from non-cancelling shifts in alerted windows (95% cluster bootstrap over seeds) ==")
    for _, r in auc.iterrows():
        print(f"{r['dataset']:10s} {r['alert']:12s} {r['population']:27s} {r['task']:30s} n={r['windows']:5d} pos={r['positives']:5d}  "
              + "  ".join(f"{s}={r[s]:.3f}[{r[s + '_lo']:.3f},{r[s + '_hi']:.3f}]" for s in SCORES))
    print("\n== Scores by geometry (alerted windows, EncMonitor-joint; clean = all held-out windows) ==")
    print(geo.to_string(index=False, float_format=f))
    print("\n== Primary AUROC by family (EncMonitor-joint alerts) ==")
    print(fam.to_string(index=False, float_format=f))
    print(f"\n== Fixed rule: C_corr < {RULE_THRESHOLD} => cancellation (share of alerted windows called cancellation) ==")
    print(rule.to_string(index=False, float_format=f))
    print(f"\nWrote auroc.csv, by_geometry.csv, by_family.csv, fixed_rule.csv to {HERE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

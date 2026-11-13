#!/usr/bin/env python3
"""Per-site-normalized EncMonitor on saved records and on regenerated weak-shift windows (one, two and
all sites).

Usage (bundle root): python experiments/normalized_weights/run.py > experiments/normalized_weights/run_log.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
import rx_common as rc  # noqa: E402
from rx_windows import SeedContext, stats_from_gram  # noqa: E402

RULES = ["pooled", "encmon_joint", "encmon_norm", "localany"]
SETS = {"mnist": ("mnist", rc.RECORD_DIRS["mnist"]), "cifar10": ("cifar10", rc.RECORD_DIRS["cifar10"]),
        "camelyon17_patch": ("camelyon17", rc.RECORD_DIRS["camelyon17"]),
        "camelyon17_slide": ("camelyon17", HERE.parent / "camelyon_slide_split" / "cache" / "camelyon17"),
        "folktables": ("folktables", HERE.parent / "census" / "records")}


def add_norm(g: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Score one seed (rx_common) and add EncMonitor-norm (joint threshold on max(D_mix/t_mix, Dn/t_norm))."""
    sc, info = rc.score_seed(g, 0.95)
    cal = g[g.condition == "clean_calibration"]
    e0 = info["e0_s"]
    S = e0.size
    dn_cal = (np.stack(cal.site_energy.to_numpy()) / e0).mean(axis=1)
    t_norm = rc.qh(dn_cal, 0.95)
    c = rc.qh(np.maximum(cal.d_mix.to_numpy() / info["t_mix"], dn_cal / t_norm), 0.95)
    dn = (np.stack(sc.site_energy.to_numpy()) / e0).mean(axis=1)
    sc["norm_score"] = np.maximum(sc.d_mix.to_numpy() / info["t_mix"], dn / t_norm) / c
    sc["encmon_norm"] = sc.norm_score > 1.0
    info["cal_norm_score"] = np.maximum(cal.d_mix.to_numpy() / info["t_mix"], dn_cal / t_norm) / c
    return sc, info


def records_part() -> pd.DataFrame:
    rows = []
    best = pd.read_csv(HERE.parent / "equal_fpr" / "best_rows.csv")
    for name, (key, d) in SETS.items():
        if not any(d.glob("seed_*/records.jsonl")):
            print(f"[{name}] no records, skipped")
            continue
        rc.RECORD_DIRS[key] = d
        df = rc.load(key)
        sc = pd.concat([add_norm(g)[0] for _, g in df.groupby("partition_seed")], ignore_index=True)
        held = sc[sc.condition == "clean_heldout"]
        cond = sc[sc.condition != "clean_heldout"].groupby(["condition", "family", "geometry"])[RULES].mean().reset_index()
        masks = rc.aggregate_masks(cond)
        if key == "folktables":   # real-drift family is not part of A1
            masks["A1_natural_opposing_mixed"] = cond.family.isin(["label_prior", "age_mix"]) & cond.geometry.isin(rc.OPP_MIX)
            masks["year_all_sites"] = (cond.family == "year") & (cond.geometry == "all")
        masks["best_case_rows"] = cond.condition.isin(set(best[best.dataset == key].condition)) if key != "folktables" else cond.condition == ""
        r = {"set": name}
        for k in RULES:
            r[f"fpr_{k}"] = float(held[k].mean())
            per = held.groupby("partition_seed")[k].mean()
            r[f"fpr_{k}_seed_min"], r[f"fpr_{k}_seed_max"] = float(per.min()), float(per.max())
        for agg, mk in masks.items():
            if mk.any():
                for k in RULES:
                    r[f"{agg}_{k}"] = float(cond[mk][k].mean())
        rows.append(r)
    return pd.DataFrame(rows)


E3_FAMILIES = {"vector": [0.1, 0.25, 0.5], "label_prior": [0.05, 0.1, 0.2]}
E3_GEOMETRIES = {"single": [1, 0, 0, 0, 0], "coherent": [1, 1, 0, 0, 0], "all": [1, 1, 1, 1, 1]}


def e3_part() -> pd.DataFrame:
    rows = []
    for ds in ("mnist", "cifar10", "camelyon17"):
        rc.RECORD_DIRS["camelyon17"] = SETS["camelyon17_patch"][1]
        rec = rc.load(ds)
        for seed, saved in rec.groupby("partition_seed"):
            ctx = SeedContext(ds, seed)
            new = []
            for fam, sevs in E3_FAMILIES.items():
                for sev in sevs:
                    for geo, dirs in E3_GEOMETRIES.items():
                        c = {"condition": f"{fam}{sev:g}__{geo}", "family": fam, "kind": fam, "severity": float(sev),
                             "geometry": geo, "directions": dirs}
                        st = stats_from_gram(ctx.run(ctx.condition_feats(c), 50, seed + 2000)["gram"], ctx.alpha)
                        for w in range(50):
                            new.append({"condition": c["condition"], "family": fam, "severity": float(sev), "geometry": geo,
                                        "window": w, "d_mix": st["d_mix"][w], "d_site": st["d_site"][w],
                                        "heterogeneity": st["heterogeneity"][w], "site_energy": st["site_energy"][w].tolist(),
                                        "contributions": st["contributions"][w].tolist(), "exact_mmd": np.nan, "partition_seed": seed})
            base = saved[saved.condition.isin(["clean_calibration", "clean_heldout"])]
            sc, info = add_norm(pd.concat([base, pd.DataFrame(new)], ignore_index=True))
            held = sc[sc.condition == "clean_heldout"]
            matched = {"pooled": rc.qh(held.d_mix, 0.95), "encmon_joint": rc.qh(held.encmon_score, 0.95),
                       "encmon_norm": rc.qh(held.norm_score, 0.95), "localany": rc.qh(held.localmax_score, 0.95)}
            col = {"pooled": "d_mix", "encmon_joint": "encmon_score", "encmon_norm": "norm_score", "localany": "localmax_score"}
            for cn, g in sc[sc.condition != "clean_heldout"].groupby("condition", sort=False):
                r = {"dataset": ds, "partition_seed": seed, "condition": cn, "family": g.family.iloc[0],
                     "severity": g.severity.iloc[0], "geometry": g.geometry.iloc[0]}
                for k in RULES:
                    r[k] = float(g[k].mean())
                    r[f"matched_{k}"] = float((g[col[k]] > matched[k]).mean())
                rows.append(r)
            print(f"[E3 windows {ds} {seed}] done", flush=True)
    per = pd.DataFrame(rows)
    per.to_csv(HERE / "e3_windows_per_seed.csv", index=False)
    return per.groupby(["dataset", "family", "severity", "geometry"], sort=False).mean(numeric_only=True).drop(columns="partition_seed").reset_index()


def main() -> int:
    rec = records_part()
    rec.to_csv(HERE / "records_summary.csv", index=False)
    e3 = e3_part()
    e3.to_csv(HERE / "e3_windows.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 80)
    f = lambda x: f"{x:.3f}"
    print(rec.set_index("set").T.to_string(float_format=f))
    print(e3.to_string(index=False, float_format=f))
    # decision rule, fixed before the analysis
    worst = {}
    for _, r in rec[rec.set.isin(["mnist", "cifar10", "camelyon17_patch"])].iterrows():
        for agg in ("A1_natural_opposing_mixed", "A4_single", "A5_coherent"):
            worst[f"{r.set}/{agg}"] = r[f"{agg}_encmon_norm"] - r[f"{agg}_encmon_joint"]
    al = e3[(e3.geometry == "all")]
    for _, r in al.iterrows():
        worst[f"E3/{r.dataset}/{r.family}{r.severity:g}"] = r.encmon_norm - r.encmon_joint
    mn = min(worst.values())
    print("\nDecision rule: worst (norm - joint) =", round(mn, 3), "at", min(worst, key=worst.get),
          "->", "ADOPT as primary" if mn >= -0.02 else "report as option")
    pd.Series(worst).to_csv(HERE / "decision_differences.csv", header=["norm_minus_joint"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

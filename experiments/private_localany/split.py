#!/usr/bin/env python3
"""Hybrid with a split false-alarm budget (split-97.5 and split-tuned).

Reuses run.py (same windows, same scoring) and adds two rules.
Usage (bundle root): python experiments/private_localany/split.py > experiments/private_localany/split_log.txt
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("p1run", HERE / "run.py")
P1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P1)
rc = P1.rc
base_score = P1.score


def score_split(g: pd.DataFrame):
    sc, info = base_score(g)
    cal = g[g.condition == "clean_calibration"]
    dm_cal = cal.d_mix.to_numpy(float)
    L_cal = (np.stack(cal.site_energy.to_numpy()) / info["tau_s"]).max(axis=1)
    L = (np.stack(sc.site_energy.to_numpy()) / info["tau_s"]).max(axis=1)
    dm = sc.d_mix.to_numpy(float)
    t_m, t_l = rc.qh(dm_cal, 0.975), rc.qh(L_cal, 0.975)
    sc["split975"] = (dm > t_m) | (L > t_l)
    best = 0.025
    for p in np.arange(0.025, 0.05 + 1e-9, 0.0005):
        u = ((dm_cal > rc.qh(dm_cal, 1 - p)) | (L_cal > rc.qh(L_cal, 1 - p))).mean()
        if u <= 0.05:
            best = float(p)
    tm2, tl2 = rc.qh(dm_cal, 1 - best), rc.qh(L_cal, 1 - best)
    sc["split_tuned"] = (dm > tm2) | (L > tl2)
    info.update({"p_tuned": best, "cal_union_975": float(((dm_cal > t_m) | (L_cal > t_l)).mean()),
                 "cal_union_tuned": float(((dm_cal > tm2) | (L_cal > tl2)).mean()),
                 "cal_mix_tuned": float((dm_cal > tm2).mean()), "cal_L_tuned": float((L_cal > tl2).mean())})
    return sc, info


P1.score = score_split
P1.RULES = P1.RULES + ["split975", "split_tuned"]


def main() -> int:
    rng = np.random.default_rng(20261022)
    checks: dict = {}
    summ, cond, fpr = P1.records_part(rng, checks)
    e3 = P1.e3_part(checks)
    # calibration diagnostics of the tuned rule
    cal_rows = []
    for name, (key, d) in P1.SETS.items():
        rc.RECORD_DIRS[key] = d
        for seed, g in rc.load(key).groupby("partition_seed"):
            _, info = score_split(g)
            cal_rows.append({"set": name, "partition_seed": int(seed), **{k: info[k] for k in
                             ("p_tuned", "cal_union_975", "cal_union_tuned", "cal_mix_tuned", "cal_L_tuned")}})
    cal = pd.DataFrame(cal_rows)
    summ.to_csv(HERE / "split_summary.csv", index=False)
    e3.to_csv(HERE / "split_e3.csv", index=False)
    fpr.to_csv(HERE / "split_fpr.csv", index=False)
    cal.to_csv(HERE / "split_calibration.csv", index=False)

    # decision rule, fixed before the analysis
    s = summ.set_index("set")
    dec = {}
    for v in ("split975", "split_tuned"):
        worst, where = 1.0, ""
        for st in s.index:
            for agg in ("A1_natural_opposing_mixed", "A4_single"):
                if f"{agg}_{v}" in s.columns and pd.notna(s.loc[st, f"{agg}_{v}"]):
                    d = s.loc[st, f"{agg}_{v}"] - max(s.loc[st, f"{agg}_encmon_joint"], s.loc[st, f"{agg}_plocal_joint"])
                    if d < worst:
                        worst, where = d, f"{st}/{agg}"
        al = e3[e3.geometry == "all"].groupby("dataset")[["encmon_joint", "plocal_joint", v]].mean()
        for ds, r in al.iterrows():
            d = r[v] - max(r.encmon_joint, r.plocal_joint)
            if d < worst:
                worst, where = d, f"E3 all-site/{ds}"
        fmax = float(fpr[fpr.method == v].fpr.max())
        dec[v] = {"worst_margin": float(worst), "at": where, "max_fpr": fmax,
                  "passes": bool(worst >= -0.02 and fmax <= 0.06)}
    (HERE / "split_decision.json").write_text(json.dumps({"decision": dec, "checks": checks}, indent=2))

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 80); pd.set_option("display.max_rows", 400)
    f = lambda x: f"{x:.3f}"
    keep = ["pooled", "encmon_joint", "plocal_joint", "hybrid", "split975", "split_tuned"]
    print("== FPR / ARL0 ==")
    print(fpr[fpr.method.isin(keep)].to_string(index=False, float_format=f))
    print("\n== Aggregates ==")
    cols = [c for c in summ.columns if any(c.endswith("_" + k) for k in keep) and c.startswith(("A1", "A4", "A5", "year"))]
    print(summ.set_index("set")[cols].T.to_string(float_format=f))
    print("\n== E3 ==")
    print(e3[["dataset", "family", "severity", "geometry"] + keep].to_string(index=False, float_format=f))
    print("\n== E3 means by geometry ==")
    print(e3.groupby(["dataset", "geometry"])[keep].mean().to_string(float_format=f))
    print("\n== Tuned calibration ==")
    print(cal.groupby("set")[["p_tuned", "cal_union_975", "cal_union_tuned", "cal_mix_tuned", "cal_L_tuned"]].agg(["min", "max"]).to_string(float_format=f))
    print("\n== Decision ==\n" + json.dumps(dec, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Census experiment, step 3 (CPU): FPR, aggregates, real-drift table and accuracy link at joint calibration.

Usage (from the repository root): python experiments/census/analyze.py > experiments/census/analysis_log.txt
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
sys.path.insert(0, str(HERE.parent / "normalized_weights"))
import rx_common as rc  # noqa: E402
from run import add_norm  # noqa: E402  (per-site-normalized variant)

RULES = ["pooled", "site", "encmon_or", "encmon_joint", "encmon_norm", "localany"]
HARM = 0.05


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="records")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    rc.RECORD_DIRS["folktables"] = HERE / args.records
    df = rc.load("folktables")
    clean_acc = {json.load(open(f))["partition_seed"]: np.array(json.load(open(f))["clean_accuracy_per_site"])
                 for f in sorted((HERE / args.records).glob("seed_*/conditions.json"))}
    sc = pd.concat([add_norm(g)[0] for _, g in df.groupby("partition_seed")], ignore_index=True)
    rng = np.random.default_rng(20261021)
    held = sc[sc.condition == "clean_heldout"]
    fpr = []
    for k in RULES:
        per = held.groupby("partition_seed")[k].mean()
        fpr.append({"rule": k, "fpr": float(held[k].mean()), "seed_min": float(per.min()), "seed_max": float(per.max())})
    fpr = pd.DataFrame(fpr)
    sh = sc[sc.condition != "clean_heldout"].copy()
    per_seed = sh.groupby(["partition_seed", "condition", "family", "geometry", "severity"])[RULES + ["hidden"]].mean().reset_index()
    cond = per_seed.groupby(["condition", "family", "geometry", "severity"])[RULES + ["hidden"]].mean().reset_index()
    masks = {"A1_natural_opposing_mixed": cond.family.isin(["label_prior", "age_mix"]) & cond.geometry.isin(rc.OPP_MIX),
             "single_injected": cond.family.isin(["label_prior", "age_mix"]) & (cond.geometry == "single"),
             "coherent_injected": cond.family.isin(["label_prior", "age_mix"]) & (cond.geometry == "coherent"),
             "year_single": (cond.family == "year") & (cond.geometry == "single"),
             "year_coherent": (cond.family == "year") & (cond.geometry == "coherent"),
             "year_all": (cond.family == "year") & (cond.geometry == "all"),
             "vector_opposing_mixed": (cond.family == "vector") & cond.geometry.isin(rc.OPP_MIX)}
    agg = []
    for name, mk in masks.items():
        cs = set(cond.condition[mk])
        ps = per_seed[per_seed.condition.isin(cs)].groupby("partition_seed")[["pooled", "encmon_joint", "encmon_norm", "localany", "hidden"]].mean()
        r = {"aggregate": name, "conditions": len(cs)}
        for k in ps.columns:
            r[k] = float(ps[k].mean())
            r[f"{k}_lo"], r[f"{k}_hi"] = rc.cluster_boot(ps[k].to_numpy(), 2000, rng)
        agg.append(r)
    agg = pd.DataFrame(agg)
    # accuracy link
    acc = np.stack(sc.acc.to_numpy())
    ref = np.stack([clean_acc[int(p)] for p in sc.partition_seed])
    sc["max_drop"] = (ref - acc).max(axis=1)
    sc["harmful"] = sc.max_drop >= HARM
    link = []
    for name, g in (("clean_heldout", sc[sc.condition == "clean_heldout"]), ("year_all_sites", sc[(sc.family == "year") & (sc.geometry == "all")]),
                    ("year_any", sc[sc.family == "year"]), ("injected_opposing_mixed", sc[sc.family.isin(["label_prior", "age_mix"]) & sc.geometry.isin(rc.OPP_MIX)])):
        h = g[g.harmful]
        link.append({"subset": name, "windows": len(g), "harmful": int(len(h)), "harmful_share": float(g.harmful.mean()),
                     **{f"detect_harmful_{k}": (float(h[k].mean()) if len(h) else float("nan")) for k in ("pooled", "encmon_joint", "localany")}})
    link = pd.DataFrame(link)
    # site attribution for real drift at one state (top-1) and two states (top-k)
    att = sc[(sc.family == "year") & sc.geometry.isin(["single", "coherent"])].groupby(["condition"])[["top1", "topk_recall"]].mean()
    # OCCP coding check (real cause candidate for the 2018 drift): share of 2018 occupation codes unseen in 2014
    occ = {}
    for y in (2015, 2016, 2017, 2018):
        o14 = set(np.unique(np.load(HERE / "cache" / "acs_2014.npz")["occp"]).tolist())
        oy = np.load(HERE / "cache" / f"acs_{y}.npz")["occp"]
        occ[str(y)] = float(np.mean([v not in o14 for v in oy]))
    fpr.to_csv(HERE / f"clean_fpr{args.tag}.csv", index=False); cond.to_csv(HERE / f"condition_rates{args.tag}.csv", index=False)
    agg.to_csv(HERE / f"aggregate{args.tag}.csv", index=False); link.to_csv(HERE / f"accuracy_link{args.tag}.csv", index=False)
    att.to_csv(HERE / f"attribution_year{args.tag}.csv")
    (HERE / "occp_unseen_share.json").write_text(json.dumps(occ, indent=2))
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    f = lambda x: f"{x:.3f}"
    print("== Clean held-out FPR ==\n" + fpr.to_string(index=False, float_format=f))
    print("\n== Aggregates (cluster bootstrap over seeds) ==\n" + agg.to_string(index=False, float_format=f))
    print("\n== Real drift (year) and injected conditions ==")
    print(cond.sort_values(["family", "geometry", "severity"]).to_string(index=False, float_format=f))
    print("\n== Accuracy link ==\n" + link.to_string(index=False, float_format=f))
    print("\n== Attribution, real drift ==\n" + att.to_string(float_format=f))
    print("\nShare of later-year rows whose OCCP code never occurs in 2014:", occ)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

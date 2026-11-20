#!/usr/bin/env python3
"""Per-seed detection and FPR of every method, setting and aggregate, from the records.

Scoring: experiments/private_localany/run.py `score` (the scoring code of the main detection table (Table 4 of the article; the same scoring is used for Tables 5-7);
release tag) + matched-FPR alerts
(threshold = 95th pct "higher" of the method's score over the seed's own 200 held-out clean windows).
Detection of an aggregate = mean over its conditions of the per-condition alert rate (equal windows per
condition, so it equals the pooled window rate). Weak all-site shifts (18 conditions, geometry "all") are not in
the records: they are regenerated per seed with the generator and seeds of private_localany `e3_part`.
Writes outputs/tables/data/per_seed_rates.csv, heldout_counts.csv, record_inventory.csv.
Usage: python analysis/tables/compute_rates.py   (first script of make_all_tables.py)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import shared as c2  # noqa: E402

DATA = c2.OUT / "tables/data"
RULES = ["pooled", "encmon_joint", "encmon_or", "localany", "hybrid", "exact", "encmon_norm"]
MRULES = ["pooled", "encmon_joint", "localany", "hybrid", "exact"]
SETS = ["mnist", "cifar10", "camelyon17_patch", "camelyon17_slide", "camelyon17_patient", "census_n250", "census_n1000"]


def rows_for(name, sc, seed, groups):
    out = []
    held = sc[sc.condition == "clean_heldout"]
    has_exact = "m_exact" in sc
    for k in RULES:
        if k == "exact" and not has_exact:
            continue
        mk = f"m_{k}" if k in MRULES else None
        out.append({"set": name, "aggregate": "clean_fpr", "method": k, "seed": seed, "conditions": 0,
                    "default": float(held[k].mean()), "matched": float(held[mk].mean()) if mk else np.nan,
                    "heldout_windows": int(len(held))})
    sh = sc[~sc.condition.str.startswith("clean")]
    cols = [k for k in RULES if k != "exact" or has_exact] + [f"m_{k}" for k in MRULES if k != "exact" or has_exact]
    cond = sh.groupby(["condition", "family", "geometry", "severity"])[cols].mean().reset_index()
    for agg, mk in groups(cond).items():
        if not mk.any():
            continue
        for k in RULES:
            if k == "exact" and not has_exact:
                continue
            m = f"m_{k}" if k in MRULES else None
            out.append({"set": name, "aggregate": agg, "method": k, "seed": seed, "conditions": int(mk.sum()),
                        "default": float(cond[mk][k].mean()), "matched": float(cond[mk][m].mean()) if m else np.nan,
                        "heldout_windows": np.nan})
    return out


def main() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    rows, inv = [], []
    for name in SETS:
        key = c2.SETS[name][0]
        seeds = c2.seeds_of(name)
        if not seeds:
            print(f"[{name}] no records, skipped", flush=True)
            continue
        for seed in seeds:
            rec = c2.load_records(name, [seed])
            sc, _ = c2.score(rec)
            conds = sorted(set(rec.condition))
            inv.append({"set": name, "seed": seed, "windows": len(rec), "conditions": len(conds) - 2,
                        "year_conditions": int(sum(c.startswith("year") for c in conds)),
                        "exact_mmd_present": bool(rec.exact_mmd.notna().all())})
            rows += rows_for(name, sc, seed, lambda cond: c2.P1.masks_for(cond, key))
            print(f"[{name} {seed}] scored", flush=True)
    # weak all-site shifts (E3 generator of private_localany), all seeds with caches
    E8 = c2.P1.E8
    for name, ds in (("mnist", "mnist"), ("cifar10", "cifar10"), ("camelyon17_patch", "camelyon17")):
        for seed in c2.seeds_of(name):
            ctx = c2.Ctx(ds, seed, c2.SETS[name][1])
            saved = c2.load_records(name, [seed])
            new = []
            for fam, sevs in E8.E3_FAMILIES.items():
                for sev in sevs:
                    for geo, dirs in E8.E3_GEOMETRIES.items():
                        c = {"condition": f"{fam}{sev:g}__{geo}", "family": fam, "kind": fam, "severity": float(sev),
                             "geometry": geo, "directions": dirs}
                        st = c2.rw.stats_from_gram(ctx.run(ctx.condition_feats(c), 50, seed + 2000)["gram"], ctx.alpha)
                        for w in range(50):
                            new.append({"condition": c["condition"], "family": fam, "severity": float(sev), "geometry": geo,
                                        "window": w, "d_mix": st["d_mix"][w], "d_site": st["d_site"][w],
                                        "heterogeneity": st["heterogeneity"][w], "site_energy": st["site_energy"][w].tolist(),
                                        "contributions": st["contributions"][w].tolist(), "exact_mmd": np.nan,
                                        "partition_seed": seed})
            base = saved[saved.condition.isin(["clean_calibration", "clean_heldout"])]
            sc, _ = c2.P1.score(pd.concat([base, pd.DataFrame(new)], ignore_index=True))
            held = sc[sc.condition == "clean_heldout"]
            for k, col in c2.METHODS.items():
                sc[f"m_{k}"] = sc[col] > c2.rc.qh(held[col], 0.95)
            sh = sc[~sc.condition.str.startswith("clean")]
            for geo in ("all", "single", "coherent"):
                g = sh[sh.geometry == geo]
                pc = g.groupby("condition")[[*c2.METHODS, *[f"m_{k}" for k in c2.METHODS], "encmon_or"]].mean()
                for k in [*c2.METHODS, "encmon_or"]:
                    rows.append({"set": f"weak_{ds}", "aggregate": f"weak_{geo}", "method": k, "seed": seed,
                                 "conditions": int(len(pc)), "default": float(pc[k].mean()),
                                 "matched": float(pc[f"m_{k}"].mean()) if k in c2.METHODS else np.nan, "heldout_windows": np.nan})
            print(f"[weak {ds} {seed}] regenerated", flush=True)
    c2.write_csv(pd.DataFrame(rows), DATA / "per_seed_rates.csv", "per-seed rates, all sets")
    c2.write_csv(pd.DataFrame(inv), DATA / "record_inventory.csv", "records per set and seed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

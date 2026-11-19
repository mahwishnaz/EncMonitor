#!/usr/bin/env python3
"""Interval comparison on the five original partition seeds (2026-2030): t-interval of the per-seed means (t = 2.776)
against the seed-cluster bootstrap interval over the same five seeds (2,000 cluster resamples), for the non-vector
opposing/mixed aggregate of each setting and the methods pooled, EncMonitor, Local-any and hybrid.
Width ratio = t-interval width / bootstrap width.

Inputs:
  --rates      per_seed_rates.csv written by analysis/tables/compute_rates.py (default outputs/tables/data/)
  --bootstrap  records_summary.csv written by experiments/private_localany/run.py (columns
               A1_natural_opposing_mixed_<method>_lo/_hi; default experiments/private_localany/records_summary.csv)
Output: <out>/interval_comparison.csv and .md (default outputs/tables/).
Usage (from the repository root): python analysis/tables/intervals.py [--rates F] [--bootstrap F] [--out DIR]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import tlib
from tlib import LABEL, SETTING, c2, pd, pick, save

SETS = ("mnist", "cifar10", "camelyon17_patch", "camelyon17_slide", "census_n250", "census_n1000")
METHODS = ("pooled", "encmon_joint", "localany", "hybrid")
AGG = "A1_natural_opposing_mixed"


def main() -> int:
    ap = argparse.ArgumentParser(description="t-interval vs seed-cluster bootstrap interval, five original seeds")
    ap.add_argument("--rates", type=Path, default=tlib.DATA / "per_seed_rates.csv")
    ap.add_argument("--bootstrap", type=Path, default=c2.CLONE / "experiments/private_localany/records_summary.csv")
    ap.add_argument("--out", type=Path, default=tlib.OUTD)
    a = ap.parse_args()
    r = c2.read_csv(a.rates)
    boot = pd.read_csv(a.bootstrap).set_index("set")
    out = []
    for s in SETS:
        g = pick(r, s, AGG, c2.OLD)
        for k in METHODS:
            sm = c2.summ(g[g.method == k]["default"].to_numpy())
            lo, hi = boot.loc[s, f"{AGG}_{k}_lo"], boot.loc[s, f"{AGG}_{k}_hi"]
            out.append({"setting": SETTING[s].replace(" (5 seeds)", ""), "method": LABEL[k], "mean_5_seeds": sm["mean"],
                        "t_lo": sm["t_lo"], "t_hi": sm["t_hi"], "boot_lo": lo, "boot_hi": hi,
                        "width_ratio_t_over_boot": (sm["t_hi"] - sm["t_lo"]) / (hi - lo)})
    df = pd.DataFrame(out)
    a.out.mkdir(parents=True, exist_ok=True)
    tlib.OUTD = a.out
    save(df, "interval_comparison", "Five original seeds: t-interval vs seed-cluster bootstrap interval",
         f"Width ratio over all rows: median {df.width_ratio_t_over_boot.median():.2f}, range "
         f"{df.width_ratio_t_over_boot.min():.2f}-{df.width_ratio_t_over_boot.max():.2f}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

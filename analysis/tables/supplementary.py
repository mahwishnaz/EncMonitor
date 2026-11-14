#!/usr/bin/env python3
"""Supplementary tables from the analyses in analysis/: weighted variants (5 seeds: uniform, (i), (ii); 20 seeds: (ii)),
clipping sensitivity (C = 8 vs 16), hospital patient/patch table and assignment counts. Reads outputs/analysis/*.
Usage: python analysis/tables/supplementary.py   (after the analysis scripts). Writes outputs/tables/supp_*."""
from tlib import SETTING, c2, pd, save

RES = c2.RESULTS
AGG = {"clean_fpr": "clean FPR", "A1_natural_opposing_mixed": "natural opposing/mixed", "opposing_all_families": "opposing (all families)",
       "A4_single": "single", "A5_coherent": "coherent", "year_all": "real drift, all states", "year_one_or_two": "real drift, one/two states"}


def summarize(d, cols, labels):
    out = []
    for (s, a), g in d.groupby(["set", "aggregate"], sort=False):
        if a not in AGG:
            continue
        row = {"setting": SETTING.get(s, s), "aggregate": AGG[a], "seeds": g.seed.nunique()}
        for k in cols:
            sm = c2.summ(g[k].to_numpy())
            row[labels[k]] = f"{sm['mean']:.3f} [{sm['t_lo']:.3f}, {sm['t_hi']:.3f}]"
        out.append(row)
    return pd.DataFrame(out)


def main():
    lab = {"pooled": "Pooled (uniform)", "encmon_joint": "EncMonitor (uniform)", "encmon_norm": "EncMonitor (i)",
           "encmon_w": "EncMonitor (ii)", "pooled_w": "Pooled (ii) weights", "localany": "Local-any",
           "m_encmon_joint": "EncMonitor uniform, matched", "m_encmon_norm": "EncMonitor (i), matched", "m_encmon_w": "EncMonitor (ii), matched"}
    f5 = RES / "weighted/per_seed_5seeds.csv"
    if f5.exists():
        save(summarize(c2.read_csv(f5), list(lab), lab), "supp_weighted_5seeds",
             "Weighted variants on the five existing seeds: uniform, (i) site-energy term normalized, (ii) normalized weights in D_mix and D_site",
             "Mean over seeds [t-interval]. (ii): w_s ∝ 1/clean energy (calibration windows), Σw = 1, same w in D_mix and D_site.")
    f20 = RES / "weighted/per_seed_all.csv"
    if f20.exists():
        save(summarize(c2.read_csv(f20), ["encmon_joint", "encmon_w", "pooled", "pooled_w", "m_encmon_joint", "m_encmon_w"], lab),
             "supp_weighted_20seeds", "Variant (ii) on all available seeds (separate variant; the main results use uniform weights)")
        w = c2.read_csv(RES / "weighted/weights_all.csv")
        save(w.groupby("set").agg(seeds=("seed", "nunique"), w_max_over_min_min=("w_max_over_min", "min"),
                                  w_max_over_min_max=("w_max_over_min", "max")).reset_index(), "supp_weighted_weight_spread",
             "Variant (ii): ratio of the largest to the smallest site weight")
    fc = RES / "clipping/per_seed.csv"
    if fc.exists():
        d = c2.read_csv(fc)
        out = []
        for (s, C, a), g in d.groupby(["set", "C", "aggregate"], sort=False):
            row = {"setting": SETTING[s], "C": C, "aggregate": a, "seeds": g.seed.nunique(), "bandwidth": g.bandwidth.mean()}
            for k in ("pooled", "encmon_joint", "localany", "hybrid", "m_encmon_joint", "m_localany"):
                sm = c2.summ(g[k].to_numpy())
                row[k] = f"{sm['mean']:.3f} [{sm['t_lo']:.3f}, {sm['t_hi']:.3f}]"
            out.append(row)
        save(pd.DataFrame(out), "supp_clipping", "Clipping sensitivity: C = 8 (main) vs C = 16, five seeds")
        fr = c2.read_csv(RES / "clipping/fractions.csv")
        save(fr.groupby(["set", "C"]).agg(fraction_clipped_mean=("fraction_clipped", "mean"), fraction_min=("fraction_clipped", "min"),
                                          fraction_max=("fraction_clipped", "max"), median_norm=("median_norm", "mean")).reset_index(),
             "supp_clipping_fractions", "Clipped fraction of the standardized representations (reference + monitoring pools)")
    fp = RES / "patient_split/pool_summary.csv"
    if fp.exists():
        save(c2.read_csv(fp), "supp_hospital_patient_table", "Camelyon17 patient split: patients, slides and patches per hospital, pool and seed")
        save(c2.read_csv(RES / "patient_split/assignments.csv"), "supp_hospital_assignments", "Patient-level assignments per hospital")
        save(c2.read_csv(RES / "patient_split/overlap.csv"), "supp_hospital_overlap", "Patch overlap between seeds per hospital")


if __name__ == "__main__":
    main()

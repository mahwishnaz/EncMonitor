#!/usr/bin/env python3
"""Patient-level split (Camelyon17): counts and overlaps.

- Number of possible patient-level assignments per hospital (from the WILDS metadata, 5 of 10 slides in the
  reference pool; ordered, and ignoring the swap of the two pools) and the number of distinct assignments drawn over
  the seeds (from each seed's cache_meta.json).
- Patch overlap between seeds per hospital: |P_a ∩ P_b| for the patches used by seeds a and b (reference +
  monitoring), for the patient split and the patch split (hospital 0 has 6,011 held-out patches).
- Supplementary table: patients and patches per hospital, pool and seed (from the per-seed patient_table.csv).
- Assertion re-check: no patient in both pools, from the saved manifests and the WILDS metadata.
Writes outputs/analysis/patient_split/{assignments.csv, overlap.csv, patient_table_all.csv, pool_summary.csv}.
Usage: python analysis/patient_split_summary.py   (needs data/wilds/camelyon17_v1.0/metadata.csv)
"""
from __future__ import annotations

import itertools
import json
import sys

import numpy as np
import pandas as pd

import shared as c2

OUTD = c2.RESULTS / "patient_split"


def main() -> int:
    OUTD.mkdir(parents=True, exist_ok=True)
    md = pd.read_csv(c2.CLONE / "data/wilds/camelyon17_v1.0/metadata.csv", index_col=0)
    patient = md["patient"].to_numpy()
    pdir = c2.SETS["camelyon17_patient"][1]
    seeds = sorted(int(p.name.split("_")[1]) for p in pdir.glob("seed_*") if (p / "cache_done.json").exists())
    # possible assignments from the metadata
    poss = {}
    for h, g in md.groupby("center"):
        sl = g.groupby("patient")["node"].nunique()
        v = sl.to_numpy()
        cnt = sum(1 for r in range(1, len(v)) for c in itertools.combinations(range(len(v)), r) if v[list(c)].sum() == v.sum() // 2)
        poss[int(h)] = {"patients": len(v), "slides": int(v.sum()), "slides_per_patient": sorted(v.tolist(), reverse=True),
                        "assignments_ordered": cnt, "assignments_ignoring_swap": cnt // 2}
    drawn = {h: [] for h in poss}
    tables, checks = [], []
    for s in seeds:
        d = pdir / f"seed_{s}"
        meta = json.load(open(d / "cache_meta.json"))
        man = np.load(d / "manifest.npz")
        for h in poss:
            ps = meta["patient_split"][str(h)]
            drawn[h].append(tuple(ps["reference_patients"]))
            assert ps["possible_assignments"] == poss[h]["assignments_ordered"]
            ref, mon = man[f"baseline_site_{h:03d}"], man[f"monitor_site_{h:03d}"]
            both = set(patient[ref]) & set(patient[mon])
            checks.append({"seed": s, "hospital": h, "patients_in_both_pools": len(both)})
        t = pd.read_csv(d / "patient_table.csv")
        tables.append(t)
    asg = pd.DataFrame([{"hospital": h, **{k: (str(v) if isinstance(v, list) else v) for k, v in poss[h].items()},
                         "seeds": len(drawn[h]), "distinct_assignments_drawn": len(set(drawn[h])),
                         "most_frequent_assignment_count": max(pd.Series(drawn[h]).value_counts()) if drawn[h] else 0}
                        for h in poss])
    # overlap of the used patches between seeds
    ov = []
    for split, root in (("patient", pdir), ("patch", c2.SETS["camelyon17_patch"][1])):
        ss = sorted(int(p.name.split("_")[1]) for p in root.glob("seed_*") if (p / "manifest.npz").exists())
        used = {s: np.load(root / f"seed_{s}" / "manifest.npz") for s in ss}
        for h in poss:
            sets = {s: set(np.concatenate([used[s][f"baseline_site_{h:03d}"], used[s][f"monitor_site_{h:03d}"]]).tolist()) for s in ss}
            pairs = [len(sets[a] & sets[b]) for a, b in itertools.combinations(ss, 2)]
            sizes = [len(v) for v in sets.values()]
            ov.append({"split": split, "hospital": h, "seeds": len(ss), "patches_used_per_seed_min": min(sizes),
                       "patches_used_per_seed_max": max(sizes), "pairwise_shared_min": min(pairs), "pairwise_shared_mean": float(np.mean(pairs)),
                       "pairwise_shared_max": max(pairs),
                       "heldout_patches_of_hospital": int(((md.center == h) & (md.split == 1)).sum()) if h in (0, 3, 4) else int((md.center == h).sum())})
    allt = pd.concat(tables, ignore_index=True)
    pool = allt.groupby(["partition_seed", "hospital", "pool"]).agg(patients=("patient", "nunique"), slides=("slides", "sum"),
                                                                     patches_drawn=("patches_drawn", "sum")).reset_index()
    c2.write_csv(asg, OUTD / "assignments.csv", "possible and drawn patient-level assignments per hospital")
    c2.write_csv(pd.DataFrame(ov), OUTD / "overlap.csv", "patch overlap between seeds per hospital")
    c2.write_csv(allt, OUTD / "patient_table_all.csv", "patients and patches per hospital, pool and seed")
    c2.write_csv(pool, OUTD / "pool_summary.csv", "patients, slides and patches per hospital, pool and seed")
    c2.write_csv(pd.DataFrame(checks), OUTD / "no_shared_patient_check.csv", "patients present in both pools (must be 0)")
    print(asg.to_string(index=False))
    print(pd.DataFrame(ov).to_string(index=False))
    print("max patients in both pools:", max(c["patients_in_both_pools"] for c in checks))
    return 0


if __name__ == "__main__":
    sys.exit(main())

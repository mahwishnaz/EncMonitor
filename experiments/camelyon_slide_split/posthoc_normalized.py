#!/usr/bin/env python3
"""Post-hoc analysis (not pre-specified): per-site normalized D_site.

Under the slide-level split, a few sites have clean energies 7-58 times higher than under the
patch-level split, and they dominate D_site, while Local-any (each site against its own clean level) keeps more
power. The CKKS circuit accepts any public weights, so D_site can use w_s = alpha_s / e0_s, where e0_s is the
site's mean clean energy over the calibration windows (a public calibration constant). Same depth, same release.
Score: max(D_mix / tau_mix, D_site_norm / tau_norm), jointly calibrated exactly like EncMonitor-joint.
Reported for all datasets and both Camelyon17 splits, next to EncMonitor-joint and Local-any.

Usage (bundle root): python experiments/camelyon_slide_split/posthoc_normalized.py > experiments/camelyon_slide_split/posthoc_normalized_log.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
import rx_common as rc  # noqa: E402

SETS = {"mnist": rc.RECORD_DIRS["mnist"], "cifar10": rc.RECORD_DIRS["cifar10"],
        "camelyon17 (patch split)": rc.RECORD_DIRS["camelyon17"], "camelyon17 (slide split)": HERE / "cache" / "camelyon17"}


def main() -> int:
    rows = []
    for name, d in SETS.items():
        key = name.split(" ")[0]
        rc.RECORD_DIRS[key] = d
        df = rc.load(key)
        parts = []
        for seed, g in df.groupby("partition_seed"):
            sc, info = rc.score_seed(g, 0.95)
            cal = g[g.condition == "clean_calibration"]
            e0 = info["e0_s"]
            norm_cal = (np.stack(cal.site_energy.to_numpy()) / e0).mean(axis=1)
            t_norm = rc.qh(norm_cal, 0.95)
            s_cal = np.maximum(cal.d_mix.to_numpy() / info["t_mix"], norm_cal / t_norm)
            c = rc.qh(s_cal, 0.95)
            norm = (np.stack(sc.site_energy.to_numpy()) / e0).mean(axis=1)
            sc["encmon_norm"] = np.maximum(sc.d_mix.to_numpy() / info["t_mix"], norm / t_norm) > c
            parts.append(sc)
        sc = pd.concat(parts, ignore_index=True)
        held = sc[sc.condition == "clean_heldout"]
        cond = sc[sc.condition != "clean_heldout"].groupby(["condition", "family", "geometry"])[["pooled", "encmon_joint", "encmon_norm", "localany"]].mean().reset_index()
        masks = rc.aggregate_masks(cond)
        r = {"set": name}
        for k in ("pooled", "encmon_joint", "encmon_norm", "localany"):
            r[f"fpr_{k}"] = float(held[k].mean())
            for agg in ("A1_natural_opposing_mixed", "A4_single", "A5_coherent"):
                r[f"{agg}_{k}"] = float(cond[masks[agg]][k].mean())
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "posthoc_normalized.csv", index=False)
    pd.set_option("display.width", 250)
    print(out.set_index("set").T.to_string(float_format=lambda x: f"{x:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

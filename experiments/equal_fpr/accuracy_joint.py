#!/usr/bin/env python3
"""Accuracy link and Camelyon17 global-reference comparison under joint calibration.

The accuracy-link numbers of scripts 27 and 34 and their global-reference comparison use the OR rule
(FPR 0.07-0.08). Here the same windows are re-thresholded with the jointly calibrated EncMonitor rule and
with Local-any, so that all methods run at about the same clean FPR. Inputs are read only:
  MNIST/CIFAR-10: outputs/main/analysis/accuracy_link/<ds>/windows_accuracy.csv (+ saved records)
  Camelyon17: records.jsonl (field `acc`) and conditions.json (`clean_accuracy_per_site`) of a records dir.

Usage (from the repository root): python experiments/equal_fpr/accuracy_joint.py [--camelyon-dir DIR --tag TAG]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
import rx_common as rc  # noqa: E402

HARM = 0.05
METHODS = ["pooled", "encmon_or", "encmon_joint", "localany"]


def link(g: pd.DataFrame) -> dict:
    h = g[g.harmful]
    r = {"windows": int(len(g)), "harmful": int(len(h))}
    for k in METHODS:
        r[f"detect_harmful_{k}"] = float(h[k].mean()) if len(h) else float("nan")
    ah = h[h.encmon_joint]
    r["top1_A_is_worst_site_in_harmful_joint_alerts"] = float(ah.top1_A_is_worst_site.mean()) if len(ah) else float("nan")
    only = g[g.encmon_joint & ~g.pooled]
    r["joint_only_alerts"] = int(len(only))
    r["joint_only_harmful_share"] = float(only.harmful.mean()) if len(only) else float("nan")
    return r


def scored_with_acc_g2(ds: str) -> pd.DataFrame:
    sc, _ = rc.load_scored(ds)
    wa = pd.read_csv(rc.ROOT / f"outputs/main/analysis/accuracy_link/{ds}/windows_accuracy.csv")
    m = sc.merge(wa[["partition_seed", "condition", "window", "max_drop", "harmful", "d_mix", "top1_A_is_worst_site"]],
                 on=["partition_seed", "condition", "window"], suffixes=("", "_acc"))
    assert len(m) == len(sc), (len(m), len(sc))
    assert float(np.abs(m.d_mix - m.d_mix_acc).max()) < 1e-12
    return m


def scored_with_acc_camelyon(rec_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rc.RECORD_DIRS["camelyon17"] = rec_dir
    df = rc.load("camelyon17")
    clean_acc = {int(json.load(open(f))["partition_seed"]): np.array(json.load(open(f))["clean_accuracy_per_site"])
                 for f in sorted(rec_dir.glob("seed_*/conditions.json"))}
    parts, gparts = [], []
    for seed, g in df.groupby("partition_seed"):
        s, _ = rc.score_seed(g, 0.95)
        parts.append(s)
        gg = g.copy()   # same windows scored against one global reference (records fields *_global)
        gg["d_mix"], gg["d_site"], gg["heterogeneity"] = gg.d_mix_global, gg.d_site_global, gg.heterogeneity_global
        sg, _ = rc.score_seed(gg, 0.95)
        gparts.append(sg)
    sc = pd.concat(parts, ignore_index=True)
    acc = np.stack(sc.acc.to_numpy())
    ref = np.stack([clean_acc[int(p)] for p in sc.partition_seed])
    sc["max_drop"] = (ref - acc).max(axis=1)
    sc["harmful"] = sc.max_drop >= HARM
    sc["top1_A_is_worst_site"] = np.stack(sc.contributions.to_numpy()).argmax(axis=1) == (ref - acc).argmax(axis=1)
    return sc, pd.concat(gparts, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camelyon-dir", default=str(rc.RECORD_DIRS["camelyon17"]))
    ap.add_argument("--tag", default="")
    ap.add_argument("--skip-g2", action="store_true")
    args = ap.parse_args()
    rows = []
    frames = {} if args.skip_g2 else {ds: scored_with_acc_g2(ds) for ds in ("mnist", "cifar10")}
    cam, cam_g = scored_with_acc_camelyon(Path(args.camelyon_dir))
    frames["camelyon17"] = cam
    for ds, sc in frames.items():
        sh = sc[sc.condition != "clean_heldout"]
        cl = sc[sc.condition == "clean_heldout"]
        for name, g in (("all_shifted", sh), ("opposing_mixed", sh[sh.geometry.isin(rc.OPP_MIX)])):
            rows.append({"dataset": ds, "subset": name, **link(g),
                         "clean_fpr_joint": float(cl.encmon_joint.mean()), "clean_fpr_pooled": float(cl.pooled.mean()),
                         "clean_harmful_share": float(cl.harmful.mean())})
        for fam, g in sh[sh.geometry.isin(rc.OPP_MIX)].groupby("family"):
            rows.append({"dataset": ds, "subset": f"opposing_mixed:{fam}", **link(g)})
    out = pd.DataFrame(rows)
    out.to_csv(HERE / f"accuracy_joint{args.tag}.csv", index=False)

    # Camelyon17 under joint calibration: site references vs one global reference
    h = []
    held, heldg = cam[cam.condition == "clean_heldout"], cam_g[cam_g.condition == "clean_heldout"]
    h.append({"condition": "clean_heldout", "joint_site_ref": held.encmon_joint.mean(), "joint_global_ref": heldg.encmon_joint.mean(),
              "or_site_ref": held.encmon_or.mean(), "or_global_ref": heldg.encmon_or.mean(), "pooled_site_ref": held.pooled.mean()})
    for cn, g in cam[cam.geometry.isin(rc.OPP_MIX)].groupby("condition"):
        gg = cam_g[cam_g.condition == cn]
        h.append({"condition": cn, "family": g.family.iloc[0], "joint_site_ref": g.encmon_joint.mean(), "joint_global_ref": gg.encmon_joint.mean(),
                  "or_site_ref": g.encmon_or.mean(), "or_global_ref": gg.encmon_or.mean(), "pooled_site_ref": g.pooled.mean()})
    h = pd.DataFrame(h)
    h.to_csv(HERE / f"camelyon_h2_joint{args.tag}.csv", index=False)
    pd.set_option("display.width", 250)
    print(out.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(h.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

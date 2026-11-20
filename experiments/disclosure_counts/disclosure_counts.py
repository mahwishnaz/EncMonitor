#!/usr/bin/env python3
"""Exact alert counts behind the default-policy disclosure counts E[V] = 3 + S x alert rate.

Re-uses the scoring of private_localany/run.py (joint EncMonitor rule, OR rule) on the saved records.
Counts held-out clean windows and the windows of the non-vector opposing/mixed aggregate (all shift families except the vector shift; for census the
injected label-prior/age-mix opposing/mixed conditions). Reads only saved records; writes only into this folder.
Usage (from the repository root): python experiments/disclosure_counts/disclosure_counts.py
"""
import importlib.util
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RX = HERE.parent
spec = importlib.util.spec_from_file_location("p1", RX / "private_localany" / "run.py")
P1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P1)
rc = P1.rc
S = 5

rows = []
for name, (key, d) in P1.SETS.items():
    rc.RECORD_DIRS[key] = d
    df = rc.load(key)
    sc = pd.concat([P1.score(g)[0] for _, g in df.groupby("partition_seed")], ignore_index=True)
    # OR rule: both statistics at their own 95th percentile (column name in rx_common scoring)
    or_col = "encmon_or" if "encmon_or" in sc.columns else None
    held = sc[sc.condition == "clean_heldout"]
    sh = sc[~sc.condition.str.startswith("clean")]
    cond = sh.groupby(["condition", "family", "geometry", "severity"]).size().reset_index()
    mk = P1.masks_for(cond, key)["A1_natural_opposing_mixed"]
    a1 = sh[sh.condition.isin(set(cond[mk].condition))]
    for part, sub in (("clean_heldout", held), ("A1_opposing_mixed", a1)):
        n, k = len(sub), int(sub.encmon_joint.sum())
        r = {"set": name, "windows": part, "n_windows": n, "n_conditions": int(sub.condition.nunique()),
             "joint_alerts": k, "joint_rate": k / n, "EV_joint": 3 + S * k / n}
        if or_col:
            ko = int(sub[or_col].sum())
            r.update({"or_alerts": ko, "or_rate": ko / n, "EV_or": 3 + S * ko / n})
        rows.append(r)
    print(f"[{name}] done; columns: {[c for c in sc.columns if 'enc' in c]}", flush=True)

out = pd.DataFrame(rows)
out.to_csv(HERE / "disclosure_counts.csv", index=False)
pd.set_option("display.width", 200)
print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

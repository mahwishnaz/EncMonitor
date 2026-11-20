#!/usr/bin/env python3
"""Joint threshold multipliers per seed (EncMonitor c_joint vs hybrid c_h), to explain the hybrid's lower all-site power.

Usage (from the repository root): python experiments/private_localany/thresholds.py
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("p1run", HERE / "run.py")
P1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P1)
rc = P1.rc

rows = []
for name, (key, d) in P1.SETS.items():
    rc.RECORD_DIRS[key] = d
    for seed, g in rc.load(key).groupby("partition_seed"):
        _, info = P1.score(g)
        rows.append({"set": name, "partition_seed": int(seed), "c_joint_encmonitor": info["c_joint"], "c_hybrid": info["c_h"],
                     "tau_lm_localany": info["tau_lm"], "alpha_sidak": info["alpha_sidak"]})
df = pd.DataFrame(rows)
df.to_csv(HERE / "thresholds.csv", index=False)
print(df.groupby("set")[["c_joint_encmonitor", "c_hybrid", "tau_lm_localany"]].agg(["min", "max"]).round(3).to_string())
print("all sets:", df.c_joint_encmonitor.min().round(3), df.c_joint_encmonitor.max().round(3), df.c_hybrid.min().round(3), df.c_hybrid.max().round(3))

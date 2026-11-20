#!/usr/bin/env python3
"""Per-seed mean over natural opposing+mixed conditions, slide split.
Usage (from the repository root): python experiments/camelyon_slide_split/per_seed_a1.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
import rx_common as rc  # noqa: E402

rc.RECORD_DIRS["camelyon17"] = HERE / "cache" / "camelyon17"
sc, _ = rc.load_scored("camelyon17")
nat = sc[(sc.condition != "clean_heldout") & (sc.family != "vector") & sc.geometry.isin(rc.OPP_MIX)]
out = nat.groupby("partition_seed")[["pooled", "encmon_joint", "localany"]].mean()
out.to_csv(HERE / "per_seed_a1.csv")
print(out.round(3))

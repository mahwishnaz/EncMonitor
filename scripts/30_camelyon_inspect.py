#!/usr/bin/env python3
"""Download (optional) and inspect Camelyon17-WILDS.

Camelyon17-WILDS: 96x96 H&E patches from lymph-node slides of five hospitals,
label 1 = the central 32x32 region contains tumor tissue. License: CC0.

This script only reads the metadata (and opens a few images to test the files).
It prints, for every hospital and every official split, the number of patches
and the tumor fraction, and writes the same table to
    outputs/camelyon/inspect.json
The later scripts (31-34) use this table to choose the training data and the
five site pools.

Usage (from the repository root):
  python scripts/30_camelyon_inspect.py --root data/wilds --download   # first time
  python scripts/30_camelyon_inspect.py --root data/wilds              # afterwards
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description="Download / inspect Camelyon17-WILDS")
    ap.add_argument("--root", default="data/wilds")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--out", default="outputs/camelyon/inspect.json")
    args = ap.parse_args()

    from wilds import get_dataset

    t0 = time.time()
    ds = get_dataset(dataset="camelyon17", root_dir=args.root, download=args.download)
    fields = list(ds.metadata_fields)
    meta = ds.metadata_array.numpy()
    hosp = meta[:, fields.index("hospital")].astype(int)
    slide = meta[:, fields.index("slide")].astype(int)
    y = np.asarray(ds.y_array).astype(int)
    split = np.asarray(ds.split_array).astype(int)
    split_names = {v: k for k, v in ds.split_dict.items()}

    table = []
    print(f"\nCamelyon17-WILDS: {len(y):,} patches, metadata fields = {fields}")
    print(f"{'hospital':>8} {'split':>8} {'patches':>9} {'slides':>7} {'tumor %':>8}")
    for h in sorted(np.unique(hosp)):
        for sp in sorted(np.unique(split[hosp == h])):
            sel = (hosp == h) & (split == sp)
            row = {"hospital": int(h), "split": split_names.get(int(sp), str(sp)),
                   "patches": int(sel.sum()), "slides": int(np.unique(slide[sel]).size),
                   "tumor_fraction": float(y[sel].mean())}
            table.append(row)
            print(f"{h:>8} {row['split']:>8} {row['patches']:>9,} {row['slides']:>7} {100 * row['tumor_fraction']:>7.1f}%")

    train_hosp = sorted({r["hospital"] for r in table if r["split"] == "train"})
    print(f"\nhospitals in the official training split: {train_hosp}")

    # open a few images to make sure the files are complete
    sizes = set()
    rng = np.random.default_rng(0)
    for i in rng.choice(len(y), size=20, replace=False):
        img = ds.get_input(int(i))
        sizes.add(tuple(img.size) + (img.mode,))
    print(f"image check (20 random patches): {sorted(sizes)}")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "n_patches": int(len(y)), "metadata_fields": fields, "split_dict": dict(ds.split_dict),
        "train_hospitals": train_hosp, "table": table, "image_sizes": [list(s) for s in sorted(sizes)],
        "data_dir": str(ds.data_dir), "seconds": round(time.time() - t0, 1),
    }, indent=2))
    print(f"\nwritten: {out}   ({round(time.time() - t0, 1)} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

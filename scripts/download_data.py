#!/usr/bin/env python3
"""Download the public datasets used by the experiments (nothing is redistributed with this repository).

  MNIST, CIFAR-10      torchvision            -> data/
  Camelyon17-WILDS 1.0 wilds.get_dataset      -> data/wilds   (about 10 GB)
  ACS census 2014-2018 Folktables             -> experiments/census/data (via experiments/census/download.py)

Usage (repository root):
  python scripts/download_data.py            # download what is missing
  python scripts/download_data.py --check    # only report what is present
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description="Download MNIST, CIFAR-10, Camelyon17-WILDS and the ACS census files")
    ap.add_argument("--root", default="data")
    ap.add_argument("--check", action="store_true", help="do not download; report presence only")
    ap.add_argument("--skip", default="", help="comma-separated subset of mnist,cifar10,camelyon17,census to skip")
    args = ap.parse_args()
    skip = {s for s in args.skip.split(",") if s}
    root = ROOT / args.root
    from encmonitor.data import load_dataset

    for name in ("mnist", "cifar10"):
        if name in skip:
            continue
        try:
            load_dataset(name, "test", str(root), download=not args.check)
            load_dataset(name, "train", str(root), download=not args.check)
            print(f"{name}: present in {root}")
        except Exception as exc:
            print(f"{name}: missing ({exc})")
    if "camelyon17" not in skip:
        try:
            from wilds import get_dataset
            get_dataset(dataset="camelyon17", root_dir=str(root / "wilds"), download=not args.check)
            print(f"camelyon17: present in {root / 'wilds'}")
        except Exception as exc:
            print(f"camelyon17: missing ({exc})")
    if "census" not in skip:
        if args.check:
            d = ROOT / "experiments" / "census" / "data"
            print(f"census: {'present' if d.exists() and any(d.rglob('*.csv')) else 'missing'} in {d}")
        else:
            subprocess.run([sys.executable, str(ROOT / "experiments" / "census" / "download.py")], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

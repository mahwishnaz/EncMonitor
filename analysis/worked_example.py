#!/usr/bin/env python3
"""Worked example of Figure 1: one window of the saved per-window records.

Reads the records as they are stored in the data archive (encmonitor_data_mnist_cifar10.tar.gz):
<records_dir>/seed_<seed>/records.jsonl, one JSON line per window. Default setting: MNIST, S = 5, m = 64, n = 250,
condition vector2__opposing (vector shift eta = 2, +2v at site 0, -2v at site 1).

Per partition seed, from the 500 rows with condition clean_calibration (95th percentile, numpy method "higher"):
tau_mix of d_mix, tau_site of d_site, and c of max(d_mix / tau_mix, d_site / tau_site).
EncMonitor alert: max(D_mix / tau_mix, D_site / tau_site) > c. Pooled alert: D_mix > tau_mix.
Selected window: the first window, by index, of the condition in seed --seed in which the EncMonitor alert fires
and the pooled alert is silent. Printed: window index; D_mix/tau_mix, D_site/tau_site, score/c; A_s (field
contributions); residual norms ||delta_s|| = sqrt(site_energy); alert rates over the windows of the condition in
seed --seed and their mean over all seeds found in <records_dir>.

Usage (from the repository root):
  python analysis/worked_example.py encmonitor_data/mnist
  (or outputs/main/mnist, the folder written by scripts/15_run_screening.py)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def q95(v) -> float:
    return float(np.quantile(np.asarray(v, dtype=float), 0.95, method="higher"))


def read(path: Path) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def seed_windows(recs: list[dict], condition: str) -> dict:
    cal = [r for r in recs if r["condition"] == "clean_calibration"]
    t_mix = q95([r["d_mix"] for r in cal])
    t_site = q95([r["d_site"] for r in cal])
    c = q95([max(r["d_mix"] / t_mix, r["d_site"] / t_site) for r in cal])
    win = sorted((r for r in recs if r["condition"] == condition), key=lambda r: r["window"])
    score = np.array([max(r["d_mix"] / t_mix, r["d_site"] / t_site) for r in win])
    return {"cal": len(cal), "t_mix": t_mix, "t_site": t_site, "c": c, "win": win,
            "encmon": score > c, "pooled": np.array([r["d_mix"] > t_mix for r in win]), "score": score}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("records_dir", help="folder with seed_<seed>/records.jsonl (e.g. encmonitor_data/mnist)")
    ap.add_argument("--seed", type=int, default=2026, help="partition seed of the example window")
    ap.add_argument("--condition", default="vector2__opposing", help="condition of the example window")
    args = ap.parse_args()

    root = Path(args.records_dir)
    files = sorted(root.glob("seed_*/records.jsonl"))
    if not files:
        print(f"No seed_*/records.jsonl under {root}")
        return 1
    per_seed = {}
    for f in files:
        recs = read(f)
        per_seed[int(recs[0]["partition_seed"])] = seed_windows(recs, args.condition)
    if args.seed not in per_seed:
        print(f"Seed {args.seed} not found under {root}")
        return 1
    d = per_seed[args.seed]
    if not d["win"]:
        print(f"Condition {args.condition} not found for seed {args.seed}")
        return 1
    sel = np.flatnonzero(d["encmon"] & ~d["pooled"])
    print(f"records: {root} ({len(per_seed)} seeds); condition {args.condition}; seed {args.seed}")
    print(f"calibration windows: {d['cal']}; tau_mix = {d['t_mix']:.6g}, tau_site = {d['t_site']:.6g}, c = {d['c']:.6g}")
    if sel.size == 0:
        print("No window with EncMonitor alert and silent pooled alert.")
    else:
        r = d["win"][sel[0]]
        S = len(r["contributions"])
        print(f"selected window (first by index with EncMonitor alert and pooled alert silent): {r['window']}")
        print(f"D_mix/tau_mix = {r['d_mix'] / d['t_mix']:.2f}  D_site/tau_site = {r['d_site'] / d['t_site']:.2f}  "
              f"score/c = {d['score'][sel[0]] / d['c']:.2f}")
        print(f"pooled alert: {bool(r['d_mix'] > d['t_mix'])}  EncMonitor alert: {bool(d['score'][sel[0]] > d['c'])}")
        print("site            " + "  ".join(f"{s:>8d}" for s in range(S)))
        print("shifted         " + "  ".join(f"{str(bool(x)):>8s}" for x in r["shifted_mask"]))
        print("A_s             " + "  ".join(f"{a:8.4f}" for a in r["contributions"]))
        print("||delta_s||     " + "  ".join(f"{np.sqrt(e):8.3f}" for e in r["site_energy"]))
    n = len(d["win"])
    both = d["encmon"] & ~d["pooled"]
    print(f"seed {args.seed}, {n} windows: EncMonitor alert rate {d['encmon'].mean():.2f}, pooled alert rate "
          f"{d['pooled'].mean():.2f}, EncMonitor alert with pooled silent {both.mean():.0%}")
    rates = np.array([[v["encmon"].mean(), v["pooled"].mean(), (v["encmon"] & ~v["pooled"]).mean()] for v in per_seed.values()])
    print(f"mean over {len(per_seed)} seeds ({min(per_seed)}-{max(per_seed)}): EncMonitor alert rate {rates[:, 0].mean():.2f}, "
          f"pooled alert rate {rates[:, 1].mean():.2f}, EncMonitor alert with pooled silent {rates[:, 2].mean():.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

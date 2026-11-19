#!/usr/bin/env python3
"""Export the per-window outputs of the weak shifts (vector and label-prior shifts of small severity at one, two or
all five sites) that compute_rates.py generates in memory.

The windows are produced by the same code as in compute_rates.py (generator of experiments/private_localany/run.py
`e3_part`): the same seed context (analysis/shared.py Ctx), the same conditions (E3_FAMILIES x E3_GEOMETRIES of
experiments/normalized_weights/run.py: 2 families x 3 severities x 3 geometries = 18 conditions per seed), the same
window seed (partition seed + 2000), 50 windows per condition and the same statistics (rx_windows.stats_from_gram).
This script only adds the writing of one record per window. The calibration and held-out clean windows are the saved
`clean_calibration` and `clean_heldout` rows of records.jsonl of the same seed; compute_rates.py scores the weak
windows against them.

Inputs (read only): the per-seed cache folders written by scripts/14 and scripts/32 (preprocessor.npz, rff_map.npz,
baselines.npz, monitor_clean.npz, pieces.npz) below <outputs>/main/mnist, <outputs>/main/cifar10 and
<outputs>/camelyon/main/camelyon17. CPU only.

Output: <out>/<experiment>/seed_<seed>/weak_shift_records.jsonl with <experiment> = mnist, cifar10,
camelyon17/patch_split. Fields per window: condition, family, severity, geometry, window, window_seed, partition_seed,
d_mix, d_site, heterogeneity, contributions, directional, site_energy (lists of 5), shifted_mask, exact_mmd (null:
not computed for these windows), top1, topk_recall, ndcg, spearman (attribution scores from the contributions, as in
scripts/15). d_mix, d_site, heterogeneity, contributions and site_energy are the values that compute_rates.py scores;
directional and the attribution scores are computed from the same residual Gram matrices and are not used for the
rates; an undefined score (Spearman correlation when all sites are shifted equally) is null. No accuracy is
computed (the generator does not apply the model).

Usage (from the repository root):
  python analysis/tables/export_weak_all_windows.py --outputs outputs --out <folder> [--seeds 2026 2027 ...] [--jobs 4]
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import shared as c2  # noqa: E402
from encmonitor.metrics import top1_accuracy, topk_recall  # noqa: E402

# set of compute_rates.py -> (dataset of the window generator, folder in the export)
EXPERIMENTS = {"mnist": ("mnist", "mnist"), "cifar10": ("cifar10", "cifar10"),
               "camelyon17_patch": ("camelyon17", "camelyon17/patch_split")}
WINDOWS = 50


def _num(v: float):
    """NaN (e.g. Spearman correlation when all five sites are shifted equally) is written as null."""
    return None if v != v else float(v)


def export_one(name: str, seed: int, outputs: Path, out: Path) -> tuple[str, int, int]:
    ds, folder = EXPERIMENTS[name]
    cache = outputs / c2.SETS[name][1].relative_to(c2.OUT)
    ctx = c2.Ctx(ds, seed, cache)
    E8 = c2.P1.E8
    rows = []
    # loops, condition dictionaries, window counts and seeds exactly as in compute_rates.py
    for fam, sevs in E8.E3_FAMILIES.items():
        for sev in sevs:
            for geo, dirs in E8.E3_GEOMETRIES.items():
                c = {"condition": f"{fam}{sev:g}__{geo}", "family": fam, "kind": fam, "severity": float(sev),
                     "geometry": geo, "directions": dirs}
                gram = ctx.run(ctx.condition_feats(c), WINDOWS, seed + 2000)["gram"]
                st = c2.rw.stats_from_gram(gram, ctx.alpha)
                # added fields (not used for the rates): G_s = alpha_s <r_s, sum_t alpha_t r_t>, shifted mask, attribution
                directional = ctx.alpha[None, :] * np.einsum("wst,t->ws", gram, ctx.alpha)
                d = np.asarray(dirs, dtype=float)
                mask = d != 0
                for w in range(WINDOWS):
                    a = st["contributions"][w]
                    rows.append({"condition": c["condition"], "family": fam, "severity": float(sev), "geometry": geo,
                                 "window": w, "window_seed": seed + 2000, "partition_seed": seed,
                                 "d_mix": float(st["d_mix"][w]), "d_site": float(st["d_site"][w]),
                                 "heterogeneity": float(st["heterogeneity"][w]),
                                 "contributions": [float(v) for v in a],
                                 "directional": [float(v) for v in directional[w]],
                                 "site_energy": [float(v) for v in st["site_energy"][w]],
                                 "shifted_mask": [bool(v) for v in mask], "exact_mmd": None,
                                 "top1": top1_accuracy(a, mask), "topk_recall": topk_recall(a, mask),
                                 "ndcg": _num(c2.rw.M15.ndcg(a, np.abs(d))), "spearman": _num(c2.rw.M15.spearman(a, np.abs(d)))})
    dst = out / folder / f"seed_{seed}" / "weak_shift_records.jsonl"
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return name, seed, len(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Export the per-window outputs of the weak shifts of compute_rates.py")
    ap.add_argument("--outputs", type=Path, default=c2.OUT, help="folder with main/ and camelyon/ (default: outputs)")
    ap.add_argument("--out", type=Path, required=True, help="destination folder")
    ap.add_argument("--sets", nargs="*", default=list(EXPERIMENTS), choices=list(EXPERIMENTS))
    ap.add_argument("--seeds", nargs="*", type=int, default=None, help="default: every seed with a cache folder")
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    outputs, out = a.outputs.resolve(), a.out.resolve()
    tasks = []
    for name in a.sets:
        cache = outputs / c2.SETS[name][1].relative_to(c2.OUT)
        have = sorted(int(p.name.split("_")[1]) for p in cache.glob("seed_*") if (p / "rff_map.npz").exists())
        tasks += [(name, s) for s in (a.seeds or have) if s in have]
    with ProcessPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        for name, seed, n in ex.map(export_one, *zip(*tasks), [outputs] * len(tasks), [out] * len(tasks)):
            print(f"[{name} {seed}] {n} windows", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

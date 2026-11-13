#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from encmonitor.artifacts import write_json
from encmonitor.data import labels_array, load_dataset
from encmonitor.site_partition import allocate_with_class_to_site, sample_class_to_site_probabilities


def main() -> int:
    ap = argparse.ArgumentParser(description="Create persistent non-IID site partitions")
    ap.add_argument("--dataset", choices=["mnist", "cifar10"], default="mnist")
    ap.add_argument("--root", default="data")
    ap.add_argument("--sites", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=20260818)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--baseline-fraction", type=float, default=0.5)
    ap.add_argument("--out", default="outputs/manifests/mnist_s5_a0.5")
    args = ap.parse_args()

    train = load_dataset(args.dataset, "train", args.root, args.download)
    test = load_dataset(args.dataset, "test", args.root, args.download)
    y_train, y_test = labels_array(train), labels_array(test)
    n_classes = int(max(y_train.max(), y_test.max()) + 1)
    p = sample_class_to_site_probabilities(n_classes, args.sites, args.alpha, args.seed)
    part_train = allocate_with_class_to_site(y_train, p, args.seed + 1, min_per_site=10)
    part_test = allocate_with_class_to_site(y_test, p, args.seed + 2, min_per_site=10)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not 0.1 <= args.baseline_fraction <= 0.9:
        raise ValueError("--baseline-fraction must be in [0.1,0.9]")
    baseline_sites, monitor_sites = [], []
    for arr in part_test.site_indices:
        cut = max(1, min(arr.size - 1, int(round(arr.size * args.baseline_fraction))))
        baseline_sites.append(arr[:cut])
        monitor_sites.append(arr[cut:])

    np.savez_compressed(
        out.with_suffix(".npz"),
        class_to_site=p,
        train_class_proportions=part_train.class_proportions,
        test_class_proportions=part_test.class_proportions,
        **{f"train_site_{s:03d}": v for s, v in enumerate(part_train.site_indices)},
        **{f"test_site_{s:03d}": v for s, v in enumerate(part_test.site_indices)},
        **{f"baseline_site_{s:03d}": v for s, v in enumerate(baseline_sites)},
        **{f"monitor_site_{s:03d}": v for s, v in enumerate(monitor_sites)},
    )
    summary = {
        "dataset": args.dataset,
        "n_sites": args.sites,
        "dirichlet_alpha": args.alpha,
        "seed": args.seed,
        "train_sizes": [int(x.size) for x in part_train.site_indices],
        "test_sizes": [int(x.size) for x in part_test.site_indices],
        "baseline_sizes": [int(x.size) for x in baseline_sites],
        "monitor_sizes": [int(x.size) for x in monitor_sites],
        "baseline_fraction": args.baseline_fraction,
        "train_class_proportions": part_train.class_proportions.tolist(),
        "test_class_proportions": part_test.class_proportions.tolist(),
        "npz": str(out.with_suffix(".npz")),
    }
    write_json(out.with_suffix(".json"), summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

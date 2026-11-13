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
from encmonitor.embeddings import EmbeddingStandardizer
from encmonitor.rff import RFFMap, median_bandwidth


def main() -> int:
    ap = argparse.ArgumentParser(description="Fit the public preprocessing and RFF map")
    ap.add_argument("--embeddings", required=True, help="baseline embeddings NPZ")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--clip-radius", type=float, default=8.0)
    ap.add_argument("--bandwidth-multiplier", type=float, default=1.0)
    ap.add_argument("--max-bandwidth-samples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1729)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    z = np.load(args.embeddings)
    site_keys = sorted(k for k in z.files if k.startswith("site_"))
    if not site_keys:
        raise RuntimeError("No site_* arrays found")
    x = np.concatenate([np.asarray(z[k], dtype=np.float64) for k in site_keys], axis=0)
    standardizer = EmbeddingStandardizer.fit(x, clip_radius=args.clip_radius)
    x_std = standardizer.transform(x)
    sigma0 = median_bandwidth(x_std, max_samples=args.max_bandwidth_samples, seed=args.seed)
    sigma = sigma0 * args.bandwidth_multiplier
    rff = RFFMap.fit(x_std.shape[1], args.dim, sigma, args.seed)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    standardizer.save(out / "preprocessor.npz")
    rff.save(out / "rff_map.npz")
    meta = {
        "source": args.embeddings,
        "n_samples": int(x.shape[0]),
        "input_dim": int(x.shape[1]),
        "rff_dim": args.dim,
        "clip_radius": args.clip_radius,
        "median_bandwidth": sigma0,
        "bandwidth_multiplier": args.bandwidth_multiplier,
        "bandwidth": sigma,
        "seed": args.seed,
        "preprocessor": str(out / "preprocessor.npz"),
        "rff_map": str(out / "rff_map.npz"),
    }
    write_json(out / "metadata.json", meta)
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

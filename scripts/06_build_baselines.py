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
from encmonitor.rff import RFFMap


def main() -> int:
    ap = argparse.ArgumentParser(description="Build fixed site-specific reference sketches")
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--preprocessor", required=True)
    ap.add_argument("--rff", required=True)
    ap.add_argument("--max-per-site", type=int, default=0, help="0 uses every reference sample")
    ap.add_argument("--seed", type=int, default=20260818)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    z = np.load(args.embeddings)
    prep = EmbeddingStandardizer.load(args.preprocessor)
    rff = RFFMap.load(args.rff)
    rng = np.random.default_rng(args.seed)
    sketches = {}
    counts = {}
    for key in sorted(k for k in z.files if k.startswith("site_")):
        x = np.asarray(z[key], dtype=np.float64)
        if args.max_per_site > 0 and x.shape[0] > args.max_per_site:
            x = x[rng.choice(x.shape[0], args.max_per_site, replace=False)]
        sketch = rff.mean_sketch(prep.transform(x))
        sketches[key] = sketch.astype(np.float64)
        counts[key] = int(x.shape[0])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **sketches)
    meta = {"source": args.embeddings, "counts": counts, "sketch_dim": rff.output_dim, "npz": str(out)}
    write_json(out.with_suffix(".json"), meta)
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

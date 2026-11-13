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
from encmonitor.data import IndexedShiftDataset, load_dataset
from encmonitor.models import build_model
from encmonitor.shifts import ShiftSpec


def parse_directions(raw: str, n_sites: int) -> list[float]:
    if not raw.strip():
        return [0.0] * n_sites
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if len(vals) != n_sites:
        raise ValueError(f"Expected {n_sites} directions, got {len(vals)}")
    return vals


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract site-local plaintext embeddings")
    ap.add_argument("--dataset", choices=["mnist", "cifar10"], required=True)
    ap.add_argument("--model", choices=["lenet", "resnet18_cifar"], required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--partition", choices=["baseline", "monitor"], required=True)
    ap.add_argument("--root", default="data")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--shift", choices=["none", "rotation", "brightness", "contrast", "blur", "noise"], default="none")
    ap.add_argument("--severity", type=float, default=0.0)
    ap.add_argument("--directions", default="", help="comma-separated per-site signs; 0 leaves a site clean")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=20260818)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader

    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(args.model)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval().to(device)
    dataset = load_dataset(args.dataset, "test", args.root, args.download)
    manifest = np.load(args.manifest)
    site_keys = sorted(k for k in manifest.files if k.startswith(f"{args.partition}_site_"))
    if not site_keys:
        raise RuntimeError(f"No {args.partition}_site_* keys in {args.manifest}")
    directions = parse_directions(args.directions, len(site_keys))

    arrays: dict[str, np.ndarray] = {}
    summary_sites = []
    with torch.no_grad():
        for site, key in enumerate(site_keys):
            direction = directions[site]
            shift = None
            if args.shift != "none" and direction != 0:
                shift = ShiftSpec(args.shift, args.severity, direction)
            view = IndexedShiftDataset(dataset, manifest[key], shift=shift, seed=args.seed + 10_000 * site)
            loader = DataLoader(view, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
            emb, labels, indices = [], [], []
            for x, y, idx in loader:
                feat = model.forward_features(x.to(device)).detach().cpu().numpy()
                emb.append(feat)
                labels.append(y.numpy())
                indices.append(idx.numpy())
            e = np.concatenate(emb, axis=0).astype(np.float32)
            y = np.concatenate(labels).astype(np.int64)
            idx = np.concatenate(indices).astype(np.int64)
            tag = f"site_{site:03d}"
            arrays[tag] = e
            arrays[f"labels_{site:03d}"] = y
            arrays[f"indices_{site:03d}"] = idx
            summary_sites.append({
                "site": site,
                "samples": int(e.shape[0]),
                "embedding_dim": int(e.shape[1]),
                "direction": direction,
                "shift_applied": None if shift is None else shift.__dict__,
            })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    meta = {
        "dataset": args.dataset,
        "model": args.model,
        "checkpoint": args.checkpoint,
        "manifest": args.manifest,
        "partition": args.partition,
        "shift": args.shift,
        "severity": args.severity,
        "directions": directions,
        "sites": summary_sites,
        "npz": str(out),
    }
    write_json(out.with_suffix(".json"), meta)
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

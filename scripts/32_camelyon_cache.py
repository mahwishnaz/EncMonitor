#!/usr/bin/env python3
"""Camelyon17 cache: site pools, embeddings, transform, RFF map, shifted pieces.

Same structure and file names as 14_build_screening_cache.py, so that the window runner
and the analysis script 16 work in the same way. Differences, all forced by the data:
  * a site is a real hospital (site s = hospital s), not a Dirichlet partition;
  * per partition seed, each hospital contributes `pool_per_hospital` held-out patches
    (never used for training), split at random into reference and monitoring halves;
  * image shifts are applied on the GPU to whole batches (stain = haematoxylin intensity).

The standardizer, median bandwidth, RFF map and site baselines use the encmonitor library
functions unchanged (EmbeddingStandardizer, median_bandwidth, RFFMap).

Per seed it writes outputs/camelyon/main/camelyon17/seed_<p>/:
  manifest.npz, baseline_clean.npz, monitor_clean.npz (site_XXX embeddings, labels_XXX,
  indices_XXX = WILDS patch ids), preprocessor.npz, rff_map.npz, baselines.npz,
  global_baseline.npy, pieces.npz, cache_meta.json, cache_done.json

Usage (from the repository root):
  CUDA_VISIBLE_DEVICES=0 python scripts/32_camelyon_cache.py --root data/wilds
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import camelyon_common as cc  # noqa: E402
from encmonitor.artifacts import environment_snapshot, write_json  # noqa: E402
from encmonitor.embeddings import EmbeddingStandardizer  # noqa: E402
from encmonitor.rff import RFFMap, median_bandwidth  # noqa: E402


def needed_factors(cfg, family):
    need: dict[int, set[float]] = {}
    for g in cfg["families"][family]["geometries"]:
        for s, f in enumerate(cfg["geometries"][g]):
            if f != 0:
                need.setdefault(s, set()).add(float(f))
    return need


def piece_key(family, severity, factor, site):
    return f"{family}|{severity:g}|{factor:+g}|site_{site:03d}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the Camelyon17 cache")
    ap.add_argument("--config", default="configs/camelyon.yaml")
    ap.add_argument("--root", default="data/wilds")
    ap.add_argument("--checkpoint", default="outputs/camelyon/model/resnet18_camelyon.pt")
    ap.add_argument("--out-dir", default="outputs/camelyon/main/camelyon17")
    ap.add_argument("--seeds", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    import torch

    cfg = yaml.safe_load(open(ROOT / args.config))
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()] or [int(s) for s in cfg["partition_seeds"]]
    if args.device != "cpu" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available -- stop here. Use --device cpu only on purpose.")
    device = args.device
    S, m = int(cfg["sites"]), int(cfg["rff_dim"])

    ckpt = torch.load(ROOT / args.checkpoint, map_location="cpu", weights_only=False)
    model = cc.build_model(pretrained=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval().to(device)

    ds = cc.load_wilds(args.root)
    md = cc.metadata(ds)
    held = cc.heldout_by_hospital(md)
    train_used = set(np.asarray(ckpt["train_indices"]).tolist())
    for h, idx in held.items():  # safety: no monitoring patch may come from the training subset
        assert not train_used.intersection(idx.tolist()), f"hospital {h}: overlap with training patches"
    image_families = [f for f, spec in cfg["families"].items() if spec["kind"] == "image"]
    P, frac = int(cfg["pool_per_hospital"]), float(cfg["baseline_fraction"])

    for seed in seeds:
        out_dir = ROOT / args.out_dir / f"seed_{seed}"
        if (out_dir / "cache_done.json").exists() and not args.force:
            print(f"[seed {seed}] cache exists, skipping")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()

        man, imgs = {}, {}
        for s in range(S):
            pool = held[s]
            pick = np.random.default_rng([seed, s]).choice(pool, size=min(P, pool.size), replace=False)
            cut = int(round(pick.size * frac))
            man[f"baseline_site_{s:03d}"], man[f"monitor_site_{s:03d}"] = pick[:cut], pick[cut:]
            imgs[s] = cc.load_images(ds, pick, args.workers)
        np.savez_compressed(out_dir / "manifest.npz", **man)

        clean = {"baseline": {}, "monitor": {}}
        mon_imgs = {}
        for s in range(S):
            cut = man[f"baseline_site_{s:03d}"].size
            for part, sl in (("baseline", slice(0, cut)), ("monitor", slice(cut, None))):
                idx = man[f"{part}_site_{s:03d}"]
                clean[part][f"site_{s:03d}"] = cc.embed(model, imgs[s][sl], device)
                clean[part][f"labels_{s:03d}"] = md["y"][idx]
                clean[part][f"indices_{s:03d}"] = idx
            mon_imgs[s] = imgs[s][cut:]
        for part in ("baseline", "monitor"):
            np.savez_compressed(out_dir / f"{part}_clean.npz", **clean[part])

        x = np.concatenate([clean["baseline"][f"site_{s:03d}"].astype(np.float64) for s in range(S)], axis=0)
        prep = EmbeddingStandardizer.fit(x, clip_radius=float(cfg["clip_radius"]))
        sigma0 = median_bandwidth(prep.transform(x), max_samples=2000, seed=int(cfg["rff_seed"]))
        rff = RFFMap.fit(x.shape[1], m, sigma0 * float(cfg["bandwidth_multiplier"]), int(cfg["rff_seed"]))
        prep.save(out_dir / "preprocessor.npz")
        rff.save(str(out_dir / "rff_map.npz"))
        baselines = {f"site_{s:03d}": rff.mean_sketch(prep.transform(clean["baseline"][f"site_{s:03d}"].astype(np.float64)))
                     for s in range(S)}
        np.savez_compressed(out_dir / "baselines.npz", **baselines)
        np.save(out_dir / "global_baseline.npy", rff.mean_sketch(prep.transform(x)))   # global-reference comparison

        pieces, listing = {}, []
        for fam in image_families:
            need = needed_factors(cfg, fam)
            for sev in cfg["families"][fam]["severities"]:
                for s, factors in sorted(need.items()):
                    for f in sorted(factors):
                        key = piece_key(fam, float(sev), f, s)
                        pieces[key] = cc.embed(model, mon_imgs[s], device, shift=cc.make_shift(fam, float(sev), f))
                        listing.append(key)
        np.savez_compressed(out_dir / "pieces.npz", **pieces)

        sizes = {p: [int(man[f"{p}_site_{s:03d}"].size) for s in range(S)] for p in ("baseline", "monitor")}
        meta = {"dataset": "camelyon17", "checkpoint": args.checkpoint, "partition_seed": seed, "sizes": sizes,
                "embedding_dim": int(x.shape[1]), "median_bandwidth": sigma0, "bandwidth": rff.bandwidth,
                "rff_dim": m, "n_pieces": len(listing), "pieces": listing,
                "tumor_fraction_monitor": [float(clean["monitor"][f"labels_{s:03d}"].mean()) for s in range(S)],
                "seconds": round(time.time() - t0, 1), "environment": environment_snapshot()}
        write_json(out_dir / "cache_meta.json", meta)
        write_json(out_dir / "cache_done.json", {"seed": seed, "seconds": meta["seconds"]})
        print(f"[camelyon17 seed {seed}] OK  baseline/monitor per site = {sizes['baseline'][0]}/{sizes['monitor'][0]}  "
              f"bandwidth={sigma0:.4f}  pieces={len(listing)}  {meta['seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

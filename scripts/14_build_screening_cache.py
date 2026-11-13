#!/usr/bin/env python3
"""Screening cache of the main experiment: per partition seed, build everything the window runner needs.

For each partition seed p (configs/main.yaml) and one dataset:
  1. site manifest            -- same code path as 02_build_manifest.py (seed p, p+1, p+2)
  2. clean baseline/monitor embeddings -- same code path as 04_extract_embeddings.py
  3. public transform + RFF map        -- same code path as 05_fit_rff.py
  4. site-specific baseline sketches   -- same code path as 06_build_baselines.py
  5. shifted monitor "pieces" for every image-shift family, severity and per-site
     direction factor needed by the geometries (e.g. mixed needs -0.5 and +0.75).
     04_extract_embeddings.py only uses the SIGN of a direction; mixed needs
     |direction| x severity, which is why pieces are extracted per (site, factor).

The model is the frozen checkpoint (trained once, never retrained). Scripts 02-06 are not
modified; this file re-uses their library functions in one process.

--verify-prefix checks, for partition seed 2026 (the seed of the artifacts in outputs/shared),
that this script reproduces the outputs of scripts 02-06 exactly.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from encmonitor.artifacts import environment_snapshot, write_json
from encmonitor.data import IndexedShiftDataset, labels_array, load_dataset
from encmonitor.embeddings import EmbeddingStandardizer
from encmonitor.models import build_model
from encmonitor.rff import RFFMap, median_bandwidth
from encmonitor.shifts import ShiftSpec
from encmonitor.site_partition import allocate_with_class_to_site, sample_class_to_site_probabilities


def build_manifest(y_train, y_test, n_sites, alpha, seed, baseline_fraction):
    """Identical logic to scripts/02_build_manifest.py."""
    n_classes = int(max(y_train.max(), y_test.max()) + 1)
    p = sample_class_to_site_probabilities(n_classes, n_sites, alpha, seed)
    part_train = allocate_with_class_to_site(y_train, p, seed + 1, min_per_site=10)
    part_test = allocate_with_class_to_site(y_test, p, seed + 2, min_per_site=10)
    baseline_sites, monitor_sites = [], []
    for arr in part_test.site_indices:
        cut = max(1, min(arr.size - 1, int(round(arr.size * baseline_fraction))))
        baseline_sites.append(arr[:cut])
        monitor_sites.append(arr[cut:])
    arrays = dict(
        class_to_site=p,
        train_class_proportions=part_train.class_proportions,
        test_class_proportions=part_test.class_proportions,
        **{f"train_site_{s:03d}": v for s, v in enumerate(part_train.site_indices)},
        **{f"test_site_{s:03d}": v for s, v in enumerate(part_test.site_indices)},
        **{f"baseline_site_{s:03d}": v for s, v in enumerate(baseline_sites)},
        **{f"monitor_site_{s:03d}": v for s, v in enumerate(monitor_sites)},
    )
    return arrays


class Extractor:
    """Identical per-site extraction to scripts/04_extract_embeddings.py."""

    def __init__(self, dataset, model, device, batch_size, workers, extract_seed):
        self.dataset, self.model, self.device = dataset, model, device
        self.batch_size, self.workers, self.extract_seed = batch_size, workers, extract_seed

    def site(self, indices, site, shift):
        import torch
        from torch.utils.data import DataLoader

        view = IndexedShiftDataset(self.dataset, indices, shift=shift, seed=self.extract_seed + 10_000 * site)
        loader = DataLoader(view, batch_size=self.batch_size, shuffle=False, num_workers=self.workers)
        emb, labels, idxs = [], [], []
        with torch.no_grad():
            for x, y, idx in loader:
                emb.append(self.model.forward_features(x.to(self.device)).detach().cpu().numpy())
                labels.append(y.numpy())
                idxs.append(idx.numpy())
        return (np.concatenate(emb).astype(np.float32), np.concatenate(labels).astype(np.int64),
                np.concatenate(idxs).astype(np.int64))


def needed_factors(cfg, family):
    """Per-site direction factors used by this family's geometries -> {site: set(factors)}."""
    need: dict[int, set[float]] = {}
    for g in cfg["families"][family]["geometries"]:
        for s, f in enumerate(cfg["geometries"][g]):
            if f != 0:
                need.setdefault(s, set()).add(float(f))
    return need


def piece_key(family, severity, factor, site):
    return f"{family}|{severity:g}|{factor:+g}|site_{site:03d}"


def verify(out: dict, shared: Path, prefix: str, m: int) -> dict:
    res = {}
    man = np.load(shared / f"manifests/{prefix}_s5_a0.5.npz")
    res["manifest_identical"] = all(np.array_equal(man[k], out["manifest"][k]) for k in man.files)
    for part in ("baseline", "monitor"):
        ref = np.load(shared / f"embeddings/{prefix}_{part}_clean.npz")
        cur = out[f"{part}_clean"]
        res[f"{part}_embeddings_max_abs_diff"] = float(max(np.abs(ref[k].astype(np.float64) - cur[k]).max()
                                                        for k in ref.files if k.startswith("site_")))
        res[f"{part}_indices_identical"] = all(np.array_equal(ref[k], cur[k]) for k in ref.files if k.startswith("indices_"))
    rff_ref = RFFMap.load(str(shared / f"transforms/{prefix}_m{m}/rff_map.npz"))
    res["rff_bandwidth_ref"] = rff_ref.bandwidth
    res["rff_bandwidth_new"] = out["rff"].bandwidth
    res["rff_omega_max_abs_diff"] = float(np.abs(rff_ref.omega - out["rff"].omega).max())
    base_ref = np.load(shared / f"baselines/{prefix}_m{m}.npz")
    res["baseline_sketch_max_abs_diff"] = float(max(np.abs(base_ref[k] - out["baselines"][k]).max() for k in base_ref.files))
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the screening cache (manifests, embeddings, transforms, shift pieces)")
    ap.add_argument("--config", default="configs/main.yaml")
    ap.add_argument("--dataset", choices=["mnist", "cifar10"], required=True)
    ap.add_argument("--model", choices=["lenet", "resnet18_cifar"], required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--root", default="data")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--seeds", default="", help="comma-separated subset of partition seeds (default: all in config)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out-dir", required=True, help="e.g. outputs/main/mnist")
    ap.add_argument("--verify-prefix", default="", help="mnist or cifar10: compare seed 2026 with outputs/shared")
    ap.add_argument("--shared-dir", default="outputs/shared")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    import torch

    cfg = yaml.safe_load(open(args.config))
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()] or [int(s) for s in cfg["partition_seeds"]]
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(args.model)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval().to(device)
    train = load_dataset(args.dataset, "train", args.root, args.download)
    test = load_dataset(args.dataset, "test", args.root, args.download)
    y_train, y_test = labels_array(train), labels_array(test)
    ex = Extractor(test, model, device, args.batch_size, args.workers, int(cfg["extract_seed"]))
    S, m = int(cfg["sites"]), int(cfg["rff_dim"])
    image_families = [f for f, spec in cfg["families"].items() if spec["kind"] == "image"]

    for seed in seeds:
        out_dir = Path(args.out_dir) / f"seed_{seed}"
        done = out_dir / "cache_done.json"
        if done.exists() and not args.force:
            print(f"[seed {seed}] cache exists, skipping")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()

        man = build_manifest(y_train, y_test, S, float(cfg["dirichlet_alpha"]), seed, float(cfg["baseline_fraction"]))
        np.savez_compressed(out_dir / "manifest.npz", **man)
        sizes = {p: [int(man[f"{p}_site_{s:03d}"].size) for s in range(S)] for p in ("baseline", "monitor")}

        clean = {}
        for part in ("baseline", "monitor"):
            arrays = {}
            for s in range(S):
                e, y, idx = ex.site(man[f"{part}_site_{s:03d}"], s, None)
                arrays[f"site_{s:03d}"], arrays[f"labels_{s:03d}"], arrays[f"indices_{s:03d}"] = e, y, idx
            np.savez_compressed(out_dir / f"{part}_clean.npz", **arrays)
            clean[part] = arrays

        x = np.concatenate([clean["baseline"][f"site_{s:03d}"].astype(np.float64) for s in range(S)], axis=0)
        prep = EmbeddingStandardizer.fit(x, clip_radius=float(cfg["clip_radius"]))
        sigma0 = median_bandwidth(prep.transform(x), max_samples=2000, seed=int(cfg["rff_seed"]))
        rff = RFFMap.fit(x.shape[1], m, sigma0 * float(cfg["bandwidth_multiplier"]), int(cfg["rff_seed"]))
        prep.save(out_dir / "preprocessor.npz")
        rff.save(str(out_dir / "rff_map.npz"))
        baselines = {f"site_{s:03d}": rff.mean_sketch(prep.transform(clean["baseline"][f"site_{s:03d}"].astype(np.float64)))
                     for s in range(S)}
        np.savez_compressed(out_dir / "baselines.npz", **baselines)

        pieces, listing = {}, []
        for fam in image_families:
            need = needed_factors(cfg, fam)
            for sev in cfg["families"][fam]["severities"]:
                for s, factors in sorted(need.items()):
                    for f in sorted(factors):
                        spec = ShiftSpec(fam, float(sev) * abs(f), 1.0 if f > 0 else -1.0)
                        e, _, _ = ex.site(man[f"monitor_site_{s:03d}"], s, spec)
                        key = piece_key(fam, float(sev), f, s)
                        pieces[key] = e
                        listing.append(key)
        np.savez_compressed(out_dir / "pieces.npz", **pieces)

        meta = {
            "dataset": args.dataset, "model": args.model, "checkpoint": args.checkpoint, "partition_seed": seed,
            "sizes": sizes, "embedding_dim": int(x.shape[1]), "median_bandwidth": sigma0,
            "bandwidth": rff.bandwidth, "rff_dim": m, "n_pieces": len(listing), "pieces": listing,
            "seconds": round(time.time() - t0, 1), "environment": environment_snapshot(),
        }
        if args.verify_prefix and seed == 2026:
            meta["verification_vs_scripts_02_06"] = verify(
                {"manifest": man, "baseline_clean": clean["baseline"], "monitor_clean": clean["monitor"],
                 "rff": rff, "baselines": baselines}, Path(args.shared_dir), args.verify_prefix, m)
        write_json(out_dir / "cache_meta.json", meta)
        write_json(done, {"seed": seed, "seconds": meta["seconds"]})
        line = f"[{args.dataset} seed {seed}] OK  min baseline/monitor site = {min(sizes['baseline'])}/{min(sizes['monitor'])}  " \
               f"bandwidth={sigma0:.4f}  pieces={len(listing)}  {meta['seconds']}s"
        print(line)
        if "verification_vs_scripts_02_06" in meta:
            print("   verification vs scripts 02-06:", json.dumps(meta["verification_vs_scripts_02_06"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Ablation caches: embeddings for other site counts (S=10) and other layers (early, logits).

Re-uses the cache code of 14_build_screening_cache.py (same manifest, extraction and
piece logic) and only changes (a) the number of sites and (b) which layer is read out.
Only the shift pieces needed by the representative ablation conditions are extracted.
The S=5 / penultimate cache is not rebuilt: the ablation re-uses the main-experiment cache.

Layers:
  penultimate : model.forward_features (LeNet 128-D, ResNet-18 512-D)  -- main representation
  early       : spatially averaged conv features (LeNet output of `features`, 64-D; ResNet-18 layer3, 256-D)
  logits      : final classifier output (10-D)

--verify-shared: for S=10 and seed 2026, the manifest is compared with the 10-site manifest built
by 02_build_manifest.py (outputs/shared/manifests/<dataset>_s10_a0.5.npz).
"""
from __future__ import annotations

import argparse
import importlib.util
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
from encmonitor.data import labels_array, load_dataset
from encmonitor.models import build_model
from encmonitor.shifts import ShiftSpec


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


g2 = _load("14_build_screening_cache")


class LayerView:
    """Expose one layer of the frozen model through forward_features (what the Extractor calls)."""

    def __init__(self, model, model_name, layer):
        self.model, self.model_name, self.layer = model, model_name, layer

    def forward_features(self, x):
        import torch

        m = self.model
        if self.layer == "penultimate":
            return m.forward_features(x)
        if self.layer == "logits":
            return m(x)
        if self.layer == "early":
            if self.model_name == "lenet":
                h = m.features(m.normalize(x))
            else:
                b = m.backbone
                h = b.layer3(b.layer2(b.layer1(b.maxpool(b.relu(b.bn1(b.conv1(m.normalize(x))))))))
            return torch.flatten(torch.nn.functional.adaptive_avg_pool2d(h, 1), 1)
        raise ValueError(self.layer)


def parse_condition(name):
    fam_sev, geo = name.split("__")
    i = next(k for k, ch in enumerate(fam_sev) if ch.isdigit())
    return fam_sev[:i], float(fam_sev[i:]), geo


def main() -> int:
    ap = argparse.ArgumentParser(description="Build ablation caches (S=10, early layer, logits)")
    ap.add_argument("--config", default="configs/ablation.yaml")
    ap.add_argument("--main-config", default="configs/main.yaml", help="for geometry and family definitions")
    ap.add_argument("--dataset", choices=["mnist", "cifar10"], required=True)
    ap.add_argument("--model", choices=["lenet", "resnet18_cifar"], required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--root", default="data")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--caches", default="S10_penultimate,S5_early,S5_logits")
    ap.add_argument("--seeds", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--shared-dir", default="outputs/shared")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    import torch

    cfg = yaml.safe_load(open(args.config))
    main_cfg = yaml.safe_load(open(args.main_config))
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()] or [int(s) for s in cfg["partition_seeds"]]
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(args.model)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval().to(device)
    train = load_dataset(args.dataset, "train", args.root, args.download)
    test = load_dataset(args.dataset, "test", args.root, args.download)
    y_train, y_test = labels_array(train), labels_array(test)

    image_conds = []
    for name in cfg["conditions"][args.dataset]:
        fam, sev, geo = parse_condition(name)
        if main_cfg["families"][fam]["kind"] == "image":
            image_conds.append((fam, sev, geo))

    for cache_name in [c for c in args.caches.split(",") if c.strip()]:
        spec = cfg["caches"][cache_name]
        S, layer = int(spec["sites"]), spec["layer"]
        base_dir = Path(spec["dir"].format(dataset=args.dataset))
        ex = g2.Extractor(test, LayerView(model, args.model, layer), device, args.batch_size, args.workers,
                          int(cfg["extract_seed"]))
        for seed in seeds:
            out_dir = base_dir / f"seed_{seed}"
            done = out_dir / "cache_done.json"
            if done.exists() and not args.force:
                print(f"[{args.dataset} {cache_name} seed {seed}] cache exists, skipping")
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            man = g2.build_manifest(y_train, y_test, S, float(cfg["dirichlet_alpha"]), seed, float(cfg["baseline_fraction"]))
            np.savez_compressed(out_dir / "manifest.npz", **man)
            sizes = {p: [int(man[f"{p}_site_{s:03d}"].size) for s in range(S)] for p in ("baseline", "monitor")}
            for part in ("baseline", "monitor"):
                arrays = {}
                for s in range(S):
                    e, y, idx = ex.site(man[f"{part}_site_{s:03d}"], s, None)
                    arrays[f"site_{s:03d}"], arrays[f"labels_{s:03d}"], arrays[f"indices_{s:03d}"] = e, y, idx
                np.savez_compressed(out_dir / f"{part}_clean.npz", **arrays)
            pieces, listing = {}, []
            for fam, sev, geo in image_conds:
                dirs = list(main_cfg["geometries"][geo]) + [0] * (S - len(main_cfg["geometries"][geo]))
                for s, f in enumerate(dirs):
                    if f == 0:
                        continue
                    key = g2.piece_key(fam, sev, float(f), s)
                    if key in pieces:
                        continue
                    e, _, _ = ex.site(man[f"monitor_site_{s:03d}"], s, ShiftSpec(fam, sev * abs(f), 1.0 if f > 0 else -1.0))
                    pieces[key] = e
                    listing.append(key)
            np.savez_compressed(out_dir / "pieces.npz", **pieces)
            meta = {"dataset": args.dataset, "cache": cache_name, "sites": S, "layer": layer, "partition_seed": seed,
                    "sizes": sizes, "embedding_dim": int(ex.site(man["monitor_site_000"][:2], 0, None)[0].shape[1]),
                    "pieces": listing, "seconds": round(time.time() - t0, 1), "environment": environment_snapshot()}
            if S == 10 and seed == 2026:
                ref = Path(args.shared_dir) / f"manifests/{args.dataset}_s10_a0.5.npz"
                if ref.exists():
                    r = np.load(ref)
                    meta["manifest_identical_to_script_02"] = all(np.array_equal(r[k], man[k]) for k in r.files)
            write_json(out_dir / "cache_meta.json", meta)
            write_json(done, {"seed": seed, "seconds": meta["seconds"]})
            extra = f"  manifest identical to script 02: {meta['manifest_identical_to_script_02']}" \
                if "manifest_identical_to_script_02" in meta else ""
            print(f"[{args.dataset} {cache_name} seed {seed}] OK  dim={meta['embedding_dim']}  "
                  f"min baseline/monitor site = {min(sizes['baseline'])}/{min(sizes['monitor'])}  "
                  f"pieces={len(listing)}  {meta['seconds']}s{extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

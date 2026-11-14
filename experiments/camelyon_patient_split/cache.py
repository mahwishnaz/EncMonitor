#!/usr/bin/env python3
"""Copy of experiments/camelyon_slide_split/cache.py with a patient-level reference/monitoring split.

Per hospital and partition seed, whole patients (WILDS metadata field `patient`) are assigned to the reference
or the monitoring pool, so that no patient has patches in both pools (asserted). The slide counts of the slide
split are kept: the reference pool holds exactly half of the hospital's slides (5 of 10). The assignment is drawn
uniformly among all patient subsets whose slides sum to that number (rng [seed, s, 31]); up to 2,500 patches are
then drawn from each pool (rng [seed, s, 32]), as in the slide split. Everything else is unchanged.

Per seed it also writes patient_table.csv (hospital, pool, patient, slides, held-out patches of the patient,
patches drawn) and the number of possible assignments per hospital in cache_meta.json.

Usage (repository root):
  CUDA_VISIBLE_DEVICES=0 python experiments/camelyon_patient_split/cache.py --root data/wilds
then the windows with scripts/33_camelyon_windows.py --cache-dir outputs/camelyon/patient_split/camelyon17
"""
from __future__ import annotations

import argparse
import csv
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
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
    ap.add_argument("--out-dir", default="outputs/camelyon/patient_split/camelyon17")
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
    patient = ds._metadata_df["patient"].to_numpy().astype(np.int64)   # WILDS metadata, row order = patch ids
    assert (ds._metadata_df["slide"].to_numpy() == md["slide"]).all()
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

        man, imgs, slides, table = {}, {}, {}, []
        half = int(round(P * frac))
        for s in range(S):
            # patient-level split: whole patients, half of the hospital's slides in each pool
            pool = held[s]
            pat_of = patient[pool]
            pats = np.unique(pat_of)
            n_sl = {int(p): int(np.unique(md["slide"][pool][pat_of == p]).size) for p in pats}
            total = sum(n_sl.values())
            options = [c for r in range(1, pats.size) for c in itertools.combinations(pats.tolist(), r)
                       if sum(n_sl[p] for p in c) == total // 2]
            assert options, f"hospital {s}: no patient subset with {total // 2} slides"
            ref_p = np.array(sorted(options[int(np.random.default_rng([seed, s, 31]).integers(len(options)))]))
            mon_p = np.setdiff1d(pats, ref_p)
            assert not set(ref_p.tolist()) & set(mon_p.tolist()), f"hospital {s}: patient in both pools"
            ref_sl = np.unique(md["slide"][pool][np.isin(pat_of, ref_p)])
            mon_sl = np.unique(md["slide"][pool][np.isin(pat_of, mon_p)])
            rng = np.random.default_rng([seed, s, 32])
            parts = []
            for chosen in (ref_sl, mon_sl):
                cand = pool[np.isin(md["slide"][pool], chosen)]
                parts.append(rng.choice(cand, size=min(half, cand.size), replace=False))
            assert not set(patient[parts[0]].tolist()) & set(patient[parts[1]].tolist()), \
                f"hospital {s}: a patient has patches in both pools"
            pick = np.concatenate(parts)
            cut = parts[0].size
            man[f"baseline_site_{s:03d}"], man[f"monitor_site_{s:03d}"] = parts[0], parts[1]
            slides[s] = {"reference_patients": ref_p.tolist(), "monitoring_patients": mon_p.tolist(),
                         "reference_slides": ref_sl.tolist(), "monitoring_slides": mon_sl.tolist(),
                         "possible_assignments": len(options),
                         "tumor_reference": float(md["y"][parts[0]].mean()),
                         "tumor_monitoring": float(md["y"][parts[1]].mean())}
            for pool_name, pp, part in (("reference", ref_p, parts[0]), ("monitoring", mon_p, parts[1])):
                for p in pp.tolist():
                    table.append({"partition_seed": seed, "hospital": s, "pool": pool_name, "patient": int(p),
                                  "slides": n_sl[int(p)], "heldout_patches": int((pat_of == p).sum()),
                                  "patches_drawn": int((patient[part] == p).sum())})
            imgs[s] = cc.load_images(ds, pick, args.workers)
        np.savez_compressed(out_dir / "manifest.npz", **man)
        with open(out_dir / "patient_table.csv", "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(table[0]))
            wr.writeheader()
            wr.writerows(table)

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
                "patient_split": {str(k): v for k, v in slides.items()},
                "tumor_fraction_monitor": [float(clean["monitor"][f"labels_{s:03d}"].mean()) for s in range(S)],
                "seconds": round(time.time() - t0, 1), "environment": environment_snapshot()}
        write_json(out_dir / "cache_meta.json", meta)
        write_json(out_dir / "cache_done.json", {"seed": seed, "seconds": meta["seconds"]})
        print(f"[camelyon17 seed {seed}] OK  baseline/monitor per site = {sizes['baseline'][0]}/{sizes['monitor'][0]}  "
              f"bandwidth={sigma0:.4f}  pieces={len(listing)}  {meta['seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

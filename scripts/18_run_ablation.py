#!/usr/bin/env python3
"""Ablation runner.

For every setting generated from configs/ablation.yaml (single-factor axes and the S x m x n core
grid) and every partition seed, this script
  * refits the public preprocessing + RFF map for the setting's m and bandwidth multiplier on the
    baseline embeddings (same code path as 05_fit_rff.py),
  * builds site-specific baselines (06_build_baselines.py) or, for the reference ablation, one global
    baseline shared by all sites,
  * runs 500 clean calibration, 200 held-out clean and 50 windows per representative condition with
    exactly the window logic and seeds of the main experiment (15_run_screening.py).
The base setting (S=5, penultimate, m=64, n=250, bandwidth x1.0, site baselines, uniform weights) is
identical to the main experiment; the script checks that it reproduces the saved records and prints the
maximum difference.
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from encmonitor.artifacts import write_json
from encmonitor.embeddings import EmbeddingStandardizer
from encmonitor.rff import RFFMap, median_bandwidth


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


main_run = _load("15_run_screening")
KEYS = ["sites", "layer", "rff_dim", "window_size", "bandwidth_multiplier", "baseline", "weights"]


def setting_id(s):
    return (f"S{s['sites']}_{s['layer']}_m{s['rff_dim']}_n{s['window_size']}_bw{float(s['bandwidth_multiplier']):g}"
            f"_{s['baseline']}_{s['weights']}")


def settings_from(cfg):
    base = dict(cfg["base"])
    out = {}
    for axis, spec in cfg["axes"].items():
        keys = list(spec)
        for vals in itertools.product(*[spec[k] for k in keys]):
            s = dict(base)
            s.update(dict(zip(keys, vals)))
            sid = setting_id(s)
            if sid not in out:
                out[sid] = {**s, "axes": [axis]}
            elif axis not in out[sid]["axes"]:
                out[sid]["axes"].append(axis)
    return base, out


def cache_dir(cfg, dataset, sites, layer):
    for spec in cfg["caches"].values():
        if int(spec["sites"]) == int(sites) and spec["layer"] == layer:
            return Path(spec["dir"].format(dataset=dataset))
    raise KeyError((sites, layer))


def parse_condition(name):
    fam_sev, geo = name.split("__")
    i = next(k for k, ch in enumerate(fam_sev) if ch.isdigit())
    return fam_sev[:i], float(fam_sev[i:]), geo


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the ablation windows for one dataset")
    ap.add_argument("--config", default="configs/ablation.yaml")
    ap.add_argument("--main-config", default="configs/main.yaml")
    ap.add_argument("--dataset", choices=["mnist", "cifar10"], required=True)
    ap.add_argument("--out-dir", default="outputs/ablation/runs")
    ap.add_argument("--seeds", default="")
    ap.add_argument("--only", default="", help="comma-separated setting ids")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    main_cfg = yaml.safe_load(open(args.main_config))
    try:
        import torch
        device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()] or [int(s) for s in cfg["partition_seeds"]]
    base, settings = settings_from(cfg)
    if args.only:
        settings = {k: v for k, v in settings.items() if k in set(args.only.split(","))}
    exact_for = cfg.get("exact_mmd_for", {})
    conds = []
    for name in cfg["conditions"][args.dataset]:
        fam, sev, geo = parse_condition(name)
        conds.append({"condition": name, "family": fam, "kind": main_cfg["families"][fam]["kind"], "severity": sev,
                      "geometry": geo, "directions5": [float(v) for v in main_cfg["geometries"][geo]]})

    out_root = Path(args.out_dir) / args.dataset
    out_root.mkdir(parents=True, exist_ok=True)
    loaded = {}      # (sites, layer, seed) -> embeddings, labels, pieces, standardizer, sigma0
    total_t0 = time.time()
    for sid, st in settings.items():
        out = out_root / f"{sid}.jsonl"
        if out.exists() and not args.force:
            print(f"[{args.dataset} {sid}] exists, skipping")
            continue
        t0 = time.time()
        S, n, m = int(st["sites"]), int(st["window_size"]), int(st["rff_dim"])
        use_exact = all(st.get(k) == v for k, v in exact_for.items()) if exact_for else False
        rows_all, replaced_sites = [], 0
        for seed in seeds:
            key = (S, st["layer"], seed)
            if key not in loaded:
                d = cache_dir(cfg, args.dataset, S, st["layer"]) / f"seed_{seed}"
                bc, mc, pc = np.load(d / "baseline_clean.npz"), np.load(d / "monitor_clean.npz"), np.load(d / "pieces.npz")
                bx = [bc[f"site_{s:03d}"].astype(np.float64) for s in range(S)]
                prep = EmbeddingStandardizer.fit(np.concatenate(bx), clip_radius=float(cfg["clip_radius"]))
                sigma0 = median_bandwidth(prep.transform(np.concatenate(bx)), max_samples=2000, seed=int(cfg["rff_seed"]))
                loaded = {key: {"bx": bx, "mx": [mc[f"site_{s:03d}"].astype(np.float64) for s in range(S)],
                                "labels": [mc[f"labels_{s:03d}"] for s in range(S)],
                                "pieces": {k: pc[k] for k in pc.files}, "prep": prep, "sigma0": sigma0}}
            L = loaded[key]
            prep = L["prep"]
            rff = RFFMap.fit(L["bx"][0].shape[1], m, L["sigma0"] * float(st["bandwidth_multiplier"]), int(cfg["rff_seed"]))
            bz = [prep.transform(x) for x in L["bx"]]
            if st["baseline"] == "site":
                base_sk = [rff.mean_sketch(z) for z in bz]
            else:
                g = rff.mean_sketch(np.concatenate(bz))
                base_sk = [g for _ in range(S)]
            sizes = np.array([x.shape[0] for x in L["mx"]], dtype=float)
            weights = np.ones(S) / S if st["weights"] == "uniform" else sizes / sizes.sum()
            replaced_sites += int((sizes < n).sum())
            mmd = main_run.ExactMMD(rff.bandwidth, device) if use_exact else None

            def fz(raw):
                z = prep.transform(raw)
                return rff.transform(z), z

            clean = [fz(x) for x in L["mx"]]
            rows = []
            meta_common = {"setting": sid, **{k: st[k] for k in KEYS}}
            for cname, nwin, wseed in (("clean_calibration", int(cfg["calibration_windows"]), seed),
                                       ("clean_heldout", int(cfg["heldout_windows"]), seed + 1000)):
                main_run.run_windows([f for f, _ in clean], [z for _, z in clean], base_sk, bz, weights, n, nwin, wseed, mmd,
                                  {"condition": cname, "family": "clean", "severity": 0.0, "geometry": "clean"}, rows)
            labels = L["labels"]
            classes = np.unique(np.concatenate(labels))
            target = int(classes[np.argmax([min(np.sum(labels[s] == c) for s in range(3)) for c in classes])])
            v = np.random.default_rng([seed, 7]).standard_normal(L["mx"][0].shape[1])
            v /= np.linalg.norm(v)
            for c in conds:
                dirs = c["directions5"] + [0.0] * (S - len(c["directions5"]))
                feats, stds = [], []
                for s in range(S):
                    f = dirs[s]
                    if f == 0:
                        pair = clean[s]
                    elif c["kind"] == "image":
                        pair = fz(L["pieces"][f"{c['family']}|{c['severity']:g}|{f:+g}|site_{s:03d}"].astype(np.float64))
                    elif c["kind"] == "label_prior":
                        q = c["severity"] * abs(f)
                        rng = np.random.default_rng([seed, 3, s, int(round(q * 1000)), int(f > 0)])
                        raw, _ = main_run.label_prior_pool(L["mx"][s], labels[s], target, q, np.sign(f), rng)
                        pair = fz(raw)
                    else:
                        pair = fz(L["mx"][s] + f * c["severity"] * v * prep.scale)
                    feats.append(pair[0]); stds.append(pair[1])
                meta = {k: c[k] for k in ("condition", "family", "severity", "geometry")}
                meta["directions"] = dirs
                main_run.run_windows(feats, stds, base_sk, bz, weights, n, int(cfg["windows_per_condition"]), seed + 2000, mmd, meta, rows)
            for r in rows:
                r.update(meta_common)
                r["partition_seed"] = seed
            rows_all.extend(rows)
        tmp = out.with_suffix(".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            for r in rows_all:
                fh.write(json.dumps(r) + "\n")
        tmp.replace(out)
        note = f"  site pools smaller than n (sampled with replacement, summed over seeds): {replaced_sites}" if replaced_sites else ""
        print(f"[{args.dataset} {sid}] OK  {len(rows_all)} windows  {round(time.time() - t0, 1)}s"
              f"{'  +exact MMD' if use_exact else ''}{note}")

        if sid == setting_id(base):
            diffs = []
            for seed in seeds:
                main_file = cache_dir(cfg, args.dataset, 5, "penultimate") / f"seed_{seed}" / "records.jsonl"
                if not main_file.exists():
                    continue
                ref = {(r["condition"], r["window"]): r for r in map(json.loads, open(main_file))
                       if r["condition"] in {c["condition"] for c in conds} | {"clean_calibration", "clean_heldout"}}
                for r in rows_all:
                    if r["partition_seed"] == seed and (r["condition"], r["window"]) in ref:
                        q = ref[(r["condition"], r["window"])]
                        diffs.append(max(abs(r["d_mix"] - q["d_mix"]), abs(r["d_site"] - q["d_site"])))
            if diffs:
                print(f"   base setting vs main-experiment records: {len(diffs)} windows compared, max abs diff = {max(diffs):.3e}")
    write_json(out_root / "run_meta.json", {"dataset": args.dataset, "settings": settings, "seeds": seeds,
                                            "seconds": round(time.time() - total_t0, 1)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

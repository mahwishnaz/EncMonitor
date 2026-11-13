#!/usr/bin/env python3
"""Export selected residual windows of the main experiment for the CKKS replay.

The saved records contain window scores but not the residual vectors, so this script re-derives the
residual matrices of selected windows from the cache with exactly the window logic and seeds of
15_run_screening.py, and CHECKS every exported window against the saved record (D_mix and D_site must
agree to floating-point precision). Windows are never regenerated independently. Thresholds are
calibrated per seed from that seed's clean-calibration records with encmonitor.thresholds.calibrate_thresholds.

Output per dataset: outputs/ckks/replay/<dataset>_replay.npz with
  residuals [N,S,m], weights [S], shifted_mask [N,S], seed [N], window [N], condition [N],
  tau_<q> [N,3] (tau_mix, tau_site, tau_H) for every configured quantile.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from encmonitor.artifacts import write_json
from encmonitor.embeddings import EmbeddingStandardizer
from encmonitor.rff import RFFMap
from encmonitor.statistics import compute_dual_statistics
from encmonitor.thresholds import calibrate_thresholds


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


main_run = _load("15_run_screening")


def parse_condition(name):
    fam_sev, geo = name.split("__")
    i = next(k for k, ch in enumerate(fam_sev) if ch.isdigit())
    return fam_sev[:i], float(fam_sev[i:]), geo


def windows(feats, base, n, count, seed):
    """Exactly the window draw of 15_run_screening.py; returns residual matrices [count,S,m]."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(count):
        rows = []
        for s in range(len(feats)):
            N = feats[s].shape[0]
            idx = rng.choice(N, n, replace=N < n)
            rows.append(feats[s][idx].mean(axis=0) - base[s])
        out.append(np.stack(rows))
    return np.stack(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="Export selected main-experiment residual windows for CKKS replay")
    ap.add_argument("--config", default="configs/ckks.yaml")
    ap.add_argument("--main-config", default="configs/main.yaml")
    ap.add_argument("--dataset", choices=["mnist", "cifar10"], required=True)
    ap.add_argument("--main-dir", default="outputs/main")
    ap.add_argument("--out-dir", default="outputs/ckks/replay")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))["replay"]
    main_cfg = yaml.safe_load(open(args.main_config))
    n, S = int(main_cfg["window_size"]), int(main_cfg["sites"])
    weights = np.ones(S) / S
    conds = []
    for name in cfg["conditions"][args.dataset]:
        fam, sev, geo = parse_condition(name)
        conds.append({"condition": name, "family": fam, "kind": main_cfg["families"][fam]["kind"], "severity": sev,
                      "directions": [float(v) for v in main_cfg["geometries"][geo]]})

    R, masks, seeds_col, win_col, cond_col = [], [], [], [], []
    taus = {q: [] for q in cfg["threshold_quantiles"]}
    max_diff = 0.0
    for seed in cfg["partition_seeds"]:
        d = Path(args.main_dir) / args.dataset / f"seed_{seed}"
        prep = EmbeddingStandardizer.load(d / "preprocessor.npz")
        rff = RFFMap.load(str(d / "rff_map.npz"))
        bz = np.load(d / "baselines.npz")
        base = [np.asarray(bz[f"site_{s:03d}"], dtype=np.float64) for s in range(S)]
        mc, pc = np.load(d / "monitor_clean.npz"), np.load(d / "pieces.npz")
        mx = [mc[f"site_{s:03d}"].astype(np.float64) for s in range(S)]
        labels = [mc[f"labels_{s:03d}"] for s in range(S)]
        records = [json.loads(l) for l in open(d / "records.jsonl")]
        ref = {(r["condition"], r["window"]): r for r in records}
        th = {q: calibrate_thresholds([r for r in records if r["condition"] == "clean_calibration"], q)
              for q in cfg["threshold_quantiles"]}

        feat = lambda raw: rff.transform(prep.transform(raw))
        clean = [feat(x) for x in mx]
        classes = np.unique(np.concatenate(labels))
        target = int(classes[np.argmax([min(np.sum(labels[s] == c) for s in range(3)) for c in classes])])
        v = np.random.default_rng([seed, 7]).standard_normal(mx[0].shape[1])
        v /= np.linalg.norm(v)

        jobs = [("clean_heldout", clean, [0.0] * S, seed + 1000, int(cfg["clean_windows_per_seed"]))]
        for c in conds:
            feats = []
            for s in range(S):
                f = c["directions"][s]
                if f == 0:
                    feats.append(clean[s])
                elif c["kind"] == "image":
                    feats.append(feat(pc[f"{c['family']}|{c['severity']:g}|{f:+g}|site_{s:03d}"].astype(np.float64)))
                elif c["kind"] == "label_prior":
                    q = c["severity"] * abs(f)
                    rng = np.random.default_rng([seed, 3, s, int(round(q * 1000)), int(f > 0)])
                    raw, _ = main_run.label_prior_pool(mx[s], labels[s], target, q, np.sign(f), rng)
                    feats.append(feat(raw))
                else:
                    feats.append(feat(mx[s] + f * c["severity"] * v * prep.scale))
            jobs.append((c["condition"], feats, c["directions"], seed + 2000, int(cfg["windows_per_condition_per_seed"])))

        for cname, feats, dirs, wseed, count in jobs:
            res = windows(feats, base, n, count, wseed)
            for w in range(count):
                st = compute_dual_statistics(res[w], weights)
                r = ref[(cname, w)]
                max_diff = max(max_diff, abs(st.d_mix - r["d_mix"]), abs(st.d_site - r["d_site"]))
                R.append(res[w]); masks.append(np.asarray(dirs) != 0); seeds_col.append(seed)
                win_col.append(w); cond_col.append(cname)
                for q in taus:
                    taus[q].append([th[q].d_mix, th[q].d_site, th[q].heterogeneity])

    if max_diff > 1e-12:
        raise RuntimeError(f"Exported windows do not reproduce the main-experiment records (max diff {max_diff:.3e})")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    arrays = {"residuals": np.stack(R), "weights": weights, "shifted_mask": np.stack(masks),
              "seed": np.asarray(seeds_col), "window": np.asarray(win_col), "condition": np.asarray(cond_col)}
    for q, t in taus.items():
        arrays[f"tau_{q:g}"] = np.asarray(t)
    np.savez_compressed(out / f"{args.dataset}_replay.npz", **arrays)
    summary = {"dataset": args.dataset, "windows": len(R), "sites": S, "rff_dim": int(arrays["residuals"].shape[2]),
               "per_condition": {c: cond_col.count(c) for c in dict.fromkeys(cond_col)},
               "max_diff_vs_g2_records": max_diff, "quantiles": list(taus)}
    write_json(out / f"{args.dataset}_replay.json", summary)
    print(f"[{args.dataset}] exported {len(R)} windows  {summary['per_condition']}  "
          f"max diff vs main-experiment records = {max_diff:.1e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

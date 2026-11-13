#!/usr/bin/env python3
"""Export real Camelyon17 residual windows for the CKKS replay (as 20_export_replay.py).

Re-derives the residual matrices of selected windows from the Camelyon cache with exactly the
window logic and seeds of 33_camelyon_windows.py, and CHECKS every exported window against the
saved records (D_mix and D_site must agree to floating-point precision). Thresholds per seed come
from that seed's clean calibration records (encmonitor.thresholds.calibrate_thresholds).

Selection (fixed before the replay, same sizes as for MNIST and CIFAR-10): per seed the first 20
held-out clean windows and the first 5 windows of four representative conditions -> 200 windows.

Output: outputs/camelyon/ckks/camelyon17_replay.npz (+ .json), the same format as 20, so the CKKS
replay script 21_run_ckks_replay.py runs on it unchanged:
  python scripts/21_run_ckks_replay.py --replay outputs/camelyon/ckks/camelyon17_replay.npz \
      --out-dir outputs/camelyon/ckks/results
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

from encmonitor.artifacts import write_json  # noqa: E402
from encmonitor.embeddings import EmbeddingStandardizer  # noqa: E402
from encmonitor.rff import RFFMap  # noqa: E402
from encmonitor.statistics import compute_dual_statistics  # noqa: E402
from encmonitor.thresholds import calibrate_thresholds  # noqa: E402

CONDITIONS = ["exchange0.5__opposing", "stain0.2__opposing", "label_prior0.35__opposing", "stain0.2__coherent"]
QUANTILES = [0.95, 0.975]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser(description="Export Camelyon17 residual windows for the CKKS replay")
    ap.add_argument("--config", default="configs/camelyon.yaml")
    ap.add_argument("--cache-dir", default="outputs/camelyon/main/camelyon17")
    ap.add_argument("--out-dir", default="outputs/camelyon/ckks")
    ap.add_argument("--clean-per-seed", type=int, default=20)
    ap.add_argument("--per-condition", type=int, default=5)
    args = ap.parse_args()

    w33 = _load(ROOT / "scripts" / "33_camelyon_windows.py", "w33")
    m15 = w33._load15()
    cfg = yaml.safe_load(open(ROOT / args.config))
    n, S = int(cfg["window_size"]), int(cfg["sites"])
    weights = np.ones(S) / S
    target = int(cfg["label_prior_target"])
    conds = {c["condition"]: c for c in m15.conditions(cfg)}

    R, masks, seeds_col, win_col, cond_col = [], [], [], [], []
    taus = {q: [] for q in QUANTILES}
    max_diff = 0.0
    for seed in cfg["replay_partition_seeds"]:
        d = ROOT / args.cache_dir / f"seed_{seed}"
        prep = EmbeddingStandardizer.load(d / "preprocessor.npz")
        rff = RFFMap.load(str(d / "rff_map.npz"))
        bz = np.load(d / "baselines.npz")
        base = np.stack([np.asarray(bz[f"site_{s:03d}"], dtype=np.float64) for s in range(S)])
        mc, pc = np.load(d / "monitor_clean.npz"), np.load(d / "pieces.npz")
        mx = [mc[f"site_{s:03d}"].astype(np.float64) for s in range(S)]
        labels = [mc[f"labels_{s:03d}"] for s in range(S)]
        records = [json.loads(line) for line in open(d / "records.jsonl")]
        ref = {(r["condition"], r["window"]): r for r in records}
        th = {q: calibrate_thresholds([r for r in records if r["condition"] == "clean_calibration"], q) for q in QUANTILES}
        feat = lambda raw: rff.transform(prep.transform(raw))
        clean = [feat(x) for x in mx]
        v = np.random.default_rng([seed, 7]).standard_normal(mx[0].shape[1])
        v /= np.linalg.norm(v)

        jobs = [("clean_heldout", clean, [0.0] * S, seed + 1000, args.clean_per_seed)]
        for name in CONDITIONS:
            c = conds[name]
            src = cfg["exchange_sources"].get(c["geometry"], {}) if c["kind"] == "exchange" else {}
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
                    idx, _ = w33.label_prior_idx(labels[s], target, q, np.sign(f), rng)
                    feats.append(feat(mx[s][idx]))
                elif c["kind"] == "exchange":
                    h, q = int(src[s]), c["severity"] * abs(f)
                    rng = np.random.default_rng([seed, 5, s, int(round(q * 1000))])
                    N = mx[s].shape[0]
                    k = int(round(q * N))
                    keep = rng.choice(N, N - k, replace=False)
                    M = mx[h].shape[0]
                    take = rng.choice(M, k, replace=k > M)
                    feats.append(feat(np.concatenate([mx[s][keep], mx[h][take]])))
                else:  # vector
                    feats.append(feat(mx[s] + f * c["severity"] * v * prep.scale))
            jobs.append((name, feats, c["directions"], seed + 2000, args.per_condition))

        for cname, feats, dirs, wseed, count in jobs:
            rng = np.random.default_rng(wseed)          # the window draw of 15 / 33
            for w in range(count):
                means = []
                for s in range(S):
                    N = feats[s].shape[0]
                    idx = rng.choice(N, n, replace=N < n)
                    means.append(feats[s][idx].mean(axis=0))
                res = np.stack(means) - base
                st = compute_dual_statistics(res, weights)
                r = ref[(cname, w)]
                max_diff = max(max_diff, abs(st.d_mix - r["d_mix"]), abs(st.d_site - r["d_site"]))
                R.append(res); masks.append(np.asarray(dirs) != 0); seeds_col.append(seed)
                win_col.append(w); cond_col.append(cname)
                for q in taus:
                    taus[q].append([th[q].d_mix, th[q].d_site, th[q].heterogeneity])

    if max_diff > 1e-12:
        raise RuntimeError(f"Exported windows do not reproduce the Camelyon records (max diff {max_diff:.3e})")
    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    arrays = {"residuals": np.stack(R), "weights": weights, "shifted_mask": np.stack(masks),
              "seed": np.asarray(seeds_col), "window": np.asarray(win_col), "condition": np.asarray(cond_col)}
    for q, t in taus.items():
        arrays[f"tau_{q:g}"] = np.asarray(t)
    np.savez_compressed(out / "camelyon17_replay.npz", **arrays)
    summary = {"dataset": "camelyon17", "windows": len(R), "sites": S, "rff_dim": int(arrays["residuals"].shape[2]),
               "per_condition": {c: cond_col.count(c) for c in dict.fromkeys(cond_col)},
               "max_diff_vs_records": max_diff, "quantiles": QUANTILES}
    write_json(out / "camelyon17_replay.json", summary)
    print(f"[camelyon17] exported {len(R)} windows  {summary['per_condition']}  max diff vs records = {max_diff:.1e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

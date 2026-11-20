#!/usr/bin/env python3
"""Camelyon17 window runner (same window logic as 15_run_screening.py).

For every partition seed cache from 32_camelyon_cache.py:
  * 500 clean calibration windows and 200 independent held-out clean windows,
  * 50 windows for every condition in configs/camelyon.yaml (family x severity x geometry),
  * per window: D_mix, D_site, H, A_s, G_s, per-site energy, exact-kernel MMD^2 (pooled current
    vs pooled reference sample), attribution metrics -- exactly the fields of the MNIST/CIFAR-10
    records, so 16_analyze_screening.py can analyse them unchanged;
  * two additions used only for Camelyon:
      - the same window scored against ONE global reference (global-reference comparison on real hospitals),
      - the frozen model's accuracy per site on the window's own patches (accuracy link).

Window sampling, label-prior resampling and the controlled vector shift follow 15 exactly.
New family 'exchange': a receiving site draws a fraction q of its pool from another hospital's
monitoring pool (configs/camelyon.yaml: exchange_sources); no image is changed.

Usage (from the repository root):
  CUDA_VISIBLE_DEVICES=0 python scripts/33_camelyon_windows.py
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

from encmonitor.artifacts import write_json  # noqa: E402
from encmonitor.embeddings import EmbeddingStandardizer  # noqa: E402
from encmonitor.metrics import top1_accuracy, topk_recall  # noqa: E402
from encmonitor.rff import RFFMap  # noqa: E402
from encmonitor.statistics import compute_dual_statistics  # noqa: E402


def _load15():
    spec = importlib.util.spec_from_file_location("screening15", ROOT / "scripts" / "15_run_screening.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def label_prior_idx(labels, target, q, sign, rng):
    """Same random calls as label_prior_pool() in 15, returning indices (labels are needed)."""
    classes = np.unique(labels)
    p = np.array([np.mean(labels == c) for c in classes])
    onehot = (classes == target).astype(float)
    if sign > 0:
        p_new = (1 - q) * p + q * onehot
    else:
        rest = p * (1 - onehot)
        p_new = (1 - q) * p + q * rest / rest.sum()
    counts = rng.multinomial(labels.size, p_new)
    idx = np.concatenate([rng.choice(np.flatnonzero(labels == c), k, replace=True)
                          for c, k in zip(classes, counts) if k > 0])
    rng.shuffle(idx)
    return idx, float(np.mean(labels[idx] == target) - np.mean(labels == target))


def run_windows(m15, pools, base, gbase, ref_std, weights, n, windows, seed, mmd, meta, out_rows):
    """pools[s] = (rff features, standardized pool, correct-prediction flags). Same sampling as 15."""
    rng = np.random.default_rng(seed)
    rng_ref = np.random.default_rng([seed, 99])
    S = len(pools)
    dirs = np.asarray(meta.get("directions", [0.0] * S))
    mask = dirs != 0
    for w in range(windows):
        means, cur_pool, ref_pool, acc = [], [], [], []
        for s in range(S):
            feats, std, correct = pools[s]
            N = feats.shape[0]
            idx = rng.choice(N, n, replace=N < n)
            means.append(feats[idx].mean(axis=0))
            acc.append(float(correct[idx].mean()))
            if mmd is not None:
                cur_pool.append(std[idx])
                R = ref_std[s].shape[0]
                ref_pool.append(ref_std[s][rng_ref.choice(R, n, replace=R < n)])
        means = np.stack(means)
        res = means - np.stack(base)
        st = compute_dual_statistics(res, weights)
        sg = compute_dual_statistics(means - gbase[None, :], weights)
        row = {k: meta[k] for k in ("condition", "family", "severity", "geometry") if k in meta}
        row.update({
            "window": w, "window_seed": seed,
            "d_mix": st.d_mix, "d_site": st.d_site, "heterogeneity": st.heterogeneity,
            "contributions": st.contributions.tolist(), "directional": st.directional.tolist(),
            "site_energy": np.einsum("sm,sm->s", res, res).tolist(),
            "shifted_mask": mask.tolist(),
            "exact_mmd": mmd(np.concatenate(cur_pool), np.concatenate(ref_pool)) if mmd is not None else None,
            "d_mix_global": sg.d_mix, "d_site_global": sg.d_site, "heterogeneity_global": sg.heterogeneity,
            "acc": acc,
        })
        if mask.any():
            row["top1"] = top1_accuracy(st.contributions, mask)
            row["topk_recall"] = topk_recall(st.contributions, mask)
            row["ndcg"] = m15.ndcg(st.contributions, np.abs(dirs))
            row["spearman"] = m15.spearman(st.contributions, np.abs(dirs))
        out_rows.append(row)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the Camelyon17 windows")
    ap.add_argument("--config", default="configs/camelyon.yaml")
    ap.add_argument("--cache-dir", default="outputs/camelyon/main/camelyon17")
    ap.add_argument("--checkpoint", default="outputs/camelyon/model/resnet18_camelyon.pt")
    ap.add_argument("--seeds", default="")
    ap.add_argument("--device", default="cuda", help="device for the exact-kernel MMD (torch)")
    ap.add_argument("--only", default="", help="comma-separated families")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    import torch

    m15 = _load15()
    cfg = yaml.safe_load(open(ROOT / args.config))
    if args.device != "cpu" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available -- stop here. Use --device cpu only on purpose.")
    device = args.device
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()] or [int(s) for s in cfg["partition_seeds"]]
    n, S = int(cfg["window_size"]), int(cfg["sites"])
    weights = np.ones(S) / S
    conds = m15.conditions(cfg)
    if args.only:
        conds = [c for c in conds if c["family"] in set(args.only.split(","))]
    sd = torch.load(ROOT / args.checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    W = sd["backbone.fc.weight"].double().numpy()
    b = sd["backbone.fc.bias"].double().numpy()
    target = int(cfg["label_prior_target"])

    for seed in seeds:
        d = ROOT / args.cache_dir / f"seed_{seed}"
        out = d / "records.jsonl"
        if out.exists() and not args.force:
            print(f"[seed {seed}] records exist, skipping")
            continue
        t0 = time.time()
        prep = EmbeddingStandardizer.load(d / "preprocessor.npz")
        rff = RFFMap.load(str(d / "rff_map.npz"))
        bz = np.load(d / "baselines.npz")
        base = [np.asarray(bz[f"site_{s:03d}"], dtype=np.float64) for s in range(S)]
        gbase = np.load(d / "global_baseline.npy")
        bc, mc = np.load(d / "baseline_clean.npz"), np.load(d / "monitor_clean.npz")
        pieces = np.load(d / "pieces.npz")
        ref_std = [prep.transform(bc[f"site_{s:03d}"].astype(np.float64)) for s in range(S)]
        clean_raw = [mc[f"site_{s:03d}"].astype(np.float64) for s in range(S)]
        labels = [mc[f"labels_{s:03d}"] for s in range(S)]
        mmd = m15.ExactMMD(rff.bandwidth, device) if cfg["exact_mmd"]["enabled"] else None

        def pool(raw, lab):
            z = prep.transform(raw)
            correct = (raw @ W.T + b).argmax(axis=1) == lab
            return rff.transform(z), z, correct

        clean_pools = [pool(clean_raw[s], labels[s]) for s in range(S)]
        clean_acc = [float(p[2].mean()) for p in clean_pools]
        v = np.random.default_rng([seed, 7]).standard_normal(clean_raw[0].shape[1])
        v /= np.linalg.norm(v)

        rows: list[dict] = []
        run_windows(m15, clean_pools, base, gbase, ref_std, weights, n, int(cfg["calibration_windows"]), seed, mmd,
                    {"condition": "clean_calibration", "family": "clean", "severity": 0.0, "geometry": "clean"}, rows)
        run_windows(m15, clean_pools, base, gbase, ref_std, weights, n, int(cfg["heldout_windows"]), seed + 1000, mmd,
                    {"condition": "clean_heldout", "family": "clean", "severity": 0.0, "geometry": "clean"}, rows)

        cond_meta = []
        for c in conds:
            pools, eff = [], [0.0] * S
            src = cfg["exchange_sources"].get(c["geometry"], {}) if c["kind"] == "exchange" else {}
            for s in range(S):
                f = c["directions"][s]
                if f == 0:
                    pools.append(clean_pools[s])
                    continue
                if c["kind"] == "image":
                    raw = pieces[f"{c['family']}|{c['severity']:g}|{f:+g}|site_{s:03d}"].astype(np.float64)
                    lab = labels[s]
                elif c["kind"] == "label_prior":
                    q = c["severity"] * abs(f)
                    rng = np.random.default_rng([seed, 3, s, int(round(q * 1000)), int(f > 0)])
                    idx, eff[s] = label_prior_idx(labels[s], target, q, np.sign(f), rng)
                    raw, lab = clean_raw[s][idx], labels[s][idx]
                elif c["kind"] == "vector":
                    raw = clean_raw[s] + f * c["severity"] * v * prep.scale
                    lab = labels[s]
                elif c["kind"] == "exchange":
                    h = int(src[s])
                    q = c["severity"] * abs(f)
                    rng = np.random.default_rng([seed, 5, s, int(round(q * 1000))])
                    N = clean_raw[s].shape[0]
                    k = int(round(q * N))
                    keep = rng.choice(N, N - k, replace=False)
                    M = clean_raw[h].shape[0]
                    take = rng.choice(M, k, replace=k > M)
                    raw = np.concatenate([clean_raw[s][keep], clean_raw[h][take]])
                    lab = np.concatenate([labels[s][keep], labels[h][take]])
                    eff[s] = k / N
                else:
                    raise ValueError(c["kind"])
                pools.append(pool(raw, lab))
            run_windows(m15, pools, base, gbase, ref_std, weights, n, int(cfg["windows_per_condition"]), seed + 2000, mmd, c, rows)
            meta = dict(c)
            if c["kind"] in ("label_prior", "exchange"):
                meta["effective_change"] = eff
            if c["kind"] == "exchange":
                meta["exchange_sources"] = {str(k): int(v) for k, v in src.items()}
            cond_meta.append(meta)

        tmp = out.with_suffix(".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            for r in rows:
                r["partition_seed"] = seed
                fh.write(json.dumps(r) + "\n")
        tmp.replace(out)
        write_json(d / "conditions.json", {"partition_seed": seed, "label_prior_target_class": target,
                                           "clean_accuracy_per_site": clean_acc, "conditions": cond_meta,
                                           "windows": len(rows), "seconds": round(time.time() - t0, 1),
                                           "mmd_device": device})
        print(f"[seed {seed}] OK  {len(rows)} windows  {len(cond_meta)} conditions  "
              f"clean accuracy per site {np.round(clean_acc, 3).tolist()}  {round(time.time() - t0, 1)}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

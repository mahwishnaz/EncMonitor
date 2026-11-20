#!/usr/bin/env python3
"""Window runner of the main experiment (MNIST, CIFAR-10).

For each partition seed cache built by 14_build_screening_cache.py, this script
  * runs clean calibration windows (500) and independent held-out clean windows (200),
  * assembles every condition in configs/main.yaml (family x severity x geometry),
  * scores every window with the dual statistics (per site, per window,
    rng.choice(N_s, n) -> RFF mean sketch -> minus site baseline),
  * and on the SAME windows records the baselines' raw scores:
      - per-site residual energy ||delta_s||^2   (local alarms + max / majority)
      - centralized exact-kernel RBF MMD^2        (pooled current vs pooled reference)
D_mix is also the plaintext pooled-only / secure-aggregation / EncMonitor-global-only score.

Label-prior shift: the cached clean monitor pool of a shifted site is
resampled (with replacement, same size) from a class mixture
    '+' : p' = (1-q) p + q e_c          (mass moves TOWARD target class c)
    '-' : p' = (1-q) p + q p_notc       (mass moves AWAY from c, proportional to the other classes)
with q = severity x |direction|; c = the class with the largest minimum count over sites 0-2.
Controlled vector shift: + direction x eta standardized units along a fixed random unit vector v.
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

from encmonitor.artifacts import write_json
from encmonitor.embeddings import EmbeddingStandardizer
from encmonitor.metrics import top1_accuracy, topk_recall
from encmonitor.rff import RFFMap
from encmonitor.statistics import compute_dual_statistics


class ExactMMD:
    """Biased RBF MMD^2 between two samples, k(a,b)=exp(-||a-b||^2 / (2 sigma^2)) (the kernel the RFF map approximates)."""

    def __init__(self, sigma: float, device: str = "cpu"):
        self.gamma = 1.0 / (2.0 * sigma * sigma)
        self.device = device
        try:
            import torch
            self.torch = torch
        except Exception:
            self.torch = None

    def __call__(self, a: np.ndarray, b: np.ndarray) -> float:
        if self.torch is not None:
            t = self.torch
            A = t.as_tensor(a, dtype=t.float64, device=self.device)
            B = t.as_tensor(b, dtype=t.float64, device=self.device)
            k = lambda u, v: t.exp(-self.gamma * t.cdist(u, v).pow(2)).mean()
            return float(k(A, A) + k(B, B) - 2.0 * k(A, B))
        def k(u, v):
            d2 = (u * u).sum(1)[:, None] + (v * v).sum(1)[None, :] - 2.0 * u @ v.T
            return np.exp(-self.gamma * np.maximum(d2, 0.0)).mean()
        return float(k(a, a) + k(b, b) - 2.0 * k(a, b))


def label_prior_pool(emb, labels, target, q, sign, rng):
    classes = np.unique(labels)
    p = np.array([np.mean(labels == c) for c in classes])
    onehot = (classes == target).astype(float)
    if sign > 0:
        p_new = (1 - q) * p + q * onehot
    else:
        rest = p * (1 - onehot)
        p_new = (1 - q) * p + q * rest / rest.sum()
    n = labels.size
    counts = rng.multinomial(n, p_new)
    idx = np.concatenate([rng.choice(np.flatnonzero(labels == c), k, replace=True)
                          for c, k in zip(classes, counts) if k > 0])
    rng.shuffle(idx)
    eff = float(np.mean(labels[idx] == target) - np.mean(labels == target))
    return emb[idx], eff


def conditions(cfg):
    out = []
    for fam, spec in cfg["families"].items():
        for sev in spec["severities"]:
            for g in spec["geometries"]:
                dirs = [float(v) for v in cfg["geometries"][g]]
                out.append({"condition": f"{fam}{sev:g}__{g}", "family": fam, "kind": spec["kind"],
                            "severity": float(sev), "geometry": g, "directions": dirs})
    return out


def ndcg(scores, gains):
    order = np.argsort(-np.asarray(scores))
    g = np.asarray(gains, dtype=float)
    disc = 1.0 / np.log2(np.arange(2, g.size + 2))
    ideal = (np.sort(g)[::-1] * disc).sum()
    return float((g[order] * disc).sum() / ideal) if ideal > 0 else float("nan")


def spearman(a, b):
    from scipy.stats import spearmanr
    b = np.asarray(b, dtype=float)
    if np.all(b == b[0]):
        return float("nan")
    return float(spearmanr(a, b).statistic)


def run_windows(feats, raw_std, base, ref_std, weights, n, windows, seed, mmd, meta, out_rows):
    """feats[s]: RFF features of the site's current pool; raw_std[s]: standardized+clipped current pool."""
    rng = np.random.default_rng(seed)             # window sampling
    rng_ref = np.random.default_rng([seed, 99])   # reference subsample for exact MMD (separate stream)
    S = len(feats)
    dirs = np.asarray(meta.get("directions", [0.0] * S))
    mask = dirs != 0
    for w in range(windows):
        residuals, cur_pool, ref_pool = [], [], []
        for s in range(S):
            N = feats[s].shape[0]
            idx = rng.choice(N, n, replace=N < n)
            residuals.append(feats[s][idx].mean(axis=0) - base[s])
            if mmd is not None:
                cur_pool.append(raw_std[s][idx])
                R = ref_std[s].shape[0]
                ref_pool.append(ref_std[s][rng_ref.choice(R, n, replace=R < n)])
        res = np.stack(residuals)
        st = compute_dual_statistics(res, weights)
        row = {k: meta[k] for k in ("condition", "family", "severity", "geometry") if k in meta}
        row.update({
            "window": w, "window_seed": seed,
            "d_mix": st.d_mix, "d_site": st.d_site, "heterogeneity": st.heterogeneity,
            "contributions": st.contributions.tolist(), "directional": st.directional.tolist(),
            "site_energy": np.einsum("sm,sm->s", res, res).tolist(),
            "shifted_mask": mask.tolist(),
            "exact_mmd": mmd(np.concatenate(cur_pool), np.concatenate(ref_pool)) if mmd is not None else None,
        })
        if mask.any():
            row["top1"] = top1_accuracy(st.contributions, mask)
            row["topk_recall"] = topk_recall(st.contributions, mask)
            row["ndcg"] = ndcg(st.contributions, np.abs(dirs))
            row["spearman"] = spearman(st.contributions, np.abs(dirs))
        out_rows.append(row)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the screening windows for one dataset")
    ap.add_argument("--config", default="configs/main.yaml")
    ap.add_argument("--cache-dir", required=True, help="e.g. outputs/main/mnist")
    ap.add_argument("--seeds", default="")
    ap.add_argument("--device", default="cuda", help="device for exact-kernel MMD (torch)")
    ap.add_argument("--only", default="", help="comma-separated families to run")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    try:
        import torch
        device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()] or [int(s) for s in cfg["partition_seeds"]]
    n, S = int(cfg["window_size"]), int(cfg["sites"])
    weights = np.ones(S) / S
    conds = conditions(cfg)
    if args.only:
        keep = set(args.only.split(","))
        conds = [c for c in conds if c["family"] in keep]

    for seed in seeds:
        d = Path(args.cache_dir) / f"seed_{seed}"
        out = d / "records.jsonl"
        if out.exists() and not args.force:
            print(f"[seed {seed}] records exist, skipping")
            continue
        t0 = time.time()
        prep = EmbeddingStandardizer.load(d / "preprocessor.npz")
        rff = RFFMap.load(str(d / "rff_map.npz"))
        base_z = np.load(d / "baselines.npz")
        base = [np.asarray(base_z[f"site_{s:03d}"], dtype=np.float64) for s in range(S)]
        bc, mc = np.load(d / "baseline_clean.npz"), np.load(d / "monitor_clean.npz")
        pieces = np.load(d / "pieces.npz")
        ref_std = [prep.transform(bc[f"site_{s:03d}"].astype(np.float64)) for s in range(S)]
        clean_raw = [mc[f"site_{s:03d}"].astype(np.float64) for s in range(S)]
        labels = [mc[f"labels_{s:03d}"] for s in range(S)]
        mmd = ExactMMD(rff.bandwidth, device) if cfg["exact_mmd"]["enabled"] else None

        def pool_feats(raw):
            z = prep.transform(raw)
            return rff.transform(z), z

        clean_fz = [pool_feats(x) for x in clean_raw]
        # label-prior target class: largest minimum count over sites 0..2 (all shifted sites of 'mixed')
        classes = np.unique(np.concatenate(labels))
        target = int(classes[np.argmax([min(np.sum(labels[s] == c) for s in range(3)) for c in classes])])
        # controlled vector direction (fixed per seed)
        v = np.random.default_rng([seed, 7]).standard_normal(clean_raw[0].shape[1])
        v /= np.linalg.norm(v)

        rows: list[dict] = []
        feats = [f for f, _ in clean_fz]; stds = [z for _, z in clean_fz]
        run_windows(feats, stds, base, ref_std, weights, n, int(cfg["calibration_windows"]), seed, mmd,
                    {"condition": "clean_calibration", "family": "clean", "severity": 0.0, "geometry": "clean"}, rows)
        run_windows(feats, stds, base, ref_std, weights, n, int(cfg["heldout_windows"]), seed + 1000, mmd,
                    {"condition": "clean_heldout", "family": "clean", "severity": 0.0, "geometry": "clean"}, rows)

        cond_meta = []
        for c in conds:
            feats, stds, eff = [], [], [0.0] * S
            for s in range(S):
                f = c["directions"][s]
                if f == 0:
                    fz = clean_fz[s]
                elif c["kind"] == "image":
                    raw = pieces[f"{c['family']}|{c['severity']:g}|{f:+g}|site_{s:03d}"].astype(np.float64)
                    fz = pool_feats(raw)
                elif c["kind"] == "label_prior":
                    q = c["severity"] * abs(f)
                    rng = np.random.default_rng([seed, 3, s, int(round(q * 1000)), int(f > 0)])
                    raw, eff[s] = label_prior_pool(clean_raw[s], labels[s], target, q, np.sign(f), rng)
                    fz = pool_feats(raw)
                elif c["kind"] == "vector":
                    raw = clean_raw[s] + f * c["severity"] * v * prep.scale
                    fz = pool_feats(raw)
                else:
                    raise ValueError(c["kind"])
                feats.append(fz[0]); stds.append(fz[1])
            run_windows(feats, stds, base, ref_std, weights, n, int(cfg["windows_per_condition"]), seed + 2000, mmd, c, rows)
            m = dict(c)
            if c["kind"] == "label_prior":
                m.update({"target_class": target, "effective_target_mass_change": eff})
            cond_meta.append(m)

        tmp = out.with_suffix(".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            for r in rows:
                r["partition_seed"] = seed
                fh.write(json.dumps(r) + "\n")
        tmp.replace(out)
        write_json(d / "conditions.json", {"partition_seed": seed, "label_prior_target_class": target,
                                           "conditions": cond_meta, "windows": len(rows),
                                           "seconds": round(time.time() - t0, 1), "mmd_device": device})
        print(f"[seed {seed}] OK  {len(rows)} windows  {len(cond_meta)} conditions  {round(time.time() - t0, 1)}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Clipping sensitivity (Supplementary). CIFAR-10 and Camelyon17 patch split, seeds 2026-2030,
clip radius C = 8 (main setting) and C = 16 (same C/sqrt(d) as MNIST: 8/sqrt(128) = 16/sqrt(512)).

For C = 16 the standardizer, median bandwidth, RFF map and site baselines are refitted exactly as in scripts 14/32
(only the clip radius differs); the windows use the same caches, seeds and streams. Every window is turned into a
record (D_mix, D_site, H, site energies) and scored with the Table 1 code (private_localany `score`).
C = 8 is regenerated the same way and checked against the saved records (relative difference < 1e-6).
Also reports the clipped fraction of the standardized representations at each C.
Writes outputs/analysis/clipping/{per_seed.csv, checks.csv, fractions.csv}.
Usage: python analysis/clipping_check.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import shared as c2

OUTD = c2.RESULTS / "clipping"
RULES = ["pooled", "encmon_joint", "localany", "hybrid", "m_pooled", "m_encmon_joint", "m_localany", "m_hybrid"]


def records_from(grams, conds, seed):
    meta = {c["condition"]: c for c in conds}
    rows = []
    for name, G in grams.items():
        S = G.shape[1]
        a = np.ones(S) / S
        e = np.einsum("wss->ws", G)
        dm = np.einsum("s,wst,t->w", a, G, a)
        ds = e @ a
        c = meta.get(name, {"family": "clean", "geometry": "clean", "severity": 0.0})
        for w in range(G.shape[0]):
            rows.append({"condition": name, "family": c["family"], "geometry": c["geometry"], "severity": c["severity"],
                         "window": w, "d_mix": dm[w], "d_site": ds[w], "heterogeneity": ds[w] - dm[w],
                         "site_energy": e[w].tolist(), "contributions": (e[w] * a).tolist(), "exact_mmd": np.nan,
                         "partition_seed": seed})
    return pd.DataFrame(rows)


def main() -> int:
    OUTD.mkdir(parents=True, exist_ok=True)
    rows, checks, fr = [], [], []
    for name in ("cifar10", "camelyon17_patch"):
        key, d, ds = c2.SETS[name]
        for seed in c2.OLD:
            for C in (8.0, 16.0):
                ctx = c2.Ctx(ds, seed, d, clip=C)
                conds = c2.rw.conditions(ds)
                grams = ctx.all_grams(conds)
                rec = records_from(grams, conds, seed)
                if C == 8.0:
                    saved = c2.load_records(name, [seed])
                    same = len(saved) == len(rec) and (saved.condition.to_numpy() == rec.condition.to_numpy()).all()
                    rd = float(np.max(np.abs(saved.d_mix.to_numpy() - rec.d_mix.to_numpy()) / np.abs(saved.d_mix.to_numpy()))) if same else np.inf
                    checks.append({"set": name, "seed": seed, "same_sequence": bool(same), "max_rel_d_mix_C8_vs_saved": rd})
                sc, _ = c2.score(rec)
                held = sc[sc.condition == "clean_heldout"]
                r0 = {"set": name, "seed": seed, "C": C, "bandwidth": float(ctx.rff.bandwidth)}
                rows.append({**r0, "aggregate": "clean_fpr", **{k: float(held[k].mean()) for k in RULES}})
                sh = sc[~sc.condition.str.startswith("clean")]
                cond = sh.groupby(["condition", "family", "geometry", "severity"])[RULES].mean().reset_index()
                m = c2.P1.masks_for(cond, key)
                m["vector_all"] = cond.family == "vector"
                m["vector_opposing_mixed"] = (cond.family == "vector") & cond.geometry.isin(["opposing", "mixed"])
                for agg, mk in m.items():
                    rows.append({**r0, "aggregate": agg, "conditions": int(mk.sum()), **{k: float(cond[mk][k].mean()) for k in RULES}})
                # clipped fraction of the standardized representations (reference + monitoring pools)
                bc, mc = np.load(ctx.cache_dir / "baseline_clean.npz"), np.load(ctx.cache_dir / "monitor_clean.npz")
                z = [(x[f"site_{s:03d}"].astype(np.float64) - ctx.prep.mean) / ctx.prep.scale for x in (bc, mc) for s in range(ctx.S)]
                nrm = np.linalg.norm(np.concatenate(z), axis=1)
                fr.append({"set": name, "seed": seed, "C": C, "dim": int(nrm.size and z[0].shape[1]),
                           "fraction_clipped": float((nrm > C).mean()), "median_norm": float(np.median(nrm))})
                print(f"[{name} {seed} C={C:g}] done", flush=True)
    c2.write_csv(pd.DataFrame(rows), OUTD / "per_seed.csv", "clipping sensitivity, per seed")
    c2.write_csv(pd.DataFrame(checks), OUTD / "checks.csv", "C = 8 regenerated vs saved records")
    c2.write_csv(pd.DataFrame(fr), OUTD / "fractions.csv", "clipped fraction of standardized representations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

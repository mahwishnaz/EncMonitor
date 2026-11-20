#!/usr/bin/env python3
"""Weighted variants of EncMonitor.

Per record set and seed, the residual Gram matrices of every window are regenerated from the caches
(rx_windows / CensusCtx, same seeds and streams as the window runners). Pass criterion: the regenerated uniform D_mix and
D_site must equal the saved values (relative difference < 1e-6); a set/seed that fails is excluded and reported.

  uniform     : saved records, private_localany `score`
                (the scoring code of the main detection table (Table 4 of the article; the same scoring is used for Tables 5-7))
  variant (i) : `encmon_norm` of the same scoring = D_mix unchanged, site-energy term divided by the clean energy
                (only the site-energy term is weighted)
  variant (ii): w_s ∝ 1/ē⁰_s (calibration windows of the seed), Σ w_s = 1, used in BOTH D_mix_w = wᵀGw and
                D_site_w = Σ w_s G_ss; thresholds and joint rule calibrated as for uniform weights (95th pct of the
                500 calibration windows; joint constant = 95th pct of max(D_mix_w/t, D_site_w/t)).
Writes outputs/analysis/weighted/{per_seed.csv, weights.csv, checks.json, summary.csv}.
Usage: python analysis/weighted_variants.py [--sets a,b] [--seeds 2026,...]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

import shared as c2

OUTD = c2.RESULTS / "weighted"
TOL = 1e-6


def qh(v):
    return c2.rc.qh(v, 0.95)


def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    den = np.maximum(np.abs(a), np.abs(b))
    return float(np.max(np.where(den > 0, np.abs(a - b) / np.where(den > 0, den, 1), 0)))


def masks(cond: pd.DataFrame, key: str) -> dict:
    m = c2.P1.masks_for(cond, key)
    m["opposing_all_families"] = cond.geometry == "opposing"
    return m


def one(name: str, seed: int) -> tuple[list[dict], dict, dict]:
    key = c2.SETS[name][0]
    rec = c2.load_records(name, [seed])
    grams, conds = c2.grams_for(name, seed)
    order = ["clean_calibration", "clean_heldout"] + [c["condition"] for c in conds]
    G = np.concatenate([grams[k] for k in order])
    names = np.concatenate([[k] * grams[k].shape[0] for k in order])
    S = G.shape[1]
    a = np.ones(S) / S
    dmix_u = np.einsum("s,wst,t->w", a, G, a)
    dsite_u = np.einsum("wss->ws", G) @ a
    chk = {"set": name, "seed": seed, "windows_saved": len(rec), "windows_regenerated": len(G),
           "same_condition_sequence": bool(len(rec) == len(G) and (rec.condition.to_numpy() == names).all())}
    if chk["same_condition_sequence"]:
        chk["max_rel_d_mix"] = rel(dmix_u, rec.d_mix.to_numpy())
        chk["max_rel_d_site"] = rel(dsite_u, rec.d_site.to_numpy())
        chk["pass"] = chk["max_rel_d_mix"] < TOL and chk["max_rel_d_site"] < TOL
    else:
        chk["pass"] = False
    if not chk["pass"]:
        return [], chk, {}
    sc, info = c2.score(rec)                    # uniform + variant (i), windows except calibration
    cal = names == "clean_calibration"
    w = c2.w_inverse_energy(G[cal])
    dmix_w = np.einsum("s,wst,t->w", w, G, w)
    dsite_w = np.einsum("wss->ws", G) @ w
    t_mix, t_site = qh(dmix_w[cal]), qh(dsite_w[cal])
    s_w = np.maximum(dmix_w / t_mix, dsite_w / t_site)
    c_w = qh(s_w[cal])
    keep = ~cal
    sc = sc.reset_index(drop=True)
    sc["pooled_w"] = dmix_w[keep] > t_mix
    sc["encmon_w"] = s_w[keep] > c_w
    sc["encmon_w_score"] = s_w[keep]
    held = sc.condition == "clean_heldout"
    sc["m_encmon_w"] = sc.encmon_w_score > qh(sc.encmon_w_score[held])
    sc["m_pooled_w"] = dmix_w[keep] > qh(dmix_w[keep][held.to_numpy()])
    sc["m_encmon_norm"] = sc.norm_score > qh(sc.norm_score[held])
    rules = ["pooled", "encmon_joint", "encmon_norm", "encmon_w", "pooled_w", "localany",
             "m_encmon_joint", "m_encmon_norm", "m_encmon_w", "m_pooled", "m_pooled_w", "m_localany"]
    rows = [{"set": name, "seed": seed, "aggregate": "clean_fpr", **{k: float(sc[held][k].mean()) for k in rules}}]
    sh = sc[~sc.condition.str.startswith("clean")]
    cond = sh.groupby(["condition", "family", "geometry", "severity"])[rules].mean().reset_index()
    for agg, mk in masks(cond, key).items():
        if mk.any():
            rows.append({"set": name, "seed": seed, "aggregate": agg, "conditions": int(mk.sum()),
                         **{k: float(cond[mk][k].mean()) for k in rules}})
    wr = {"set": name, "seed": seed, **{f"w_{s}": float(w[s]) for s in range(S)},
          "w_max_over_min": float(w.max() / w.min()), "t_mix_w": t_mix, "t_site_w": t_site, "c_w": c_w}
    return rows, chk, wr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="mnist,cifar10,camelyon17_patch,camelyon17_slide,camelyon17_patient,census_n250,census_n1000")
    ap.add_argument("--seeds", default="")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    OUTD.mkdir(parents=True, exist_ok=True)
    want = [int(s) for s in args.seeds.split(",") if s]
    rows, checks, ws = [], [], []
    for name in args.sets.split(","):
        for seed in c2.seeds_of(name):
            if want and seed not in want:
                continue
            t0 = time.time()
            r, chk, wr = one(name, seed)
            chk["seconds"] = round(time.time() - t0, 1)
            rows += r
            checks.append(chk)
            if wr:
                ws.append(wr)
            print(f"[{name} {seed}] pass criterion {'PASS' if chk['pass'] else 'FAIL'} "
                  f"(d_mix {chk.get('max_rel_d_mix', float('nan')):.1e}, d_site {chk.get('max_rel_d_site', float('nan')):.1e}) {chk['seconds']}s",
                  flush=True)
    sfx = f"_{args.tag}" if args.tag else ""
    c2.write_csv(pd.DataFrame(rows), OUTD / f"per_seed{sfx}.csv", "weighted variants per seed")
    c2.write_csv(pd.DataFrame(ws), OUTD / f"weights{sfx}.csv", "variant (ii) weights per seed")
    c2.write_csv(pd.DataFrame(checks), OUTD / f"checks{sfx}.csv", "pass criterion: regenerated uniform D_mix/D_site vs saved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

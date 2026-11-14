#!/usr/bin/env python3
"""Worked example for Figure 1: one real monitoring window, m = 4.

Candidates, in this order (CANDIDATES): (1) three sites, vector shift eta = 2 opposing; (2) five sites, label
prior 0.5 opposing; (3) five sites, vector eta = 2 opposing. A window is selected if the global alert fires and
the pooled alert is silent; the first such window of the first candidate that has one is reported, and later
candidates are not computed.
Usage: python analysis/worked_example.py
Writes outputs/analysis/worked_example/{candidates.csv, residuals.csv, statistics.csv, thresholds.csv,
window_indices.csv}.

Details:

Data: MNIST, partition seed 2026, sites 0, 1, 2 of the main-experiment cache (frozen LeNet representations,
standardizer of the cache, clip radius 8). RFF map with m = 4 (same bandwidth and RFF seed 1729 as the paper's map).
Each site's reference sketch μ_s0 is the mean RFF feature of its reference pool. Condition label_prior 0.5
opposing (site 0: more of the target class, site 1: less; site 2 unchanged); its 50 windows are drawn exactly as
in script 15 (window seed 2026 + 2000, n = 250 per site). Selection rule (fixed before looking at the values): the
first window, by index, in which the global alert fires while the pooled alert is silent; the alert rates over all
50 windows are reported as well. Plaintext implementation: encmonitor.statistics.
Thresholds for this 3-site, m = 4 configuration: 95th percentile over 500 clean calibration windows (window seed
2026), joint constant c as in the paper. Release decision: L2 always; L3 (A_s) if the global alert fires.
Writes results/worked_example/{residuals.csv, statistics.csv, thresholds.csv, window_indices.csv}.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import shared as c2  # noqa: E402
from encmonitor.rff import RFFMap
from encmonitor.statistics import compute_dual_statistics

OUTD = c2.RESULTS / "worked_example"
M = 4
# RULE.md, in this order; directions as in configs/main.yaml (opposing = +1, -1, 0, 0, 0)
CANDIDATES = [
    {"name": "1: 3 sites, vector eta=2 opposing", "sites": [0, 1, 2], "kind": "vector", "severity": 2.0},
    {"name": "2: 5 sites, label prior 0.5 opposing", "sites": [0, 1, 2, 3, 4], "kind": "label_prior", "severity": 0.5},
    {"name": "3: 5 sites, vector eta=2 opposing", "sites": [0, 1, 2, 3, 4], "kind": "vector", "severity": 2.0},
]
DIRECTIONS = [1.0, -1.0, 0.0, 0.0, 0.0]


def main() -> int:
    seed = 2026
    ctx = c2.Ctx("mnist", seed, c2.SETS["mnist"][1])
    cand_rows = []
    for cd in CANDIDATES:
        out = evaluate(ctx, seed, cd)
        cand_rows.append({"candidate": cd["name"], "computed": True, "selected_window": out["k"],
                          "global_alert_rate_50": out["glob"].mean(), "pooled_alert_rate_50": out["pool_al"].mean(),
                          "windows_global_and_not_pooled": int((out["glob"] & ~out["pool_al"]).sum())})
        if out["k"] is not None:
            break
    for cd in CANDIDATES[len(cand_rows):]:
        cand_rows.append({"candidate": cd["name"], "computed": False})
    c2.write_csv(pd.DataFrame(cand_rows), OUTD / "candidates.csv", "worked example v2: candidates in RULE.md order")
    print(pd.DataFrame(cand_rows).to_string(index=False))
    if out["k"] is None:
        print("No candidate has a window with a global alert and a silent pooled alert; no window is selected.")
        return 0
    write(ctx, out)
    return 0


def evaluate(ctx, seed, cd):
    SITES = cd["sites"]
    rff = RFFMap.fit(ctx.rff.input_dim, M, ctx.rff.bandwidth, 1729)
    bc = np.load(ctx.cache_dir / "baseline_clean.npz")
    base = np.stack([rff.mean_sketch(ctx.prep.transform(bc[f"site_{s:03d}"].astype(np.float64))) for s in SITES])
    feats = lambda raw: rff.transform(ctx.prep.transform(raw))
    clean = [feats(ctx.clean_raw[s]) for s in SITES]
    w = np.ones(len(SITES)) / len(SITES)
    n = ctx.n

    def windows(pools, W, wseed, keep_idx=False):
        rng = np.random.default_rng(wseed)
        out, idxs = [], []
        for _ in range(W):
            res, ii = [], []
            for p in pools:
                idx = rng.choice(p.shape[0], n, replace=p.shape[0] < n)
                res.append(p[idx].mean(0))
                ii.append(idx)
            out.append(np.stack(res) - base)
            idxs.append(ii)
        return (np.stack(out), idxs) if keep_idx else np.stack(out)

    cal = windows(clean, 500, seed)
    st = [compute_dual_statistics(r, w) for r in cal]
    q = lambda v: c2.rc.qh(v, 0.95)
    t_mix, t_site = q([s.d_mix for s in st]), q([s.d_site for s in st])
    c = q([max(s.d_mix / t_mix, s.d_site / t_site) for s in st])
    m0_mix = float(np.mean([s.d_mix for s in st]))
    e0 = np.mean([np.einsum("sm,sm->s", r, r) for r in cal], axis=0)

    cond = {"kind": cd["kind"], "family": cd["kind"], "severity": cd["severity"], "directions": DIRECTIONS}
    pools = []
    for s in SITES:
        f = cond["directions"][s]
        if f == 0:
            pools.append(clean[SITES.index(s)])
        elif cond["kind"] == "vector":     # rx_windows.SeedContext.site_feats, kind "vector"
            pools.append(feats(ctx.clean_raw[s] + f * cond["severity"] * ctx.v * ctx.prep.scale))
        else:
            qv = cond["severity"] * abs(f)
            rng = np.random.default_rng([seed, 3, s, int(round(qv * 1000)), int(f > 0)])
            idx = c2.rw.label_prior_idx(ctx.labels[s], ctx.target, qv, np.sign(f), rng)
            pools.append(feats(ctx.clean_raw[s][idx]))
    R, idxs = windows(pools, 50, seed + 2000, keep_idx=True)
    sts = [compute_dual_statistics(r, w) for r in R]
    glob = np.array([max(x.d_mix / t_mix, x.d_site / t_site) > c for x in sts])
    pool_al = np.array([x.d_mix > t_mix for x in sts])
    cand = np.flatnonzero(glob & ~pool_al)
    k = int(cand[0]) if cand.size else None
    return dict(locals())


def write(ctx, o):
    SITES, k, R, idxs, w, base, n = o["SITES"], o["k"], o["R"], o["idxs"], o["w"], o["base"], o["n"]
    t_mix, t_site, c, m0_mix, e0, glob, pool_al, cd = (o[x] for x in ("t_mix", "t_site", "c", "m0_mix", "e0", "glob", "pool_al", "cd"))
    delta = R[k]
    idxs = [idxs[k]]
    s_ = compute_dual_statistics(delta, w)
    E = np.einsum("sm,sm->s", delta, delta)
    score = max(s_.d_mix / t_mix, s_.d_site / t_site)
    alert = score > c
    C = s_.d_mix / (np.sqrt(w * s_.contributions).sum() ** 2)
    ee = np.maximum(E - e0, 0)
    Ccorr = min(1.0, max(s_.d_mix - m0_mix, 0) / ((w * np.sqrt(ee)).sum() ** 2))   # clipped to [0, 1] as in h_diagnostic

    res_rows = []
    for i, s in enumerate(SITES):
        f = DIRECTIONS[s]
        shift = "none" if f == 0 else (f"{'+' if f > 0 else '-'}{cd['severity']:g} " +
                                        ("along +v" if (cd["kind"] == "vector" and f > 0) else "along -v" if cd["kind"] == "vector"
                                         else "toward target class" if f > 0 else "away from target class"))
        res_rows.append({"site": s, "shift": shift,
                         "n": n, **{f"window_sketch_{j + 1}": float(delta[i, j] + base[i, j]) for j in range(M)},
                         **{f"reference_sketch_{j + 1}": float(base[i, j]) for j in range(M)},
                         **{f"residual_{j + 1}": float(delta[i, j]) for j in range(M)},
                         "residual_norm_sq": float(E[i]), "weight": float(w[i])})
    res_rows.append({"site": "pooled (weighted mean)", "shift": "", "n": n * len(SITES),
                     **{f"residual_{j + 1}": float((w @ delta)[j]) for j in range(M)}, "residual_norm_sq": float(s_.d_mix)})
    stat = pd.DataFrame([
        {"quantity": "D_mix", "value": s_.d_mix, "threshold": t_mix, "ratio_to_threshold": s_.d_mix / t_mix},
        {"quantity": "D_site", "value": s_.d_site, "threshold": t_site, "ratio_to_threshold": s_.d_site / t_site},
        {"quantity": "H = D_site - D_mix", "value": s_.heterogeneity, "threshold": np.nan, "ratio_to_threshold": np.nan},
        {"quantity": "H / D_site", "value": s_.heterogeneity / s_.d_site, "threshold": np.nan, "ratio_to_threshold": np.nan},
        {"quantity": "EncMonitor score max(D_mix/t_mix, D_site/t_site)", "value": score, "threshold": c, "ratio_to_threshold": score / c},
        {"quantity": "global alert (score > c)", "value": float(alert), "threshold": np.nan, "ratio_to_threshold": np.nan},
        {"quantity": "pooled alert (D_mix > t_mix)", "value": float(s_.d_mix > t_mix), "threshold": np.nan, "ratio_to_threshold": np.nan},
        *[{"quantity": f"A_{s} (released only after an alert)", "value": float(a), "threshold": np.nan, "ratio_to_threshold": np.nan}
          for s, a in zip(SITES, s_.contributions)],
        {"quantity": "coherence C", "value": C, "threshold": np.nan, "ratio_to_threshold": np.nan},
        {"quantity": "coherence C_corr (noise-corrected)", "value": Ccorr, "threshold": 0.5, "ratio_to_threshold": np.nan},
        {"quantity": "release", "value": np.nan, "threshold": np.nan, "ratio_to_threshold": np.nan,
         "note": ("L2 (D_mix, D_site, H) + L3 (" + ", ".join(f"A_{s}" for s in SITES) + ")") if alert else "L2 only (D_mix, D_site, H)"},
    ])
    stat = pd.concat([stat, pd.DataFrame([
        {"quantity": "candidate", "value": np.nan, "note": cd["name"]},
        {"quantity": "D_mix / t_mix", "value": s_.d_mix / t_mix},
        {"quantity": "D_site / t_site", "value": s_.d_site / t_site},
        {"quantity": "selected window index (of 50)", "value": k},
        {"quantity": "global alert rate over the 50 windows", "value": glob.mean()},
        {"quantity": "pooled alert rate over the 50 windows", "value": pool_al.mean()}])], ignore_index=True)
    stat["identity_check |D_site - D_mix - H|"] = abs(s_.d_site - s_.d_mix - s_.heterogeneity)
    c2.write_csv(pd.DataFrame(res_rows), OUTD / "residuals.csv", "worked example: sketches and residuals, MNIST seed 2026, m = 4")
    c2.write_csv(stat, OUTD / "statistics.csv", "worked example: statistics, thresholds and release decision")
    c2.write_csv(pd.DataFrame([{"t_mix": t_mix, "t_site": t_site, "c_joint": c, "calibration_windows": 500, "candidate": cd["name"],
                                "bandwidth": ctx.rff.bandwidth, "rff_seed": 1729, "m": M, "sites": str(SITES)}]),
                 OUTD / "thresholds.csv", "worked example: thresholds of the 3-site m = 4 configuration")
    c2.write_csv(pd.DataFrame({f"site_{s}": idxs[0][i] for i, s in enumerate(SITES)}), OUTD / "window_indices.csv",
                 "row indices of the window within each site's monitoring pool (after the shift for sites 0, 1)")
    print(pd.DataFrame(res_rows).round(4).to_string(index=False))
    print(stat.to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())

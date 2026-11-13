#!/usr/bin/env python3
"""Leakage of the released values (a, b) and a differential-privacy noise trade-off (c). CPU only.

Usage (bundle root): python experiments/leakage_dp/run.py > experiments/leakage_dp/run_log.txt
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
from rx_common import NATURAL_EXCLUDE, OPP_MIX, RECORD_DIRS, load, qh  # noqa: E402
from rx_windows import SeedContext, clip_gram, conditions, stats_from_gram  # noqa: E402

DATASETS = ["mnist", "cifar10", "camelyon17"]
EPSILONS = [0.1, 0.3, 1.0, 3.0, 10.0]
DELTA = 1e-5
CACHE = HERE / "windows"
BEST = HERE.parent / "equal_fpr" / "best_rows.csv"


# ---------------------------------------------------------------- (a) site identity
def part_a(rng_seed: int = 20261023) -> pd.DataFrame:
    rows = []
    for ds in DATASETS:
        for seed in sorted(int(p.name.split("_")[1]) for p in RECORD_DIRS[ds].glob("seed_*")):
            ctx = SeedContext(ds, seed)
            for fam, sev in (("vector", 1.0), ("label_prior", 0.35)):
                X2, X3, Xp, y = [], [], [], []
                for s in range(ctx.S):
                    dirs = [0.0] * ctx.S
                    dirs[s] = 1.0
                    c = {"kind": fam, "family": fam, "severity": sev, "directions": dirs}
                    out = ctx.run(ctx.condition_feats(c), 50, seed + 6000 + s, keep_pooled=True)
                    st = stats_from_gram(out["gram"], ctx.alpha)
                    X2.append(np.log(np.c_[st["d_mix"], st["d_site"], np.maximum(st["heterogeneity"], 1e-12)]))
                    X3.append(st["contributions"])
                    Xp.append(out["pooled"])
                    y.append(np.full(50, s))
                X2, X3, Xp, y = map(np.concatenate, (X2, X3, Xp, y))
                cv = StratifiedKFold(5, shuffle=True, random_state=rng_seed)
                clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
                r = {"dataset": ds, "partition_seed": seed, "family": fam, "severity": sev, "windows": len(y), "chance": 1 / ctx.S}
                for name, X in (("L2_scalars", X2), ("L3_A_s", X3), ("SecAgg_pooled_sketch", Xp)):
                    r[name] = float(cross_val_score(clf, X, y, cv=cv).mean())
                r["L3_argmax_no_training"] = float((X3.argmax(axis=1) == y).mean())
                rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- (b) repeated L3 releases
def part_b(rng) -> pd.DataFrame:
    rows, site_rows = [], []
    for ds in DATASETS:
        rec = load(ds)
        for seed, g in rec.groupby("partition_seed"):
            cal = g[g.condition == "clean_calibration"]
            A = np.stack(cal.contributions.to_numpy())
            mc = np.load(RECORD_DIRS[ds] / f"seed_{seed}" / "monitor_clean.npz")
            for s in range(A.shape[1]):
                lab = mc[f"labels_{s:03d}"]
                p = np.bincount(lab) / lab.size
                p = p[p > 0]
                site_rows.append({"dataset": ds, "partition_seed": seed, "site": s, "mean_clean_A_s": float(A[:, s].mean()),
                                  "pool_size": int(lab.size), "class_entropy": float(-(p * np.log(p)).sum())})
    sites = pd.DataFrame(site_rows)
    for ds, g in sites.groupby("dataset"):
        for prop in ("pool_size", "class_entropy"):
            if g[prop].nunique() < 2:
                continue
            rho = spearmanr(g.mean_clean_A_s, g[prop]).statistic
            perm = np.array([spearmanr(g.mean_clean_A_s, rng.permutation(g[prop].to_numpy())).statistic for _ in range(10000)])
            within = [spearmanr(gs.mean_clean_A_s, gs[prop]).statistic for _, gs in g.groupby("partition_seed") if gs[prop].nunique() > 1]
            rows.append({"dataset": ds, "property": prop, "sites": len(g), "spearman_all_sites": float(rho),
                         "perm_p_two_sided": float((np.abs(perm) >= abs(rho)).mean()),
                         "spearman_within_seed_mean": float(np.mean(within)), "within_seed_min": float(np.min(within)),
                         "within_seed_max": float(np.max(within))})
    sites.to_csv(HERE / "b_site_properties.csv", index=False)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- (c) Gaussian mechanism
def analytic_gaussian_sigma(sens: float, eps: float, delta: float) -> float:
    """Smallest sigma of the analytic Gaussian mechanism (Balle and Wang, 2018) for L2 sensitivity `sens`."""
    def dlt(sig):
        return norm.cdf(sens / (2 * sig) - eps * sig / sens) - np.exp(eps) * norm.cdf(-sens / (2 * sig) - eps * sig / sens)
    lo, hi = 1e-12 * sens, 100.0 * sens / eps + sens
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        lo, hi = (lo, mid) if dlt(mid) <= delta else (mid, hi)
    return hi


def regenerate(ds: str, seed: int, saved: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
    """All windows of one dataset/seed with the seeds of scripts 15/33; returns (condition per window, gram, check)."""
    f = CACHE / f"{ds}_seed{seed}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=False)
        return z["condition"], z["gram"], float(z["check"])
    ctx = SeedContext(ds, seed)
    cfg = ctx.cfg
    names, grams = [], []
    specs = [("clean_calibration", ctx.clean_feats, int(cfg["calibration_windows"]), seed),
             ("clean_heldout", ctx.clean_feats, int(cfg["heldout_windows"]), seed + 1000)]
    for name, feats, w, ws in specs:
        grams.append(ctx.run(feats, w, ws)["gram"])
        names += [name] * w
    for c in conditions(ds):
        grams.append(ctx.run(ctx.condition_feats(c), int(cfg["windows_per_condition"]), seed + 2000)["gram"])
        names += [c["condition"]] * int(cfg["windows_per_condition"])
    gram, names = np.concatenate(grams), np.array(names)
    st = stats_from_gram(gram, ctx.alpha)
    # check against saved records in the same order (condition, window)
    sv = saved.set_index(["condition", "window"])
    win = np.concatenate([np.arange((names == n).sum()) for n in dict.fromkeys(names)])
    ref = sv.loc[list(zip(names, win)), ["d_mix", "d_site"]].to_numpy()
    check = float(np.abs(ref - np.c_[st["d_mix"], st["d_site"]]).max())
    CACHE.mkdir(exist_ok=True)
    np.savez_compressed(f, condition=names, gram=gram, check=check)
    return names, gram, check


def part_c() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    best = pd.read_csv(BEST)
    rows, sens_rows, checks = [], [], {}
    for ds in DATASETS:
        rec = load(ds)
        cmeta = {c["condition"]: c for c in conditions(ds)}
        best_conds = set(best[best.dataset == ds].condition)
        a1 = {n for n, c in cmeta.items() if c["geometry"] in OPP_MIX and c["family"] not in NATURAL_EXCLUDE}
        per = []                                                  # per seed and epsilon: results
        for seed, saved in rec.groupby("partition_seed"):
            names, gram, chk = regenerate(ds, seed, saved)
            checks[f"{ds}/seed_{seed}"] = chk
            S, n = gram.shape[1], 250      # window_size in both configs
            alpha = np.ones(S) / S
            cal_mask = names == "clean_calibration"
            E_cal = np.einsum("wss->ws", gram[cal_mask])
            R = float(np.sqrt(10 * max(qh(E_cal[:, s], 0.95) for s in range(S))))
            Dl = 2 * np.sqrt(2) / n
            s_E = 2 * R * Dl + Dl ** 2
            s_mix, s_site, s_A = 2 * R * alpha[0] * Dl + (alpha[0] * Dl) ** 2, alpha[0] * s_E, alpha[0] * s_E
            st = stats_from_gram(clip_gram(gram, R), alpha)
            st_raw = stats_from_gram(gram, alpha)
            clip_frac = float((np.einsum("wss->ws", gram) > R ** 2).mean())
            med_clean = float(np.median(st["d_site"][cal_mask]))
            for eps in [np.inf] + EPSILONS:
                if np.isinf(eps):
                    sig2 = sig3 = 0.0
                else:
                    sig2 = analytic_gaussian_sigma(np.hypot(s_mix, s_site), eps, DELTA)
                    sig3 = analytic_gaussian_sigma(s_A, eps, DELTA)
                sens_rows.append({"dataset": ds, "partition_seed": seed, "epsilon": eps, "R": R, "R2_over_median_clean_Dsite": R ** 2 / med_clean,
                                  "share_site_residuals_clipped": clip_frac, "sens_D_mix": s_mix, "sens_D_site": s_site, "sens_A_s": s_A,
                                  "sigma_L2": sig2, "sigma_L3": sig3, "sigma_L2_over_median_clean_Dsite": sig2 / med_clean})
                variants = [("clipped", st)] + ([("unclipped", st_raw)] if np.isinf(eps) else [])
                for vname, sv in variants:
                    rng = np.random.default_rng([seed, int(1000 * eps) if np.isfinite(eps) else 0, len(ds)])
                    W = names.size
                    dm = sv["d_mix"] + sig2 * rng.standard_normal(W)
                    dsi = sv["d_site"] + sig2 * rng.standard_normal(W)
                    A = sv["contributions"] + sig3 * rng.standard_normal((W, S))
                    t_mix, t_site = qh(dm[cal_mask], 0.95), qh(dsi[cal_mask], 0.95)
                    score = np.maximum(dm / t_mix, dsi / t_site)
                    alert = score > qh(score[cal_mask], 0.95)
                    pooled = dm > t_mix
                    held = names == "clean_heldout"
                    r = {"dataset": ds, "partition_seed": seed, "epsilon": eps, "variant": vname,
                         "fpr_joint": float(alert[held].mean()), "fpr_pooled": float(pooled[held].mean())}
                    for agg, cset in (("A1", a1), ("best", best_conds)):
                        rates = [alert[names == cn].mean() for cn in cset]
                        r[f"tpr_joint_{agg}"] = float(np.mean(rates))
                        r[f"tpr_pooled_{agg}"] = float(np.mean([pooled[names == cn].mean() for cn in cset]))
                    t1, tk = [], []
                    for cn, c in cmeta.items():
                        mask = np.asarray(c["directions"]) != 0
                        sel = (names == cn) & alert
                        if not sel.any():
                            continue
                        order = np.argsort(-A[sel], axis=1)
                        k = int(mask.sum())
                        hit = mask[order[:, :k]].sum(axis=1) / k
                        (t1 if c["geometry"] == "single" else tk).append(hit.mean())
                    r["top1_single"], r["topk_other"] = float(np.mean(t1)), float(np.mean(tk))
                    per.append(r)
        rows += per
    return pd.DataFrame(rows), pd.DataFrame(sens_rows), checks


def main() -> int:
    rng = np.random.default_rng(20261023)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    f = lambda x: f"{x:.3f}"

    a = part_a()
    a.to_csv(HERE / "a_site_identity_per_seed.csv", index=False)
    a_sum = a.groupby(["dataset", "family", "severity"])[["chance", "L2_scalars", "L3_A_s", "L3_argmax_no_training", "SecAgg_pooled_sketch"]].agg(["mean", "min", "max"])
    a_sum.to_csv(HERE / "a_site_identity.csv")
    print("== (a) Which site shifted? Cross-validated accuracy of an attacker with a labeled history (chance 0.2); mean [min, max] over seeds ==")
    print(a_sum.to_string(float_format=f))

    b = part_b(rng)
    b.to_csv(HERE / "b_repeated_L3.csv", index=False)
    print("\n== (b) Mean clean A_s vs site properties (Spearman over all sites; permutation p) ==")
    print(b.to_string(index=False, float_format=f))

    c, sens, checks = part_c()
    c.to_csv(HERE / "c_dp_per_seed.csv", index=False)
    sens.to_csv(HERE / "c_sensitivity.csv", index=False)
    (HERE / "reproduction_check.json").write_text(json.dumps({"max_abs_diff_vs_saved_records": checks}, indent=2))
    cs = c.groupby(["dataset", "variant", "epsilon"])[[k for k in c.columns if k not in ("dataset", "partition_seed", "epsilon", "variant")]].mean()
    cs.to_csv(HERE / "c_dp_tradeoff.csv")
    print("\nReproduction check (regenerated vs saved, max |diff| of D_mix, D_site):", max(checks.values()))
    print("\n== (c) Sensitivity and noise scale (mean over seeds) ==")
    print(sens.groupby(["dataset", "epsilon"])[["R2_over_median_clean_Dsite", "share_site_residuals_clipped", "sigma_L2_over_median_clean_Dsite"]]
          .mean().to_string(float_format=lambda x: f"{x:.4g}"))
    print("\n== (c) Detection and attribution with Gaussian noise on released scalars (per-window epsilon, delta = 1e-5; mean over seeds) ==")
    print(cs.to_string(float_format=f))
    print(f"\nWrote a_*.csv, b_*.csv, c_*.csv, reproduction_check.json and windows/ cache to {HERE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

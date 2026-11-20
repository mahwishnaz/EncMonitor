#!/usr/bin/env python3
"""Falsified-residual test.

What does the monitor see when one site submits a fabricated or altered residual? The protocol provides no
integrity protection; this test only measures what the released outputs show. Plaintext statistics of the same
circuit (encmonitor.statistics.compute_dual_statistics), thresholds calibrated on honest calibration windows of the
seed (the scoring code of the main detection table (Table 4 of the article; the same scoring is used for Tables 5-7)). A few windows are re-evaluated with the CKKS circuit (OpenFHE, depth 2) to confirm that the
encrypted statistics and the decisions are the same.

Data: MNIST, CIFAR-10, Camelyon17 patch split; seeds 2026-2030; window sampling identical to scripts 15/33.
Base windows (honest): clean held-out (200 per seed), and a real shift at site 0 only
(label_prior 0.35 single; vector 1.0 single; 50 windows each).
Falsifications (site f submits x instead of its honest residual δ_f):
  hide_replay  : f = 0 (the drifted site) submits a residual of one of its own clean calibration windows
  zero         : f = 0 submits the zero vector
  inflate_x3   : f = 1 (honest, unshifted) submits 3·δ_1
  random       : f = 1 submits a random direction with norm 2 × the 95th pct of its clean residual norm
  cancel       : f = 1 submits −δ_0 (insider who knows site 0's residual; worst case for the pooled score)
  fake_shift   : f = 1 submits δ_1 + Δ, Δ = mean residual of site 1 under the vector shift η = 1 (fabricated drift)
Measured: alert rates (Pooled, EncMonitor joint, Local-any), top-1 attribution A_s among alerted windows (share
pointing at the falsifying site), noise-corrected coherence C_corr (Section 5 of the article; C_corr < 0.5 =
cancellation flag of experiments/h_diagnostic), H/D_site, and a lower-tail check D_site < 1st pct of calibration.
Writes outputs/analysis/falsified/{per_seed.csv, summary.csv, ckks_check.json}.
Usage: python analysis/falsified_residual.py
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

import shared as c2

OUTD = c2.RESULTS / "falsified"
SCEN = ["honest", "hide_replay", "zero", "inflate_x3", "random", "cancel", "fake_shift"]
BASES = {"clean": None, "label_prior0.35_single": ("label_prior", 0.35), "vector1_single": ("vector", 1.0)}


def residuals(ctx, feats, windows, wseed):
    """Same draws as rx_windows.SeedContext.run; returns (W, S, m) residuals."""
    rng = np.random.default_rng(wseed)
    out = np.empty((windows, ctx.S, ctx.base.shape[1]))
    for w in range(windows):
        for s in range(ctx.S):
            N = feats[s].shape[0]
            out[w, s] = feats[s][rng.choice(N, ctx.n, replace=N < ctx.n)].mean(axis=0) - ctx.base[s]
    return out


def stats(R, a):
    pooled = np.einsum("s,wsm->wm", a, R)
    E = np.einsum("wsm,wsm->ws", R, R)
    return {"d_mix": np.einsum("wm,wm->w", pooled, pooled), "d_site": E @ a, "E": E}


def falsify(R, scen, cal_R, delta_fake, q95_norm1, rng):
    R = R.copy()
    W = R.shape[0]
    if scen == "hide_replay":
        R[:, 0] = cal_R[rng.integers(0, cal_R.shape[0], W), 0]
    elif scen == "zero":
        R[:, 0] = 0.0
    elif scen == "inflate_x3":
        R[:, 1] = 3.0 * R[:, 1]
    elif scen == "random":
        v = rng.standard_normal(R[:, 1].shape)
        R[:, 1] = v / np.linalg.norm(v, axis=1, keepdims=True) * 2.0 * q95_norm1
    elif scen == "cancel":
        R[:, 1] = -R[:, 0]
    elif scen == "fake_shift":
        R[:, 1] = R[:, 1] + delta_fake
    return R


def main() -> int:
    OUTD.mkdir(parents=True, exist_ok=True)
    rows, ckks_inputs = [], []
    rng = np.random.default_rng(20261113)
    for name, ds in (("mnist", "mnist"), ("cifar10", "cifar10"), ("camelyon17_patch", "camelyon17")):
        for seed in c2.OLD:
            ctx = c2.Ctx(ds, seed, c2.SETS[name][1])
            S, a = ctx.S, ctx.alpha
            cal = residuals(ctx, ctx.clean_feats, 500, seed)
            st = stats(cal, a)
            q = lambda v: c2.rc.qh(v, 0.95)
            t_mix, t_site = q(st["d_mix"]), q(st["d_site"])
            c_j = q(np.maximum(st["d_mix"] / t_mix, st["d_site"] / t_site))
            tau_s = np.array([q(st["E"][:, s]) for s in range(S)])
            tau_lm = q((st["E"] / tau_s).max(axis=1))
            m0_mix, e0 = st["d_mix"].mean(), st["E"].mean(axis=0)
            dsite_p01 = float(np.quantile(st["d_site"], 0.01))
            q95_norm1 = float(np.quantile(np.sqrt(st["E"][:, 1]), 0.95))
            # fabricated drift for site 1: mean residual of site 1 under the vector shift eta = 1
            fake = residuals(ctx, [ctx.site_feats("vector", "vector", 1.0, 1.0 if s == 1 else 0.0, s) for s in range(S)], 50, seed + 2000)
            delta_fake = fake[:, 1].mean(axis=0)
            for base, spec in BASES.items():
                if spec is None:
                    R0 = residuals(ctx, ctx.clean_feats, 200, seed + 1000)
                else:
                    fam, sev = spec
                    feats = [ctx.site_feats(fam, fam, sev, 1.0 if s == 0 else 0.0, s) for s in range(S)]
                    R0 = residuals(ctx, feats, 50, seed + 2000)
                for scen in SCEN:
                    R = R0 if scen == "honest" else falsify(R0, scen, cal, delta_fake, q95_norm1, rng)
                    s_ = stats(R, a)
                    enc = np.maximum(s_["d_mix"] / t_mix, s_["d_site"] / t_site) > c_j
                    pooled = s_["d_mix"] > t_mix
                    local = (s_["E"] / tau_s).max(axis=1) > tau_lm
                    A = s_["E"] * a
                    top = A.argmax(axis=1)
                    f = 0 if scen in ("hide_replay", "zero") else 1
                    e = np.maximum(s_["E"] - e0, 0.0)
                    den = (a * np.sqrt(e)).sum(axis=1) ** 2
                    ccorr = np.clip(np.where(den > 0, np.maximum(s_["d_mix"] - m0_mix, 0.0) / np.where(den > 0, den, 1), 1.0), 0, 1)
                    rows.append({"set": name, "seed": seed, "base": base, "scenario": scen, "falsifier_site": f,
                                 "windows": len(R), "pooled_alert": pooled.mean(), "encmon_alert": enc.mean(),
                                 "localany_alert": local.mean(),
                                 "falsifier_local_bit": (s_["E"][:, f] / tau_s[f] > tau_lm).mean(),
                                 "top1_is_falsifier_given_alert": (top[enc] == f).mean() if enc.any() else np.nan,
                                 "top1_is_site0_given_alert": (top[enc] == 0).mean() if enc.any() else np.nan,
                                 "ccorr_lt_0.5_given_alert": (ccorr[enc] < 0.5).mean() if enc.any() else np.nan,
                                 "median_ccorr_given_alert": float(np.median(ccorr[enc])) if enc.any() else np.nan,
                                 "median_H_over_Dsite": float(np.median((s_["d_site"] - s_["d_mix"]) / s_["d_site"])),
                                 "dsite_below_cal_p01": (s_["d_site"] < dsite_p01).mean(),
                                 "median_falsifier_energy_over_tau": float(np.median(s_["E"][:, f] / tau_s[f]))})
                    if seed == 2026 and name == "mnist" and base != "clean" or (seed == 2026 and name == "mnist" and scen in ("inflate_x3", "random")):
                        ckks_inputs.append((name, base, scen, R[:3], (t_mix, t_site, c_j)))
            print(f"[{name} {seed}] done", flush=True)
    df = pd.DataFrame(rows)
    c2.write_csv(df, OUTD / "per_seed.csv", "falsified-residual test, per seed")
    cols = [c for c in df.columns if c not in ("set", "seed", "base", "scenario", "falsifier_site", "windows")]
    summ = df.groupby(["set", "base", "scenario"], sort=False)[cols].mean().reset_index()
    c2.write_csv(summ, OUTD / "summary.csv", "falsified-residual test, mean over seeds 2026-2030")
    ckks_check(ckks_inputs)
    return 0


def ckks_check(items):
    """Re-evaluate a few falsified windows with the CKKS circuit (encmonitor.he) and compare with plaintext."""
    from encmonitor.he.backend import OpenFHEBackend
    from encmonitor.he.circuit import evaluate_dual_statistics
    from encmonitor.he.context import build_context, decrypt_scalar, encrypt_vector
    from encmonitor.statistics import compute_dual_statistics
    m = items[0][3].shape[2]
    b = build_context(m, multiplicative_depth=2, scaling_mod_size=50)
    be = OpenFHEBackend(b.cc)
    out, max_err, agree, n = [], 0.0, 0, 0
    for name, base, scen, R, (t_mix, t_site, c_j) in items:
        for w in range(R.shape[0]):
            ref = compute_dual_statistics(R[w], np.ones(R.shape[1]))
            o = evaluate_dual_statistics(be, [encrypt_vector(b, x) for x in R[w]], np.ones(R.shape[1]), width=m, disclose_contributions=True)
            v = [decrypt_scalar(b, o.d_mix), decrypt_scalar(b, o.d_site), decrypt_scalar(b, o.heterogeneity)]
            Acts = [decrypt_scalar(b, x) for x in o.contributions]
            err = max(abs(v[0] - ref.d_mix), abs(v[1] - ref.d_site), abs(v[2] - ref.heterogeneity),
                      max(abs(x - y) for x, y in zip(Acts, ref.contributions)))
            dec_he = max(v[0] / t_mix, v[1] / t_site) > c_j
            dec_pt = max(ref.d_mix / t_mix, ref.d_site / t_site) > c_j
            top_same = int(np.argmax(Acts)) == int(np.argmax(ref.contributions))
            max_err = max(max_err, err)
            agree += int(dec_he == dec_pt and top_same)
            n += 1
            out.append({"set": name, "base": base, "scenario": scen, "window": w, "max_abs_error": err,
                        "alert_he": bool(dec_he), "alert_plain": bool(dec_pt), "top1_same": top_same})
    c2.json_dump({"windows": n, "max_abs_error": max_err, "decision_and_top1_agreement": agree / n, "rows": out,
                  "ring_dimension": b.ring_dimension}, OUTD / "ckks_check.json")
    print(f"CKKS check: {n} windows, max abs error {max_err:.2e}, agreement {agree}/{n}")


if __name__ == "__main__":
    sys.exit(main())

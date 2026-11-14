#!/usr/bin/env python3
"""Chronological streams with the complete 2-of-3 confirmation and reset policy.

Streams are built at the level of records (individual rows / patches) arriving at each site; a window is n
consecutive records of every site. Two window layouts: non-overlapping (step n) and 50 % overlapping (step n/2,
consecutive windows share half of their records, so their statistics are autocorrelated).
Policy per method (applied after decryption): a window alerts if the method's rule fires (thresholds of the seed,
calibrated on its 500 clean calibration windows exactly as in Table 1); an alarm is CONFIRMED when at least 2 of the
last 3 windows alerted; after a confirmed alarm the 3-window history is cleared (reset) and monitoring continues.
Disclosure (default policy L2 every window, L3 after a global alert): L3 frequency = share of windows with an alert;
also reported: L3 only after a confirmed alarm.

(a) Census (n = 250 and n = 1,000): each site's records come from its own state's pools in chronological order
    2014 (monitoring pool) → 2015 → 2016 → 2017 → 2018, equal stream time per year (real drift). Injected event at
    the middle of the stream (inside 2016): opposing label-prior shift q = 0.35 at sites 0 (+) and 1 (−), applied to
    the current year's pool (same resampling as the census windows). No-shift streams: the same chronology without
    injection, and a stationary 2014-only stream for the empirical ARL0.
(b) Camelyon17 patient-level split (patient-disjoint reference and monitoring pools): stationary clean stream for the
    ARL0, and streams with an injected event at the middle: patient exchange q = 0.5 opposing (sites 0 and 1 swap
    patients) and stain 0.2 coherent (sites 0 and 1).
Every window is a uniform without-replacement sample of n records of the active pool, exactly as a calibration
window of the window runners; non-overlapping windows are drawn independently, and overlapping windows share half
of their records (the only dependence between windows). Records are generated in blocks of `step` records; a year
or event change takes effect at the first block that starts at or after it.
Event streams: 60 windows, event at window 30 (record position 30·step); delay = windows from the first window
that contains shifted records to the confirmed alarm; missed = no confirmation by the end of the stream (≥ 29
windows after the event). No-shift streams: 400 windows; empirical ARL0 = mean index of the first confirmed alarm
(censored at 400, count reported); false alarms per 1,000 windows with reset.
Replications: every seed with records (census 20, patient split 20) × 25 streams per layout and stream type.
Writes outputs/analysis/streams/{per_stream.csv, summary.csv, autocorr.csv}.
Usage: python analysis/streams.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import shared as c2

OUTD = c2.RESULTS / "streams"
REPS = 25
T_EVENT, T0, T_NULL = 60, 30, 400
METHODS = ["pooled", "encmon", "localany", "hybrid"]


class Calib:
    def __init__(self, G):   # G: calibration Gram matrices (500, S, S)
        S = G.shape[1]
        self.a = np.ones(S) / S
        E = np.einsum("wss->ws", G)
        dm = np.einsum("s,wst,t->w", self.a, G, self.a)
        ds = E @ self.a
        q = lambda v: c2.rc.qh(v, 0.95)
        self.t_mix, self.t_site = q(dm), q(ds)
        self.c = q(np.maximum(dm / self.t_mix, ds / self.t_site))
        self.tau_s = np.array([q(E[:, s]) for s in range(S)])
        self.tau_lm = q((E / self.tau_s).max(axis=1))
        self.c_h = q(np.maximum(dm / self.t_mix, (E / self.tau_s).max(axis=1)))

    def alerts(self, R):   # R: (W, S, m) residuals
        pooled = np.einsum("s,wsm->wm", self.a, R)
        dm = np.einsum("wm,wm->w", pooled, pooled)
        E = np.einsum("wsm,wsm->ws", R, R)
        ds = E @ self.a
        lm = (E / self.tau_s).max(axis=1)
        return {"pooled": dm > self.t_mix,
                "encmon": np.maximum(dm / self.t_mix, ds / self.t_site) > self.c,
                "localany": lm > self.tau_lm,
                "hybrid": ((E > self.c_h * self.tau_s).any(axis=1)) | (dm > self.c_h * self.t_mix),
                "encmon_score": np.maximum(dm / self.t_mix, ds / self.t_site)}


def confirm(bits):
    """2-of-3 confirmation with reset; returns the indices of the windows at which an alarm is confirmed."""
    out, hist = [], []
    for t, b in enumerate(bits):
        hist = (hist + [bool(b)])[-3:]
        if sum(hist) >= 2:
            out.append(t)
            hist = []
    return out


def stream_residuals(pool_of, base, n, step, T, rng):
    """pool_of(s, p) -> feature array of the pool for site s at record position p (constant within segments).
    Returns (T, S, m) residuals of windows [k*step, k*step + n)."""
    # Records arrive in blocks of `step` records from the pool that is active at the block's first record (segment
    # changes are aligned to blocks). Each block is drawn WITHOUT replacement (as every window of the window runners:
    # rng.choice(N, n, replace=N < n)); with 50 % overlap a block also excludes the records of the previous block
    # (same pool), so that every window of n records is a uniform without-replacement sample of the pool, i.e. it
    # has exactly the distribution of a calibration window. Window k = mean of blocks k .. k + n/step - 1.
    S, m = base.shape
    r = n // step
    nb = T - 1 + r
    R = np.empty((T, S, m))
    for s in range(S):
        B = np.empty((nb, m))
        prev_F, prev_idx = None, None
        for b in range(nb):
            F, _ = pool_of(s, b * step)
            N = F.shape[0]
            if r > 1 and F is prev_F:
                mask = np.ones(N, bool)
                mask[prev_idx] = False
                idx = rng.choice(np.flatnonzero(mask), step, replace=False)
            else:
                idx = rng.choice(N, step, replace=N < step)
            B[b] = F[idx].mean(axis=0)
            prev_F, prev_idx = F, idx
        cs = np.vstack([np.zeros((1, m)), np.cumsum(B, axis=0)])
        R[:, s] = (cs[r:r + T] - cs[:T]) / r - base[s]
    return R


def evaluate(cal, R, event_window=None):
    al = cal.alerts(R)
    out = {}
    for k in METHODS:
        bits = al[k]
        conf = confirm(bits)
        r = {"alert_rate": float(bits.mean()), "confirmed": len(conf),
             "L3_after_alert": float(bits.mean()), "L3_after_confirmed": len(conf) / len(bits)}
        if event_window is None:
            r["first_confirm"] = conf[0] + 1 if conf else np.nan
        else:
            pre = [t for t in conf if t < event_window]
            post = [t for t in conf if t >= event_window]
            r["false_alarms_before_event"] = len(pre)
            r["delay"] = post[0] - event_window + 1 if post else np.nan
            r["missed"] = not post
            r["alert_rate_before_event"] = float(bits[:event_window].mean())
        out[k] = r
    return out, al


def lag1(x):
    x = np.asarray(x, float)
    x = x - x.mean()
    d = (x * x).sum()
    return float((x[1:] * x[:-1]).sum() / d) if d > 0 else np.nan


def census_part(n, rows, ac):
    for seed in c2.seeds_of(f"census_n{n}"):
        ctx = c2.CensusCtx(seed, n)
        cal = Calib(ctx.run(ctx.clean, 500, seed))
        W = ctx.W
        years = [2014] + W.YEARS
        feats = {(s, y): ctx.clean[s] if y == 2014 else ctx.feats(ctx.data[y]["rep"][ctx.pools[(s, y)]].astype(np.float64))
                 for s in range(ctx.S) for y in years}
        labels = {(s, y): ctx.mon_lab[s] if y == 2014 else ctx.data[y]["label"][ctx.pools[(s, y)]] for s in range(ctx.S) for y in years}
        raw = {(s, y): ctx.mon_raw[s] if y == 2014 else ctx.data[y]["rep"][ctx.pools[(s, y)]].astype(np.float64)
               for s in range(ctx.S) for y in years}
        shifted = {}
        for s, f in ((0, 1.0), (1, -1.0)):
            for y in years:
                idx = c2.rw.label_prior_idx(labels[(s, y)], 1, 0.35, np.sign(f), np.random.default_rng([seed, 3, s, 350, int(f > 0), y]))
                shifted[(s, y)] = ctx.feats(raw[(s, y)][idx])
        rng = np.random.default_rng([seed, 77, n])
        for step_name, step in (("non-overlapping", n), ("50% overlap", n // 2)):
            for kind in ("stationary_2014", "chronological_no_event", "chronological_event"):
                T = T_NULL if kind != "chronological_event" else T_EVENT
                L = (T - 1) * step + n
                seg = L / 5.0
                P0 = T0 * step

                def pool_of(s, p, kind=kind, seg=seg, P0=P0, L=L):
                    if kind == "stationary_2014":
                        return feats[(s, 2014)], L
                    yi = min(int(p // seg), 4)
                    end = int(np.ceil((yi + 1) * seg)) if yi < 4 else L
                    y = years[yi]
                    if kind == "chronological_event" and s in (0, 1):
                        if p >= P0:
                            return shifted[(s, y)], end
                        end = min(end, P0)
                    return feats[(s, y)], end

                for rep in range(REPS):
                    R = stream_residuals(pool_of, ctx.base, n, step, T, rng)
                    ev = T0 - (n // step - 1) if kind == "chronological_event" else None   # first window with shifted records
                    res, al = evaluate(cal, R, ev)
                    for k, r in res.items():
                        rows.append({"data": f"census_n{n}", "seed": seed, "layout": step_name, "stream": kind, "rep": rep,
                                     "method": k, "windows": T, **r})
                    if rep == 0 and kind == "stationary_2014":
                        ac.append({"data": f"census_n{n}", "seed": seed, "layout": step_name,
                                   "lag1_encmon_score": lag1(al["encmon_score"]), "lag1_encmon_alert": lag1(al["encmon"])})
        print(f"[census n={n} {seed}] streams done", flush=True)


def camelyon_part(rows, ac):
    conds = {c["condition"]: c for c in c2.rw.conditions("camelyon17")}
    events = {"exchange0.5__opposing": conds["exchange0.5__opposing"], "stain0.2__coherent": conds["stain0.2__coherent"]}
    n = 250
    for seed in c2.seeds_of("camelyon17_patient"):
        ctx = c2.Ctx("camelyon17", seed, c2.SETS["camelyon17_patient"][1])
        cal = Calib(ctx.run(ctx.clean_feats, 500, seed)["gram"])
        shifted = {e: ctx.condition_feats(c) for e, c in events.items()}
        rng = np.random.default_rng([seed, 78])
        for step_name, step in (("non-overlapping", n), ("50% overlap", n // 2)):
            for kind in ["stationary"] + [f"event:{e}" for e in events]:
                T = T_NULL if kind == "stationary" else T_EVENT
                L = (T - 1) * step + n
                P0 = T0 * step

                def pool_of(s, p, kind=kind, P0=P0, L=L):
                    if kind == "stationary":
                        return ctx.clean_feats[s], L
                    sh = shifted[kind.split(":")[1]][s]
                    return (sh, L) if p >= P0 else (ctx.clean_feats[s], P0)

                for rep in range(REPS):
                    R = stream_residuals(pool_of, ctx.base, n, step, T, rng)
                    ev = None if kind == "stationary" else T0 - (n // step - 1)
                    res, al = evaluate(cal, R, ev)
                    for k, r in res.items():
                        rows.append({"data": "camelyon17_patient", "seed": seed, "layout": step_name, "stream": kind,
                                     "rep": rep, "method": k, "windows": T, **r})
                    if rep == 0 and kind == "stationary":
                        ac.append({"data": "camelyon17_patient", "seed": seed, "layout": step_name,
                                   "lag1_encmon_score": lag1(al["encmon_score"]), "lag1_encmon_alert": lag1(al["encmon"])})
        print(f"[camelyon patient {seed}] streams done", flush=True)


def summarize(df):
    out = []
    for (data, layout, stream, k), g in df.groupby(["data", "layout", "stream", "method"], sort=False):
        r = {"data": data, "layout": layout, "stream": stream, "method": k, "streams": len(g), "seeds": g.seed.nunique(),
             "windows_per_stream": int(g.windows.iloc[0]), "alert_rate_per_window": g.alert_rate.mean(),
             "L3_share_after_alert": g.L3_after_alert.mean(), "L3_share_after_confirmed": g.L3_after_confirmed.mean()}
        if "missed" in g and g["missed"].notna().any():
            r.update({"missed_event_rate": g.missed.mean(), "mean_delay_windows": g.delay.mean(),
                      "median_delay_windows": g.delay.median(), "false_alarms_before_event_per_stream": g.false_alarms_before_event.mean(),
                      "alert_rate_before_event": g.alert_rate_before_event.mean()})
        else:
            fc = g.first_confirm
            r.update({"empirical_ARL0_mean_first_confirm": fc.mean(), "streams_without_alarm_censored": int(fc.isna().sum()),
                      "confirmed_alarms_per_1000_windows": 1000 * g.confirmed.sum() / (g.windows.sum()),
                      "empirical_ARL0_windows_per_confirmed_alarm": g.windows.sum() / max(g.confirmed.sum(), 1),
                      "markov_ARL0_from_window_rate": c2.arl_2of3(g.alert_rate.mean())})
        out.append(r)
    return pd.DataFrame(out)


def main() -> int:
    OUTD.mkdir(parents=True, exist_ok=True)
    rows, ac = [], []
    for n in (250, 1000):
        census_part(n, rows, ac)
    camelyon_part(rows, ac)
    df = pd.DataFrame(rows)
    c2.write_csv(df, OUTD / "per_stream.csv", "streams, one row per stream and method")
    c2.write_csv(summarize(df), OUTD / "summary.csv", "streams summary")
    c2.write_csv(pd.DataFrame(ac), OUTD / "autocorr.csv", "lag-1 autocorrelation in stationary streams (rep 0)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

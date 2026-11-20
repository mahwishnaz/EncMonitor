#!/usr/bin/env python3
"""Cost for larger consortia, S = 5, 10, 20, 50 (m = 64, the article's CKKS profile: depth 2,
scaling 50 bits, HEStd_128_classic, hybrid key switching; build_context of the release).

MEASURED with the CKKS implementation for every S (CPU only, OpenFHE default threads, as in
private_localany/ckks.py): residual ciphertext size, result ciphertext size, evaluation-key sizes, server time of
the full circuit (D_mix, D_site, H and the S contributions A_s), per-site encryption time, decryption time of the
3 global scalars and of the S contributions; encrypted Local-any (encrypted alert bits, CKKS depth 0 and BGV,
as in private_localany/ckks.py): bit ciphertext size, server time of the S−1 additions, decryption of the count.
Times are taken only if the 1-minute load average is below MAX_LOAD (shared server rule); otherwise only sizes
and operation counts are written and the time fields are marked "not timed".
Residual values do not affect the cost; the windows are random vectors with the norm of real residuals.

Also (semi-synthetic) check of the k/sqrt(S) rule for S = 5, 10, 20, 50: virtual sites draw their clean residual
from the pool of real MNIST clean calibration residuals (seeds 2026-2030, all sites), k sites get the same added
shift Δ; z = (mean shifted − mean clean)/SD clean for D_mix and D_site; ratio vs k/sqrt(S).
Writes outputs/analysis/cost/{cost.csv, cost.json, ksqrt.csv}.
Usage: python analysis/cost_projection.py   (CPU only; time it on an otherwise idle machine)
Note: the depth-0 CKKS bit context of private_localany/ckks.py (first modulus 35 bits, scaling 25 bits) decodes the
alert count exactly for every S used here; `count_decrypted_exactly` records this per S (random alert bits, about
S/2 alerts per window). BGV (t = 65537) is exact.
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

import shared as c2

OUTD = c2.RESULTS / "cost"
MAX_LOAD = 4.0
SITES = [5, 10, 20, 50]
REPS = 10


def ser_size(fhe, obj) -> int:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x"
        fhe.SerializeToFile(str(p), obj, fhe.BINARY)
        return p.stat().st_size


def med(x):
    return float(np.median(x))


def cost_part():
    import openfhe as fhe
    from encmonitor.he.backend import OpenFHEBackend
    from encmonitor.he.circuit import evaluate_dual_statistics
    from encmonitor.he.context import build_context, decrypt_scalar, encrypt_vector
    P1C = c2._load("p1ckks", c2.CLONE / "experiments/private_localany/ckks.py")
    m = 64
    load0 = os.getloadavg()
    timed = load0[0] < MAX_LOAD
    b = build_context(m, multiplicative_depth=2, scaling_mod_size=50)
    be = OpenFHEBackend(b.cc)
    keys = P1C.key_sizes(b.cc, b.key_pair, mult=True, rot=True)
    rng = np.random.default_rng(5)
    rows = []
    for S in SITES:
        w = np.ones(S)
        R = rng.standard_normal((REPS, S, m)) * 0.02
        te, ts, td3, tdA = [], [], [], []
        for r in range(REPS):
            t0 = time.perf_counter(); cts = [encrypt_vector(b, x) for x in R[r]]; te.append((time.perf_counter() - t0) / S)
            cnt = P1C.Counting(be) if r == 0 else be
            t0 = time.perf_counter(); o = evaluate_dual_statistics(cnt, cts, w, width=m, disclose_contributions=True); ts.append(time.perf_counter() - t0)
            if r == 0:
                ops = dict(cnt.n)
                ct_bytes, res_bytes = ser_size(fhe, cts[0]), ser_size(fhe, o.d_mix)
            t0 = time.perf_counter(); [decrypt_scalar(b, x) for x in (o.d_mix, o.d_site, o.heterogeneity)]; td3.append(time.perf_counter() - t0)
            t0 = time.perf_counter(); [decrypt_scalar(b, x) for x in o.contributions]; tdA.append(time.perf_counter() - t0)
        row = {"method": "EncMonitor (CKKS N=%d)" % b.ring_dimension, "sites": S, "m": m,
               "ciphertext_bytes_per_site": ct_bytes, "upload_bytes_per_window": S * ct_bytes,
               "result_ciphertexts": 3 + S, "result_bytes_per_window": (3 + S) * res_bytes,
               "eval_key_bytes_total": keys.get("mult_key_bytes", 0) + keys.get("rotation_key_bytes", 0),
               "public_key_bytes": keys["public_key_bytes"], "server_ops": json.dumps(ops), "timed": timed}
        if timed:
            row.update({"encrypt_sec_per_site": med(te), "server_sec_per_window": med(ts),
                        "decrypt_3_scalars_sec": med(td3), "decrypt_S_contributions_sec": med(tdA)})
        rows.append(row)
        print(f"EncMonitor S={S} done", flush=True)
    for kind in ("ckks_d0", "bgv_t65537_d0"):
        cc, kp = P1C.bit_context(kind)
        for S in SITES:
            bits = rng.integers(0, 2, (REPS, S))
            te, ts, td, exact = [], [], [], 0
            for r in range(REPS):
                t0 = time.perf_counter(); cts = [P1C.enc_bit(cc, kp, kind, x) for x in bits[r]]; te.append((time.perf_counter() - t0) / S)
                t0 = time.perf_counter(); tot = cts[0]
                for ct in cts[1:]:
                    tot = cc.EvalAdd(tot, ct)
                ts.append(time.perf_counter() - t0)
                t0 = time.perf_counter(); K = P1C.dec_count(cc, kp, kind, tot); td.append(time.perf_counter() - t0)
                exact += int(K == int(bits[r].sum()))
            cb = ser_size(fhe, cts[0])
            row = {"method": f"Local-any bits ({kind}, N={cc.GetRingDimension()})", "sites": S, "m": m,
                   "ciphertext_bytes_per_site": cb, "upload_bytes_per_window": S * cb, "result_ciphertexts": 1,
                   "result_bytes_per_window": ser_size(fhe, tot), "eval_key_bytes_total": 0,
                   "public_key_bytes": ser_size(fhe, kp.publicKey), "server_ops": json.dumps({"add": S - 1}), "timed": timed,
                   "count_decrypted_exactly": f"{exact}/{REPS}"}
            if timed:
                row.update({"encrypt_sec_per_site": med(te), "server_sec_per_window": med(ts), "decrypt_3_scalars_sec": med(td)})
            rows.append(row)
        print(f"Local-any {kind} done", flush=True)
    return rows, {"load_before": load0, "load_after": os.getloadavg(), "timed": timed, "max_load": MAX_LOAD,
                  "ring_dimension": b.ring_dimension, "reps": REPS, **keys}


def ksqrt_part():
    pool = []
    for seed in c2.OLD:
        ctx = c2.Ctx("mnist", seed, c2.SETS["mnist"][1])
        rng = np.random.default_rng(seed)
        for w in range(500):
            for s in range(ctx.S):
                N = ctx.clean_feats[s].shape[0]
                pool.append(ctx.clean_feats[s][rng.choice(N, ctx.n, replace=N < ctx.n)].mean(0) - ctx.base[s])
    pool = np.asarray(pool)
    sd_res = float(np.sqrt((pool ** 2).sum(1).mean()))
    rng = np.random.default_rng(20261113)
    rows = []
    v = rng.standard_normal(pool.shape[1]); v /= np.linalg.norm(v)
    for S in SITES:
        for k in sorted({1, 2, 3, 5, max(1, S // 2), S}):
            if k > S:
                continue
            delta = 0.5 * sd_res * v                       # shift size ~ half the RMS clean residual norm
            W = 4000
            Rc = pool[rng.integers(0, len(pool), (W, S))]
            Rs = pool[rng.integers(0, len(pool), (W, S))].copy()
            Rs[:, :k] += delta
            f = lambda R: (np.einsum("wm,wm->w", R.mean(1), R.mean(1)), np.einsum("wsm,wsm->w", R, R) / S)
            mc, sc_ = f(Rc)
            ms, ss = f(Rs)
            z_mix = (ms.mean() - mc.mean()) / mc.std()
            z_site = (ss.mean() - sc_.mean()) / sc_.std()
            rows.append({"sites": S, "k": k, "z_mix": z_mix, "z_site": z_site, "ratio": z_mix / z_site,
                         "predicted_k_over_sqrtS": k / math.sqrt(S), "windows": W})
    return rows


def main() -> int:
    OUTD.mkdir(parents=True, exist_ok=True)
    rows, meta = cost_part()            # timing first, before any other CPU work of this script
    c2.write_csv(pd.DataFrame(rows), OUTD / "cost.csv", "CKKS cost per window vs number of sites")
    c2.json_dump(meta, OUTD / "cost.json")
    ks = pd.DataFrame(ksqrt_part())
    c2.write_csv(ks, OUTD / "ksqrt.csv", "semi-synthetic check of z(D_mix)/z(D_site) = k/sqrt(S) (MNIST clean residual pool)")
    print(ks.round(3).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

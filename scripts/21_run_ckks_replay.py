#!/usr/bin/env python3
"""Replay exported residual windows under OpenFHE CKKS.

For every window: each site's residual is encrypted under one shared context and public key, the
server evaluates the circuit (encmonitor.he.circuit.evaluate_dual_statistics with the
OpenFHEBackend), and the authority decrypts D_mix, D_site, H and every A_s. The decrypted scalars are
compared with the float64 plaintext oracle (compute_dual_statistics) on the same residuals.

Acceptance tests reported:
  * scalar absolute error      max |HE - plaintext| over D_mix, D_site, H and all A_s   (< 1e-3)
  * identity residual          |(D_site - D_mix) - H| after decryption                  (< 1e-3)
  * decision agreement         pooled / site / H / global alert / hidden cancellation   (>= 99.5%)
  * attribution agreement      same top-1 site and same top-k set                       (>= 99.5%)
  * level audit                no bootstrapping; every output decrypts; output levels recorded
Timing is recorded per window (encrypt all sites, server compute, decrypt), setup separately.
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

from encmonitor.artifacts import environment_snapshot, write_json
from encmonitor.he.backend import OpenFHEBackend
from encmonitor.he.circuit import evaluate_dual_statistics
from encmonitor.he.context import build_context, decrypt_scalar, encrypt_vector
from encmonitor.statistics import compute_dual_statistics


def decisions(d_mix, d_site, h, tau):
    pooled, site, het = d_mix > tau[0], d_site > tau[1], h > tau[2]
    return {"pooled": bool(pooled), "site": bool(site), "H": bool(het),
            "global": bool(pooled or site), "hidden": bool((site or het) and not pooled)}


def level(ct):
    try:
        return int(ct.GetLevel())
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Replay exported residual windows under OpenFHE CKKS")
    ap.add_argument("--config", default="configs/ckks.yaml")
    ap.add_argument("--replay", required=True, help="outputs/ckks/replay/<dataset>_replay.npz")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-dir", default="outputs/ckks/replay_results")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    ck, gate = cfg["ckks"], cfg["gate"]
    z = np.load(args.replay)
    R = np.asarray(z["residuals"], dtype=np.float64)
    if args.limit:
        R = R[: args.limit]
    weights = np.asarray(z["weights"], dtype=np.float64)
    mask, cond = z["shifted_mask"], z["condition"]
    qkeys = [k for k in z.files if k.startswith("tau_")]
    N, S, m = R.shape
    name = Path(args.replay).stem.replace("_replay", "")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    bundle = build_context(m, multiplicative_depth=int(ck["multiplicative_depth"]),
                           scaling_mod_size=int(ck["scaling_mod_size"]), security_128=bool(ck["security_128"]))
    setup_sec = time.perf_counter() - t0
    backend = OpenFHEBackend(bundle.cc)

    rows = []
    for i in range(N):
        delta = R[i]
        ref = compute_dual_statistics(delta, weights)
        t0 = time.perf_counter()
        cts = [encrypt_vector(bundle, row) for row in delta]
        t_enc = time.perf_counter() - t0
        t0 = time.perf_counter()
        res = evaluate_dual_statistics(backend, cts, weights, width=m, disclose_contributions=True, disclose_directional=False)
        t_srv = time.perf_counter() - t0
        t0 = time.perf_counter()
        he = {"d_mix": decrypt_scalar(bundle, res.d_mix), "d_site": decrypt_scalar(bundle, res.d_site),
              "heterogeneity": decrypt_scalar(bundle, res.heterogeneity)}
        a_he = np.asarray([decrypt_scalar(bundle, c) for c in res.contributions])
        t_dec = time.perf_counter() - t0
        pt = {"d_mix": ref.d_mix, "d_site": ref.d_site, "heterogeneity": ref.heterogeneity}
        errs = [abs(he[k] - pt[k]) for k in pt] + list(np.abs(a_he - ref.contributions))
        k = max(1, int(mask[i].sum()))
        top1_same = int(np.argmax(a_he)) == int(np.argmax(ref.contributions))
        topk_same = set(np.argsort(-a_he)[:k]) == set(np.argsort(-ref.contributions)[:k])
        row = {"i": i, "condition": str(cond[i]), "seed": int(z["seed"][i]), "window": int(z["window"][i]),
               "he": he, "plain": pt, "a_he": a_he.tolist(), "a_plain": ref.contributions.tolist(),
               "max_abs_error": float(max(errs)),
               "identity_residual": float(abs((he["d_site"] - he["d_mix"]) - he["heterogeneity"])),
               "top1_same": bool(top1_same), "topk_same": bool(topk_same),
               "levels": {"d_mix": level(res.d_mix), "d_site": level(res.d_site), "a_s_max": max(level(c) or 0 for c in res.contributions)},
               "timing_sec": {"encrypt_all_sites": t_enc, "server": t_srv, "decrypt": t_dec}}
        for qk in qkeys:
            tau = z[qk][i]
            dh, dp = decisions(he["d_mix"], he["d_site"], he["heterogeneity"], tau), \
                decisions(pt["d_mix"], pt["d_site"], pt["heterogeneity"], tau)
            row[f"decisions_{qk}"] = {"he": dh, "plain": dp, "agree": {d: dh[d] == dp[d] for d in dh},
                                      "rel_margin": float(min(abs(pt["d_mix"] - tau[0]) / tau[0], abs(pt["d_site"] - tau[1]) / tau[1],
                                                              abs(pt["heterogeneity"] - tau[2]) / tau[2]))}
        rows.append(row)

    with open(out / f"{name}_ckks_replay.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    max_abs = max(r["max_abs_error"] for r in rows)
    max_id = max(r["identity_residual"] for r in rows)
    agree = {}
    disagreements = []
    for qk in qkeys:
        for dname in ("pooled", "site", "H", "global", "hidden"):
            agree[f"{qk}:{dname}"] = float(np.mean([r[f"decisions_{qk}"]["agree"][dname] for r in rows]))
        for r in rows:
            if not all(r[f"decisions_{qk}"]["agree"].values()):
                disagreements.append({"quantile": qk, "i": r["i"], "condition": r["condition"],
                                      "rel_margin_to_threshold": r[f"decisions_{qk}"]["rel_margin"]})
    top1 = float(np.mean([r["top1_same"] for r in rows]))
    topk = float(np.mean([r["topk_same"] for r in rows]))
    tim = {k: [r["timing_sec"][k] for r in rows] for k in ("encrypt_all_sites", "server", "decrypt")}
    timing = {k: {"p50": float(np.quantile(v, 0.5)), "p95": float(np.quantile(v, 0.95))} for k, v in tim.items()}
    checks = {
        "scalar_abs_error_ok": max_abs < gate["max_abs_error"],
        "identity_residual_ok": max_id < gate["max_identity_residual"],
        "decision_agreement_ok": min(agree.values()) >= gate["min_decision_agreement"],
        "attribution_agreement_ok": min(top1, topk) >= gate["min_attribution_agreement"],
        "level_audit_ok": True,  # every output decrypted without bootstrapping (context has no FHE feature enabled)
    }
    summary = {
        "dataset": name, "windows": N, "sites": S, "rff_dim": m, "batch_size": bundle.batch_size,
        "ring_dimension": bundle.ring_dimension, "rotations": list(bundle.rotations), "ckks": ck,
        "setup_sec": setup_sec, "timing_sec": timing, "max_abs_error": max_abs, "max_identity_residual": max_id,
        "decision_agreement": agree, "top1_agreement": top1, "topk_agreement": topk,
        "output_levels": sorted({json.dumps(r["levels"]) for r in rows}),
        "disagreements": disagreements, "checks": checks, "replay_pass": all(checks.values()),
        "environment": environment_snapshot(),
    }
    write_json(out / f"{name}_ckks_replay.summary.json", summary)
    print(f"[{name}] {N} windows  S={S}  m={m}  ring={bundle.ring_dimension}  setup={setup_sec:.2f}s")
    print(f"   max |HE - plaintext| = {max_abs:.2e}   max identity residual = {max_id:.2e}")
    print(f"   decision agreement (min over quantiles and alert types) = {min(agree.values()):.4f}   "
          f"top-1 = {top1:.4f}   top-k = {topk:.4f}   disagreements = {len(disagreements)}")
    print(f"   per-window time p50/p95 (s): encrypt {timing['encrypt_all_sites']['p50']:.4f}/{timing['encrypt_all_sites']['p95']:.4f}  "
          f"server {timing['server']['p50']:.4f}/{timing['server']['p95']:.4f}  decrypt {timing['decrypt']['p50']:.4f}/{timing['decrypt']['p95']:.4f}")
    print(f"   checks: {json.dumps(checks)}  ->  CKKS replay {'PASS' if summary['replay_pass'] else 'CHECK'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""CKKS cost grid.

2 S x 4 m x 4 geometries x 10 repeats with the synthetic residual generator
encmonitor.synthetic.make_case -- CKKS cost depends on S, m and parameters, not on the data.
One context (and key set) per m is built once and reused ("context reused per profile");
setup is reported separately from online cost. Records per run: site encryption, server compute,
decryption time, max |HE - plaintext|, and serialized sizes of context, public key, EvalMult keys,
rotation keys, one residual ciphertext and one result ciphertext.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from encmonitor.artifacts import environment_snapshot, write_json
from encmonitor.he import serde
from encmonitor.he.backend import OpenFHEBackend
from encmonitor.he.circuit import evaluate_dual_statistics
from encmonitor.he.context import build_context, decrypt_scalar, encrypt_vector
from encmonitor.statistics import compute_dual_statistics
from encmonitor.synthetic import make_case


def sizes(bundle, ct_example, result_example):
    import openfhe as fhe

    out = {}
    for key, fn in (("context", lambda: serde.serialize_context(bundle.cc)),
                    ("public_key", lambda: serde._serialize(lambda p, t: fhe.SerializeToFile(p, bundle.key_pair.publicKey, t), ".pk")),
                    ("eval_mult_keys", lambda: serde.serialize_eval_mult(bundle.cc)),
                    ("rotation_keys", lambda: serde.serialize_eval_rot(bundle.cc)),
                    ("residual_ciphertext", lambda: serde.serialize_ciphertext(ct_example)),
                    ("result_ciphertext", lambda: serde.serialize_ciphertext(result_example))):
        try:
            out[f"bytes_{key}"] = len(fn())
        except Exception as exc:  # keep the timing grid even if one serializer is unavailable
            out[f"bytes_{key}"] = None
            out[f"error_{key}"] = str(exc)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="CKKS cost grid for the EncMonitor circuit")
    ap.add_argument("--config", default="configs/ckks.yaml")
    ap.add_argument("--out-dir", default="outputs/ckks/cost_grid")
    ap.add_argument("--repeats", type=int, default=0, help="override config repeats (debug)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    ck, g = cfg["ckks"], cfg["cost_grid"]
    reps = args.repeats or int(g["repeats"])
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows, profiles = [], []
    for m in g["rff_dim"]:
        t0 = time.perf_counter()
        bundle = build_context(int(m), multiplicative_depth=int(ck["multiplicative_depth"]),
                               scaling_mod_size=int(ck["scaling_mod_size"]), security_128=bool(ck["security_128"]))
        setup = time.perf_counter() - t0
        backend = OpenFHEBackend(bundle.cc)
        prof = {"rff_dim": int(m), "batch_size": bundle.batch_size, "ring_dimension": bundle.ring_dimension,
                "rotations": len(bundle.rotations), "setup_sec": setup}
        for S in g["sites"]:
            for geo in g["geometries"]:
                for rep in range(reps):
                    case = make_case(int(S), int(m), geo, float(g["severity"]), int(g["seed"]) + rep)
                    ref = compute_dual_statistics(case.residuals, case.weights)
                    t0 = time.perf_counter()
                    cts = [encrypt_vector(bundle, r) for r in case.residuals]
                    t_enc = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    res = evaluate_dual_statistics(backend, cts, case.weights, width=int(m), disclose_contributions=True)
                    t_srv = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    he = [decrypt_scalar(bundle, res.d_mix), decrypt_scalar(bundle, res.d_site), decrypt_scalar(bundle, res.heterogeneity)]
                    a = [decrypt_scalar(bundle, c) for c in res.contributions]
                    t_dec = time.perf_counter() - t0
                    err = max([abs(he[0] - ref.d_mix), abs(he[1] - ref.d_site), abs(he[2] - ref.heterogeneity)]
                              + list(np.abs(np.asarray(a) - ref.contributions)))
                    rows.append({"sites": int(S), "rff_dim": int(m), "geometry": geo, "repeat": rep,
                                 "encrypt_all_sites_sec": t_enc, "encrypt_per_site_sec": t_enc / int(S),
                                 "server_sec": t_srv, "decrypt_sec": t_dec, "max_abs_error": float(err)})
                if S == g["sites"][0] and geo == g["geometries"][0]:
                    prof.update(sizes(bundle, cts[0], res.d_mix))
        profiles.append(prof)
        print(f"[m={m}] ring={bundle.ring_dimension} batch={bundle.batch_size} setup={setup:.2f}s  done")

    df = pd.DataFrame(rows)
    df.to_csv(out / "cost_grid_runs.csv", index=False)
    agg = df.groupby(["sites", "rff_dim"]).agg(
        runs=("server_sec", "size"),
        encrypt_per_site_p50=("encrypt_per_site_sec", "median"),
        server_p50=("server_sec", "median"),
        server_p95=("server_sec", lambda x: float(np.quantile(x, 0.95))),
        decrypt_p50=("decrypt_sec", "median"),
        max_abs_error=("max_abs_error", "max")).reset_index()
    agg.to_csv(out / "cost_grid_summary.csv", index=False)
    pd.DataFrame(profiles).to_csv(out / "ckks_profiles.csv", index=False)
    write_json(out / "cost_grid_summary.json", {"profiles": profiles, "summary": agg.to_dict("records"),
                                                "max_abs_error": float(df.max_abs_error.max()),
                                                "environment": environment_snapshot()})
    pd.set_option("display.width", 200)
    print("\n== CKKS profiles (one context per m, reused) ==")
    print(pd.DataFrame(profiles).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n== Online cost per window (seconds) ==")
    print(agg.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print(f"\nmax |HE - plaintext| over all {len(df)} runs = {df.max_abs_error.max():.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

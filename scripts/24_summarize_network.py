#!/usr/bin/env python3
"""Collect all network session summaries into one table and write the acceptance decision.

Network profiles 1 ms/1 Gbps, 20 ms/100 Mbps and 80 ms/20 Mbps are MODELED from the measured wire bytes
(latency + bytes/bandwidth), because Linux netem needs root rights that were not available.
Measured numbers are loopback (same host, separate processes and TCP sockets).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize network sessions")
    ap.add_argument("--sessions-dir", default="outputs/network/sessions")
    ap.add_argument("--out-dir", default="outputs/network")
    args = ap.parse_args()

    rows, decision = [], {}
    for f in sorted(Path(args.sessions_dir).glob("*/session_summary.json")):
        s = json.load(open(f))
        pw, setup = s["per_window"], s["setup"]
        row = {"session": s["label"], "sites": s["sites"], "m": s["m"], "windows": s["windows"],
               "ring_dimension": setup["ring_dimension"],
               "keygen_sec": setup["keygen_sec"], "key_distribution_sec": setup["distribute_keys_sec"],
               "bytes_context": setup["bytes"]["INIT_CONTEXT"], "bytes_public_key": setup["bytes"]["INIT_PUBLIC_KEY"],
               "bytes_eval_mult": setup["bytes"]["INIT_EVAL_MULT"], "bytes_eval_rot": setup["bytes"]["INIT_EVAL_ROT"],
               "upload_MB_per_window": pw["upload_wire_bytes_all_sites"] / 1e6,
               "result_MB_per_window": pw["result_wire_bytes"] / 1e6,
               "site_encrypt_ms_p50": pw["site_encrypt_sec_p50_p95"][0] * 1e3,
               "server_compute_s_p50": pw["server_compute_sec_p50_p95"][0],
               "server_compute_s_p95": pw["server_compute_sec_p50_p95"][1],
               "round_trip_s_p50": pw["compute_round_trip_sec_p50_p95"][0],
               "round_trip_s_p95": pw["compute_round_trip_sec_p50_p95"][1],
               "authority_decrypt_s_p50": pw["authority_decrypt_sec_p50_p95"][0],
               "max_abs_err_vs_plaintext": s["max_abs_err_vs_plaintext"],
               "max_abs_diff_vs_inprocess_ckks": s["max_abs_diff_vs_inprocess_ckks"],
               "network_pass": s["network_pass"]}
        for name, v in s["modeled_network_profiles"].items():
            row[f"modeled_net_s_{name}"] = v["network_total_sec"]
        rows.append(row)
        decision[s["label"]] = {"pass": s["network_pass"], "checks": s["checks"]}
    df = pd.DataFrame(rows)
    out = Path(args.out_dir)
    df.to_csv(out / "g5_network_summary.csv", index=False)
    all_pass = bool(len(df)) and all(d["pass"] for d in decision.values())
    (out / "g5_decision.json").write_text(json.dumps({"sessions": decision, "network_pass": all_pass}, indent=2))
    pd.set_option("display.width", 250)
    cols = ["session", "sites", "m", "windows", "upload_MB_per_window", "result_MB_per_window", "site_encrypt_ms_p50",
            "server_compute_s_p50", "round_trip_s_p50", "round_trip_s_p95", "max_abs_err_vs_plaintext",
            "max_abs_diff_vs_inprocess_ckks", "network_pass"]
    print("== Measured (loopback, separate processes over TCP) ==")
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    mcols = ["session"] + [c for c in df.columns if c.startswith("modeled_net_s_")]
    print("\n== Modeled network time per window (s) from measured bytes: latency + bytes/bandwidth ==")
    print(df[mcols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nNetwork decision: {'PASS' if all_pass else 'CHECK'}  ({sum(d['pass'] for d in decision.values())}/{len(decision)} sessions pass)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

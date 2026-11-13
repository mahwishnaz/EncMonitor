#!/usr/bin/env python3
"""Evaluation-key sizes per CKKS profile, each measured in a fresh process (OpenFHE serializes the
evaluation keys of every context held in a process).

Usage (bundle root): python experiments/private_localany/keysizes.py > experiments/private_localany/keysizes_log.txt
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

CHILD = r'''
import json, sys, tempfile
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import importlib.util
import openfhe as fhe
from encmonitor.he.context import build_context, encrypt_vector
from encmonitor.he.backend import OpenFHEBackend
from encmonitor.he.circuit import evaluate_dual_statistics
m, ks = int(sys.argv[2]), sys.argv[3]
if ks == "hybrid":
    b = build_context(m, multiplicative_depth=2, scaling_mod_size=50)
else:
    spec = importlib.util.spec_from_file_location("e10", Path(sys.argv[1]) / "experiments/ckks_keyswitch/run.py")
    e10 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e10)
    b = e10.build_context_bv(m)
ct = encrypt_vector(b, [0.01] * m)
res = evaluate_dual_statistics(OpenFHEBackend(b.cc), [ct] * 5, [0.2] * 5, width=m)
out = {"m": m, "key_switch": ks, "ring_dimension": b.ring_dimension, "rotations": len(b.rotations)}
with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    fhe.SerializeToFile(str(d / "ct"), ct, fhe.BINARY)
    fhe.SerializeToFile(str(d / "res"), res.d_mix, fhe.BINARY)
    fhe.SerializeToFile(str(d / "pk"), b.key_pair.publicKey, fhe.BINARY)
    b.cc.SerializeEvalMultKey(str(d / "mult"), fhe.BINARY)
    b.cc.SerializeEvalAutomorphismKey(str(d / "rot"), fhe.BINARY)
    for k, f in (("residual_ciphertext", "ct"), ("result_ciphertext", "res"), ("public_key", "pk"),
                 ("mult_keys", "mult"), ("rotation_keys", "rot")):
        out[k + "_bytes"] = (d / f).stat().st_size
print(json.dumps(out))
'''


def main() -> int:
    rows = []
    for m, ks in [(32, "hybrid"), (64, "hybrid"), (128, "hybrid"), (256, "hybrid"), (64, "bv")]:
        r = subprocess.run([sys.executable, "-c", CHILD, str(ROOT), str(m), ks], capture_output=True, text=True, check=True)
        row = json.loads(r.stdout.strip().splitlines()[-1])
        row["eval_keys_bytes"] = row["mult_keys_bytes"] + row["rotation_keys_bytes"]
        rows.append(row)
        print(json.dumps(row), flush=True)
    grid = json.load(open(ROOT / "outputs/ckks/cost_grid/cost_grid_summary.json"))["profiles"]
    for p in grid:
        print("CKKS cost grid (same process):", {k: p[k] for k in p if "bytes" in k or k == "rff_dim"})
    (HERE / "key_sizes.json").write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

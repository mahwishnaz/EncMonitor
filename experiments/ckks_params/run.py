#!/usr/bin/env python3
"""CKKS parameters of the EncMonitor profile and whether N = 8192 is allowed (parameters only, no timing).

Builds the context with the library function used by every CKKS run
(encmonitor.he.context.build_context) and reports ring dimension, modulus chain, key-switching and scaling
technique, and the OpenMP thread count. Then asks OpenFHE for the same circuit with N = 8192 at
HEStd_128_classic under several scaling choices; OpenFHE raises an error when a ring dimension is too
small for the requested security level. No timings are measured.

Usage (bundle root): python experiments/ckks_params/run.py > experiments/ckks_params/run_log.txt
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import openfhe as fhe  # noqa: E402

from encmonitor.he.context import build_context  # noqa: E402


def describe(cc) -> dict:
    p = cc.GetCryptoParameters() if hasattr(cc, "GetCryptoParameters") else None
    out = {"ring_dimension": int(cc.GetRingDimension())}
    try:
        out["log2_Q"] = float(math.log2(int(cc.GetModulus()))) if hasattr(cc, "GetModulus") else None
    except Exception as exc:  # noqa: BLE001
        out["log2_Q"] = f"n/a ({exc})"
    for name in ("GetScalingTechnique", "GetKeySwitchTechnique", "GetMultiplicativeDepth", "GetScalingModSize",
                 "GetFirstModSize", "GetNumLargeDigits", "GetSecurityLevel"):
        if p is not None and hasattr(p, name):
            try:
                out[name] = str(getattr(p, name)())
            except Exception as exc:  # noqa: BLE001
                out[name] = f"n/a ({exc})"
    return out


def params(m: int, ring: int | None, scaling, first_mod: int | None):
    pr = fhe.CCParamsCKKSRNS()
    pr.SetMultiplicativeDepth(2)
    pr.SetScalingModSize(50)
    pr.SetBatchSize(m)
    pr.SetSecurityLevel(fhe.SecurityLevel.HEStd_128_classic)
    pr.SetScalingTechnique(scaling)
    if first_mod:
        pr.SetFirstModSize(first_mod)
    if ring:
        pr.SetRingDim(ring)
    return pr


def main() -> int:
    res = {"OMP_NUM_THREADS_env": os.environ.get("OMP_NUM_THREADS"), "cpu_count": os.cpu_count()}
    try:
        import openfhe
        res["openfhe_version"] = getattr(openfhe, "__version__", None)
    except Exception:  # noqa: BLE001
        pass
    paper = {}
    for m in (32, 64, 128, 256):
        b = build_context(m)
        paper[m] = describe(b.cc)
        paper[m]["printed_params"] = str(b.cc.GetCryptoParameters()) if hasattr(b.cc, "GetCryptoParameters") else None
    res["paper_profiles"] = paper
    trials = []
    for tech in ("FLEXIBLEAUTO", "FLEXIBLEAUTOEXT", "FIXEDAUTO", "FIXEDMANUAL"):
        if not hasattr(fhe.ScalingTechnique, tech):
            continue
        for first in (None, 60, 55):
            for ring in (None, 8192):
                t = {"scaling": tech, "first_mod_size": first, "requested_ring": ring}
                try:
                    cc = fhe.GenCryptoContext(params(64, ring, getattr(fhe.ScalingTechnique, tech), first))
                    t.update({"ok": True, **describe(cc)})
                except Exception as exc:  # noqa: BLE001
                    t.update({"ok": False, "error": str(exc).splitlines()[0][:300]})
                trials.append(t)
    res["trials_m64"] = trials
    (HERE / "params.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

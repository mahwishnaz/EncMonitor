#!/usr/bin/env python3
"""Ring dimension chosen by OpenFHE for the EncMonitor circuit under HYBRID vs BV key switching
(HEStd_128_classic, depth 2, 50-bit scaling, FLEXIBLEAUTO). Parameters only, no timing.
Usage (from the repository root): python experiments/ckks_params/keyswitch.py > experiments/ckks_params/keyswitch_log.txt
"""
import json
import math
from pathlib import Path

import openfhe as fhe


def mk(ks, dnum=None, ring=None):
    p = fhe.CCParamsCKKSRNS(); p.SetMultiplicativeDepth(2); p.SetScalingModSize(50); p.SetBatchSize(64)
    p.SetSecurityLevel(fhe.SecurityLevel.HEStd_128_classic); p.SetScalingTechnique(fhe.ScalingTechnique.FLEXIBLEAUTO)
    p.SetKeySwitchTechnique(ks)
    if dnum: p.SetNumLargeDigits(dnum)
    if ring: p.SetRingDim(ring)
    return p


d = fhe.CCParamsCKKSRNS()
out = {"defaults": {"key_switch": str(d.GetKeySwitchTechnique()), "first_mod_size": d.GetFirstModSize()}, "trials": []}
for ks in (fhe.KeySwitchTechnique.HYBRID, fhe.KeySwitchTechnique.BV):
    for dnum in (None, 1, 2, 3):
        for ring in (None, 8192):
            r = {"key_switch": str(ks), "num_large_digits": dnum, "requested_ring": ring}
            try:
                cc = fhe.GenCryptoContext(mk(ks, dnum, ring))
                r.update(ok=True, ring_dimension=int(cc.GetRingDimension()), log2_Q=round(math.log2(int(cc.GetModulus())), 2))
            except Exception as e:  # noqa: BLE001
                r.update(ok=False, error=str(e).split("():")[-1].strip()[:160])
            out["trials"].append(r)
Path(__file__).with_name("keyswitch.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))

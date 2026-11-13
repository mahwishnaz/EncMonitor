#!/usr/bin/env python3
"""Replay the exported windows under CKKS with BV key switching at N = 8192 (fidelity only).

Usage (bundle root): python experiments/ckks_keyswitch/run.py > experiments/ckks_keyswitch/run_log.txt
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import openfhe as fhe  # noqa: E402

import encmonitor.he.context as ctxmod  # noqa: E402
from encmonitor.he.circuit import power_of_two_width, rotation_steps  # noqa: E402


def build_context_bv(sketch_dim, *, multiplicative_depth=2, scaling_mod_size=50, security_128=True):
    batch_size = power_of_two_width(int(sketch_dim))
    p = fhe.CCParamsCKKSRNS()
    p.SetMultiplicativeDepth(int(multiplicative_depth))
    p.SetScalingModSize(int(scaling_mod_size))
    p.SetBatchSize(int(batch_size))
    p.SetSecurityLevel(fhe.SecurityLevel.HEStd_128_classic)
    p.SetScalingTechnique(fhe.ScalingTechnique.FLEXIBLEAUTO)
    p.SetKeySwitchTechnique(fhe.KeySwitchTechnique.BV)
    p.SetRingDim(8192)
    cc = fhe.GenCryptoContext(p)
    for f in (fhe.PKESchemeFeature.PKE, fhe.PKESchemeFeature.KEYSWITCH, fhe.PKESchemeFeature.LEVELEDSHE):
        cc.Enable(f)
    kp = cc.KeyGen()
    cc.EvalMultKeyGen(kp.secretKey)
    rotations = tuple(rotation_steps(batch_size))
    cc.EvalRotateKeyGen(kp.secretKey, list(rotations))
    return ctxmod.CKKSBundle(cc=cc, key_pair=kp, batch_size=batch_size, rotations=rotations,
                             ring_dimension=int(cc.GetRingDimension()))


def sizes(m: int) -> dict:
    """Serialized sizes (bytes) of one residual ciphertext and of the evaluation keys."""
    b = build_context_bv(m)
    ct = ctxmod.encrypt_vector(b, [0.1] * m)
    out = {}
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        fhe.SerializeToFile(str(d / "ct"), ct, fhe.BINARY)
        b.cc.SerializeEvalMultKey(str(d / "mult"), fhe.BINARY)
        b.cc.SerializeEvalAutomorphismKey(str(d / "rot"), fhe.BINARY)
        fhe.SerializeToFile(str(d / "pk"), b.key_pair.publicKey, fhe.BINARY)
        for k in ("ct", "mult", "rot", "pk"):
            out[k + "_bytes"] = (d / k).stat().st_size
    out["ring_dimension"] = b.ring_dimension
    return out


def main() -> int:
    spec = importlib.util.spec_from_file_location("replay21", ROOT / "scripts" / "21_run_ckks_replay.py")
    m21 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m21)
    m21.build_context = build_context_bv            # BV key switching instead of hybrid key switching; nothing else changes
    res = {"sizes_m64": sizes(64)}
    replays = [ROOT / "outputs/ckks/replay/mnist_replay.npz", ROOT / "outputs/ckks/replay/cifar10_replay.npz",
               ROOT / "outputs/camelyon/ckks/camelyon17_replay.npz"]
    for rp in replays:
        cfg = ROOT / ("configs/ckks.yaml")
        sys.argv = ["21", "--config", str(cfg), "--replay", str(rp), "--out-dir", str(HERE / "results")]
        m21.main()
        s = json.load(open(HERE / "results" / f"{rp.stem.replace('_replay', '')}_ckks_replay.summary.json"))
        res[s["dataset"]] = {k: s[k] for k in ("windows", "ring_dimension", "max_abs_error", "max_identity_residual",
                                               "top1_agreement", "topk_agreement", "checks", "replay_pass")}
        res[s["dataset"]]["min_decision_agreement"] = min(s["decision_agreement"].values())
    (HERE / "summary.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

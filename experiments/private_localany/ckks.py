#!/usr/bin/env python3
"""CKKS cost of encrypted Local-any (encrypted bit sum), EncMonitor and the hybrid.

CPU only, OpenFHE default threads (as in 21_run_ckks_replay.py). Times only if the load average is low.
Usage (from the repository root): python experiments/private_localany/ckks.py > experiments/private_localany/ckks_log.txt
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import openfhe as fhe  # noqa: E402

from encmonitor.he.backend import OpenFHEBackend  # noqa: E402
from encmonitor.he.circuit import _weighted_sum, evaluate_dual_statistics, power_of_two_width, slot_sum  # noqa: E402
from encmonitor.he.context import build_context, decrypt_scalar, encrypt_vector  # noqa: E402
from encmonitor.statistics import compute_dual_statistics  # noqa: E402

spec = importlib.util.spec_from_file_location("e10", ROOT / "experiments" / "ckks_keyswitch" / "run.py")
E10 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E10)

REPLAY = ROOT / "outputs/ckks/replay/mnist_replay.npz"
MAX_LOAD = 4.0


def ser_size(obj) -> int:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x"
        fhe.SerializeToFile(str(p), obj, fhe.BINARY)
        return p.stat().st_size


def key_sizes(cc, kp, mult: bool, rot: bool) -> dict:
    out = {"public_key_bytes": ser_size(kp.publicKey)}
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        if mult:
            cc.SerializeEvalMultKey(str(d / "m"), fhe.BINARY)
            out["mult_key_bytes"] = (d / "m").stat().st_size
        if rot:
            cc.SerializeEvalAutomorphismKey(str(d / "r"), fhe.BINARY)
            out["rotation_key_bytes"] = (d / "r").stat().st_size
    return out


class Counting:
    """Counts operations of the circuit (wraps the OpenFHE backend)."""

    def __init__(self, inner):
        self.inner, self.n = inner, {"add": 0, "sub": 0, "mul": 0, "mul_scalar": 0, "rotate": 0}

    def add(self, a, b):
        self.n["add"] += 1
        return self.inner.add(a, b)

    def sub(self, a, b):
        self.n["sub"] += 1
        return self.inner.sub(a, b)

    def mul(self, a, b):
        self.n["mul"] += 1
        return self.inner.mul(a, b)

    def mul_scalar(self, a, v):
        self.n["mul_scalar"] += 1
        return self.inner.mul_scalar(a, v)

    def rotate(self, a, k):
        self.n["rotate"] += 1
        return self.inner.rotate(a, k)


def dmix_only(backend, cts, alpha, width):
    mean = _weighted_sum(backend, cts, alpha)
    return slot_sum(backend, backend.mul(mean, mean), power_of_two_width(width))


def bit_context(kind: str):
    # CKKS depth 0: first modulus 35 bits, scaling 25 bits, so decoded values lie in about [-2^9, 2^9).
    # The count of alert bits was checked to be exact for every count 0..64 at S = 5, 10, 16, 20, 50 and 64
    # (error below 1e-3). Ring dimension 4,096 and ciphertext size are the same as with a 30-bit first
    # modulus, which wraps at 16 alerting sites.
    if kind == "ckks_d0":
        p = fhe.CCParamsCKKSRNS()
        p.SetMultiplicativeDepth(0); p.SetScalingModSize(25); p.SetFirstModSize(35); p.SetBatchSize(1)
        p.SetScalingTechnique(fhe.ScalingTechnique.FLEXIBLEAUTO)
    else:   # bgv_t65537_d0
        p = fhe.CCParamsBGVRNS()
        p.SetPlaintextModulus(65537); p.SetMultiplicativeDepth(0)
    p.SetSecurityLevel(fhe.SecurityLevel.HEStd_128_classic)
    cc = fhe.GenCryptoContext(p)
    cc.Enable(fhe.PKESchemeFeature.PKE); cc.Enable(fhe.PKESchemeFeature.LEVELEDSHE)
    return cc, cc.KeyGen()


def enc_bit(cc, kp, kind, b):
    pt = cc.MakeCKKSPackedPlaintext([float(b)]) if kind == "ckks_d0" else cc.MakePackedPlaintext([int(b)])
    return cc.Encrypt(kp.publicKey, pt)


def dec_count(cc, kp, kind, ct) -> int:
    pt = cc.Decrypt(kp.secretKey, ct)
    pt.SetLength(1)
    v = pt.GetCKKSPackedValue()[0].real if kind == "ckks_d0" else pt.GetPackedValue()[0]
    return int(round(v))


def pct(x):
    x = np.asarray(x)
    return {"p50": float(np.median(x)), "p95": float(np.quantile(x, 0.95)), "mean": float(x.mean())}


def main() -> int:
    load0 = os.getloadavg()
    timed = load0[0] < MAX_LOAD
    z = np.load(REPLAY)
    R = np.asarray(z["residuals"], dtype=np.float64)
    W, S, m = R.shape
    alpha = np.asarray(z["weights"], float) / np.sum(z["weights"])
    E = np.einsum("wsk,wsk->ws", R, R)
    bits = (E > np.median(E)).astype(int)          # bit values do not affect time
    res = {"load_before": load0, "timed": timed, "windows": W, "sites": S, "m": m}

    # ---------- encrypted Local-any: encrypted bit sum
    for kind in ("ckks_d0", "bgv_t65537_d0"):
        cc, kp = bit_context(kind)
        r = {"ring_dimension": int(cc.GetRingDimension()), "log2_Q": math.log2(int(cc.GetModulus())),
             "depth": 0, "eval_keys": "none", "server_ops": {"add": S - 1}}
        ct0 = enc_bit(cc, kp, kind, 1)
        r["ciphertext_bytes"] = ser_size(ct0)
        r.update(key_sizes(cc, kp, mult=False, rot=False))
        te, ts, td, ok = [], [], [], 0
        for w in range(W):
            t0 = time.perf_counter(); cts = [enc_bit(cc, kp, kind, b) for b in bits[w]]; te.append(time.perf_counter() - t0)
            t0 = time.perf_counter(); tot = cts[0]
            for c in cts[1:]:
                tot = cc.EvalAdd(tot, c)
            ts.append(time.perf_counter() - t0)
            t0 = time.perf_counter(); K = dec_count(cc, kp, kind, tot); td.append(time.perf_counter() - t0)
            ok += int(K == int(bits[w].sum()))
        r["count_exact_windows"] = ok
        r["result_ciphertext_bytes"] = ser_size(tot)
        if timed:
            r["time_sec"] = {"encrypt_all_sites": pct(te), "server": pct(ts), "decrypt": pct(td)}
        res[f"private_localany_{kind}"] = r
        print(kind, json.dumps({k: v for k, v in r.items() if k != "time_sec"}), flush=True)

    # ---------- EncMonitor (article setting, N = 16384 hybrid key switching) and hybrid in the same context
    for label, builder in (("N16384_hybridKS", lambda: build_context(m, multiplicative_depth=2, scaling_mod_size=50)),
                           ("N8192_BV", lambda: E10.build_context_bv(m))):
        b = builder()
        be = OpenFHEBackend(b.cc)
        r = {"ring_dimension": b.ring_dimension, "log2_Q": math.log2(int(b.cc.GetModulus())), "depth": 2}
        ct0 = encrypt_vector(b, R[0, 0])
        r["residual_ciphertext_bytes"] = ser_size(ct0)
        r.update(key_sizes(b.cc, b.key_pair, mult=True, rot=True))
        cnt = Counting(be)
        out = evaluate_dual_statistics(cnt, [encrypt_vector(b, x) for x in R[0]], alpha, width=m, disclose_contributions=True)
        r["encmonitor_server_ops"] = dict(cnt.n)
        r["encmonitor_result_ciphertexts"] = 3 + S
        r["encmonitor_result_ciphertext_bytes_each"] = ser_size(out.d_mix)
        cnt = Counting(be)
        dm = dmix_only(cnt, [encrypt_vector(b, x) for x in R[0]], alpha, m)
        r["hybrid_dmix_server_ops"] = dict(cnt.n)
        r["hybrid_bit_in_same_context_ciphertext_bytes"] = ser_size(b.cc.Encrypt(b.key_pair.publicKey, b.cc.MakeCKKSPackedPlaintext([1.0])))
        errs, te, ts, td, th = [], [], [], [], []
        for w in range(W):
            ref = compute_dual_statistics(R[w], z["weights"])
            t0 = time.perf_counter(); cts = [encrypt_vector(b, x) for x in R[w]]; te.append(time.perf_counter() - t0)
            t0 = time.perf_counter(); o = evaluate_dual_statistics(be, cts, z["weights"], width=m, disclose_contributions=True); ts.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            v = [decrypt_scalar(b, o.d_mix), decrypt_scalar(b, o.d_site), decrypt_scalar(b, o.heterogeneity)]
            td.append(time.perf_counter() - t0)
            errs.append(max(abs(v[0] - ref.d_mix), abs(v[1] - ref.d_site), abs(v[2] - ref.heterogeneity)))
            # hybrid server work: D_mix-only circuit + bit sum in the same context
            bcts = [b.cc.Encrypt(b.key_pair.publicKey, b.cc.MakeCKKSPackedPlaintext([float(x)])) for x in bits[w]]
            t0 = time.perf_counter()
            dmv = dmix_only(be, cts, alpha, m)
            tot = bcts[0]
            for c in bcts[1:]:
                tot = b.cc.EvalAdd(tot, c)
            th.append(time.perf_counter() - t0)
        r["encmonitor_max_abs_error"] = float(max(errs))
        if timed:
            r["encmonitor_time_sec"] = {"encrypt_all_sites": pct(te), "server": pct(ts), "decrypt_3_scalars": pct(td)}
            r["hybrid_server_time_sec"] = pct(th)
        res[label] = r
        print(label, json.dumps({k: v for k, v in r.items() if "time" not in k}), flush=True)

    res["load_after"] = os.getloadavg()
    (HERE / "ckks_cost.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .circuit import power_of_two_width, rotation_steps


@dataclass(frozen=True)
class CKKSBundle:
    cc: Any
    key_pair: Any
    batch_size: int
    rotations: tuple[int, ...]
    ring_dimension: int


def require_openfhe():
    try:
        import openfhe as fhe
    except Exception as exc:
        raise RuntimeError(
            "OpenFHE Python bindings are not installed. Build openfhe-development "
            "and openfhe-python on the target machine, then rerun this script."
        ) from exc
    return fhe


def build_context(
    sketch_dim: int,
    *,
    multiplicative_depth: int = 2,
    scaling_mod_size: int = 50,
    security_128: bool = True,
) -> CKKSBundle:
    """Create a leveled CKKS context for the EncMonitor circuit.

    No bootstrapping is enabled. The batch size is the next power of two at
    least as large as sketch_dim, which makes the rotation-tree reduction
    deterministic and inexpensive.
    """
    fhe = require_openfhe()
    batch_size = power_of_two_width(int(sketch_dim))
    params = fhe.CCParamsCKKSRNS()
    params.SetMultiplicativeDepth(int(multiplicative_depth))
    params.SetScalingModSize(int(scaling_mod_size))
    params.SetBatchSize(int(batch_size))
    if security_128 and hasattr(fhe, "SecurityLevel"):
        params.SetSecurityLevel(fhe.SecurityLevel.HEStd_128_classic)
    if hasattr(fhe, "ScalingTechnique") and hasattr(fhe.ScalingTechnique, "FLEXIBLEAUTO"):
        params.SetScalingTechnique(fhe.ScalingTechnique.FLEXIBLEAUTO)

    cc = fhe.GenCryptoContext(params)
    cc.Enable(fhe.PKESchemeFeature.PKE)
    cc.Enable(fhe.PKESchemeFeature.KEYSWITCH)
    cc.Enable(fhe.PKESchemeFeature.LEVELEDSHE)

    kp = cc.KeyGen()
    if not getattr(kp, "good", lambda: True)():
        raise RuntimeError("OpenFHE KeyGen failed")
    cc.EvalMultKeyGen(kp.secretKey)
    rotations = tuple(rotation_steps(batch_size))
    if rotations:
        cc.EvalRotateKeyGen(kp.secretKey, list(rotations))
    return CKKSBundle(
        cc=cc,
        key_pair=kp,
        batch_size=batch_size,
        rotations=rotations,
        ring_dimension=int(cc.GetRingDimension()),
    )


def pad_vector(values, batch_size: int):
    import numpy as np

    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if v.size > batch_size:
        raise ValueError(f"vector length {v.size} exceeds batch_size {batch_size}")
    out = np.zeros(batch_size, dtype=np.float64)
    out[: v.size] = v
    return out


def encrypt_vector(bundle: CKKSBundle, values):
    padded = pad_vector(values, bundle.batch_size)
    pt = bundle.cc.MakeCKKSPackedPlaintext(padded.tolist())
    pt.SetLength(bundle.batch_size)
    return bundle.cc.Encrypt(bundle.key_pair.publicKey, pt)


def decrypt_vector(bundle: CKKSBundle, ciphertext):
    import numpy as np

    pt = bundle.cc.Decrypt(bundle.key_pair.secretKey, ciphertext)
    pt.SetLength(bundle.batch_size)
    vals = pt.GetCKKSPackedValue()
    return np.asarray([complex(v).real for v in vals], dtype=np.float64)


def decrypt_scalar(bundle: CKKSBundle, ciphertext) -> float:
    return float(decrypt_vector(bundle, ciphertext)[0])

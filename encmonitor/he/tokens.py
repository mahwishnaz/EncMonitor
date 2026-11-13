from __future__ import annotations

import os


def _make_instance(cls):
    try:
        return cls.__new__(cls)
    except Exception:
        return None


def token_matrix(prefer_binary: bool = True):
    import openfhe as fhe

    binary = _make_instance(getattr(fhe, "SERBINARY"))
    json_token = _make_instance(getattr(fhe, "SERJSON"))
    if binary is None or json_token is None:
        raise RuntimeError("OpenFHE serialization token construction failed")
    force_json = os.getenv("FHE_FORCE_JSON", "").lower() in {"1", "true", "yes", "on"}
    if force_json:
        return [(".json", json_token)]
    return [(".bin", binary), (".json", json_token)] if prefer_binary else [(".json", json_token), (".bin", binary)]

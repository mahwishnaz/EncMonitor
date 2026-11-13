from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Callable

from .tokens import token_matrix


def _ok(ret: Any) -> bool | None:
    if ret is None:
        return None
    if isinstance(ret, bool):
        return ret
    if isinstance(ret, tuple) and len(ret) == 2:
        return bool(ret[1])
    return bool(ret)


def _unwrap(ret: Any) -> Any:
    if isinstance(ret, tuple) and len(ret) == 2:
        return ret[0] if bool(ret[1]) else None
    if isinstance(ret, bool) or ret is None:
        return None
    return ret


def _serialize(builder: Callable[[str, Any], Any], suffix: str) -> bytes:
    last: Exception | None = None
    for _, token in token_matrix(True):
        fd, path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        try:
            ret = builder(path, token)
            if _ok(ret) is True or Path(path).stat().st_size > 0:
                return Path(path).read_bytes()
        except Exception as exc:
            last = exc
        finally:
            Path(path).unlink(missing_ok=True)
    raise RuntimeError(f"OpenFHE serialization failed: {last}")


def _temp_blob(blob: bytes, suffix: str) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    Path(path).write_bytes(blob)
    return path


def serialize_context(cc) -> bytes:
    import openfhe as fhe

    return _serialize(lambda p, t: fhe.SerializeToFile(p, cc, t), ".cc")


def deserialize_context(blob: bytes):
    import openfhe as fhe

    last = None
    for _, token in token_matrix(True):
        path = _temp_blob(blob, ".cc")
        try:
            obj = _unwrap(fhe.DeserializeCryptoContext(path, token))
            if obj is not None:
                return obj
        except Exception as exc:
            last = exc
        finally:
            Path(path).unlink(missing_ok=True)
    raise RuntimeError(f"Context deserialization failed: {last}")


def serialize_ciphertext(ct) -> bytes:
    import openfhe as fhe

    return _serialize(lambda p, t: fhe.SerializeToFile(p, ct, t), ".ct")


def deserialize_ciphertext(blob: bytes):
    import openfhe as fhe

    last = None
    for _, token in token_matrix(True):
        path = _temp_blob(blob, ".ct")
        try:
            obj = _unwrap(fhe.DeserializeCiphertext(path, token))
            if obj is not None:
                return obj
        except Exception as exc:
            last = exc
        finally:
            Path(path).unlink(missing_ok=True)
    raise RuntimeError(f"Ciphertext deserialization failed: {last}")


def serialize_eval_mult(cc) -> bytes:
    return _serialize(lambda p, t: cc.SerializeEvalMultKey(p, t), ".ekm")


def serialize_eval_rot(cc) -> bytes:
    return _serialize(lambda p, t: cc.SerializeEvalAutomorphismKey(p, t), ".eka")


def _deserialize_key(cc, blob: bytes, suffix: str, method: str) -> None:
    last = None
    for _, token in token_matrix(True):
        path = _temp_blob(blob, suffix)
        try:
            ret = getattr(cc, method)(path, token)
            if _ok(ret) is True:
                return
        except Exception as exc:
            last = exc
        finally:
            Path(path).unlink(missing_ok=True)
    raise RuntimeError(f"{method} failed: {last}")


def deserialize_eval_mult(cc, blob: bytes) -> None:
    _deserialize_key(cc, blob, ".ekm", "DeserializeEvalMultKey")


def deserialize_eval_rot(cc, blob: bytes) -> None:
    _deserialize_key(cc, blob, ".eka", "DeserializeEvalAutomorphismKey")

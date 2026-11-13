from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


class ArithmeticBackend(Protocol):
    def add(self, a: Any, b: Any) -> Any: ...
    def sub(self, a: Any, b: Any) -> Any: ...
    def mul(self, a: Any, b: Any) -> Any: ...
    def mul_scalar(self, a: Any, value: float) -> Any: ...
    def rotate(self, a: Any, steps: int) -> Any: ...


@dataclass(frozen=True)
class NumpyBackend:
    """Plaintext SIMD emulator used to test the exact circuit topology."""

    def add(self, a, b):
        return np.asarray(a) + np.asarray(b)

    def sub(self, a, b):
        return np.asarray(a) - np.asarray(b)

    def mul(self, a, b):
        return np.asarray(a) * np.asarray(b)

    def mul_scalar(self, a, value: float):
        return np.asarray(a) * float(value)

    def rotate(self, a, steps: int):
        return np.roll(np.asarray(a), int(steps))


@dataclass(frozen=True)
class OpenFHEBackend:
    """Thin adapter around an OpenFHE CryptoContext."""

    cc: Any

    def add(self, a, b):
        return self.cc.EvalAdd(a, b)

    def sub(self, a, b):
        return self.cc.EvalSub(a, b)

    def mul(self, a, b):
        return self.cc.EvalMult(a, b)

    def mul_scalar(self, a, value: float):
        return self.cc.EvalMult(a, float(value))

    def rotate(self, a, steps: int):
        return self.cc.EvalRotate(a, int(steps))

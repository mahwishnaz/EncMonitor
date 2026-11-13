"""Optional OpenFHE CKKS support for EncMonitor."""

from .backend import NumpyBackend, OpenFHEBackend
from .circuit import EncryptedDualStatistics, evaluate_dual_statistics

__all__ = ["NumpyBackend", "OpenFHEBackend", "EncryptedDualStatistics", "evaluate_dual_statistics"]

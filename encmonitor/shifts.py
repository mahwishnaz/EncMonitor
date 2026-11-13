from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import numpy as np

ShiftName = Literal["none", "rotation", "brightness", "contrast", "blur", "noise"]


@dataclass(frozen=True)
class ShiftSpec:
    name: ShiftName
    severity: float
    direction: float = 1.0


def apply_tensor_shift(x: Any, spec: ShiftSpec, *, seed: int = 0) -> Any:
    """Apply an image shift to a CHW torch tensor in [0,1]."""
    try:
        import torch
        from torchvision.transforms import functional as TF
    except Exception as exc:
        raise RuntimeError("torch and torchvision are required for image shifts") from exc

    if spec.name == "none":
        return x
    sign = 1.0 if spec.direction >= 0 else -1.0
    sev = float(spec.severity)
    if spec.name == "rotation":
        return TF.rotate(x, angle=sign * sev)
    if spec.name == "brightness":
        factor = max(0.0, 1.0 + sign * sev)
        return TF.adjust_brightness(x, factor)
    if spec.name == "contrast":
        factor = max(0.0, 1.0 + sign * sev)
        return TF.adjust_contrast(x, factor)
    if spec.name == "blur":
        sigma = max(0.05, sev)
        return TF.gaussian_blur(x, kernel_size=[3, 3], sigma=[sigma, sigma])
    if spec.name == "noise":
        gen = torch.Generator(device=x.device)
        gen.manual_seed(int(seed))
        noise = torch.randn(x.shape, generator=gen, device=x.device, dtype=x.dtype)
        return torch.clamp(x + sign * sev * noise, 0.0, 1.0)
    raise ValueError(f"Unsupported shift={spec.name}")

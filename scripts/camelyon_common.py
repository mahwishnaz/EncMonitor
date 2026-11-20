"""Shared helpers for the Camelyon17-WILDS experiment (scripts 31-35).

The encmonitor library is used unchanged; these helpers add only what a real five-hospital
data set needs (data access, the model wrapper, and a stain shift for H&E images).
"""
from __future__ import annotations

import numpy as np

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ----------------------------------------------------------------------------- data
def load_wilds(root: str):
    from wilds import get_dataset

    return get_dataset(dataset="camelyon17", root_dir=root, download=False)


def metadata(ds) -> dict:
    fields = list(ds.metadata_fields)
    meta = ds.metadata_array.numpy()
    return {
        "hospital": meta[:, fields.index("hospital")].astype(np.int64),
        "slide": meta[:, fields.index("slide")].astype(np.int64),
        "y": np.asarray(ds.y_array).astype(np.int64),
        "split": np.asarray(ds.split_array).astype(np.int64),
        "split_dict": dict(ds.split_dict),
    }


def heldout_by_hospital(md: dict) -> dict[int, np.ndarray]:
    """Patches never used for training: every split except 'train'.

    For the three training hospitals this is the official id_val split; for the other
    two hospitals it is their OOD val / test split."""
    train_code = md["split_dict"]["train"]
    out = {}
    for h in sorted(np.unique(md["hospital"])):
        out[int(h)] = np.flatnonzero((md["hospital"] == h) & (md["split"] != train_code))
    return out


def load_images(ds, indices: np.ndarray, workers: int = 8, batch: int = 512) -> np.ndarray:
    """Decode patches into one uint8 array [N, 96, 96, 3] (order = indices)."""
    import torch
    from torch.utils.data import DataLoader, Dataset

    class View(Dataset):
        def __len__(self):
            return int(len(indices))

        def __getitem__(self, i):
            return torch.from_numpy(np.array(ds.get_input(int(indices[i])).convert("RGB"), dtype=np.uint8))

    out = []
    for x in DataLoader(View(), batch_size=batch, shuffle=False, num_workers=workers):
        out.append(x.numpy())
    return np.concatenate(out, axis=0)


def to_float(batch_uint8, device):
    """uint8 [B,H,W,3] (numpy or tensor) -> float [B,3,H,W] in [0,1] on device."""
    import torch

    t = torch.as_tensor(batch_uint8).to(device, non_blocking=True)
    return t.permute(0, 3, 1, 2).float().div_(255.0)


# ----------------------------------------------------------------------------- model
def build_model(pretrained: bool = False):
    """ResNet-18 for 96x96 patches, 2 classes, ImageNet normalisation inside the model.

    forward_features() returns the 512-d penultimate (average-pooled) representation,
    the layer monitored in the article; classifier = backbone.fc."""
    import torch
    import torch.nn as nn
    from torchvision.models import resnet18

    weights = None
    if pretrained:
        from torchvision.models import ResNet18_Weights

        weights = ResNet18_Weights.IMAGENET1K_V1

    class CamelyonResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, -1, 1, 1))
            self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, -1, 1, 1))
            self.backbone = resnet18(weights=weights)
            self.backbone.fc = nn.Linear(512, 2)

        @property
        def classifier(self):
            return self.backbone.fc

        def forward_features(self, x):
            b = self.backbone
            x = (x - self.mean) / self.std
            x = b.maxpool(b.relu(b.bn1(b.conv1(x))))
            x = b.layer4(b.layer3(b.layer2(b.layer1(x))))
            return torch.flatten(b.avgpool(x), 1)

        def forward(self, x):
            return self.backbone.fc(self.forward_features(x))

    return CamelyonResNet()


def embed(model, images_uint8: np.ndarray, device, batch: int = 512, shift=None):
    """Penultimate embeddings [N,512] (float32) of uint8 images, with an optional
    batched shift function f(x_float_BCHW) -> x_float_BCHW applied on the device."""
    import torch

    out = []
    with torch.no_grad():
        for start in range(0, len(images_uint8), batch):
            x = to_float(images_uint8[start:start + batch], device)
            if shift is not None:
                x = shift(x).clamp_(0.0, 1.0)
            out.append(model.forward_features(x).float().cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


# ----------------------------------------------------------------------------- shifts
# Ruifrok & Johnston H&E-DAB stain vectors (the same matrix as skimage.color.rgb2hed).
_RGB_FROM_HED = np.array([[0.65, 0.70, 0.29],
                          [0.07, 0.99, 0.11],
                          [0.27, 0.57, 0.78]], dtype=np.float64)
_HED_FROM_RGB = np.linalg.inv(_RGB_FROM_HED)
_LOG_ADJUST = np.log(1e-6)


def stain_shift(x, factor: float):
    """Scale the haematoxylin (H) channel of H&E images by (1 + factor).

    x: float tensor [B,3,H,W] in [0,1]. factor > 0 darker / more nuclear stain,
    factor < 0 lighter. Colour deconvolution as in skimage.color.rgb2hed/hed2rgb, but
    without clipping negative stain values, so that factor = 0 returns the image unchanged."""
    import torch

    dev, dt = x.device, torch.float32
    hed_from_rgb = torch.as_tensor(_HED_FROM_RGB, dtype=dt, device=dev)
    rgb_from_hed = torch.as_tensor(_RGB_FROM_HED, dtype=dt, device=dev)
    rgb = x.permute(0, 2, 3, 1).clamp(min=1e-6)
    hed = (torch.log(rgb) / _LOG_ADJUST) @ hed_from_rgb
    hed[..., 0] = hed[..., 0] * max(0.0, 1.0 + float(factor))
    out = torch.exp((hed * _LOG_ADJUST) @ rgb_from_hed)
    return out.clamp(0.0, 1.0).permute(0, 3, 1, 2).contiguous()


def make_shift(family: str, severity: float, factor: float):
    """Batched shift function for an image family; factor = signed geometry factor."""
    from encmonitor.shifts import ShiftSpec, apply_tensor_shift

    sev = float(severity) * abs(float(factor))
    sign = 1.0 if factor > 0 else -1.0
    if family == "stain":
        return lambda x: stain_shift(x, sign * sev)
    if family in ("brightness", "contrast", "blur"):
        spec = ShiftSpec(family, sev, sign)
        return lambda x: apply_tensor_shift(x, spec)
    raise ValueError(f"not an image family: {family}")

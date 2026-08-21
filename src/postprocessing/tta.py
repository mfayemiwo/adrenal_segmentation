"""Test-time augmentation, carried over from the prior published pipeline.

Applies a fixed set of invertible spatial transforms, averages the sigmoid
probabilities in the original orientation, and returns a single fused
probability map per slice.
"""
from __future__ import annotations

import torch


def _flip_h(x):
    return torch.flip(x, dims=[-1])


def _flip_v(x):
    return torch.flip(x, dims=[-2])


def _rot90(x):
    return torch.rot90(x, k=1, dims=[-2, -1])


_TRANSFORMS = {
    "hflip": (_flip_h, _flip_h),
    "vflip": (_flip_v, _flip_v),
    "rot90": (_rot90, lambda x: torch.rot90(x, k=-1, dims=[-2, -1])),
}


@torch.no_grad()
def tta_predict(model, images: torch.Tensor, transforms: tuple[str, ...] = ("hflip", "vflip", "rot90")) -> torch.Tensor:
    """images: (B, C, H, W). Returns fused probability map (B, num_classes, H, W)."""
    model.eval()
    probs = torch.sigmoid(model(images))

    for name in transforms:
        if name not in _TRANSFORMS:
            raise KeyError(f"unknown TTA transform '{name}'")
        forward, inverse = _TRANSFORMS[name]
        augmented = forward(images)
        pred = torch.sigmoid(model(augmented))
        probs = probs + inverse(pred)

    return probs / (len(transforms) + 1)

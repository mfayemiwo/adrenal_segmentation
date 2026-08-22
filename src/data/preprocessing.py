"""CT preprocessing utilities: resampling, HU windowing, normalisation.

These mirror the preprocessing stage described in the prior published
pipeline (Fayemiwo et al., 2025) so that Stage A (gate) and Stage B
(segmenter) see CT volumes in a consistent, comparable representation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import SimpleITK as sitk
except ImportError:  # pragma: no cover - optional at import time for unit tests
    sitk = None


@dataclass
class PreprocessConfig:
    spacing: tuple[float, float, float] = (1.0, 1.0, 2.0)
    hu_window: tuple[float, float] = (-135.0, 215.0)


def resample_volume(volume, original_spacing, target_spacing, is_mask: bool = False):
    """Resample a numpy volume (Z, Y, X) to `target_spacing`.

    Requires SimpleITK. Kept as a thin, swappable wrapper so a different
    resampling backend (e.g. scipy.ndimage.zoom) can be substituted without
    touching callers.
    """
    if sitk is None:
        raise ImportError("SimpleITK is required for resample_volume(); pip install SimpleITK")

    image = sitk.GetImageFromArray(volume.astype(np.float32))
    image.SetSpacing(tuple(reversed(original_spacing)))  # sitk is (x, y, z)

    original_size = image.GetSize()
    resample_factor = [o / t for o, t in zip(image.GetSpacing(), reversed(target_spacing))]
    new_size = [int(round(sz * f)) for sz, f in zip(original_size, resample_factor)]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(tuple(reversed(target_spacing)))
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_mask else sitk.sitkBSpline)

    resampled = resampler.Execute(image)
    return sitk.GetArrayFromImage(resampled)


def apply_hu_window(volume: np.ndarray, hu_window: tuple[float, float]) -> np.ndarray:
    """Clip to a soft-tissue HU window and rescale to [0, 1]."""
    lo, hi = hu_window
    clipped = np.clip(volume, lo, hi)
    return (clipped - lo) / max(hi - lo, 1e-6)


def normalise_intensity(volume: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance normalisation after HU windowing."""
    mean, std = volume.mean(), volume.std()
    if std < 1e-6:
        return volume - mean
    return (volume - mean) / std


def preprocess_volume(volume: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    windowed = apply_hu_window(volume, cfg.hu_window)
    return normalise_intensity(windowed)

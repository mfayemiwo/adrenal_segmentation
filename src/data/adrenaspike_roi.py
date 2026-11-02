"""Ground-truth-free gland-specific ROI localisation and reversible cropping."""
from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter


def locate(prob: np.ndarray) -> tuple[int, int, int]:
    """Use *predicted* probability only. Never inspect mask to choose a crop."""
    if prob.ndim != 3:
        raise ValueError("Expected (z,y,x) probability volume")
    smooth = gaussian_filter(prob.astype(np.float32), sigma=(1.5, 3, 3))
    return tuple(int(i) for i in np.unravel_index(np.argmax(smooth), smooth.shape))


def bounds(center, patch_shape, volume_shape):
    result = []
    for c, p, n in zip(center, patch_shape, volume_shape):
        start = c - p//2
        end = start+p
        v0, v1 = max(start, 0), min(end, n)
        result.append((v0, v1, v0-start, v0-start+v1-v0))
    return result


def crop(volume: np.ndarray, center, shape, pad=0):
    if volume.ndim != 3:
        raise ValueError("crop expects 3D array")
    b = bounds(center, shape, volume.shape)
    out = np.full(shape, pad, dtype=volume.dtype)
    out[tuple(slice(t[2], t[3]) for t in b)] = volume[
        tuple(slice(t[0], t[1]) for t in b)
    ]
    return out


def insert(volume: np.ndarray, patch: np.ndarray, center, blend=True):
    if volume.ndim != 3 or patch.ndim != 3:
        raise ValueError("insert expects (z,y,x) arrays")
    b = bounds(center, patch.shape, volume.shape)
    dst = tuple(slice(t[0],t[1]) for t in b)
    src = tuple(slice(t[2],t[3]) for t in b)
    if blend:
        windows = [np.maximum(np.hanning(n+2)[1:-1], 0.02) for n in patch.shape]
        taper = windows[0][:,None,None]*windows[1][None,:,None]*windows[2][None,None,:]
        weight = taper[src]
        volume[dst] = volume[dst]*(1-weight)+patch[src]*weight
    else:
        volume[dst] = patch[src]
    return volume

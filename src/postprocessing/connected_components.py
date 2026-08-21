"""Connected-component pruning, carried over from the prior published pipeline.

Removes small disconnected mask fragments (a common source of false
positives on anatomically similar but incorrect structures) and reports the
false-positive-component count, which is one of the primary end-to-end
outcome metrics for this project (see src/evaluation/evaluate_pipeline.py).
"""
from __future__ import annotations

import numpy as np

try:
    from scipy import ndimage
except ImportError as exc:  # pragma: no cover
    raise ImportError("scipy is required for connected-component pruning; pip install scipy") from exc


def remove_small_components(mask: np.ndarray, min_voxels: int = 30) -> tuple[np.ndarray, int]:
    """mask: binary array (2D slice or 3D volume). Returns (pruned_mask,
    num_components_removed)."""
    labeled, n_components = ndimage.label(mask)
    if n_components == 0:
        return mask, 0

    sizes = ndimage.sum(mask, labeled, index=range(1, n_components + 1))
    removed = 0
    pruned = mask.copy()
    for component_id, size in enumerate(sizes, start=1):
        if size < min_voxels:
            pruned[labeled == component_id] = 0
            removed += 1
    return pruned, removed


def keep_largest_k_components(mask: np.ndarray, k: int = 2) -> np.ndarray:
    """Retain only the `k` largest connected components (e.g. k=2 for left +
    right adrenal glands, discarding everything else regardless of size)."""
    labeled, n_components = ndimage.label(mask)
    if n_components <= k:
        return mask

    sizes = ndimage.sum(mask, labeled, index=range(1, n_components + 1))
    keep_ids = np.argsort(sizes)[::-1][:k] + 1
    return np.isin(labeled, keep_ids).astype(mask.dtype)


def count_components(mask: np.ndarray) -> int:
    _, n_components = ndimage.label(mask)
    return int(n_components)

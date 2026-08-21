"""Uncertainty-driven adaptive slice buffering — novel contribution #2.

The design review recommended a *fixed* safety buffer ("if slices 50-58 are
positive, send ~47-61 to the segmenter"). We keep the fixed buffer as the
floor (ablation arm F disables everything beyond it) but add an adaptive
extension: near a positive run's boundary, if the gate's probability is
still close to the decision threshold (i.e. the model is unsure whether the
gland has really ended), the buffer keeps growing until the model becomes
confident the slice is negative, up to a hard cap. This directly targets the
answer's stated failure mode — incomplete organ coverage at ambiguous
boundary slices — with a mechanism that adapts to how ambiguous each
patient's anatomy actually is, rather than one constant margin for everyone.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BufferConfig:
    fixed_buffer: int = 4
    max_expansion: int = 6
    uncertainty_threshold: float = 0.15  # expand while distance-from-threshold < this
    operating_threshold: float = 0.5     # decision threshold chosen for target sensitivity


def slice_uncertainty(probs: np.ndarray, operating_threshold: float) -> np.ndarray:
    """Distance-from-decision-boundary uncertainty in [0, 1]; 1 = maximally
    uncertain (prob == operating_threshold), 0 = fully confident (prob in {0,1})."""
    denom = max(operating_threshold, 1 - operating_threshold, 1e-6)
    return 1.0 - (np.abs(probs - operating_threshold) / denom)


def positive_runs(probs: np.ndarray, operating_threshold: float) -> list[tuple[int, int]]:
    """Contiguous [start, end) index ranges where probs >= operating_threshold."""
    positive = probs >= operating_threshold
    runs = []
    start = None
    for i, is_pos in enumerate(positive):
        if is_pos and start is None:
            start = i
        elif not is_pos and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(probs)))
    return runs


def expand_run(run: tuple[int, int], probs: np.ndarray, cfg: BufferConfig) -> tuple[int, int]:
    """Grow a positive run outward: first by the fixed buffer, then further
    while the boundary slice's uncertainty exceeds `cfg.uncertainty_threshold`,
    up to `cfg.max_expansion` additional slices per side."""
    start, end = run
    n = len(probs)
    uncertainty = slice_uncertainty(probs, cfg.operating_threshold)

    lo = max(0, start - cfg.fixed_buffer)
    hi = min(n, end + cfg.fixed_buffer)

    expanded_left = 0
    while expanded_left < cfg.max_expansion and lo > 0 and uncertainty[lo - 1] > cfg.uncertainty_threshold:
        lo -= 1
        expanded_left += 1

    expanded_right = 0
    while expanded_right < cfg.max_expansion and hi < n and uncertainty[hi] > cfg.uncertainty_threshold:
        hi += 1
        expanded_right += 1

    return lo, hi


def build_slice_inclusion_mask(probs: np.ndarray, cfg: BufferConfig) -> np.ndarray:
    """Return a boolean mask over the full volume: True for slices that
    should be sent to the segmenter (positive run + fixed buffer + any
    uncertainty-driven expansion)."""
    mask = np.zeros(len(probs), dtype=bool)
    for run in positive_runs(probs, cfg.operating_threshold):
        lo, hi = expand_run(run, probs, cfg)
        mask[lo:hi] = True
    return mask


def buffering_report(probs: np.ndarray, cfg: BufferConfig) -> dict:
    """Diagnostics for the ablation study: how many slices the fixed buffer
    alone would include vs. how many the uncertainty expansion added."""
    fixed_only_cfg = BufferConfig(fixed_buffer=cfg.fixed_buffer, max_expansion=0,
                                   uncertainty_threshold=cfg.uncertainty_threshold,
                                   operating_threshold=cfg.operating_threshold)
    fixed_mask = build_slice_inclusion_mask(probs, fixed_only_cfg)
    full_mask = build_slice_inclusion_mask(probs, cfg)
    return {
        "fixed_buffer_slice_count": int(fixed_mask.sum()),
        "expanded_slice_count": int(full_mask.sum()),
        "additional_slices_from_uncertainty": int(full_mask.sum() - fixed_mask.sum()),
    }

import numpy as np

from src.training.uncertainty_buffer import (
    BufferConfig,
    build_slice_inclusion_mask,
    buffering_report,
    positive_runs,
    slice_uncertainty,
)


def test_positive_runs_detects_contiguous_block():
    probs = np.array([0.1, 0.1, 0.9, 0.95, 0.9, 0.1, 0.1])
    runs = positive_runs(probs, operating_threshold=0.5)
    assert runs == [(2, 5)]


def test_uncertainty_expansion_grows_beyond_fixed_buffer():
    # A positive run at [10, 15) with a slow, ambiguous decay in probability
    # just past the fixed buffer boundary should trigger extra expansion.
    probs = np.zeros(30)
    probs[10:15] = 0.95
    probs[15:18] = 0.45  # near the 0.5 operating threshold: high uncertainty
    probs[18:] = 0.02

    cfg = BufferConfig(fixed_buffer=2, max_expansion=6, uncertainty_threshold=0.15, operating_threshold=0.5)
    mask = build_slice_inclusion_mask(probs, cfg)

    # fixed buffer alone would stop at index 17 (15 + 2); uncertainty should push further
    assert mask[17], "uncertainty expansion should include index 17"
    report = buffering_report(probs, cfg)
    assert report["additional_slices_from_uncertainty"] > 0


def test_slice_uncertainty_bounds():
    probs = np.array([0.0, 0.5, 1.0])
    u = slice_uncertainty(probs, operating_threshold=0.5)
    assert np.isclose(u[1], 1.0)  # maximally uncertain at the threshold
    assert np.isclose(u[0], 0.0)
    assert np.isclose(u[2], 0.0)

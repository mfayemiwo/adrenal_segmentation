import numpy as np

from src.evaluation.metrics import (
    dice_score,
    f_beta_score,
    hausdorff_distance_95,
    iou_score,
    normalized_surface_dice,
    npv,
    pr_auc,
    select_threshold_for_sensitivity,
    sensitivity,
    specificity,
)


def test_dice_and_iou_perfect_overlap():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:10, 5:10] = 1
    assert dice_score(mask, mask) == 1.0
    assert iou_score(mask, mask) == 1.0


def test_dice_both_empty_is_perfect():
    empty = np.zeros((20, 20), dtype=np.uint8)
    assert dice_score(empty, empty) == 1.0


def test_hd95_and_nsd_zero_for_identical_masks():
    mask = np.zeros((30, 30), dtype=np.uint8)
    mask[10:20, 10:20] = 1
    assert hausdorff_distance_95(mask, mask) == 0.0
    assert normalized_surface_dice(mask, mask, tolerance_mm=1.0) == 1.0


def test_sensitivity_specificity_npv():
    probs = np.array([0.9, 0.8, 0.2, 0.1, 0.6])
    labels = np.array([1, 1, 0, 0, 1])
    assert sensitivity(probs, labels, 0.5) == 1.0
    assert specificity(probs, labels, 0.5) == 1.0
    assert npv(probs, labels, 0.5) == 1.0


def test_f_beta_favours_recall():
    probs = np.array([0.9, 0.9, 0.9, 0.1])
    labels = np.array([1, 1, 0, 0])  # one false positive, no false negatives
    f2 = f_beta_score(probs, labels, 0.5, beta=2.0)
    assert f2 > 0.8  # recall is perfect, so F2 should stay high despite the FP


def test_pr_auc_range():
    probs = np.array([0.9, 0.1, 0.8, 0.2])
    labels = np.array([1, 0, 1, 0])
    score = pr_auc(probs, labels)
    assert 0.0 <= score <= 1.0


def test_select_threshold_for_sensitivity_meets_target():
    rng = np.random.RandomState(0)
    labels = rng.binomial(1, 0.3, size=200)
    probs = np.clip(labels * 0.6 + rng.normal(0, 0.2, size=200), 0, 1)
    threshold = select_threshold_for_sensitivity(probs, labels, target_sensitivity=0.9)
    assert sensitivity(probs, labels, threshold) >= 0.9

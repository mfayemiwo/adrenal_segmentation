"""Evaluation metrics for both the gate (Stage A) and the segmenter (Stage B),
plus the full-volume outcome metrics that motivate the whole cascade design.

Segmentation overlap/boundary metrics (Dice, IoU, HD95, NSD, volume error)
are computed per the standard definitions used in nnU-Net/MONAI-style
benchmarking. Gate metrics deliberately foreground sensitivity/NPV/F2/PR-AUC
over accuracy, per the design review's core recommendation.
"""
from __future__ import annotations

import numpy as np

try:
    from scipy import ndimage
except ImportError as exc:  # pragma: no cover
    raise ImportError("scipy is required for metrics; pip install scipy") from exc

try:
    from sklearn.metrics import average_precision_score, brier_score_loss
except ImportError as exc:  # pragma: no cover
    raise ImportError("scikit-learn is required for PR-AUC/Brier metrics; pip install scikit-learn") from exc


# ---------------------------------------------------------------------------
# Segmentation overlap / boundary metrics
# ---------------------------------------------------------------------------

def dice_score(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    pred, target = pred.astype(bool), target.astype(bool)
    intersection = np.logical_and(pred, target).sum()
    denom = pred.sum() + target.sum()
    if denom == 0:
        return 1.0  # both empty: trivially correct, avoids penalising true-negative slices
    return float((2 * intersection + eps) / (denom + eps))


def iou_score(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    pred, target = pred.astype(bool), target.astype(bool)
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    if union == 0:
        return 1.0
    return float((intersection + eps) / (union + eps))


def _surface_distances(pred: np.ndarray, target: np.ndarray, spacing: tuple[float, ...] | None = None):
    pred, target = pred.astype(bool), target.astype(bool)
    if not pred.any() or not target.any():
        return None  # undefined when either mask is empty

    pred_surface = np.logical_xor(pred, ndimage.binary_erosion(pred))
    target_surface = np.logical_xor(target, ndimage.binary_erosion(target))

    target_dt = ndimage.distance_transform_edt(~target_surface, sampling=spacing)
    pred_dt = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)

    pred_to_target = target_dt[pred_surface]
    target_to_pred = pred_dt[target_surface]
    return pred_to_target, target_to_pred


def hausdorff_distance_95(pred: np.ndarray, target: np.ndarray, spacing: tuple[float, ...] | None = None) -> float:
    distances = _surface_distances(pred, target, spacing)
    if distances is None:
        return float("nan")
    pred_to_target, target_to_pred = distances
    all_distances = np.concatenate([pred_to_target, target_to_pred])
    return float(np.percentile(all_distances, 95))


def normalized_surface_dice(pred: np.ndarray, target: np.ndarray, tolerance_mm: float = 2.0,
                             spacing: tuple[float, ...] | None = None) -> float:
    """Fraction of predicted/reference surface within `tolerance_mm` of the
    other surface — the standard NSD/"surface Dice" formulation."""
    distances = _surface_distances(pred, target, spacing)
    if distances is None:
        return float("nan")
    pred_to_target, target_to_pred = distances
    within_tol = np.sum(pred_to_target <= tolerance_mm) + np.sum(target_to_pred <= tolerance_mm)
    total = len(pred_to_target) + len(target_to_pred)
    return float(within_tol / total) if total else float("nan")


def volume_error(pred: np.ndarray, target: np.ndarray, voxel_volume_mm3: float = 1.0) -> float:
    """Signed volume error in the same units as `voxel_volume_mm3` (e.g. mm^3
    if spacing-derived, or voxel count if left at 1.0)."""
    return float((pred.sum() - target.sum()) * voxel_volume_mm3)


# ---------------------------------------------------------------------------
# Gate metrics — sensitivity-first, per the design review
# ---------------------------------------------------------------------------

def confusion_counts(probs: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    preds = (probs >= threshold).astype(int)
    tp = int(np.sum((preds == 1) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def sensitivity(probs: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    c = confusion_counts(probs, labels, threshold)
    denom = c["tp"] + c["fn"]
    return c["tp"] / denom if denom else float("nan")


def specificity(probs: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    c = confusion_counts(probs, labels, threshold)
    denom = c["tn"] + c["fp"]
    return c["tn"] / denom if denom else float("nan")


def npv(probs: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    c = confusion_counts(probs, labels, threshold)
    denom = c["tn"] + c["fn"]
    return c["tn"] / denom if denom else float("nan")


def f_beta_score(probs: np.ndarray, labels: np.ndarray, threshold: float, beta: float = 2.0) -> float:
    """F2 weights recall (sensitivity) 4x more than precision — appropriate
    given the review's asymmetric cost argument (missed positive slices are
    far worse than wasted segmentation attempts)."""
    c = confusion_counts(probs, labels, threshold)
    precision_denom = c["tp"] + c["fp"]
    recall_denom = c["tp"] + c["fn"]
    precision = c["tp"] / precision_denom if precision_denom else 0.0
    recall = c["tp"] / recall_denom if recall_denom else 0.0
    if precision + recall == 0:
        return 0.0
    beta2 = beta ** 2
    return (1 + beta2) * precision * recall / (beta2 * precision + recall)


def pr_auc(probs: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(average_precision_score(labels, probs))


def calibration_brier(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(brier_score_loss(labels, probs))


def select_threshold_for_sensitivity(probs: np.ndarray, labels: np.ndarray, target_sensitivity: float = 0.98) -> float:
    """Sweep candidate thresholds and return the highest one that still
    achieves >= target_sensitivity on the validation set — i.e. the
    threshold is chosen to satisfy the sensitivity constraint first, and only
    then to minimise unnecessary segmentation attempts (specificity) as a
    secondary objective."""
    candidates = np.unique(probs)
    best_threshold, best_specificity = 0.0, -1.0
    for t in candidates:
        sens = sensitivity(probs, labels, t)
        if sens >= target_sensitivity:
            spec = specificity(probs, labels, t)
            if spec > best_specificity:
                best_threshold, best_specificity = float(t), spec
    return best_threshold


# ---------------------------------------------------------------------------
# Full-volume outcome metrics — the pipeline-level payoff of the gate design
# ---------------------------------------------------------------------------

def empty_slice_false_positive_rate(pred_masks: list[np.ndarray], target_masks: list[np.ndarray]) -> float:
    """Fraction of slices with an *empty* ground-truth mask on which the
    pipeline nonetheless predicted a non-empty mask. This is the headline
    metric the slice-gate is designed to improve, and prior segmentation-only
    papers typically do not report it."""
    fp_count, empty_count = 0, 0
    for pred, target in zip(pred_masks, target_masks):
        if not target.any():
            empty_count += 1
            if pred.any():
                fp_count += 1
    return fp_count / empty_count if empty_count else float("nan")


def false_positive_components_per_scan(pred_masks: list[np.ndarray], target_masks: list[np.ndarray]) -> float:
    """Mean number of predicted connected components that do not overlap any
    ground-truth component, averaged per scan."""
    from src.postprocessing.connected_components import count_components

    total_fp_components = 0
    for pred, target in zip(pred_masks, target_masks):
        labeled_pred, n_pred = ndimage.label(pred)
        if n_pred == 0:
            continue
        for component_id in range(1, n_pred + 1):
            component_mask = labeled_pred == component_id
            if not np.any(component_mask & target.astype(bool)):
                total_fp_components += 1
    return total_fp_components / max(len(pred_masks), 1)

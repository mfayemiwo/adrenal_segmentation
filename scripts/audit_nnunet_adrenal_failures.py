#!/usr/bin/env python3
"""CPU-only audit of low-Dice nnU-Net adrenal CT test cases.

This is a diagnostic of *already evaluated* cases, not a mechanism for
discarding failures or learning postprocessing settings on held-out patients.
Inspect GT/prediction independently for image-label corruption before any
prespecified exclusion. Requires only existing numpy/nibabel environment.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np


def dice(a: np.ndarray, b: np.ndarray) -> float:
    denom = int(a.sum()) + int(b.sum())
    return float(2 * np.logical_and(a, b).sum() / denom) if denom else 1.0


def describe(mask: np.ndarray, affine: np.ndarray) -> str:
    voxels = int(mask.sum())
    if not voxels:
        return "0 voxels; no centroid"
    # Compute centroid in world space, not just an uncalibrated z index.
    centroid = np.asarray(np.nonzero(mask), dtype=np.float64).mean(axis=1)
    world = nib.affines.apply_affine(affine, centroid)
    return (f"{voxels} voxels; centroid mm="
            f"({world[0]:.1f}, {world[1]:.1f}, {world[2]:.1f})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path,
                        default=Path("runs/nnunet3d_holdout_metrics_fold0.csv"))
    parser.add_argument("--dataset-dir", type=Path,
                        default=Path("nnunet_raw/Dataset701_AMOSAdrenalCT"))
    parser.add_argument("--pred-dir", type=Path,
                        default=Path("runs/nnunet3d_holdout_predictions_fold0"))
    parser.add_argument("--worst", type=int, default=5)
    parser.add_argument("--case-id", type=str, default=None)
    args = parser.parse_args()

    with args.metrics.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise RuntimeError("No metrics found")
    if args.case_id:
        selected = [r for r in rows if r["case_id"] == args.case_id]
        if not selected:
            raise ValueError(f"Unknown case ID: {args.case_id}")
    else:
        # Prioritise a zero-Dice gland and then low mean bilateral Dice.
        selected = sorted(
            rows,
            key=lambda row: (min(float(row["dice_left"]),
                                 float(row["dice_right"])),
                             (float(row["dice_left"]) +
                              float(row["dice_right"])) / 2),
        )[:args.worst]

    print(f"Reviewing {len(selected)} already-evaluated cases "
          "(never exclude a case solely for low Dice).", flush=True)
    for row in selected:
        case_id = row["case_id"]
        gt_path = args.dataset_dir / "labelsTs" / f"{case_id}.nii.gz"
        pred_path = args.pred_dir / f"{case_id}.nii.gz"
        reference = nib.load(str(gt_path))
        prediction = nib.load(str(pred_path))
        if prediction.shape != reference.shape:
            raise ValueError(f"{case_id}: pred/GT volume shape mismatch")
        gt = np.asarray(reference.dataobj)
        pred = np.asarray(prediction.dataobj)
        if not np.isin(gt, (0, 1, 2)).all() or not np.isin(pred, (0, 1, 2)).all():
            raise ValueError(f"{case_id}: unexpected segmentation labels")
        affine_equal = bool(np.allclose(
            reference.affine, prediction.affine, rtol=0, atol=1e-3
        ))
        print(f"\nCase {case_id}: original bilateral Dice="
              f"{(float(row['dice_left']) + float(row['dice_right'])) / 2:.4f}")
        print(f"  shape={reference.shape}, affines_match={affine_equal}, "
              f"spacing={reference.header.get_zooms()[:3]}")
        for class_id, gland in ((1, "left"), (2, "right")):
            gm, pm = gt == class_id, pred == class_id
            # An unexpectedly large wrong-class overlap can suggest a
            # laterality-label issue; it is NOT automatic evidence of one.
            opposite = 3 - class_id
            swap_overlap = dice(gm, pred == opposite)
            print(f"  {gland} GT: {describe(gm, reference.affine)}")
            print(f"  {gland} pred: {describe(pm, prediction.affine)}")
            print(f"  {gland} Dice={dice(gm, pm):.4f}; "
                  f"GT vs opposite predicted class Dice={swap_overlap:.4f}")
        if not affine_equal:
            print("  WARNING: GT and prediction affines differ; examine"
                  " registration/header convention before interpreting Dice")
        if min(float(row["dice_left"]), float(row["dice_right"])) <= 1e-7:
            print("  FLAG: complete non-overlap in at least one gland;"
                  " inspect the CT and both masks visually.")

    print("\nDo not remove cases based on predictions; retain the"
          " full 100-patient primary evaluation.", flush=True)


if __name__ == "__main__":
    main()

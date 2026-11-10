#!/usr/bin/env python3
"""Score the frozen external TotalSegmentator v3 test cohort using AMOS metrics.

Full-volume, patient-level, class-wise Dice, HD95 and NSD at 2mm. No
postprocessing, threshold tuning or optional GPU benchmark hooks. Requires
prediction and reference CT geometry to match; checks exactly the pre-registered
test IDs. Includes bootstrap 95% CIs, empty-output accounting and the QC status
from the external manifest. This is NOT a heldout AMOS paired comparison.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evaluation.metrics import hausdorff_distance_95, normalized_surface_dice


def score_case(reference_path: Path, prediction_path: Path, image_path: Path):
    ref = nib.load(str(reference_path))
    pred = nib.load(str(prediction_path))
    source = nib.load(str(image_path))
    for name, obj in (("prediction", pred), ("input CT", source)):
        if obj.shape != ref.shape or not np.allclose(
            obj.affine, ref.affine, atol=1e-3, rtol=0
        ):
            raise ValueError(f"{reference_path.name}: {name} geometry mismatch")
    y = np.asarray(ref.dataobj, dtype=np.uint8)
    p = np.asarray(pred.dataobj, dtype=np.uint8)
    if not np.isin(y, (0, 1, 2)).all() or not np.isin(p, (0, 1, 2)).all():
        raise ValueError("Unexpected gland label beyond 0, 1, 2")
    # Transpose matches the existing nnU-Net evaluator and its mm spacings.
    truth, out = y.transpose(2, 0, 1), p.transpose(2, 0, 1)
    sx, sy, sz = ref.header.get_zooms()[:3]
    result = {}
    for gland_id, side in ((1, "left"), (2, "right")):
        gt, pr = truth == gland_id, out == gland_id
        gt_vox = int(gt.sum())
        pr_vox = int(pr.sum())
        if gt_vox == 0:
            raise ValueError(f"{reference_path.name}: empty external GT {side}")
        result[f"gt_voxels_{side}"] = gt_vox
        result[f"pred_voxels_{side}"] = pr_vox
        result[f"dice_{side}"] = (
            2.0 * np.logical_and(gt, pr).sum() / (gt_vox + pr_vox)
            if gt_vox + pr_vox else 1.
        )
        if gt_vox and pr_vox:
            extent = np.argwhere(pr | gt)
            lo = np.maximum(extent.min(axis=0) - 2, 0)
            hi = np.minimum(extent.max(axis=0) + 3, gt.shape)
            box = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
            result[f"hd95_{side}_mm"] = float(
                hausdorff_distance_95(pr[box], gt[box], spacing=(sz, sx, sy))
            )
            result[f"nsd2mm_{side}"] = float(
                normalized_surface_dice(
                    pr[box], gt[box], tolerance_mm=2., spacing=(sz, sx, sy)
                )
            )
        else:
            result[f"hd95_{side}_mm"] = float("nan")
            result[f"nsd2mm_{side}"] = float("nan")
    result["dice_bilateral"] = (
        result["dice_left"] + result["dice_right"]
    ) / 2
    return result


def finite_mean_and_n(rows, key):
    values = [float(row[key]) for row in rows if math.isfinite(float(row[key]))]
    return {"mean": float(np.mean(values)) if values else None,
            "valid_cases": len(values), "total_cases": len(rows)}


def bootstrap_mean_ci(rows, key, *, seed=20261110, draws=10000):
    values = np.array([float(row[key]) for row in rows], dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return {"mean": None if not values.size else float(values.mean()),
                "ci95": None, "n": int(values.size)}
    rng = np.random.default_rng(seed)
    estimates = np.mean(
        values[rng.integers(0, len(values), size=(draws, len(values)))],
        axis=1,
    )
    return {
        "mean": float(values.mean()),
        "ci95": [float(x) for x in np.quantile(estimates, [0.025, 0.975])],
        "n": len(values), "draws": draws, "seed": seed,
    }


def run(args):
    manifest_path = args.dataset_dir / "external_test_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected = tuple(sorted(manifest["case_ids"]))
    if len(expected) != len(set(expected)):
        raise ValueError("External test manifest contains duplicate cases")
    if len(expected) != 76 and not args.allow_nonstandard_count:
        raise ValueError("Expected exactly 76 frozen external test CT cases")
    if manifest.get("source_partition") != "test":
        raise ValueError("Not the official TotalSegmentator test split")
    expected_nifti = {cid + ".nii.gz" for cid in expected}
    references = {p.name for p in (args.dataset_dir / "labelsTs").glob("*.nii.gz")}
    predictions = {p.name for p in args.pred_dir.glob("*.nii.gz")}
    images = {p.name for p in (args.dataset_dir / "imagesTs").glob("*_0000.nii.gz")}
    if references != expected_nifti or predictions != expected_nifti:
        raise ValueError("Missing or unexpected ground-truth/predicted external-test NIfTIs")
    if images != {cid + "_0000.nii.gz" for cid in expected}:
        raise ValueError("External test images do not match frozen manifest")
    rows = []
    for i, case_id in enumerate(expected, 1):
        metrics = score_case(
            args.dataset_dir / "labelsTs" / f"{case_id}.nii.gz",
            args.pred_dir / f"{case_id}.nii.gz",
            args.dataset_dir / "imagesTs" / f"{case_id}_0000.nii.gz",
        )
        rows.append({"case_id": case_id, **metrics})
        print(f"Scored {i}/{len(expected)} {case_id}: "
              f"bilateral Dice={metrics['dice_bilateral']:.4f}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    score_keys = (
        "dice_left", "dice_right", "dice_bilateral",
        "hd95_left_mm", "hd95_right_mm",
        "nsd2mm_left", "nsd2mm_right",
    )
    score_summary = {key: finite_mean_and_n(rows, key) for key in score_keys}
    intervals = {key: bootstrap_mean_ci(rows, key)
                 for key in ("dice_left", "dice_right", "dice_bilateral")}
    report = {
        "model": "original AMOS-only dataset701 3d_fullres fold0 nnU-Net",
        "test_dataset": "TotalSegmentator v3 official test subset, bilateral labels",
        "cohort_size": len(rows),
        "cohort_status": manifest["qc_status"],
        "predefined_method": "no retraining or postprocessing, patient-level full-volume Dice/HD95/NSD",
        "scores": score_summary,
        "bootstrap_95pct_CI": intervals,
        "empty_prediction_cases": {
            side: sum(r[f"pred_voxels_{side}"] == 0 for r in rows)
            for side in ("left", "right")
        },
        "warnings": [
            "This is a different dataset from AMOS; mean values are NOT paired comparisons.",
            "Unreviewed cases and annotation source limitations must be disclosed.",
            "HD95 and NSD are NaN when prediction empty; valid denominators reported.",
            "Do not tune model/postprocessor or retrospectively exclude hard test cases.",
            "A high Dice here alone does not establish a CNN-SNN contribution.",
        ],
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "test_cases": len(rows), "mean_bilateral_dice":
            score_summary["dice_bilateral"]["mean"],
        "external_summary": str(summary_path),
        "cohort_status": report["cohort_status"],
    }, indent=2), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", type=Path,
                   default=Path("nnunet_raw/ExternalTotalSegV3AdrenalTest_v1"))
    p.add_argument("--pred-dir", type=Path,
                   default=Path("runs/nnunet3d_external_tsv3_predictions_fold0"))
    p.add_argument("--output", type=Path,
                   default=Path("runs/nnunet3d_external_tsv3_metrics_fold0.csv"))
    p.add_argument("--allow-nonstandard-count", action="store_true",
                   help="For synthetic tests only")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()

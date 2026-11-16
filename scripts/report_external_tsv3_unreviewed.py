#!/usr/bin/env python3
"""DESCRIPTIVE audit of completed exploratory TotalSegmentator external scores.

Uses the fixed 76 patient-level per-case score CSV and geometry manifest, not
the CT voxels. No test-set tuning, filtering, outlier deletion, or relabelling.
Collects left/right Dice and HD95/NSD valid denominators plus evidence about
nine geometrically regridded cases. Outputs one immutable aggregate JSON for
manuscript preparation, separate from the original raw metric CSV.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def sha(path):
    dig = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            dig.update(chunk)
    return dig.hexdigest()


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mean_sd(x):
    return {
        "n": len(x), "mean": statistics.mean(x) if x else None,
        "sample_sd": statistics.stdev(x) if len(x) > 1 else None,
        "median": statistics.median(x) if x else None,
        "min": min(x) if x else None,
        "max": max(x) if x else None,
    }


def summary(rows, key):
    v = [float(r[key]) for r in rows if math.isfinite(float(r[key]))]
    return {**mean_sd(v), "not_defined_count": len(rows) - len(v)}


def report(metrics_path, original_path, geometry_path):
    manifest = json.loads(original_path.read_text(encoding="utf-8"))
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    original_ids = set(manifest["case_ids"])
    rows = read_rows(metrics_path)
    actual_ids = {r["case_id"] for r in rows}
    if (len(rows) != 76 or len(actual_ids) != 76 or
        actual_ids != original_ids or
        geometry.get("case_count") != 76):
        raise ValueError("All 76 original external test cases must be present")
    changed = set(geometry["regridded_cases"])
    if (geometry.get("regridded_count") != len(changed) or
        not changed <= actual_ids):
        raise ValueError("Mismatch between native-grid metrics and frozen geometry manifest")
    original = [r for r in rows if r["case_id"] not in changed]
    resampled = [r for r in rows if r["case_id"] in changed]
    scores = [float(r["dice_bilateral"]) for r in rows]
    audit = {
        "status": "DESCRIPTIVE_COMPLETE_NO_MODEL_TUNING",
        "model": "AMOS-only Dataset701 nnU-Net 3d_fullres fold0",
        "dataset": "TotalSegmentator v3 official test bilateral-positive subset",
        "official_parent_test_cases": 109,
        "scored_bilateral_nonempty_test_cases": len(rows),
        "unreviewed_reference_annotations": True,
        "sha256": {
            "full_original_metrics": sha(metrics_path),
            "original_76_case_manifest": sha(original_path),
            "geometry_prep_manifest": sha(geometry_path),
        },
        "score_distribution": {
            key: summary(rows, key) for key in (
                "dice_bilateral", "dice_left", "dice_right",
                "hd95_left_mm", "hd95_right_mm",
                "nsd2mm_left", "nsd2mm_right"
            )
        },
        "failure_counts": {
            "bilateral_dice_below_0_20": sum(x < .2 for x in scores),
            "bilateral_dice_below_0_50": sum(x < .5 for x in scores),
            "bilateral_dice_at_least_0_90": sum(x >= .9 for x in scores),
            "bilateral_dice_exact_zero": sum(x == 0 for x in scores),
            "empty_left_predictions": sum(int(r["pred_voxels_left"]) == 0 for r in rows),
            "empty_right_predictions": sum(int(r["pred_voxels_right"]) == 0 for r in rows),
        },
        "source_CT_geometry_subgroups_DESCRIPTIVE_NOT_CAUSAL": {
            "original_itk_compatible": {
                "n": len(original), "bilateral_dice": summary(original,"dice_bilateral")
            },
            "original_required_regridding": {
                "n": len(resampled), "bilateral_dice": summary(resampled,"dice_bilateral")
            },
        },
        "failure_examples_DO_NOT_TUNE_ON_THESE_CASES": [
            {
                "case_id": r["case_id"],
                "bilateral_dice": float(r["dice_bilateral"]),
                "dice_left": float(r["dice_left"]),
                "dice_right": float(r["dice_right"]),
                "source_CT_regridded": r["case_id"] in changed,
            }
            for r in sorted(rows, key=lambda r: float(r["dice_bilateral"]))[:10]
        ],
        "limits": [
            "This is observational external-test analysis, not model selection.",
            "Never exclude or relabel low-score cases based on predictions.",
            "Nine originally sheared CT cases were orthogonalised for inference; predictions restored to native grid.",
            "Geometric subgroups and other source differences are observational, not causal.",
            "Original test cohort was 109, only 76 cases had two nonempty source adrenal labels.",
            "Source mask annotation quality unreviewed and source-patient independence unverified.",
            "HD95 and NSD spacings in the original evaluator do not fully account for off-axis/sheared physical geometry in those nine CTs; report this limitation.",
            "Scores describe an AMOS-only model, not an enlarged-data model or an SNN.",
        ],
    }
    return audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics", type=Path, default=Path(
        "runs/nnunet3d_external_tsv3_metrics_fold0_ortho_native_unreviewed.csv"
    ))
    p.add_argument("--original-manifest", type=Path, default=Path(
        "nnunet_raw/ExternalTotalSegV3AdrenalTest_v1/external_test_manifest.json"
    ))
    p.add_argument("--geometry", type=Path, default=Path(
        "nnunet_raw/ExternalTotalSegV3AdrenalOrtho_v1/geometry_manifest.json"
    ))
    p.add_argument("--output", type=Path, default=Path(
        "runs/nnunet3d_external_tsv3_failure_audit_unreviewed.json"
    ))
    args = p.parse_args()
    result = report(args.metrics, args.original_manifest, args.geometry)
    if args.output.exists():
        p.error("Refusing to overwrite frozen external descriptive audit")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "status": result["status"],
        "case_count": result["scored_bilateral_nonempty_test_cases"],
        "Dice_bilateral": result["score_distribution"]["dice_bilateral"],
        "Dice_left": result["score_distribution"]["dice_left"],
        "Dice_right": result["score_distribution"]["dice_right"],
        "failures": result["failure_counts"],
        "regridded_subgroup": result[
            "source_CT_geometry_subgroups_DESCRIPTIVE_NOT_CAUSAL"
        ]["original_required_regridding"],
        "evidence_file": str(args.output),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()

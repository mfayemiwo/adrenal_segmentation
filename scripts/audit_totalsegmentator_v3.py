#!/usr/bin/env python3
"""Audit the 1,939-case TotalSegmentator v3 CT dataset before research training.

Never assume a NIfTI mask filename means that it contains a valid adrenal label.
A blank adrenal mask can indicate out-of-field anatomy, a missed annotation or
genuine absence. A kidney mask is only a weak coverage clue, never proof that an
adrenal should have been visible. This tool ONLY reads source images/masks and
writes reports; it does not copy, alter or merge any CT or segmentation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re

import nibabel as nib
import numpy as np

DEFAULT_ROOT = Path(
    "/mnt/userapps/isrcdl_apps/mfayemiwo/DaisyProject/data/"
    "totalsegmentator_v3/extracted/Totalsegmentator_dataset_v300"
)
FIELDS = (
    "case_id", "split", "study_type", "institute", "slice_thickness_metadata_mm",
    "ct_shape", "ct_spacing_mm", "left_voxels", "right_voxels",
    "left_mm3", "right_mm3", "left_aligned", "right_aligned",
    "left_binary", "right_binary", "both_present", "one_present",
    "both_empty", "kidney_left_present", "kidney_right_present",
    "kidney_coverage_clue", "review_priority", "candidate_train_only",
    "error",
)


def metadata_for(root: Path) -> dict[str, dict]:
    path = root / "meta.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing TotalSegmentator metadata: {path}")
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file, delimiter=";")
        if not reader.fieldnames or "image_id" not in reader.fieldnames:
            raise ValueError("Metadata must have semicolon-delimited image_id")
        records = {}
        for row in reader:
            case_id = str(row.get("image_id", "")).strip()
            if case_id:
                if case_id in records:
                    raise ValueError(f"Duplicate image_id in meta.csv: {case_id}")
                records[case_id] = row
    return records


def inspect_binary_mask(path: Path, ct: nib.Nifti1Image) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    img = nib.load(str(path))
    aligned = bool(
        img.ndim == 3 and img.shape == ct.shape
        and np.allclose(img.affine, ct.affine, atol=1e-3, rtol=0)
    )
    data = np.asanyarray(img.dataobj)
    binary = bool(np.all(np.logical_or(data == 0, data == 1)))
    voxels = int(np.count_nonzero(data))
    mm3 = float(voxels * abs(np.linalg.det(img.affine[:3, :3])))
    return {"voxels": voxels, "mm3": mm3,
            "aligned": aligned, "binary": binary}


def inspect_case(case: Path, info: dict) -> dict:
    case_id = case.name
    result = dict.fromkeys(FIELDS, "")
    result.update({
        "case_id": case_id, "split": info.get("split", ""),
        "study_type": info.get("study_type", ""),
        "institute": info.get("institute", ""),
        "slice_thickness_metadata_mm": info.get("slice_thickness", ""),
        "candidate_train_only": False,
    })
    try:
        ct_path = case / "ct.nii.gz"
        ct = nib.load(str(ct_path))
        if ct.ndim != 3:
            raise ValueError(f"CT must be 3D; found {ct.shape}")
        result["ct_shape"] = "x".join(map(str, ct.shape))
        result["ct_spacing_mm"] = "x".join(
            f"{v:.4f}" for v in ct.header.get_zooms()[:3]
        )

        segmentation_dir = case / "segmentations"
        sides = {}
        for side in ("left", "right"):
            stats = inspect_binary_mask(
                segmentation_dir / f"adrenal_gland_{side}.nii.gz", ct
            )
            sides[side] = stats
            for key, value in stats.items():
                result[f"{side}_{key}"] = value

        left = sides["left"]["voxels"]
        right = sides["right"]["voxels"]
        result["both_present"] = bool(left > 0 and right > 0)
        result["one_present"] = bool((left > 0) != (right > 0))
        result["both_empty"] = bool(left == 0 and right == 0)

        # Only read additional large volumes where they inform missing-label
        # review. Kidney visibility does NOT prove adrenal coverage.
        if left == 0 or right == 0:
            for side in ("left", "right"):
                kidney = segmentation_dir / f"kidney_{side}.nii.gz"
                if kidney.exists():
                    result[f"kidney_{side}_present"] = bool(
                        inspect_binary_mask(kidney, ct)["voxels"] > 0
                    )
                else:
                    result[f"kidney_{side}_present"] = ""
        result["kidney_coverage_clue"] = (
            result["kidney_left_present"] is True
            or result["kidney_right_present"] is True
        )
        result["review_priority"] = bool(
            (left == 0 or right == 0) and result["kidney_coverage_clue"]
        )
        # Candidates for future review ONLY. Do not train on this list before
        # image-duplication audit, annotation QA and research split approval.
        result["candidate_train_only"] = bool(
            result["both_present"]
            and all(s["aligned"] and s["binary"] for s in sides.values())
            and str(result["split"]).strip().lower() == "train"
        )
    except (OSError, ValueError, TypeError, EOFError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def save_reports(rows: list[dict], output_dir: Path, metadata_count: int) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "case_audit.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    candidates = [row for row in rows if row["candidate_train_only"] is True]
    with (output_dir / "candidates_not_train_ready.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(candidates)

    split_counts = Counter(str(row["split"]) for row in rows)
    summary = {
        "dataset": "TotalSegmentator v3",
        "metadata_rows": metadata_count,
        "scanned_case_directories": len(rows),
        "split_counts": dict(sorted(split_counts.items())),
        "both_adrenals_nonempty": sum(r["both_present"] is True for r in rows),
        "only_one_adrenal_nonempty": sum(r["one_present"] is True for r in rows),
        "both_adrenals_empty": sum(r["both_empty"] is True for r in rows),
        "empty_or_one_sided_with_kidney_coverage_clue": sum(
            r["review_priority"] is True for r in rows
        ),
        "unreadable_or_missing_geometry": sum(bool(r["error"]) for r in rows),
        "train_split_nonempty_bilateral_candidates_NOT_CLEARED": len(candidates),
        "warnings": [
            "A blank adrenal mask is NOT automatically a valid negative annotation.",
            "An empty gland plus kidney visibility is only a review clue, NOT proof of an error.",
            "All CTs may be resampled to 1.5 mm; use metadata slice_thickness for original acquisition.",
            "Do not mix any data into AMOS until duplicate-patient, annotation and split audits pass.",
            "These candidates are NOT approved training inputs; preserve all 100 AMOS heldout cases.",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Per-case audit: {output_dir / 'case_audit.csv'}", flush=True)
    print(f"Review-only candidate list: {output_dir / 'candidates_not_train_ready.csv'}",
          flush=True)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--output-dir", type=Path,
                   default=Path("runs/totalsegmentator_v3_audit"))
    p.add_argument("--limit", type=int, default=0,
                   help="0=all cases, >0=first N for a quick test")
    args = p.parse_args()
    if args.limit < 0:
        p.error("--limit must be nonnegative")
    root = args.dataset.resolve()
    info = metadata_for(root)
    cases = sorted(p for p in root.iterdir()
                   if p.is_dir() and re.fullmatch(r"s\d+", p.name))
    if args.limit:
        cases = cases[:args.limit]
    if not cases:
        raise FileNotFoundError(f"No sNNNN case folders in {root}")
    results = []
    for i, case in enumerate(cases, 1):
        row = inspect_case(case, info.get(case.name, {}))
        if case.name not in info and not row["error"]:
            row["error"] = "No matching meta.csv image_id"
            row["candidate_train_only"] = False
        results.append(row)
        if i == 1 or i % 100 == 0 or i == len(cases):
            print(f"Audited {i}/{len(cases)} CT cases", flush=True)
    save_reports(results, args.output_dir.resolve(), len(info))


if __name__ == "__main__":
    main()

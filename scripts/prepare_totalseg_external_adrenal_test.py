#!/usr/bin/env python3
"""Prepare an external-only, NON-TRAINING TotalSegmentator v3 adrenal CT cohort.

Selects ONLY the existing TotalSegmentator v3 official 'test' cases with two
nonempty binary adrenal masks and valid CT geometry. Outputs CT symlinks and
two-class GT NIfTIs in a new, uniquely named external-only directory. It never
touches nnU-Net Dataset701, and does NOT train/finetune an existing model.

This preparation can be performed before annotation review; generated files
are still PRELIMINARY until blinded independent QC. No accuracy or model
prediction is consulted for inclusion. No model inference occurs in this file.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np


def file_digest(path: Path):
    hash_obj = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(4 * 1024 * 1024), b""):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def read_test_inventory(path: Path):
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("Empty external-test inventory")
    ids = [r["case_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate external-test case ID")
    if any(str(r.get("split", "")).strip().lower() != "test" for r in rows):
        raise ValueError("Refusing to include any non-test TotalSegmentator case")
    return sorted(rows, key=lambda r: r["case_id"])


def compose_left_right(ct_path: Path, left_path: Path, right_path: Path):
    ct = nib.load(str(ct_path))
    left = nib.load(str(left_path))
    right = nib.load(str(right_path))
    if ct.ndim != 3:
        raise ValueError("External CT must be 3D")
    for name, mask in (("left", left), ("right", right)):
        if mask.shape != ct.shape or not np.allclose(
            mask.affine, ct.affine, rtol=0, atol=1e-3
        ):
            raise ValueError(f"{name} adrenal mask does not align with CT")
    left_data = np.asarray(left.dataobj)
    right_data = np.asarray(right.dataobj)
    if not np.isin(left_data, (0, 1)).all() or not np.isin(right_data, (0, 1)).all():
        raise ValueError("External gland masks must be binary (0/1)")
    a, b = left_data > 0, right_data > 0
    if not a.any() or not b.any():
        raise ValueError("External gland masks must both be nonempty")
    if np.any(a & b):
        raise ValueError("External left and right reference masks overlap")
    merged = np.zeros(ct.shape, dtype=np.uint8)
    merged[a] = 1
    merged[b] = 2
    header = ct.header.copy()
    header.set_data_dtype(np.uint8)
    result = nib.Nifti1Image(merged, ct.affine, header)
    qform, qcode = ct.get_qform(coded=True)
    sform, scode = ct.get_sform(coded=True)
    if qform is not None:
        result.set_qform(qform, int(qcode))
    if sform is not None:
        result.set_sform(sform, int(scode))
    return result


def prepare_one(row: dict, root: Path, output: Path, resume: bool):
    cid = row["case_id"]
    if not cid.startswith("s") or not cid[1:].isdigit():
        raise ValueError(f"Unexpected TotalSegmentator case ID: {cid}")
    case = root / cid
    src_ct = case / "ct.nii.gz"
    left = case / "segmentations/adrenal_gland_left.nii.gz"
    right = case / "segmentations/adrenal_gland_right.nii.gz"
    dest_cid = "tsv3_" + cid
    dest_ct = output / "imagesTs" / (dest_cid + "_0000.nii.gz")
    dest_label = output / "labelsTs" / (dest_cid + ".nii.gz")
    if not all(p.is_file() for p in (src_ct, left, right)):
        raise FileNotFoundError(f"External CT and masks missing: {cid}")

    if dest_ct.is_symlink():
        if dest_ct.resolve() != src_ct.resolve():
            raise ValueError(f"Existing CT symlink points to different scan: {dest_ct}")
    elif dest_ct.exists():
        raise FileExistsError(f"CT target is not a safe symlink: {dest_ct}")
    else:
        dest_ct.symlink_to(src_ct.resolve())

    # Regenerate the expected reference labels in memory to verify any prior
    # interrupted output before deciding to skip. Never trust stale labels.
    expected = compose_left_right(src_ct, left, right)
    if dest_label.exists():
        if not resume:
            raise FileExistsError(f"{dest_label} exists; use --resume to verify")
        got = nib.load(str(dest_label))
        if (got.shape != expected.shape or
            not np.allclose(got.affine, expected.affine, atol=1e-3, rtol=0) or
            not np.array_equal(np.asarray(got.dataobj), np.asarray(expected.dataobj))):
            raise ValueError(f"Stale/corrupted external test label: {dest_label}")
    else:
        tmp = dest_label.with_name(dest_label.name + ".partial.nii.gz")
        nib.save(expected, str(tmp))
        tmp.replace(dest_label)
    return dest_cid


def run(a):
    root = a.dataset_root.resolve()
    output = a.output.resolve()
    if a.output == a.dataset_root or root in output.parents or output in root.parents:
        raise ValueError("External test output must be separate from source dataset")
    # Hard guard: never write to the existing AMOS Dataset701 model data tree.
    if "Dataset701_AMOSAdrenalCT" in output.parts:
        raise ValueError("Refusing to overwrite or nest inside AMOS Dataset701")
    selected = read_test_inventory(a.inventory)
    if len(selected) != 76 and not a.allow_nonstandard_count:
        raise ValueError(f"Expected 76 original bilateral external test cases, got {len(selected)}")

    if output.exists() and not a.resume and any(output.iterdir()):
        raise FileExistsError("External output already exists; use --resume to verify")
    (output / "imagesTs").mkdir(parents=True, exist_ok=True)
    (output / "labelsTs").mkdir(parents=True, exist_ok=True)

    done = []
    for i, row in enumerate(selected, 1):
        dst = prepare_one(row, root, output, resume=a.resume)
        done.append(dst)
        if i == 1 or i % 10 == 0 or i == len(selected):
            print(f"Prepared external test CT+bilateral reference {i}/{len(selected)}", flush=True)
    expected_files = {x + ".nii.gz" for x in done}
    present = {p.name for p in (output / "labelsTs").glob("*.nii.gz")
               if not p.name.endswith(".partial.nii.gz")}
    if expected_files != present:
        raise ValueError("External reference label inventory unexpected: extra/missing cases")
    metadata = {
        "dataset": "ExternalTotalSegmentatorV3AdrenalTestOnly",
        "source_dataset_version": "TotalSegmentator v3.0.0",
        "source_dataset_root": str(root),
        "num_test_cases": len(done),
        "case_ids": done,
        "label_mapping": {"0": "background", "1": "left_adrenal", "2": "right_adrenal"},
        "source_partition": "test",
        "input_channel": "CT",
        "source_inventory_sha256": file_digest(a.inventory),
        "qc_status": "PRELIMINARY_LABEL_QUALITY_REVIEW_PENDING",
        "restrictions": [
            "Not a nnU-Net training dataset; only imagesTs/labelsTs are present.",
            "Original AMOS-trained Dataset701 checkpoint must be used unchanged.",
            "No test labels or scores may guide model selection, hyperparameters, thresholds or augmentations.",
            "Predeclared blinded human label-review decisions needed for final external benchmark.",
        ],
    }
    (output / "external_test_manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "prepared": len(done),
        "output": str(output),
        "qc_status": metadata["qc_status"],
        "manifest": str(output / "external_test_manifest.json"),
    }, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, default=Path(
        "../data/totalsegmentator_v3/extracted/Totalsegmentator_dataset_v300"))
    p.add_argument("--inventory", type=Path, default=Path(
        "runs/totalsegmentator_v3_qc_plan/external_test_bilateral_inventory.csv"))
    p.add_argument("--output", type=Path, default=Path(
        "nnunet_raw/ExternalTotalSegV3AdrenalTest_v1"))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--allow-nonstandard-count", action="store_true",
                   help="Only for synthetic tests; NOT for final external evaluation")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prepare Dataset702: original AMOS training + TotalSegmentator v3 training only.

Consumes a PRE-FROZEN `runs/expanded_dataset702/plan.json`; never reads AMOS
imagesTs or ANY official TotalSegmentator test image. Includes internal TS
validation in imagesTr, but fold0 train/val is separately frozen and written
to preprocessed/splits_final.json ONLY AFTER nnUNetv2_plan_and_preprocess.
No ground-truth-derived ROI cropping, no laterality oracle.

Original 200 AMOS train CT and label symlinks are reused unchanged. For 1053
candidate TS *train* CTs, validate binary/aligned bilateral masks. ITK-friendly
orthonormal CTs remain source symlinks and combined label masks are made on
their native grid. For sheared/ITK-incompatible CT, CT is linearly resampled
to an orthonormal physical grid and BOTH source masks are nearest-neighbour
resampled onto the identical grid. Original CT/mask files NEVER altered.

If any CT is not safe to process or annotations become empty/overlapping after
regridding, FAIL rather than silently omit it or use an invalid label.

This is research/automated-QA-only (unreviewed labels), NOT clinical QC.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.repair_totalseg_external_ct_geometry import (
    isitk_readable, orthonormal_error, orthogonalize, sha256,
)


DEFAULT_MANIFEST = Path("runs/expanded_dataset702/plan.json")
DEFAULT_OUT = Path("nnunet_raw/Dataset702_AMOSTotalSegAdrenalCT")
DATASET_ID = 702


def frozen_plan(path):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (manifest.get("dataset_id") != DATASET_ID or
        manifest.get("status") != "FROZEN_BEFORE_EXPANDED_CNN_TRAINING" or
        manifest.get("quality_status") != "AUTOMATED_SCREEN_ONLY_UNREVIEWED"):
        raise ValueError("Invalid frozen Dataset702 planning record")
    ids = manifest["amos_source_train_case_ids"]
    ts = manifest["ts_source_train_case_ids"]
    train = manifest["fold0_train_ids"]
    val = manifest["fold0_val_ids"]
    all_ids = {"am_" + cid for cid in ids} | {"tsv3_" + cid for cid in ts}
    if (len(ids) != len(set(ids)) or len(ts) != len(set(ts)) or
        len(all_ids) != len(ids) + len(ts) or
        set(train) & set(val) or set(train) | set(val) != all_ids or
        manifest["full_imagesTr_count"] != len(all_ids)):
        raise ValueError("Frozen manifest contains overlapping or missing cases")
    if set(ts) & {s.removeprefix("tsv3_") for s in
                  manifest["external_test_case_ids_RESERVED_NOT_IN_DATASET"]}:
        raise ValueError("LEAKAGE: test patient found in TS training data")
    return manifest


def assert_frozen_sources(manifest):
    sources = manifest["source_paths"]
    old = manifest["sha256_sources"]
    paths = {
        "amos_split_manifest": Path(sources["amos_dataset701"]) / "split_manifest.json",
        "totalseg_candidate_csv": Path(sources["candidate_csv"]),
        "totalseg_full_case_audit": Path(sources["candidate_csv"]).parent / "case_audit.csv",
        "totalseg_meta_csv": Path(sources["totalseg_source"]) / "meta.csv",
        "external_test_manifest": Path(sources["external_test_manifest"]),
    }
    for label, path in paths.items():
        if not path.is_file() or sha256(path) != old[label]:
            raise ValueError(f"Source evidence drifted after Dataset702 freeze: {label}")


def source_mask_data(mask_file: Path, ct: nib.Nifti1Image, cid: str, side: str):
    mask = nib.load(str(mask_file))
    if mask.ndim != 3 or mask.shape != ct.shape or not np.allclose(
        mask.affine, ct.affine, rtol=0, atol=1e-3
    ):
        raise ValueError(f"{cid}: {side} mask geometry incompatible with original CT")
    values = np.asarray(mask.dataobj)
    if not np.isin(values, (0, 1)).all() or not np.any(values):
        raise ValueError(f"{cid}: {side} mask not binary/nonempty")
    return mask, values.astype(bool, copy=False)


def make_merged_label(ct_input: nib.Nifti1Image,
                      mask_l: nib.Nifti1Image, mask_r: nib.Nifti1Image,
                      *, resample_masks: bool, cid: str):
    if resample_masks:
        region = (ct_input.shape, ct_input.affine)
        left = np.asarray(resample_from_to(mask_l, region, order=0,
                                          mode="constant", cval=0).dataobj) > 0
        right = np.asarray(resample_from_to(mask_r, region, order=0,
                                           mode="constant", cval=0).dataobj) > 0
    else:
        left = np.asarray(mask_l.dataobj) > 0
        right = np.asarray(mask_r.dataobj) > 0
    if not left.any() or not right.any() or np.logical_and(left, right).any():
        raise ValueError(f"{cid}: invalid bilateral glands after spatial transform")
    label = np.zeros(ct_input.shape, dtype=np.uint8)
    label[left] = 1
    label[right] = 2
    header = ct_input.header.copy()
    header.set_data_dtype(np.uint8)
    result = nib.Nifti1Image(label, ct_input.affine, header=header)
    qform, qc = ct_input.get_qform(coded=True)
    sform, sc = ct_input.get_sform(coded=True)
    if qform is not None:
        result.set_qform(qform, int(qc))
    if sform is not None:
        result.set_sform(sform, int(sc))
    return result


def link_exact(src, dest):
    if dest.is_symlink():
        if dest.resolve() != src.resolve():
            raise ValueError(f"Existing image symlink points to wrong source: {dest}")
    elif dest.exists():
        raise FileExistsError(f"Cannot replace existing non-symlink output: {dest}")
    else:
        dest.symlink_to(src.resolve())


def save_expected_label(image, path, *, resume):
    if path.exists():
        if not resume:
            raise FileExistsError(f"Existing Dataset702 label: {path}")
        old = nib.load(str(path))
        if (old.shape != image.shape or not np.allclose(
                old.affine, image.affine, atol=1e-4, rtol=0) or
            not np.array_equal(np.asarray(old.dataobj), np.asarray(image.dataobj))):
            raise ValueError(f"Previously prepared label changed: {path}")
    else:
        tmp = path.with_name(path.name + ".partial.nii.gz")
        nib.save(image, str(tmp))
        tmp.replace(path)


def prepare_one_ts(cid, source_root, output_dir, *, resume, tolerance,
                   max_output_voxels):
    source = source_root / cid
    original_ct = source / "ct.nii.gz"
    original_img = nib.load(str(original_ct))
    if original_img.ndim != 3 or not np.isfinite(original_img.affine).all():
        raise ValueError(f"{cid}: invalid original CT geometry")
    left_img, left = source_mask_data(
        source / "segmentations/adrenal_gland_left.nii.gz", original_img,
        cid, "left"
    )
    right_img, right = source_mask_data(
        source / "segmentations/adrenal_gland_right.nii.gz", original_img,
        cid, "right"
    )
    if np.logical_and(left, right).any():
        raise ValueError(f"{cid}: source left/right annotation overlap")

    image_path = output_dir / "imagesTr" / f"tsv3_{cid}_0000.nii.gz"
    label_path = output_dir / "labelsTr" / f"tsv3_{cid}.nii.gz"
    itk_ok, itk_issue = isitk_readable(original_ct)
    nonorth = orthonormal_error(original_img.affine)
    resample = (not itk_ok) or nonorth > tolerance
    if resample:
        if image_path.is_symlink():
            raise ValueError(f"{cid}: expected resampled file, found symlink")
        if image_path.exists():
            if not resume:
                raise FileExistsError(image_path)
            new_img = nib.load(str(image_path))
            ok, why = isitk_readable(image_path)
            if (not ok or orthonormal_error(new_img.affine) > 1e-5):
                raise ValueError(f"{cid}: invalid saved orthogonal CT: {why}")
        else:
            orthogonalize(original_ct, image_path,
                          max_voxels=max_output_voxels)
            new_img = nib.load(str(image_path))
        method = "CT linear orthogonal resample; masks NN resample"
    else:
        link_exact(original_ct, image_path)
        new_img = original_img
        method = "original CT symlink; native masks"
    merged = make_merged_label(
        new_img, left_img, right_img, resample_masks=resample, cid=cid
    )
    save_expected_label(merged, label_path, resume=resume)
    # Ensure downstream nnU-Net/SimpleITK sees matching physical frames.
    good_label, reason = isitk_readable(label_path)
    if not good_label:
        raise ValueError(f"{cid}: prepared binary label unreadable by ITK: {reason}")
    return {
        "case_id": cid,
        "model_case_id": "tsv3_" + cid,
        "geometry_method": method,
        "source_shape": list(original_img.shape),
        "prepared_shape": list(new_img.shape),
        "original_affine": original_img.affine.tolist(),
        "prepared_affine": new_img.affine.tolist(),
        "source_affine_nonorthogonality": nonorth,
        "source_itk_readable": itk_ok,
        "source_itk_issue": itk_issue,
        "left_label_voxels_prepared": int((np.asarray(merged.dataobj) == 1).sum()),
        "right_label_voxels_prepared": int((np.asarray(merged.dataobj) == 2).sum()),
    }


def run(args):
    manifest = frozen_plan(args.manifest)
    assert_frozen_sources(manifest)
    target = args.out.resolve()
    old_root = Path(manifest["source_paths"]["amos_dataset701"]).resolve()
    ts_root = Path(manifest["source_paths"]["totalseg_source"]).resolve()
    if (target in (old_root, ts_root) or
        target in old_root.parents or target in ts_root.parents or
        old_root in target.parents or ts_root in target.parents):
        raise ValueError("Refusing to replace or nest into either source dataset")
    if target.name != "Dataset702_AMOSTotalSegAdrenalCT":
        raise ValueError("Only the new separate Dataset702_AMOSTotalSegAdrenalCT is allowed")
    if target.exists() and not args.resume and any(target.iterdir()):
        raise FileExistsError(f"Output directory nonempty; use --resume: {target}")
    for leaf in ("imagesTr", "labelsTr"):
        (target / leaf).mkdir(parents=True, exist_ok=True)
    for idx, cid in enumerate(manifest["amos_source_train_case_ids"], 1):
        dst_cid = "am_" + cid
        original_image = old_root / "imagesTr" / f"{cid}_0000.nii.gz"
        original_label = old_root / "labelsTr" / f"{cid}.nii.gz"
        link_exact(original_image, target / "imagesTr" / (dst_cid + "_0000.nii.gz"))
        link_exact(original_label, target / "labelsTr" / (dst_cid + ".nii.gz"))
        if idx == 1 or idx % 50 == 0 or idx == len(manifest["amos_source_train_case_ids"]):
            print(f"AMOS source dataset: {idx}/{len(manifest['amos_source_train_case_ids'])}",
                  flush=True)
    cases = []
    all_ts = manifest["ts_source_train_case_ids"]
    for i, cid in enumerate(all_ts, 1):
        cases.append(prepare_one_ts(
            cid, ts_root, target, resume=args.resume,
            tolerance=args.orthogonality_tolerance,
            max_output_voxels=args.max_output_voxels
        ))
        if i == 1 or i % 25 == 0 or i == len(all_ts):
            print(f"TS training+internal validation source {i}/{len(all_ts)}; "
                  f"regridded={sum('orthogonal resample' in r['geometry_method'] for r in cases)}",
                  flush=True)
    expected = set(manifest["fold0_train_ids"]) | set(manifest["fold0_val_ids"])
    images = {x.name.removesuffix("_0000.nii.gz") for x in
              (target / "imagesTr").glob("*_0000.nii.gz")}
    labels = {x.name.removesuffix(".nii.gz") for x in
              (target / "labelsTr").glob("*.nii.gz")}
    if images != expected or labels != expected:
        raise ValueError("Dataset702 source inventory incomplete or contains extra cases")
    dataset_info = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "left_adrenal": 1, "right_adrenal": 2},
        "numTraining": len(expected),
        "file_ending": ".nii.gz",
    }
    dataset_json = target / "dataset.json"
    if dataset_json.exists():
        if json.loads(dataset_json.read_text()) != dataset_info:
            raise ValueError("Existing Dataset702 dataset.json unexpected")
    else:
        dataset_json.write_text(json.dumps(dataset_info, indent=2) + "\n")
    geo = {
        "status": "COMPLETE",
        "dataset_id": DATASET_ID,
        "original_split_plan_sha256": sha256(args.manifest),
        "cases": cases,
        "num_total_TS_cases": len(cases),
        "num_regridded_TS_cases": sum(
            "orthogonal resample" in x["geometry_method"] for x in cases
        ),
        "quality_status": "AUTOMATED_SCREEN_ONLY_UNREVIEWED",
        "source_CT_and_masks_unchanged": True,
        "no_label_derived_ROI_or_crop": True,
    }
    geom_path = target / "geometry_manifest.json"
    if geom_path.exists():
        past = json.loads(geom_path.read_text())
        if (past["original_split_plan_sha256"] != geo["original_split_plan_sha256"]
            or past["num_total_TS_cases"] != geo["num_total_TS_cases"]):
            raise ValueError("Existing Dataset702 geometry report does not match frozen plan")
    else:
        geom_path.write_text(json.dumps(geo, indent=2) + "\n")
    print(json.dumps({
        "status": "COMPLETE",
        "dataset_id": DATASET_ID,
        "full_imagesTr": len(expected),
        "fold0_train": len(manifest["fold0_train_ids"]),
        "fold0_validation": len(manifest["fold0_val_ids"]),
        "resampled_ts_cts": geo["num_regridded_TS_cases"],
        "dataset_dir": str(target),
        "source_ground_truth_status": "automated checks only; no human review",
    }, indent=2), flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--orthogonality-tolerance", type=float, default=1e-4)
    p.add_argument("--max-output-voxels", type=int, default=550_000_000)
    p.add_argument("--resume", action="store_true")
    a=p.parse_args()
    run(a)


if __name__ == "__main__":
    main()

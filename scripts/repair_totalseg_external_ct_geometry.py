#!/usr/bin/env python3
"""Prepare/restore frozen TotalSegmentator v3 external test geometry, NO LABEL CROPS.

When TotalSegmentator NIfTI CT has a non-orthonormal/sheared affine, SimpleITK
and nnU-Net refuse it. Original NIfTIs and reference adrenal masks are NEVER
altered. Prepare 76 nnU-Net input CTs with an ITK header preflight: safe inputs
are symlinked unchanged, problematic images are resampled once into an
orthonormal RAS+ physical grid (linear interpolation, CT background -1024 HU).
Restore resulting segmentations into each ORIGINAL CT's native grid with
nearest-neighbour interpolation and report all geometry changes.

No reference mask is read to build CT input. GT is never used for cropping,
resampling input, selecting cases, or choosing preprocessing parameters.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

import nibabel as nib
from nibabel.processing import resample_from_to, resample_to_output
import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fd:
        for chunk in iter(lambda: fd.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_manifest(root: Path, expected_count: int = 76):
    path = root / "external_test_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    ids = manifest.get("case_ids", [])
    if (manifest.get("source_partition") != "test" or
        len(ids) != expected_count or len(set(ids)) != expected_count or
        any(not cid.startswith("tsv3_s") for cid in ids) or
        manifest.get("label_mapping") != {
            "0": "background", "1": "left_adrenal", "2": "right_adrenal"
        }):
        raise ValueError("Unexpected official external test cohort/label mapping")
    expected = {cid + "_0000.nii.gz" for cid in ids}
    actual = {p.name for p in (root / "imagesTs").glob("*_0000.nii.gz")}
    if actual != expected:
        raise ValueError("External CT files do not match original frozen 76-case manifest")
    return manifest, sha256(path)


def isitk_readable(path: Path) -> tuple[bool, str | None]:
    """Read only image metadata, not voxels, using nnU-Net's ITK backend."""
    import SimpleITK as sitk
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    try:
        reader.ReadImageInformation()
        return True, None
    except RuntimeError as exc:
        return False, str(exc).splitlines()[-1][:500]


def orthonormal_error(affine: np.ndarray) -> float:
    axes = np.asarray(affine[:3, :3], dtype=float)
    length = np.linalg.norm(axes, axis=0)
    if np.any(~np.isfinite(length)) or np.any(length <= 0):
        raise ValueError("Invalid CT affine axis scaling")
    normalized = axes / length
    return float(np.max(np.abs(normalized.T @ normalized - np.eye(3))))


def world_voxels_estimate(image: nib.spatialimages.SpatialImage, spacing):
    # Bounding box in world coordinates; guards extreme shears/affines.
    corners = np.array([
        [a, b, c] for a in (0., image.shape[0]-1.)
        for b in (0., image.shape[1]-1.)
        for c in (0., image.shape[2]-1.)
    ])
    world = nib.affines.apply_affine(image.affine, corners)
    extents = world.max(axis=0) - world.min(axis=0)
    return int(np.prod(np.ceil(extents / np.asarray(spacing)) + 2))


def orthogonalize(source: Path, target: Path, *, max_voxels: int):
    original = nib.load(str(source))
    if original.ndim != 3:
        raise ValueError(f"Expected 3D CT: {source}")
    if not np.isfinite(original.affine).all():
        raise ValueError("Non-finite CT geometry")
    voxel_sizes = tuple(float(x) for x in nib.affines.voxel_sizes(original.affine))
    if min(voxel_sizes) <= 0 or max(voxel_sizes) > 12:
        raise ValueError(f"Implausible CT voxel sizes {voxel_sizes}")
    estimated = world_voxels_estimate(original, voxel_sizes)
    if estimated > max_voxels:
        raise MemoryError(
            f"Orthogonalisation would require ~{estimated:,} voxels; "
            f"limit={max_voxels:,}. Review geometry rather than cropping.")
    fixed = resample_to_output(
        original, voxel_sizes=voxel_sizes, order=1,
        mode="constant", cval=-1024.0,
    )
    # Serialize the true output as float32. Header qform/sform are rewritten to
    # match the new orthogonal affine, never copied as the old sheared form.
    fixed_data = np.asarray(fixed.dataobj, dtype=np.float32)
    canonical = nib.Nifti1Image(fixed_data, fixed.affine)
    canonical.header.set_xyzt_units("mm")
    canonical.set_qform(fixed.affine, code=1)
    canonical.set_sform(fixed.affine, code=1)
    if orthonormal_error(canonical.affine) > 1e-5:
        raise ValueError(f"Orthogonal output unexpectedly nonorthonormal: {source}")
    tmp = target.with_name(target.name + ".partial.nii.gz")
    nib.save(canonical, str(tmp))
    good, reason = isitk_readable(tmp)
    if not good:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"Orthogonalisation did not fix ITK header: {reason}")
    tmp.replace(target)
    return {
        "source_shape": list(original.shape),
        "prepared_shape": list(canonical.shape),
        "source_affine": original.affine.tolist(),
        "prepared_affine": canonical.affine.tolist(),
        "source_orthonormal_error": orthonormal_error(original.affine),
        "prepared_sha256": sha256(target),
        "estimated_output_voxels": estimated,
    }


def preflight(args):
    root = args.original.resolve()
    out = args.prepared.resolve()
    if root == out or root in out.parents or out in root.parents:
        raise ValueError("Prepared input must be separate from original data tree")
    frozen, frozen_sha = source_manifest(root)
    ids = sorted(frozen["case_ids"])
    if out.exists() and (out / "geometry_manifest.json").is_file():
        old = json.loads((out / "geometry_manifest.json").read_text())
        if old.get("original_manifest_sha256") != frozen_sha:
            raise ValueError("Original frozen cohort changed; refusing to reuse")
    images = out / "imagesTs"
    images.mkdir(parents=True, exist_ok=True)
    cases = []
    for i, cid in enumerate(ids, 1):
        src = root / "imagesTs" / (cid + "_0000.nii.gz")
        dst = images / src.name
        img = nib.load(str(src))
        if img.ndim != 3 or not np.isfinite(img.affine).all():
            raise ValueError(f"Non-3D/invalid affine for {cid}")
        err = orthonormal_error(img.affine)
        itk_ok, itk_reason = isitk_readable(src)
        # A sheared volume may be read by one reader version but not another.
        # Require both ITK metadata acceptance and orthogonal directions.
        fix = (not itk_ok) or err > args.orthogonality_tolerance
        row = {
            "case_id": cid,
            "original_filename": str(src),
            "method": "resample_linear_to_world_RAS" if fix else "original_symlink",
            "source_shape": list(img.shape),
            "source_affine": img.affine.tolist(),
            "source_orthonormal_error": err,
            "source_itk_header_ok": itk_ok,
            "source_itk_error": itk_reason,
        }
        if fix:
            if dst.is_symlink():
                raise ValueError(f"Unexpected symlink where conversion needed: {dst}")
            if dst.exists() and args.resume:
                good, reason = isitk_readable(dst)
                if not good:
                    raise ValueError(f"Existing prepared CT broken: {dst}: {reason}")
                check = nib.load(str(dst))
                if orthonormal_error(check.affine) > 1e-5:
                    raise ValueError(f"Existing CT not orthogonal: {dst}")
                row.update({
                    "prepared_shape": list(check.shape),
                    "prepared_affine": check.affine.tolist(),
                    "prepared_sha256": sha256(dst),
                    "reused": True,
                })
            elif dst.exists():
                raise FileExistsError(f"Prepared image exists; use --resume: {dst}")
            else:
                print(f"[{i}/{len(ids)}] Regridding {cid} (ITK={itk_ok}, shear={err:.6g})",
                      flush=True)
                row.update(orthogonalize(
                    src, dst, max_voxels=args.max_output_voxels))
                row["reused"] = False
        else:
            if dst.is_symlink():
                if dst.resolve() != src.resolve():
                    raise ValueError(f"Stale symlink to wrong CT: {dst}")
            elif dst.exists():
                raise ValueError(f"Prepared path is regular file for unchanged CT: {dst}")
            else:
                dst.symlink_to(src.resolve())
            row["prepared_shape"] = list(img.shape)
            row["prepared_affine"] = img.affine.tolist()
            row["reused"] = True
        if i == 1 or i % 10 == 0 or i == len(ids):
            print(f"Prepared geometry {i}/{len(ids)}; {cid}: {row['method']}",
                  flush=True)
        cases.append(row)
    expected = {cid + "_0000.nii.gz" for cid in ids}
    present = {p.name for p in images.glob("*_0000.nii.gz")}
    if present != expected:
        raise ValueError("Prepared CT folder has extra/missing cases")
    report = {
        "status": "COMPLETE",
        "original_manifest_sha256": frozen_sha,
        "source_partition": "test",
        "case_count": len(ids),
        "qc_status": "UNREVIEWED source annotations — exploratory only",
        "preprocessing": {
            "method": "ITK header preflight; original CT symlinks if valid; "
                      "problematic NIfTI resampled in physical RAS+ world axes",
            "label_derived_cropping": False,
            "interpolation_for_ct": "linear (order=1)",
            "background_hu_for_resampled_ct": -1024,
            "target_spacing": "original physical voxel-size magnitudes",
            "interpolation_for_restored_pred": "nearest (order=0)",
            "unchanged_data": "Original source CT and both original adrenal masks",
            "orthogonality_tolerance": args.orthogonality_tolerance,
        },
        "regridded_count": sum(r["method"] != "original_symlink" for r in cases),
        "regridded_cases": [r["case_id"] for r in cases if r["method"] != "original_symlink"],
        "cases": cases,
    }
    dest_manifest = out / "geometry_manifest.json"
    if dest_manifest.exists():
        old = json.loads(dest_manifest.read_text(encoding="utf-8"))
        # Except for derived hash and reused flags, geometry must be stable.
        old_core = [(c["case_id"], c["method"], c["prepared_shape"],
                     c["prepared_affine"]) for c in old["cases"]]
        now_core = [(c["case_id"], c["method"], c["prepared_shape"],
                     c["prepared_affine"]) for c in cases]
        if old_core != now_core:
            raise ValueError("Prepared geometry has drifted from frozen prior run")
        print("Reusing stable pre-existing geometry manifest.", flush=True)
    else:
        dest_manifest.write_text(json.dumps(report, indent=2) + "\n",
                                 encoding="utf-8")
    print(json.dumps({
        "status": "COMPLETE",
        "case_count": len(ids),
        "regridded_count": report["regridded_count"],
        "regridded_cases": report["regridded_cases"],
        "prepared": str(out),
    }, indent=2), flush=True)


def restore(args):
    source = args.original.resolve()
    prepared = args.prepared.resolve()
    predictions = args.predictions.resolve()
    target = args.restored.resolve()
    frozen, original_sha = source_manifest(source)
    geom_path = prepared / "geometry_manifest.json"
    geom = json.loads(geom_path.read_text(encoding="utf-8"))
    ids = sorted(frozen["case_ids"])
    if (geom.get("original_manifest_sha256") != original_sha or
        geom.get("status") != "COMPLETE" or
        geom.get("case_count") != len(ids)):
        raise ValueError("Prepared geometry does not match original frozen cohort")
    lookup = {r["case_id"]: r for r in geom["cases"]}
    if set(lookup) != set(ids):
        raise ValueError("Prepared geometry includes unexpected CTs")
    expected_names = {cid + ".nii.gz" for cid in ids}
    observed = {p.name for p in predictions.glob("*.nii.gz")}
    if expected_names != observed:
        raise ValueError(
            f"nnU-Net predictions incomplete/extra: expected {len(ids)} "
            f"found {len(observed)}; missing={sorted(expected_names-observed)[:5]}")
    target.mkdir(parents=True, exist_ok=True)
    for i, cid in enumerate(ids, 1):
        native_ct = nib.load(str(source / "imagesTs" / (cid + "_0000.nii.gz")))
        prepared_ct = nib.load(str(prepared / "imagesTs" / (cid + "_0000.nii.gz")))
        prediction_path = predictions / (cid + ".nii.gz")
        predicted = nib.load(str(prediction_path))
        if (predicted.shape != prepared_ct.shape or
            not np.allclose(predicted.affine, prepared_ct.affine, rtol=0, atol=1e-3)):
            raise ValueError(f"Prediction geometry inconsistent with inference CT: {cid}")
        dest = target / (cid + ".nii.gz")
        if dest.exists() and args.resume:
            previous = nib.load(str(dest))
            if (previous.shape != native_ct.shape or
                not np.allclose(previous.affine, native_ct.affine, rtol=0, atol=1e-4)):
                raise ValueError(f"Existing restored prediction geometry invalid: {cid}")
            print(f"Restored {i}/{len(ids)}: reuse {cid}", flush=True)
            continue
        if dest.exists():
            raise FileExistsError(f"Refusing to overwrite existing prediction: {dest}")
        if lookup[cid]["method"] == "original_symlink":
            if (predicted.shape != native_ct.shape or
                not np.allclose(predicted.affine, native_ct.affine, rtol=0, atol=1e-3)):
                raise ValueError(f"Unchanged CT prediction must match native CT: {cid}")
            # Original-format predictions need no interpolation at all.
            shutil.copy2(prediction_path, dest)
        else:
            transformed = resample_from_to(
                predicted, (native_ct.shape, native_ct.affine),
                order=0, mode="constant", cval=0,
            )
            masks = np.asarray(transformed.dataobj, dtype=np.uint8)
            if not np.isin(masks, [0, 1, 2]).all():
                raise ValueError(f"Unexpected label after restoration: {cid}")
            header = native_ct.header.copy()
            header.set_data_dtype(np.uint8)
            saved = nib.Nifti1Image(masks, native_ct.affine, header=header)
            qmat, qcode = native_ct.get_qform(coded=True)
            smat, scode = native_ct.get_sform(coded=True)
            if qmat is not None:
                saved.set_qform(qmat, int(qcode))
            if smat is not None:
                saved.set_sform(smat, int(scode))
            tmp = dest.with_name(dest.name + ".partial.nii.gz")
            nib.save(saved, str(tmp))
            tmp.replace(dest)
        if i == 1 or i % 10 == 0 or i == len(ids):
            print(f"Restored native geometry {i}/{len(ids)}: {cid}", flush=True)
    observed_native = {p.name for p in target.glob("*.nii.gz")}
    if expected_names != observed_native:
        raise ValueError("Restored predictions folder differs from cohort")
    result = {
        "status": "COMPLETE",
        "case_count": len(ids),
        "native_ct_reference": str(source),
        "preprocessing_geometry_manifest_sha256": sha256(geom_path),
        "regridded_count": geom["regridded_count"],
        "prediction_input": str(predictions),
        "prediction_output_native": str(target),
        "qc_status": "UNREVIEWED source labels — exploratory external evaluation",
        "method": "world physical affine resampling to original CT grid order=0 "
                  "for regridded CTs; no interpolation for untouched CTs",
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    path = target.parent / "external_geometry_restore_summary.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old.get("preprocessing_geometry_manifest_sha256") != result[
            "preprocessing_geometry_manifest_sha256"]:
            raise ValueError("Restore summary for incompatible prior geometry run")
    else:
        path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "restore"))
    p.add_argument("--original", type=Path,
                   default=Path("nnunet_raw/ExternalTotalSegV3AdrenalTest_v1"))
    p.add_argument("--prepared", type=Path,
                   default=Path("nnunet_raw/ExternalTotalSegV3AdrenalOrtho_v1"))
    p.add_argument("--predictions", type=Path,
                   default=Path("runs/nnunet3d_external_tsv3_ortho_predictions_fold0"))
    p.add_argument("--restored", type=Path,
                   default=Path("runs/nnunet3d_external_tsv3_native_predictions_fold0"))
    p.add_argument("--max-output-voxels", type=int, default=550_000_000)
    p.add_argument("--orthogonality-tolerance", type=float, default=1e-4)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if args.action == "prepare":
        preflight(args)
    else:
        restore(args)


if __name__ == "__main__":
    main()

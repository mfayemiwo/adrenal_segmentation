#!/usr/bin/env python3
"""CPU-only bilateral adrenal postprocessing of *existing* nnU-Net NIfTI masks.

Does NOT train/load a neural network, run GPU inference, inspect CT intensities,
or use ground-truth labels. Class IDs: 0 background, 1 left, 2 right.

Use a separate output directory: NEVER overwrite the raw predictions. Choose
the postprocessing rule on nnU-Net's internal development validation predictions,
not by maximizing Dice on the held-out AMOS imagesTs/labelsTs cohort.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi


def postprocess_mask(
    labelmap: np.ndarray, *, mode: str = "largest",
    min_relative_size: float = 0.15, connectivity: int = 2,
) -> tuple[np.ndarray, list[dict]]:
    """Filter each gland independently using 3D connected components.

    largest: keep one largest 3D component per gland.
    relative: keep every component with >= min_relative_size * largest_size.

    3D, not slice-wise, avoids deleting a gland just because it is small on a
    particular z slice. No foreground class is ever relabelled as the other.
    """
    if labelmap.ndim != 3:
        raise ValueError("Expected a 3D segmentation")
    if not np.isin(labelmap, (0, 1, 2)).all():
        raise ValueError("Labels must be 0, 1 (left), or 2 (right)")
    if mode not in ("largest", "relative"):
        raise ValueError("mode must be largest or relative")
    if not 0 < min_relative_size <= 1 or not np.isfinite(min_relative_size):
        raise ValueError("min_relative_size must be in (0, 1]")
    if connectivity not in (1, 2, 3):
        raise ValueError("connectivity must be 1, 2, or 3")

    output = np.asarray(labelmap, dtype=np.uint8).copy()
    structure = ndi.generate_binary_structure(3, connectivity)
    records = []

    for class_id, gland in ((1, "left"), (2, "right")):
        gland_mask = labelmap == class_id
        components, count = ndi.label(gland_mask, structure=structure)
        sizes = np.bincount(components.ravel(), minlength=count + 1)
        sizes[0] = 0
        voxels_before = int(gland_mask.sum())

        if count == 0:
            kept = np.zeros_like(gland_mask, dtype=bool)
            largest = 0
        else:
            largest = int(sizes.max())
            if mode == "largest":
                chosen = np.array([int(np.argmax(sizes))], dtype=np.int32)
            else:
                chosen = np.flatnonzero(sizes >= min_relative_size * largest)
                chosen = chosen[chosen != 0]
            kept = np.isin(components, chosen)

        output[gland_mask & ~kept] = 0
        voxels_after = int(kept.sum())
        records.append({
            "gland": gland,
            "components_before": int(count),
            "largest_component_voxels": largest,
            "voxels_before": voxels_before,
            "voxels_after": voxels_after,
            "voxels_removed": voxels_before - voxels_after,
            "removed_fraction": (
                (voxels_before - voxels_after) / voxels_before
                if voxels_before else 0.
            ),
            "warning": (
                "MORE_THAN_HALF_REMOVED"
                if voxels_before and voxels_after < voxels_before / 2 else ""
            ),
        })
    return output, records


def run(args: argparse.Namespace) -> None:
    source = args.input_dir.resolve()
    dest = args.output_dir.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Missing nnU-Net prediction directory: {source}")
    if (source == dest or source in dest.parents or dest in source.parents):
        raise ValueError("Input and output directories must be separate, non-nested paths")
    files = sorted(source.glob("*.nii.gz"))
    if not files:
        raise FileNotFoundError(f"No segmentation .nii.gz files in {source}")
    if getattr(args, "resume", False) and args.force:
        raise ValueError("--resume and --force cannot both be selected")
    dest.mkdir(parents=True, exist_ok=True)

    records = []
    created = 0
    verified = 0
    for path in files:
        target = dest / path.name
        if target.exists() and not args.force and not getattr(args, "resume", False):
            raise FileExistsError(
                f"{target} exists. Use --resume to verify and reuse existing "
                "output, or --force to replace the postprocessed copy."
            )
        img = nib.load(str(path))
        labels = np.asarray(img.dataobj)
        fixed, changes = postprocess_mask(
            labels, mode=args.mode,
            min_relative_size=args.min_relative_size,
            connectivity=args.connectivity,
        )
        if target.exists() and getattr(args, "resume", False):
            # Validate the earlier, completed mask against the deterministic
            # recipe, including its geometry; never silently trust a stale
            # file from a different mode or parameter configuration.
            previous = nib.load(str(target))
            if (previous.shape != img.shape or
                not np.allclose(previous.affine, img.affine, atol=1e-3, rtol=0) or
                not np.array_equal(np.asarray(previous.dataobj), fixed)):
                raise ValueError(
                    f"Existing {target} differs from the requested cleanup. "
                    "Stop rather than overwrite; use another output directory "
                    "or explicitly review --force."
                )
            verified += 1
            print(f"{path.name}: verified existing result ({verified} verified)",
                  flush=True)
        else:
            header = img.header.copy()
            header.set_data_dtype(np.uint8)
            out_img = nib.Nifti1Image(fixed, img.affine, header)
            # Preserve spatial transforms/metadata, including qform/sform codes.
            qform, qcode = img.get_qform(coded=True)
            sform, scode = img.get_sform(coded=True)
            if qform is not None:
                out_img.set_qform(qform, int(qcode))
            if sform is not None:
                out_img.set_sform(sform, int(scode))
            tmp = dest / (path.name + ".partial.nii.gz")
            nib.save(out_img, str(tmp))
            tmp.replace(target)
            created += 1
        for change in changes:
            change["case_id"] = path.name[:-7]
            change["mode"] = args.mode
            change["connectivity"] = args.connectivity
            change["min_relative_size"] = args.min_relative_size
            records.append(change)
        print(
            f"{path.name}: "
            + " | ".join(
                f"{item['gland']}: {item['voxels_before']} -> "
                f"{item['voxels_after']} voxels, "
                f"{item['components_before']} components"
                for item in changes
            ), flush=True,
        )

    report_path = dest / "postprocessing_audit.csv"
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "case_id", "mode", "connectivity", "min_relative_size", "gland",
            "components_before", "largest_component_voxels", "voxels_before",
            "voxels_after", "voxels_removed", "removed_fraction", "warning",
        ])
        writer.writeheader()
        writer.writerows(records)
    print(f"Complete: {len(files)} masks in {dest}; "
          f"newly saved={created}, verified existing={verified}", flush=True)
    print(f"Audit: {report_path}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path,
                   default=Path("runs/nnunet3d_holdout_predictions_fold0"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--mode", choices=("largest", "relative"), default="largest")
    p.add_argument("--min-relative-size", type=float, default=0.15)
    p.add_argument("--connectivity", type=int, choices=(1, 2, 3), default=2,
                   help="1=6, 2=18, 3=26 neighbouring voxels in 3D")
    p.add_argument("--resume", action="store_true",
                   help="Verify and reuse completed output NIfTIs, computing only missing ones")
    p.add_argument("--force", action="store_true",
                   help="Overwrite only output-directory masks (never input)")
    run(p.parse_args())


if __name__ == "__main__":
    main()

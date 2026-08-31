#!/usr/bin/env python3
"""Convert AMOS22 into an nnU-Net v2 dataset containing only the adrenal glands.

nnU-Net is the standard anchor for a segmentation paper, and it is also a
diagnostic: if it reaches ~0.85 on this cohort while our pipeline sits at 0.746,
the deficit is in our pipeline and we know roughly how much is recoverable; if
it lands near 0.78, the published figures become the thing that needs
explaining. Either outcome is worth a day of compute.

What this produces
------------------
    <out>/Dataset501_AdrenalAMOS/
        imagesTr/amos_0001_0000.nii.gz      our TRAINING cases (nnU-Net runs its
        labelsTr/amos_0001.nii.gz           own 5-fold CV inside this set)
        imagesTs/amos_0507_0000.nii.gz      our VALIDATION cases, held out so the
        labelsTs/amos_0507.nii.gz           final number is comparable to ours
        dataset.json

The `_0000` suffix is nnU-Net's channel index and is mandatory; labels carry no
suffix. Images are symlinked by default (AMOS is ~16 GB and nnU-Net only reads
them); labels must be rewritten because AMOS's 15 organ classes are collapsed to
background / left / right.

Label mapping. AMOS uses 11 for the RIGHT adrenal and 12 for the LEFT. nnU-Net
requires labels contiguous from zero, so:

    0  background   (everything else, including the other 13 organs)
    1  left adrenal   (AMOS 12)
    2  right adrenal  (AMOS 11)

1 = left and 2 = right matches this project's own cache convention
(LEFT_CHANNEL_VALUE / RIGHT_CHANNEL_VALUE), so predictions can be scored by the
same code without a remapping step.

Usage
-----
    python scripts/prepare_nnunet_dataset.py --data-root ../data/amos22 \
        --out ../nnunet/nnUNet_raw
    python scripts/prepare_nnunet_dataset.py --data-root ../data/amos22 \
        --out ../nnunet/nnUNet_raw --copy      # if the filesystem dislikes symlinks
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_adrenal_segmenter import discover_cases, setup_logging   # noqa: E402

LEFT_OUT, RIGHT_OUT = 1, 2          # matches the project's cache convention


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, required=True, help="AMOS22 root")
    p.add_argument("--out", type=Path, required=True, help="nnUNet_raw directory")
    p.add_argument("--dataset-id", type=int, default=501)
    p.add_argument("--dataset-name", default="AdrenalAMOS")
    p.add_argument("--modality", choices=["ct", "mri", "all"], default="ct")
    p.add_argument("--mri-id-threshold", type=int, default=500)
    p.add_argument("--right-label", type=int, default=11, help="AMOS right adrenal id")
    p.add_argument("--left-label", type=int, default=12, help="AMOS left adrenal id")
    p.add_argument("--copy", action="store_true", help="copy images instead of symlinking")
    p.add_argument("--max-cases", type=int, default=0, help="cap for a quick check")
    p.add_argument("--overwrite", action="store_true")
    return p


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(os.path.abspath(src), dst)


def write_label(src: Path, dst: Path, left_in: int, right_in: int) -> dict:
    """Collapse AMOS's 15 organ classes to background / left / right.

    The affine and header are preserved exactly: nnU-Net checks that image and
    label geometry agree, and a rebuilt header is the usual cause of a failed
    integrity check.
    """
    import nibabel as nib

    nii = nib.load(str(src))
    data = np.asarray(nii.dataobj)
    out = np.zeros(data.shape, dtype=np.uint8)
    out[data == left_in] = LEFT_OUT
    out[data == right_in] = RIGHT_OUT

    result = nib.Nifti1Image(out, nii.affine, nii.header)
    result.set_data_dtype(np.uint8)
    nib.save(result, str(dst))
    return {"left": int((out == LEFT_OUT).sum()), "right": int((out == RIGHT_OUT).sum())}


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    folder = args.out / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    if folder.exists() and not args.overwrite:
        print(f"{folder} already exists. Pass --overwrite to rebuild it.", file=sys.stderr)
        return 2
    for sub in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
        (folder / sub).mkdir(parents=True, exist_ok=True)

    logger = setup_logging(folder / "convert.log")
    disc = SimpleNamespace(
        data_root=args.data_root, seed=42, val_fraction=0.2,
        max_train_cases=args.max_cases, max_val_cases=args.max_cases,
        modality=args.modality, mri_id_threshold=args.mri_id_threshold,
    )
    train_records, val_records = discover_cases(disc, logger)
    logger.info("Converting %d training and %d held-out cases (modality '%s')",
                len(train_records), len(val_records), args.modality)

    empty, stats = [], {"train": 0, "test": 0}
    for split, records, img_dir, lab_dir in (
        ("train", train_records, "imagesTr", "labelsTr"),
        ("test", val_records, "imagesTs", "labelsTs"),
    ):
        for i, rec in enumerate(records, 1):
            cid = rec["case_id"]
            suffix = "".join(Path(rec["image_path"].name).suffixes[-2:]) or ".nii.gz"
            link_or_copy(rec["image_path"], folder / img_dir / f"{cid}_0000{suffix}", args.copy)
            counts = write_label(rec["label_path"], folder / lab_dir / f"{cid}{suffix}",
                                 args.left_label, args.right_label)
            if counts["left"] == 0 and counts["right"] == 0:
                empty.append(cid)
            stats[split] += 1
            if i % 25 == 0 or i == len(records):
                logger.info("  %s: %d/%d", split, i, len(records))

    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "left_adrenal": LEFT_OUT, "right_adrenal": RIGHT_OUT},
        "numTraining": stats["train"],
        "file_ending": ".nii.gz",
        "description": ("AMOS22 adrenal glands only. Every other organ is background. "
                        "imagesTs holds this project's validation split, kept out of "
                        "nnU-Net's own cross-validation so the final number is "
                        "comparable with the 2.5D pipeline."),
    }
    (folder / "dataset.json").write_text(json.dumps(dataset_json, indent=2), encoding="utf-8")

    logger.info("-" * 70)
    logger.info("Wrote %s", folder)
    logger.info("  imagesTr / labelsTr   %d cases  (nnU-Net cross-validates within these)",
                stats["train"])
    logger.info("  imagesTs / labelsTs   %d cases  (held out; score these against ours)",
                stats["test"])
    logger.info("  labels                0 background, %d left (AMOS %d), %d right (AMOS %d)",
                LEFT_OUT, args.left_label, RIGHT_OUT, args.right_label)
    if empty:
        logger.warning("%d case(s) contain NO adrenal voxels after relabelling: %s",
                       len(empty), ", ".join(empty[:8]) + ("..." if len(empty) > 8 else ""))
        logger.warning("Check --left-label / --right-label against your dataset.json if this "
                       "is unexpected - silently empty labels train a model that predicts "
                       "nothing and report a plausible-looking Dice of 0.")
    else:
        logger.info("  every case has adrenal voxels")
    logger.info("-" * 70)
    logger.info("Next: nnUNetv2_plan_and_preprocess -d %d --verify_dataset_integrity",
                args.dataset_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

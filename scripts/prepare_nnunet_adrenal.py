#!/usr/bin/env python3
"""Export original full-volume AMOS CT to nnU-Net v2, with separate holdout."""
import argparse
import json
from pathlib import Path
import nibabel as nib
import numpy as np


def discover(root, ims, labs, limit):
    found = []
    for im in sorted((root / ims).glob("*.nii.gz")):
        cid = im.name[:-7]
        digits = "".join(x for x in cid if x.isdigit())
        if not digits or int(digits) >= limit:
            continue
        lb = root / labs / im.name
        if not lb.is_file():
            raise FileNotFoundError(lb)
        found.append((cid, im, lb))
    if not found:
        raise ValueError(f"No CT cases in {root / ims}")
    return found


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=Path("../data/amos22"))
    p.add_argument("--output-root", type=Path, default=Path("../nnunet_raw"))
    p.add_argument("--dataset-id", type=int, default=701)
    p.add_argument("--left-label", type=int, default=12)
    p.add_argument("--right-label", type=int, default=11)
    p.add_argument("--ct-id-limit", type=int, default=500)
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    tr = discover(a.data_root, "imagesTr", "labelsTr", a.ct_id_limit)
    va = discover(a.data_root, "imagesVa", "labelsVa", a.ct_id_limit)
    ids_tr, ids_va = {x[0] for x in tr}, {x[0] for x in va}
    if ids_tr & ids_va:
        raise ValueError("Train and heldout patient IDs overlap")
    out = a.output_root / f"Dataset{a.dataset_id:03d}_AMOSAdrenalCT"
    if out.exists() and not a.force:
        raise FileExistsError(f"{out} exists; use --force to recreate files")
    for part in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
        (out / part).mkdir(parents=True, exist_ok=True)
    for dataset, imdir, lbdir in ((tr, "imagesTr", "labelsTr"), (va, "imagesTs", "labelsTs")):
        for cid, im, lb in dataset:
            target_im = out / imdir / f"{cid}_0000.nii.gz"
            if target_im.exists() or target_im.is_symlink():
                target_im.unlink()
            target_im.symlink_to(im.resolve())
            vol = nib.load(str(lb))
            orig = np.asarray(vol.dataobj)
            mask = np.zeros(orig.shape, dtype=np.uint8)
            mask[orig == a.left_label] = 1
            mask[orig == a.right_label] = 2
            if not mask.any():
                raise ValueError(f"Missing adrenal glands in {cid}")
            header = vol.header.copy()
            header.set_data_dtype(np.uint8)
            nib.save(nib.Nifti1Image(mask, vol.affine, header),
                     str(out / lbdir / f"{cid}.nii.gz"))
    (out / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "left_adrenal": 1, "right_adrenal": 2},
        "numTraining": len(tr), "file_ending": ".nii.gz"
    }, indent=2) + "\n")
    (out / "split_manifest.json").write_text(json.dumps({
        "train_cases": sorted(ids_tr), "heldout_cases": sorted(ids_va),
        "note": "Original full abdominal scans, no label-derived crop. Do not select hyperparameters on holdout."
    }, indent=2) + "\n")
    print(f"Prepared {out}; {len(tr)} training CT, {len(va)} heldout CT")


if __name__ == "__main__":
    main()

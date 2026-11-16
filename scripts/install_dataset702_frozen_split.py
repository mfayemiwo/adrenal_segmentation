#!/usr/bin/env python3
"""Install the PRE-FROZEN Dataset702 development fold into nnU-Net preprocessed.

nnU-Net v2 looks for `splits_final.json` in the dataset preprocessed folder.
We explicitly freeze one fold, including all AMOS original training CTs and
only the development-side TotalSegmentator cases. Reject output overwrites
unless file content is identical. Never put AMOS 100 heldout or official 109
TotalSegmentator test cases in fold0 train/validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def install(plan_path: Path, raw_root: Path, preprocessed_root: Path):
    m = json.loads(plan_path.read_text(encoding="utf-8"))
    if m.get("dataset_id") != 702 or m.get("status") != "FROZEN_BEFORE_EXPANDED_CNN_TRAINING":
        raise ValueError("Unrecognised frozen Dataset702 cohort")
    train, val = m["fold0_train_ids"], m["fold0_val_ids"]
    if (len(train) != 1095 or len(val) != 158 or
        set(train) & set(val) or
        len(set(train + val)) != 1253):
        raise ValueError("Frozen fold0 differs from approved 1095+158 development cohort")
    if any(s.startswith("am_") for s in val):
        raise ValueError("All original 200 AMOS training scans must remain in fold0 train")
    raw = raw_root / "Dataset702_AMOSTotalSegAdrenalCT"
    pre = preprocessed_root / "Dataset702_AMOSTotalSegAdrenalCT"
    geometry = raw / "geometry_manifest.json"
    if not (raw / "dataset.json").is_file() or not geometry.is_file():
        raise FileNotFoundError("First complete CPU Dataset702 preparation")
    g = json.loads(geometry.read_text(encoding="utf-8"))
    h = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    if g.get("original_split_plan_sha256") != h or g.get("status") != "COMPLETE":
        raise ValueError("Prepared Dataset702 does not match frozen manifest")
    names = {p.name.removesuffix("_0000.nii.gz") for p in
             (raw / "imagesTr").glob("*_0000.nii.gz")}
    if names != set(train) | set(val):
        raise ValueError("Raw Dataset702 cases differ from frozen fold")
    if not (pre / "nnUNetPlans.json").is_file():
        raise FileNotFoundError(
            "First run nnUNetv2_plan_and_preprocess -d 702 "
            "--verify_dataset_integrity -c 3d_fullres")
    split_path = pre / "splits_final.json"
    intended = [{"train": sorted(train), "val": sorted(val)}]
    if split_path.exists():
        if json.loads(split_path.read_text(encoding="utf-8")) != intended:
            raise ValueError(
                "Existing nnU-Net splits_final.json differs from frozen cohort; "
                "refusing to overwrite.")
        action = "reused"
    else:
        split_path.write_text(json.dumps(intended, indent=2) + "\n",
                              encoding="utf-8")
        action = "installed"
    print(json.dumps({
        "status": "SUCCESS", "action": action,
        "fold": 0, "train_count": len(train), "val_count": len(val),
        "split_path": str(split_path.resolve()),
        "raw_dataset": str(raw.resolve())
    }, indent=2))
    return intended


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, default=Path(
        "runs/expanded_dataset702/plan.json"
    ))
    p.add_argument("--nnunet-raw", type=Path, default=Path("nnunet_raw"))
    p.add_argument("--nnunet-preprocessed", type=Path,
                   default=Path("nnunet_preprocessed"))
    a = p.parse_args()
    install(a.plan, a.nnunet_raw, a.nnunet_preprocessed)


if __name__ == "__main__":
    main()

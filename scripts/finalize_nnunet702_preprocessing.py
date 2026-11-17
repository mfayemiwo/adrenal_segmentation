#!/usr/bin/env python3
"""Complete or verify Dataset702 ground-truth backup in nnU-Net preprocessed.

Standard nnU-Net stores the original training reference segmentations under
nnUNet_preprocessed/Dataset702.../gt_segmentations for validation/export.
Because we run safe, resumable per-case preprocessing, this script performs
that standard copy separately, using atomic writes and SHA-256 comparison.

Only runs after all 1253 nnUNetPlans_3d_fullres preprocessed cases are ready.
No CT is copied. No originals or full-volume test data are modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.resume_nnunet702_preprocessing import required_case_files, sha256


def complete_ground_truth(raw: Path, pre: Path, frozen_plan: Path):
    record = json.loads(frozen_plan.read_text(encoding="utf-8"))
    expected = sorted(set(record["fold0_train_ids"]) |
                      set(record["fold0_val_ids"]))
    if (record.get("dataset_id") != 702 or len(expected) != 1253 or
        len(record["fold0_train_ids"]) != 1095 or
        len(record["fold0_val_ids"]) != 158):
        raise ValueError("Frozen Dataset702 development split changed")
    preproc = pre / "nnUNetPlans_3d_fullres"
    incomplete = [case for case in expected if required_case_files(preproc, case) is None]
    if incomplete:
        raise ValueError(
            f"Cannot copy GT/finish until all cases preprocessed; "
            f"{len(incomplete)} missing, e.g. {incomplete[:4]}")
    out = pre / "gt_segmentations"
    out.mkdir(parents=True, exist_ok=True)
    for idx, case in enumerate(expected, 1):
        source = raw / "labelsTr" / (case + ".nii.gz")
        target = out / source.name
        if not source.is_file():
            raise FileNotFoundError(f"Missing original Dataset702 training label {case}")
        source_sha = sha256(source)
        if target.is_file():
            if sha256(target) != source_sha:
                raise ValueError(f"Existing copied GT changed unexpectedly: {target}")
        else:
            tmp = target.with_name(target.name + ".partial")
            try:
                shutil.copy2(source, tmp)
                if sha256(tmp) != source_sha:
                    raise IOError(f"GT copy checksum mismatch: {case}")
                os.replace(tmp, target)
            finally:
                tmp.unlink(missing_ok=True)
        if idx % 100 == 0 or idx == len(expected):
            print(f"Original GT segmentations safely copied/verified: "
                  f"{idx}/{len(expected)}", flush=True)
    present = {p.name for p in out.glob("*.nii.gz")}
    if present != {c + ".nii.gz" for c in expected}:
        raise ValueError("Preprocessed gt_segmentations inventory differs from frozen cohort")
    print("SUCCESS: all 1253 original Dataset702 training GT labels backed up exactly.", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nnunet-raw", type=Path, default=Path("nnunet_raw"))
    p.add_argument("--nnunet-preprocessed", type=Path,
                   default=Path("nnunet_preprocessed"))
    p.add_argument("--frozen-plan", type=Path,
                   default=Path("runs/expanded_dataset702/plan.json"))
    args = p.parse_args()
    complete_ground_truth(
        args.nnunet_raw.resolve() / "Dataset702_AMOSTotalSegAdrenalCT",
        args.nnunet_preprocessed.resolve() / "Dataset702_AMOSTotalSegAdrenalCT",
        args.frozen_plan.resolve())


if __name__ == "__main__":
    main()

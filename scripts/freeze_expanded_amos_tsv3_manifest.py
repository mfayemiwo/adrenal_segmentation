#!/usr/bin/env python3
"""Freeze an AMOS+TotalSegmentator v3 development-only nnU-Net dataset-702 split.

Does not move or read medical image voxels; reads only original dataset case
IDs, automated training-annotation inventory, TotalSegmentator metadata and
source split manifests. NEVER includes original AMOS 100 heldout or ANY of the
109 TotalSegmentator official test cases. Creates an immutable development
manifest with one internal fold0 (AMOS 200 train + 85% TS train; 15% TS val).

IMPORTANT: the 1053 v3 train-side candidates have automated presence/geometry
checks but NO manual source-label review. They are NOT certified clean data.
All external test results have already been seen, therefore do not select
training architecture/hyperparameters on external 76-case scores.

Use separate dataset ID 702; do not touch original Dataset701.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
from math import floor
from pathlib import Path
import subprocess

AMOS_DEFAULT = Path("nnunet_raw/Dataset701_AMOSAdrenalCT")
TS_ROOT_DEFAULT = Path(
    "../data/totalsegmentator_v3/extracted/Totalsegmentator_dataset_v300"
)
CANDIDATES_DEFAULT = Path(
    "runs/totalsegmentator_v3_audit/candidates_not_train_ready.csv"
)
AUDIT_DEFAULT = Path("runs/totalsegmentator_v3_audit/case_audit.csv")
EXTERNAL_DEFAULT = Path(
    "nnunet_raw/ExternalTotalSegV3AdrenalTest_v1/external_test_manifest.json"
)
OUTPUT_DEFAULT = Path("runs/expanded_dataset702/plan.json")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hashed_rank(seed, name):
    return hashlib.sha256(f"{seed}|{name}".encode("utf-8")).hexdigest()


def read_csv(path, delimiter=","):
    with path.open(newline="", encoding="utf-8-sig") as fp:
        return list(csv.DictReader(fp, delimiter=delimiter))


def read_manifest(path):
    return json.loads(path.read_text(encoding="utf-8"))


def fixed_validation_by_institute(records, n_valid, seed):
    """Deterministic largest-remainder stratification by source institute.

    Ensures every multi-case source still has at least one *training* case.
    Doesn't rely on segmentation accuracy, external test errors or anatomy ROI.
    """
    if n_valid < 0 or n_valid >= len(records):
        raise ValueError("Internal validation must leave some TS training cases")
    groups = defaultdict(list)
    for record in records:
        groups[record["institute"]].append(record["case_id"])
    planned = {}
    for name, ids in groups.items():
        expected = n_valid * len(ids) / len(records)
        planned[name] = min(floor(expected), len(ids) - 1)
    room = n_valid - sum(planned.values())
    while room > 0:
        possible = [
            (n_valid * len(ids) / len(records) - planned[name],
             hashed_rank(seed, "site|" + name), name)
            for name, ids in groups.items()
            if planned[name] < len(ids) - 1
        ]
        if not possible:
            raise ValueError("Cannot allocate requested validation while preserving training site coverage")
        _, _, name = max(possible)
        planned[name] += 1
        room -= 1
    selected = set()
    for name, ids in sorted(groups.items()):
        ordered = sorted(ids, key=lambda cid: hashed_rank(seed, cid))
        selected.update(ordered[:planned[name]])
    if len(selected) != n_valid:
        raise AssertionError("Stratified selection count mismatch")
    return sorted(selected)


def generate_plan(amos_dir: Path, ts_root: Path, candidates_path: Path,
                  audit_path: Path, external_path: Path, *,
                  seed=20261116, val_fraction=.15,
                  expected_amos=200, expected_amos_holdout=100,
                  expected_ts=1053, expected_external=76):
    if not 0.05 <= val_fraction <= 0.30:
        raise ValueError("Validation must be between 5% and 30% of TS train candidates")
    original = read_manifest(amos_dir / "split_manifest.json")
    amos_train = sorted(original["train_cases"])
    amos_test = sorted(original["heldout_cases"])
    if len(amos_train) != expected_amos or len(amos_test) != expected_amos_holdout:
        raise ValueError("Original AMOS 200 train / 100 heldout source split changed")
    if (set(amos_train) & set(amos_test) or
        len(set(amos_train)) != len(amos_train) or
        len(set(amos_test)) != len(amos_test)):
        raise ValueError("Original AMOS cohort is not patient-disjoint")
    for cid in amos_train:
        image = amos_dir / "imagesTr" / (cid + "_0000.nii.gz")
        label = amos_dir / "labelsTr" / (cid + ".nii.gz")
        if not image.is_file() or not label.is_file():
            raise FileNotFoundError(f"Original AMOS training image/label missing: {cid}")

    audit = read_csv(audit_path)
    audited = {r["case_id"]: r for r in audit}
    if len(audited) != len(audit):
        raise ValueError("Duplicated TotalSegmentator audit IDs")
    if any(r.get("split", "").lower() not in ("train", "test")
           for r in audit):
        raise ValueError("Unknown TotalSegmentator official split")
    candidates = read_csv(candidates_path)
    if len(candidates) != expected_ts or len({
        r["case_id"] for r in candidates
    }) != expected_ts:
        raise ValueError("Expected 1053 unique TotalSegmentator training candidates")
    meta = {row["image_id"]: row for row in
            read_csv(ts_root / "meta.csv", delimiter=";")}
    if len(meta) < expected_ts:
        raise ValueError("Original TotalSegmentator metadata incomplete")
    ts = []
    for row in candidates:
        cid = row["case_id"]
        if cid not in audited or cid not in meta:
            raise ValueError(f"Candidate missing from official audit/metadata: {cid}")
        if (audited[cid].get("split", "").lower() != "train" or
            row.get("split", "").lower() != "train"):
            raise ValueError(f"Official TotalSegmentator test in training candidates: {cid}")
        if any(str(row.get(field, "")).lower() != "true" for field in (
            "both_present", "left_aligned", "right_aligned", "left_binary",
            "right_binary", "candidate_train_only"
        )):
            raise ValueError(f"Candidate did not pass automated mask checks: {cid}")
        if row.get("error"):
            raise ValueError(f"Invalid candidate row: {cid}: {row['error']}")
        folder = ts_root / cid
        if not all((folder / rel).is_file() for rel in (
            "ct.nii.gz", "segmentations/adrenal_gland_left.nii.gz",
            "segmentations/adrenal_gland_right.nii.gz"
        )):
            raise FileNotFoundError(f"Missing CT or adrenal mask: {cid}")
        ts.append({
            "case_id": cid,
            "institute": (meta[cid].get("institute") or "UNKNOWN").strip() or "UNKNOWN",
            "study_type": (meta[cid].get("study_type") or "").strip(),
            "original_slice_thickness": meta[cid].get("slice_thickness", ""),
        })
    external = read_manifest(external_path)
    if (external.get("source_partition") != "test" or
        len(external.get("case_ids", [])) != expected_external or
        len(set(external["case_ids"])) != expected_external):
        raise ValueError("External TotalSegmentator frozen cohort unexpected")
    external_ids = {cid.removeprefix("tsv3_") for cid in external["case_ids"]}
    if any(cid not in audited or audited[cid]["split"].lower() != "test"
           for cid in external_ids):
        raise ValueError("External official test IDs incorrect")
    if set(x["case_id"] for x in ts) & external_ids:
        raise ValueError("LEAKAGE: TotalSegmentator official test found in train")
    if len(audit) > 1 and len(audited) != len(audit):
        raise ValueError("Duplicate audit IDs")
    valid_n = round(expected_ts * val_fraction)
    valid = fixed_validation_by_institute(ts, valid_n, seed)
    held_internally = set(valid)
    ts_train = sorted(x["case_id"] for x in ts if x["case_id"] not in held_internally)
    train_names = sorted(
        ["am_" + cid for cid in amos_train] +
        ["tsv3_" + cid for cid in ts_train]
    )
    val_names = sorted("tsv3_" + cid for cid in valid)
    all_cases = set(train_names) | set(val_names)
    if len(all_cases) != len(amos_train) + len(ts) or set(train_names) & set(val_names):
        raise ValueError("Fold0 is not a disjoint partition of new model inputs")
    # Candidate IDs are fully frozen, including all 1053, but split is
    # development-only and never examines external predictions or Dice.
    return {
        "status": "FROZEN_BEFORE_EXPANDED_CNN_TRAINING",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": 702,
        "dataset_name": "Dataset702_AMOSTotalSegAdrenalCT",
        "seed": seed, "ts_validation_fraction": val_fraction,
        "train_count": len(train_names), "validation_count": len(val_names),
        "full_imagesTr_count": len(all_cases),
        "amos_train_count": len(amos_train),
        "ts_train_count": len(ts_train),
        "ts_validation_count": len(valid),
        "amos_source_train_case_ids": amos_train,
        "ts_source_train_case_ids": sorted(x["case_id"] for x in ts),
        "ts_source_validation_case_ids": valid,
        "fold0_train_ids": train_names, "fold0_val_ids": val_names,
        "external_test_case_ids_RESERVED_NOT_IN_DATASET": sorted(external["case_ids"]),
        "original_amos_test_case_ids_RESERVED_NOT_IN_DATASET": amos_test,
        "total_seg_institute_counts": dict(sorted({
            site: sum(x["institute"] == site for x in ts)
            for site in {x["institute"] for x in ts}
        }.items())),
        "source_paths": {
            "amos_dataset701": str(amos_dir.resolve()),
            "totalseg_source": str(ts_root.resolve()),
            "candidate_csv": str(candidates_path.resolve()),
        },
        "sha256_sources": {
            "amos_split_manifest": sha(amos_dir / "split_manifest.json"),
            "totalseg_candidate_csv": sha(candidates_path),
            "totalseg_full_case_audit": sha(audit_path),
            "totalseg_meta_csv": sha(ts_root / "meta.csv"),
            "external_test_manifest": sha(external_path),
        },
        "quality_status": "AUTOMATED_SCREEN_ONLY_UNREVIEWED",
        "geometry_policy": (
            "Original CT if ITK-orthonormal; otherwise CT-only world-space "
            "linear resample. Apply exact same geometric transform to source "
            "adrenal masks using nearest neighbour. No GT-derived ROI crop."
        ),
        "experiment_notes": [
            "Original Dataset701 200 AMOS training CTs reused; 100 AMOS test CTs never used.",
            "All TotalSegmentator source official-test CTs remain external.",
            "75% to 95% TotalSegmentator training cases are in fold0 training; rest development validation.",
            "All 1053 TotalSegmentator cases are AUTO-QA candidates, NOT human-label certified.",
            "The 76 external TotalSegmentator test scores were seen historically; no tuning based on them.",
            "External original 76/109 eligibility is restricted to two non-empty labels.",
            "This manifest alone does not start GPU training or certify any clinical labels.",
            "True CNN-SNN contribution requires matched analog control on identical data and CT workload.",
        ],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--amos", type=Path, default=AMOS_DEFAULT)
    p.add_argument("--totalseg", type=Path, default=TS_ROOT_DEFAULT)
    p.add_argument("--candidates", type=Path, default=CANDIDATES_DEFAULT)
    p.add_argument("--audit", type=Path, default=AUDIT_DEFAULT)
    p.add_argument("--external", type=Path, default=EXTERNAL_DEFAULT)
    p.add_argument("--out", type=Path, default=OUTPUT_DEFAULT)
    p.add_argument("--seed", type=int, default=20261116)
    p.add_argument("--ts-validation-fraction", type=float, default=.15)
    args = p.parse_args()
    if args.out.exists():
        p.error(f"Refusing to rewrite frozen development manifest: {args.out}")
    plan = generate_plan(
        args.amos, args.totalseg, args.candidates, args.audit, args.external,
        seed=args.seed, val_fraction=args.ts_validation_fraction
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan, indent=2) + "\n",
                        encoding="utf-8")
    print(json.dumps({
        "manifest": str(args.out.resolve()),
        "dataset_id": plan["dataset_id"],
        "full_imagesTr_count": plan["full_imagesTr_count"],
        "fold0_train": plan["train_count"],
        "fold0_validation": plan["validation_count"],
        "amos_original_train": plan["amos_train_count"],
        "ts_dev_train": plan["ts_train_count"],
        "ts_dev_validation": plan["ts_validation_count"],
        "ts_reserved_external_official_test": len(
            plan["external_test_case_ids_RESERVED_NOT_IN_DATASET"]),
        "train_label_status": plan["quality_status"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()

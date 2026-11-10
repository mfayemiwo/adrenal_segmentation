#!/usr/bin/env python3
"""Deterministically plan a BLINDED stratified TotalSegmentator adrenal-label QC.

Uses only dataset metadata and existing gland label voxel volumes, NEVER model
predictions or test performance. Outputs local review checklists and provenance.
It does not approve or exclude any CT automatically. Keep official test split
reserved: the test review is purely label/geometry assessment before inference.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from math import isfinite, sqrt
from pathlib import Path
from statistics import quantiles

from audit_totalsegmentator_v3 import DEFAULT_ROOT, metadata_for

ROOT_DEFAULT = Path("runs/totalsegmentator_v3_audit")
OUTPUT_FIELDS = [
    "case_id", "split", "age", "age_stratum",
    "institute", "study_type", "pathology", "scanner_model",
    "original_slice_thickness_mm", "thickness_stratum",
    "left_mm3", "right_mm3", "volume_stratum",
    "left_voxels", "right_voxels", "selection_reason",
    "image_path", "left_label_path", "right_label_path",
    "review_status", "reviewer", "review_date", "review_reason",
    "laterality_check", "boundary_check", "complete_gland_check",
]


def number(value):
    try:
        out = float(value)
        return out if isfinite(out) else None
    except (ValueError, TypeError):
        return None


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def age_bin(age):
    if age is None:
        return "age_unknown"
    if age < 18:
        return "pediatric"
    if age < 65:
        return "adult_18_64"
    return "older_65_plus"


def thick_bin(thick):
    if thick is None or thick <= 0:
        return "thick_unknown"
    if thick <= 1.5:
        return "thin_le1p5"
    if thick <= 3:
        return "intermediate_le3"
    return "thick_gt3"


def quantile_limits(numbers):
    values = sorted(x for x in numbers if x is not None and x > 0)
    if len(values) < 20:
        return (None, None)
    def at(p):
        return values[int((len(values) - 1) * p)]
    return at(.05), at(.95)


def build_inventory(audit_rows: list[dict], meta: dict, dataset: Path):
    choices = []
    for r in audit_rows:
        if str(r.get("both_present", "")).lower() != "true":
            continue
        if r.get("error"):
            continue
        if not all(str(r.get(k, "")).lower() == "true" for k in (
            "left_aligned", "right_aligned", "left_binary", "right_binary"
        )):
            continue
        split = str(r.get("split", "")).strip().lower()
        if split not in ("train", "test"):
            continue
        cid = r["case_id"]
        m = meta.get(cid, {})
        l, rr = number(r.get("left_mm3")), number(r.get("right_mm3"))
        if l is None or rr is None or l <= 0 or rr <= 0:
            continue
        age, thickness = number(m.get("age")), number(m.get("slice_thickness"))
        choices.append({
            "case_id": cid, "split": split,
            "age": age if age is not None else "",
            "age_stratum": age_bin(age),
            "institute": (m.get("institute") or "UNKNOWN").strip() or "UNKNOWN",
            "study_type": (m.get("study_type") or "UNKNOWN").strip() or "UNKNOWN",
            "pathology": (m.get("pathology") or "UNKNOWN").strip() or "UNKNOWN",
            "scanner_model": (m.get("scanner_model") or "UNKNOWN").strip() or "UNKNOWN",
            "original_slice_thickness_mm": thickness if thickness is not None else "",
            "thickness_stratum": thick_bin(thickness),
            "left_mm3": l, "right_mm3": rr,
            "left_voxels": r["left_voxels"], "right_voxels": r["right_voxels"],
            "image_path": str(dataset / cid / "ct.nii.gz"),
            "left_label_path": str(dataset / cid / "segmentations" / "adrenal_gland_left.nii.gz"),
            "right_label_path": str(dataset / cid / "segmentations" / "adrenal_gland_right.nii.gz"),
            "review_status": "UNREVIEWED", "reviewer": "", "review_date": "",
            "review_reason": "", "laterality_check": "",
            "boundary_check": "", "complete_gland_check": "",
        })
    return sorted(choices, key=lambda x: x["case_id"])


def add_volume_features(cases: list[dict]):
    left_limits = quantile_limits([r["left_mm3"] for r in cases])
    right_limits = quantile_limits([r["right_mm3"] for r in cases])
    for r in cases:
        flags = []
        if left_limits[0] is not None:
            if r["left_mm3"] <= left_limits[0]:
                flags.append("low_left_volume")
            if r["left_mm3"] >= left_limits[1]:
                flags.append("high_left_volume")
        if right_limits[0] is not None:
            if r["right_mm3"] <= right_limits[0]:
                flags.append("low_right_volume")
            if r["right_mm3"] >= right_limits[1]:
                flags.append("high_right_volume")
        ratio = r["left_mm3"] / r["right_mm3"]
        if ratio >= 3 or ratio <= (1 / 3):
            flags.append("volume_asymmetry")
        r["volume_stratum"] = "|".join(flags) if flags else "ordinary_volume"


def features(row: dict):
    base = {
        "age:" + row["age_stratum"],
        "thickness:" + row["thickness_stratum"],
        "institute:" + row["institute"],
        "study:" + row["study_type"],
        "scanner:" + row["scanner_model"],
    }
    path = row["pathology"].lower()
    base.add("pathology:reported" if path not in ("unknown", "unclear", "")
             else "pathology:unspecified")
    base.update("volume:" + v for v in row["volume_stratum"].split("|"))
    return base


def select_stratified(cases: list[dict], count: int, seed: int):
    if count < 0 or count > len(cases):
        raise ValueError(f"Sample size {count} cannot exceed {len(cases)}")
    if not cases:
        return []
    category_sets = {r["case_id"]: features(r) for r in cases}
    freq = Counter(f for fs in category_sets.values() for f in fs)
    selected = []
    seen = Counter()
    remaining = {r["case_id"]: r for r in cases}
    for _ in range(count):
        def score(row):
            fs = category_sets[row["case_id"]]
            coverage = sum(
                (1.0 / sqrt(freq[f])) / (1.0 + 3.0 * seen[f])
                for f in fs
            )
            # Tie-breaking is deterministic across Python versions and hosts.
            jitter = int(hashlib.sha256(
                f"{seed}|{row['case_id']}".encode()
            ).hexdigest()[:12], 16) / (16 ** 12)
            return coverage, jitter
        picked = max(remaining.values(), key=score)
        selected.append(picked)
        for feature in category_sets[picked["case_id"]]:
            seen[feature] += 1
        remaining.pop(picked["case_id"])
    for r in selected:
        r["selection_reason"] = ("deterministic maximum-diversity sampling: "
                                  + ", ".join(sorted(category_sets[r["case_id"]])))
    return selected


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--audit", type=Path,
                   default=ROOT_DEFAULT / "case_audit.csv")
    p.add_argument("--out-dir", type=Path,
                   default=Path("runs/totalsegmentator_v3_qc_plan"))
    p.add_argument("--train-review-n", type=int, default=60)
    p.add_argument("--test-review-n", type=int, default=15)
    p.add_argument("--seed", type=int, default=20261110)
    args = p.parse_args()
    if args.train_review_n < 0 or args.test_review_n < 0:
        p.error("Review sample sizes cannot be negative")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        p.error("QC output directory exists and is nonempty; do not overwrite reviewed samples")
    with args.audit.open(newline="", encoding="utf-8") as fp:
        audit_rows = list(csv.DictReader(fp))
    info = metadata_for(args.dataset)
    inventory = build_inventory(audit_rows, info, args.dataset.resolve())
    train = [r for r in inventory if r["split"] == "train"]
    test = [r for r in inventory if r["split"] == "test"]
    add_volume_features(train)
    add_volume_features(test)
    train_selected = select_stratified(train, args.train_review_n, args.seed)
    test_selected = select_stratified(test, args.test_review_n, args.seed + 1)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, records in (
        ("training_bilateral_inventory.csv", train),
        ("external_test_bilateral_inventory.csv", test),
        ("training_label_review_sample.csv", train_selected),
        ("external_test_label_review_sample.csv", test_selected),
    ):
        write_csv(args.out_dir / name, records)
    if set(r["case_id"] for r in train) & set(r["case_id"] for r in test):
        raise ValueError("Official train/test cases overlap")
    result = {
        "seed": args.seed,
        "audit_sha256": sha256(args.audit),
        "metadata_sha256": sha256(args.dataset / "meta.csv"),
        "eligible_train_cases": len(train),
        "eligible_test_cases": len(test),
        "training_review_cases": len(train_selected),
        "test_review_cases": len(test_selected),
        "selection_method": "deterministic greedy metadata-diversity; no model predictions",
        "split_policy": "Official test split stays external; no model tuning using external labels",
        "review_status": "UNREVIEWED; no automatic approval",
        "warnings": [
            "Sampled review does not establish correctness of all unreviewed masks.",
            "Clinical orientation, gland completeness and boundaries require human review.",
            "A reviewer must record deviations and predeclared exclusion criteria.",
            "If suitability is uncertain, do not make clinical-quality claims.",
        ],
    }
    (args.out_dir / "plan_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

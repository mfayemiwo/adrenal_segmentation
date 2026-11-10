#!/usr/bin/env python3
"""Audit blinded source-label QC and gate external test inference.

This script never reads model predictions or external Dice. It validates a
completed, separately saved copy of the pre-selected source-only external
review CSV against the frozen plan, records a non-sensitive human attestation
of source provenance limitations, and writes a locally auditable gate. For
each of the 15 sampled test cases, REQUIRE all checks PASS and ACCEPT. An
unreviewed or uncertain case cannot be silently counted as satisfactory.

This confirms completion of the *sampled* review only, NOT clinical label
quality for all 76 cases, nor patient-level independence.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


EXPECTED_TEST_SAMPLE = "external_test_label_review_sample.csv"
ALLOWABLE_CHECKS = {"PASS", "FAIL", "UNCERTAIN"}
REVIEW_FIELDS = {
    "review_status", "reviewer", "review_date", "review_reason",
    "laterality_check", "boundary_check", "complete_gland_check",
}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for buf in iter(lambda: fp.read(4 * 1024 * 1024), b""):
            h.update(buf)
    return h.hexdigest()


def csv_rows(path):
    with path.open(newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"Invalid/duplicated CSV fields: {path}")
        rows = list(reader)
    ids = [r["case_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate sample case IDs in {path}")
    return reader.fieldnames, {r["case_id"]: r for r in rows}


def check_review(original, reviewed, expected_count=15):
    source_fields, selected = csv_rows(original)
    result_fields, submitted = csv_rows(reviewed)
    if len(selected) != expected_count or len(submitted) != expected_count:
        raise ValueError(
            f"Frozen source-review sample must contain {expected_count} cases")
    if source_fields != result_fields or selected.keys() != submitted.keys():
        raise ValueError("Reviewer copy must preserve original headers and exact IDs")
    for cid, expected in selected.items():
        actual = submitted[cid]
        for name in source_fields:
            if name not in REVIEW_FIELDS and expected[name] != actual[name]:
                raise ValueError(f"{cid}: frozen data field changed: {name}")
        if actual["split"] != "test":
            raise ValueError(f"{cid}: case is not from official test partition")

    counts = {"ACCEPT": 0, "QUERY": 0, "REJECT": 0, "UNREVIEWED": 0}
    incomplete = []
    for cid, data in submitted.items():
        status = data["review_status"].strip().upper()
        if status not in counts:
            raise ValueError(f"{cid}: invalid review_status={status!r}")
        counts[status] += 1
        checks = [data[name].strip().upper() for name in (
            "laterality_check", "boundary_check", "complete_gland_check"
        )]
        reviewer = data["reviewer"].strip()
        review_date = data["review_date"].strip()
        if status == "UNREVIEWED":
            incomplete.append(cid)
            continue
        if not reviewer or not review_date:
            incomplete.append(cid)
            continue
        try:
            date.fromisoformat(review_date)
        except ValueError:
            incomplete.append(cid)
            continue
        if any(value not in ALLOWABLE_CHECKS for value in checks):
            incomplete.append(cid)
            continue
        if status != "ACCEPT" or any(value != "PASS" for value in checks):
            incomplete.append(cid)
            continue
    return counts, incomplete


def verify_gate(gate_path, plan_path, reviewed_path, manifest_path,
                attestation_path):
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != "SAMPLED_SOURCE_LABEL_QC_ACCEPTED_FOR_EXPLORATORY_EVAL":
        raise ValueError("External evaluation not signed off as a sampled-label screen")
    for key, path in (
        ("plan_sha256", plan_path),
        ("review_sha256", reviewed_path),
        ("manifest_sha256", manifest_path),
        ("attestation_sha256", attestation_path),
    ):
        if sha(path) != gate[key]:
            raise ValueError(f"Frozen QC evidence changed: {key}")
    counts, incomplete = check_review(plan_path, reviewed_path)
    if incomplete or counts != {"ACCEPT": 15, "QUERY": 0, "REJECT": 0,
                                "UNREVIEWED": 0}:
        raise ValueError("Review does not approve the frozen test sample")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if len(manifest.get("case_ids", [])) != 76 or manifest.get("source_partition") != "test":
        raise ValueError("Frozen 76-case official external-test manifest invalid")
    ids = set(manifest["case_ids"])
    if len(ids) != 76:
        raise ValueError("Duplicate external test cases")
    _, sample = csv_rows(plan_path)
    if any("tsv3_" + cid not in ids for cid in sample):
        raise ValueError("Sampled source QA is not a subset of external test manifest")
    statement = attestation_path.read_text(encoding="utf-8").strip()
    if len(statement) < 80 or "review" not in statement.lower():
        raise ValueError("Missing substantive human provenance/review attestation")
    print("PASS: local QA gate and unchanged frozen references validated; "
          "sampled label review is NOT certification of all 76 test masks.",
          flush=True)
    return gate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan-dir", type=Path, default=Path(
        "runs/totalsegmentator_v3_qc_plan"))
    p.add_argument("--review-csv", type=Path, default=Path(
        "runs/totalsegmentator_v3_qc_plan/reviews/external_test_review_completed.csv"))
    p.add_argument("--attestation", type=Path, default=Path(
        "runs/totalsegmentator_v3_qc_plan/reviews/reviewer_attestation.txt"))
    p.add_argument("--manifest", type=Path, default=Path(
        "nnunet_raw/ExternalTotalSegV3AdrenalTest_v1/external_test_manifest.json"))
    p.add_argument("--gate", type=Path, default=Path(
        "runs/totalsegmentator_v3_qc_plan/reviews/external_review_gate.json"))
    p.add_argument("--verify-only", action="store_true")
    args = p.parse_args()
    original = args.plan_dir / EXPECTED_TEST_SAMPLE
    if args.verify_only:
        if not args.gate.is_file():
            p.error("External-test QC signoff gate does not exist")
        verify_gate(args.gate, original, args.review_csv,
                    args.manifest, args.attestation)
        return

    if args.gate.exists():
        p.error("Existing signed gate will not be overwritten")
    counts, incomplete = check_review(original, args.review_csv)
    frozen_manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if len(frozen_manifest.get("case_ids", [])) != 76:
        raise ValueError("External-test source manifest not 76 cases")
    statement = args.attestation.read_text(encoding="utf-8").strip()
    if len(statement) < 80 or "review" not in statement.lower():
        raise ValueError(
            "Provide signed reviewer attestation (>=80 chars; mention review "
            "and independent-patient uncertainty).")
    report = {
        "status": "SAMPLED_SOURCE_LABEL_QC_ACCEPTED_FOR_EXPLORATORY_EVAL",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": sha(original),
        "review_sha256": sha(args.review_csv),
        "manifest_sha256": sha(args.manifest),
        "attestation_sha256": sha(args.attestation),
        "counts": counts,
        "sampled_test_cases": 15,
        "official_test_cases": 76,
        "scope": "Blinded 15-case source-label sample; not full annotation validation",
        "limitations": [
            "61 external cases were not included in the manual label QC sample.",
            "Image similarity screening cannot confirm patient independence.",
            "Only exploratory external evaluation is authorised; no tuning on test.",
        ],
    }
    if incomplete:
        print(json.dumps({"counts": counts,
                          "cases_not_accepted_for_exploratory_test": incomplete},
                         indent=2), flush=True)
        raise SystemExit("STOP: human label QC is unresolved; cannot sign off")
    args.gate.parent.mkdir(parents=True, exist_ok=True)
    args.gate.write_text(json.dumps(report, indent=2) + "\n",
                         encoding="utf-8")
    verify_gate(args.gate, original, args.review_csv,
                args.manifest, args.attestation)
    print(f"Signed local gate: {args.gate}", flush=True)


if __name__ == "__main__":
    main()

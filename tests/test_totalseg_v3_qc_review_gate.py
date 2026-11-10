"""Synthetic QA signoff tests: no external CTs, scores or clinical claims."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

from scripts.validate_totalseg_v3_qc_review import (
    check_review, verify_gate, sha,
)

FIELDS = [
    "case_id", "split", "institute", "review_status",
    "reviewer", "review_date", "review_reason",
    "laterality_check", "boundary_check", "complete_gland_check",
]


def make_frozen(root: Path):
    plan = root / "external_test_label_review_sample.csv"
    reviewed = root / "reviewed.csv"
    src, complete = [], []
    for n in range(15):
        name = f"s{n:04d}"
        src.append({
            "case_id": name, "split": "test", "institute": "A",
            "review_status": "UNREVIEWED", "reviewer": "",
            "review_date": "", "review_reason": "",
            "laterality_check": "", "boundary_check": "",
            "complete_gland_check": "",
        })
        complete.append({
            **src[-1], "review_status": "ACCEPT",
            "reviewer": "qualified-reviewer-1",
            "review_date": "2026-11-10",
            "review_reason": "Source-only anatomy and mask reviewed",
            "laterality_check": "PASS", "boundary_check": "PASS",
            "complete_gland_check": "PASS",
        })
    for path, rows in ((plan, src), (reviewed, complete)):
        with path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    manifest = root / "external_test_manifest.json"
    manifest.write_text(json.dumps({
        "source_partition": "test",
        "case_ids": [f"tsv3_s{n:04d}" for n in range(76)],
    }))
    attest = root / "attestation.txt"
    attest.write_text(
        "I reviewed all 15 blinded reference-only test annotations and "
        "acknowledge that unreviewed test labels and patient overlap remain "
        "uncertain. I explicitly authorise exploratory external testing."
    )
    gate = root / "gate.json"
    gate.write_text(json.dumps({
        "status": "SAMPLED_SOURCE_LABEL_QC_ACCEPTED_FOR_EXPLORATORY_EVAL",
        "plan_sha256": sha(plan), "review_sha256": sha(reviewed),
        "manifest_sha256": sha(manifest),
        "attestation_sha256": sha(attest),
    }))
    return plan, reviewed, manifest, attest, gate


class QCSignoffGateTests(unittest.TestCase):
    def test_valid_review_accepts_sample_but_not_all_76(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = make_frozen(Path(temp))
            counts, incomplete = check_review(paths[0], paths[1])
            self.assertEqual(counts["ACCEPT"], 15)
            self.assertEqual(incomplete, [])
            result = verify_gate(paths[4], *paths[:4])
            self.assertEqual(result["status"],
                             "SAMPLED_SOURCE_LABEL_QC_ACCEPTED_FOR_EXPLORATORY_EVAL")

    def test_unreviewed_case_must_block_signoff(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = make_frozen(Path(temp))
            reviewed = paths[1]
            with reviewed.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            rows[5]["review_status"] = "UNREVIEWED"
            with reviewed.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            counts, incomplete = check_review(paths[0], paths[1])
            self.assertEqual(counts["UNREVIEWED"], 1)
            self.assertIn("s0005", incomplete)
            with self.assertRaises(ValueError):
                verify_gate(paths[4], *paths[:4])

    def test_silent_metadata_change_must_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = make_frozen(Path(temp))
            with paths[1].open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            rows[0]["institute"] = "new-institute"
            with paths[1].open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaises(ValueError):
                check_review(paths[0], paths[1])

    def test_tampered_manifest_must_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = make_frozen(Path(temp))
            with paths[2].open("a", encoding="utf-8") as f:
                f.write(" ")
            with self.assertRaises(ValueError):
                verify_gate(paths[4], *paths[:4])


if __name__ == "__main__":
    unittest.main()

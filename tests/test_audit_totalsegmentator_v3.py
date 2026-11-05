"""CPU-only integration tests for the TotalSegmentator v3 no-leakage audit."""
from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from scripts.audit_totalsegmentator_v3 import (
    inspect_case, metadata_for, save_reports,
)


def make_nifti(path: Path, data: np.ndarray, affine=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(
        nib.Nifti1Image(
            data,
            np.diag([1.5, 1.5, 1.5, 1.]) if affine is None else affine,
        ),
        str(path),
    )


class TotalSegAuditTests(unittest.TestCase):
    def test_bilateral_empty_and_single_sided_cases(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            cases = (root / "s0000", root / "s0001", root / "s0002")
            for case in cases:
                make_nifti(case / "ct.nii.gz", np.zeros((9, 10, 8), np.float32))
                for side in ("left", "right"):
                    make_nifti(
                        case / "segmentations" / f"adrenal_gland_{side}.nii.gz",
                        np.zeros((9, 10, 8), np.uint8),
                    )
            # Bilateral positive and correctly aligned.
            for side in ("left", "right"):
                mask = np.zeros((9, 10, 8), np.uint8)
                mask[1:3, 2:4, 3:5] = 1
                make_nifti(
                    cases[0] / "segmentations" / f"adrenal_gland_{side}.nii.gz",
                    mask,
                )

            # Both empty, but one kidney appears -> needs review, not label.
            kidney = np.zeros((9, 10, 8), np.uint8)
            kidney[1:3, 1:4, 1:3] = 1
            make_nifti(cases[1] / "segmentations" / "kidney_left.nii.gz",
                       kidney)

            # Unilateral positive, not considered bilateral-train-ready.
            right = np.zeros((9, 10, 8), np.uint8)
            right[3:6, 3:6, 3:5] = 1
            make_nifti(cases[2] / "segmentations" /
                       "adrenal_gland_right.nii.gz", right)

            with (root / "meta.csv").open("w", newline="") as f:
                writer = csv.writer(f, delimiter=";")
                writer.writerow(["image_id", "split", "slice_thickness", "study_type"])
                writer.writerow(["s0000", "train", "5", "abdomen"])
                writer.writerow(["s0001", "train", "2", "abdomen"])
                writer.writerow(["s0002", "test", "3", "abdomen"])

            metadata = metadata_for(root)
            rows = [inspect_case(case, metadata[case.name]) for case in cases]
            self.assertTrue(rows[0]["candidate_train_only"])
            self.assertEqual(rows[0]["left_mm3"], 8 * 1.5**3)
            self.assertTrue(rows[1]["both_empty"])
            self.assertTrue(rows[1]["review_priority"])
            self.assertFalse(rows[1]["candidate_train_only"])
            self.assertTrue(rows[2]["one_present"])
            self.assertFalse(rows[2]["candidate_train_only"])

            report_dir = root / "audit"
            summary = save_reports(rows, report_dir, 3)
            self.assertEqual(summary["both_adrenals_nonempty"], 1)
            self.assertEqual(summary["only_one_adrenal_nonempty"], 1)
            self.assertEqual(summary["both_adrenals_empty"], 1)
            self.assertEqual(
                summary["train_split_nonempty_bilateral_candidates_NOT_CLEARED"], 1
            )
            self.assertTrue((report_dir / "case_audit.csv").is_file())
            self.assertTrue(
                (report_dir / "candidates_not_train_ready.csv").is_file()
            )

    def test_invalid_geometry_not_eligible(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name) / "s0000"
            make_nifti(root / "ct.nii.gz", np.zeros((6, 6, 6), np.float32))
            for side in ("left", "right"):
                mask = np.zeros((6, 6, 6), np.uint8)
                mask[1, 1, 1] = 1
                affine = np.diag([1.5, 1.5, 1.5, 1.])
                if side == "right":
                    affine[0, 3] = 6
                make_nifti(root / "segmentations" /
                           f"adrenal_gland_{side}.nii.gz", mask, affine)
            row = inspect_case(root, {"split": "train"})
            self.assertTrue(row["both_present"])
            self.assertFalse(row["right_aligned"])
            self.assertFalse(row["candidate_train_only"])

if __name__ == "__main__":
    unittest.main()

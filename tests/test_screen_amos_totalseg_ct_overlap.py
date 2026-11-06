"""CPU-only synthetic tests for AMOS vs TotalSegmentator CT overlap screening."""
import csv
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from scripts.screen_amos_totalseg_ct_overlap import (
    fingerprint_ct, thumbnail_from_array, top_candidates,
    cached_fingerprint, eligible_total_cases, amos_cts,
)


class CTOverlapScreenTests(unittest.TestCase):
    def test_exact_and_thumbnail_identical_images(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            data = np.full((16, 17, 18), -1000, dtype=np.float32)
            data[3:12, 4:13, 5:15] = np.linspace(
                -250, 350, 9 * 9 * 10, dtype=np.float32
            ).reshape(9, 9, 10)
            for name in ("a.nii.gz", "b.nii.gz"):
                nib.save(nib.Nifti1Image(data, np.eye(4)), str(root / name))
            h1, f1 = fingerprint_ct(root / "a.nii.gz")
            h2, f2 = fingerprint_ct(root / "b.nii.gz")
            self.assertEqual(h1, h2)
            self.assertAlmostEqual(float(f1 @ f2), 1.0, places=5)
            pairs = top_candidates(f1[None, :], f2[None, :],
                                   cutoff=0.98, top_k=0)
            self.assertEqual(len(pairs), 1)
            self.assertEqual((pairs[0][0], pairs[0][1]), (0, 0))
            hashed, v = cached_fingerprint(root / "a.nii.gz",
                                           root / "cache" / "a.npz", 16)
            self.assertEqual(hashed, h1)
            self.assertTrue(np.allclose(v, f1))

    def test_dissimilar_shapes_not_false_exact(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            a = np.indices((8, 9, 10))[0].astype(np.float32)
            b = np.indices((8, 9, 11))[0].astype(np.float32)
            for name, value in (("a.nii.gz", a), ("b.nii.gz", b)):
                nib.save(nib.Nifti1Image(value, np.eye(4)), str(root / name))
            h1, _ = fingerprint_ct(root / "a.nii.gz")
            h2, _ = fingerprint_ct(root / "b.nii.gz")
            self.assertNotEqual(h1, h2)

    def test_eligible_cases_keep_train_test_split(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            fields = ["case_id", "split", "both_present", "error",
                      "left_aligned", "right_aligned",
                      "left_binary", "right_binary"]
            rows = [
                ["s0001", "train", "True", "", "True", "True", "True", "True"],
                ["s0002", "test", "True", "", "True", "True", "True", "True"],
                ["s0003", "train", "False", "", "True", "True", "True", "True"],
            ]
            for cid in ("s0001", "s0002", "s0003"):
                p = root / cid
                p.mkdir()
                (p / "ct.nii.gz").touch()
            with (root / "audit.csv").open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(fields)
                writer.writerows(rows)
            chosen = eligible_total_cases(root / "audit.csv", root)
            self.assertEqual([(c, split) for c, _, split in chosen],
                             [("s0001", "train"), ("s0002", "test")])

    def test_amos_includes_ct_train_and_va_not_mri_range(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for folder, name in (("imagesTr", "amos_0123.nii.gz"),
                                 ("imagesVa", "amos_0346.nii.gz"),
                                 ("imagesTr", "amos_0522.nii.gz")):
                p = root / folder
                p.mkdir(exist_ok=True)
                (p / name).touch()
            chosen = amos_cts(root)
            self.assertEqual(sorted(c for c, _, _ in chosen),
                             ["amos_0123", "amos_0346"])


if __name__ == "__main__":
    unittest.main()

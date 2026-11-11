"""Synthetic test of a sheared NIfTI CT external workflow (no patient data)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np

from scripts.repair_totalseg_external_ct_geometry import (
    orthonormal_error, orthogonalize, isitk_readable,
    preflight, restore,
    source_manifest,
)


class NonOrthonormalTotalSegTests(unittest.TestCase):
    @staticmethod
    def make_case(source: Path, cid: str, shear: float):
        ct_dir = source / "imagesTs"
        ref_dir = source / "labelsTs"
        ct_dir.mkdir(parents=True, exist_ok=True)
        ref_dir.mkdir(parents=True, exist_ok=True)
        shape = (32, 36, 30)
        affine = np.diag([1.5, 1.5, 1.5, 1.])
        affine[0, 1] = shear
        affine[:3, 3] = [-80., 35., -40.]
        image = np.full(shape, -500.0, np.float32)
        image[5:23, 5:25, 5:20] = 80.0
        label = np.zeros(shape, np.uint8)
        label[7:16, 10:19, 12:19] = 1
        label[19:28, 10:19, 12:19] = 2
        nib.save(nib.Nifti1Image(image, affine),
                 str(ct_dir / (cid + "_0000.nii.gz")))
        nib.save(nib.Nifti1Image(label, affine),
                 str(ref_dir / (cid + ".nii.gz")))

    def test_orthogonalization_and_restore_native_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "original"
            prepared = root / "prepared"
            predictions = root / "predictions"
            restored = root / "restored"
            case_names = ("tsv3_s0001", "tsv3_s0002")
            self.make_case(original, case_names[0], shear=0.)
            self.make_case(original, case_names[1], shear=0.65)
            (original / "external_test_manifest.json").write_text(json.dumps({
                "source_partition": "test",
                "case_ids": list(case_names),
                "label_mapping": {
                    "0": "background", "1": "left_adrenal", "2": "right_adrenal"
                },
            }))
            src, _ = source_manifest(original, expected_count=2)
            self.assertEqual(len(src["case_ids"]), 2)
            # The production freeze expects exactly 76; for this synthetic
            # preflight call, patch the expected count at its own boundary.
            from unittest.mock import patch
            import scripts.repair_totalseg_external_ct_geometry as fix
            original_fn = fix.source_manifest

            def fixture_manifest(path):
                return original_fn(path, expected_count=2)

            args = SimpleNamespace(
                original=original, prepared=prepared,
                orthogonality_tolerance=1e-4,
                max_output_voxels=10_000_000, resume=True,
            )
            with patch.object(fix, "source_manifest", side_effect=fixture_manifest):
                preflight(args)
            report = json.loads((prepared / "geometry_manifest.json").read_text())
            self.assertEqual(report["case_count"], 2)
            self.assertEqual(report["regridded_count"], 1)
            self.assertEqual(report["regridded_cases"], [case_names[1]])
            symlink = prepared / "imagesTs" / (case_names[0] + "_0000.nii.gz")
            fixed = prepared / "imagesTs" / (case_names[1] + "_0000.nii.gz")
            self.assertTrue(symlink.is_symlink())
            self.assertTrue(fixed.is_file())
            self.assertFalse(fixed.is_symlink())
            self.assertLess(orthonormal_error(nib.load(str(fixed)).affine), 1e-6)
            self.assertTrue(isitk_readable(fixed)[0])

            predictions.mkdir()
            for cid in case_names:
                source_label = nib.load(str(original / "labelsTs" / (cid + ".nii.gz")))
                image = nib.load(str(prepared / "imagesTs" / (cid + "_0000.nii.gz")))
                # Emulate an ideal segmenter on the prepared grid strictly
                # for an interpolation integrity check, not a real metric.
                artificial = resample_from_to(
                    source_label, (image.shape, image.affine), order=0
                )
                nib.save(artificial, str(predictions / (cid + ".nii.gz")))
            restargs = SimpleNamespace(
                original=original, prepared=prepared,
                predictions=predictions, restored=restored, resume=True
            )
            with patch.object(fix, "source_manifest", side_effect=fixture_manifest):
                restore(restargs)
            for cid in case_names:
                native = nib.load(str(original / "labelsTs" / (cid + ".nii.gz")))
                restored_pred = nib.load(str(restored / (cid + ".nii.gz")))
                self.assertEqual(restored_pred.shape, native.shape)
                np.testing.assert_allclose(restored_pred.affine, native.affine, atol=1e-4)
                out = np.asarray(restored_pred.dataobj)
                y = np.asarray(native.dataobj)
                self.assertTrue(np.isin(out, [0, 1, 2]).all())
                for cl in (1, 2):
                    intersection = np.logical_and(out == cl, y == cl).sum()
                    dice = 2*intersection / ((out==cl).sum() + (y==cl).sum())
                    self.assertGreater(dice, 0.75,
                                       f"Geometry interpolation mismatch for {cid} class {cl}")
            # Resume is only allowed for original frozen geometry.
            with patch.object(fix, "source_manifest", side_effect=fixture_manifest):
                preflight(args)
                restore(restargs)

    def test_rejects_unexpected_source_partition(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "imagesTs").mkdir()
            (folder / "external_test_manifest.json").write_text(json.dumps({
                "source_partition": "train",
                "case_ids": ["tsv3_s0001"],
                "label_mapping": {
                    "0": "background", "1": "left_adrenal", "2": "right_adrenal"
                },
            }))
            with self.assertRaises(ValueError):
                source_manifest(folder, expected_count=1)


if __name__ == "__main__":
    unittest.main()

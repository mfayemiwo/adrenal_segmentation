"""Fast CPU tests for bilateral nnU-Net mask-only postprocessing."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np

from scripts.postprocess_nnunet_adrenal import postprocess_mask, run


def test_volume():
    x = np.zeros((20, 20, 20), dtype=np.uint8)
    # Gland 1 main component and one remote speck.
    x[1:5, 1:5, 1:5] = 1
    x[10, 10, 10] = 1
    # Gland 2 must stay independently, even though smaller than gland 1.
    x[12:15, 12:15, 12:15] = 2
    x[6, 18, 6] = 2
    return x


class BilateralPostprocessingTests(unittest.TestCase):
    def test_largest_independently_for_each_class(self):
        x = test_volume()
        out, rows = postprocess_mask(x)
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual((out == 1).sum(), 64)
        self.assertEqual((out == 2).sum(), 27)
        self.assertEqual(out[10, 10, 10], 0)
        self.assertEqual(out[6, 18, 6], 0)
        self.assertEqual(rows[0]["components_before"], 2)
        self.assertEqual(rows[1]["components_before"], 2)
        self.assertEqual(x[10, 10, 10], 1)  # source untouched

    def test_relative_mode_retains_large_secondary_fragment(self):
        x = np.zeros((18, 18, 18), dtype=np.uint8)
        x[1:5, 1:5, 1:5] = 1  # 64 voxels
        x[9:11, 9:11, 9:11] = 1  # 8 voxels
        y, _ = postprocess_mask(x, mode="relative", min_relative_size=0.1)
        self.assertEqual((y == 1).sum(), 72)
        z, _ = postprocess_mask(x, mode="relative", min_relative_size=0.2)
        self.assertEqual((z == 1).sum(), 64)

    def test_does_not_hallucinate_missing_gland(self):
        x = np.zeros((5, 5, 5), dtype=np.uint8)
        y, records = postprocess_mask(x)
        np.testing.assert_array_equal(x, y)
        self.assertEqual(records[0]["voxels_removed"], 0)

    def test_refuses_unknown_labels(self):
        x = np.zeros((5, 5, 5), dtype=np.uint8)
        x[0, 0, 0] = 4
        with self.assertRaises(ValueError):
            postprocess_mask(x)

    def test_preserves_affine_and_original_predictions(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root)
            source = folder / "raw"
            output = folder / "processed"
            source.mkdir()
            x = test_volume()
            affine = np.array([
                [0.9, 0., 0., 3.],
                [0., -0.9, 0., -2.],
                [0., 0., 2.5, 10.],
                [0., 0., 0., 1.],
            ])
            img = nib.Nifti1Image(x, affine)
            img.set_qform(affine, code=1)
            img.set_sform(affine, code=1)
            nib.save(img, str(source / "ct001.nii.gz"))
            run(SimpleNamespace(
                input_dir=source, output_dir=output, mode="largest",
                min_relative_size=0.15, connectivity=2, force=False,
            ))
            before = nib.load(str(source / "ct001.nii.gz"))
            after = nib.load(str(output / "ct001.nii.gz"))
            self.assertEqual(int(np.asarray(before.dataobj)[10, 10, 10]), 1)
            self.assertEqual(int(np.asarray(after.dataobj)[10, 10, 10]), 0)
            np.testing.assert_allclose(after.affine, affine)
            self.assertEqual(after.get_qform(coded=True)[1], 1)
            self.assertEqual(after.get_sform(coded=True)[1], 1)
            self.assertTrue((output / "postprocessing_audit.csv").exists())
            # An interrupted run can resume; completed files must be
            # verified and reused, not overwritten or silently trusted.
            existing_file = output / "ct001.nii.gz"
            old_bytes = existing_file.read_bytes()
            run(SimpleNamespace(
                input_dir=source, output_dir=output, mode="largest",
                min_relative_size=0.15, connectivity=2,
                force=False, resume=True,
            ))
            self.assertEqual(existing_file.read_bytes(), old_bytes)
            self.assertTrue((output / "postprocessing_audit.csv").exists())
            # A stale/corrupted mask must stop the resume safely.
            broken = np.zeros_like(x)
            nib.save(nib.Nifti1Image(broken, affine), str(existing_file))
            with self.assertRaises(ValueError):
                run(SimpleNamespace(
                    input_dir=source, output_dir=output, mode="largest",
                    min_relative_size=0.15, connectivity=2,
                    force=False, resume=True,
                ))
            # Recreate a valid processed output for original existing-file checks.
            run(SimpleNamespace(
                input_dir=source, output_dir=output, mode="largest",
                min_relative_size=0.15, connectivity=2, force=True,
            ))
            with self.assertRaises(FileExistsError):
                run(SimpleNamespace(
                    input_dir=source, output_dir=output, mode="largest",
                    min_relative_size=0.15, connectivity=2, force=False,
                ))
            with self.assertRaises(ValueError):
                run(SimpleNamespace(
                    input_dir=source, output_dir=source, mode="largest",
                    min_relative_size=0.15, connectivity=2, force=True,
                ))


if __name__ == "__main__":
    unittest.main()

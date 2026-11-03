"""Test results packaging without any GPU, real patient data, or SLURM."""
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.collect_adrenal_results import collect


class CollectResultsTests(unittest.TestCase):
    def test_collect_metrics_logs_not_images_or_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "runs" / "energy_nnunet").mkdir(parents=True)
            (root / "runs" / "energy_nnunet" / "nnunet_energy_trials.csv").write_text("energy_j\n100\n")
            (root / "runs" / "model.pt").write_bytes(b"weights")
            (root / "runs" / "patient.nii.gz").write_bytes(b"patient")
            (root / "nnunet3d_all.42.out").write_text("job finished\n")
            export = root / "results_exports" / "test.zip"
            result = collect(root, export, max_file_mb=2, max_total_mb=10)
            self.assertEqual(result["included_count"], 2)
            with zipfile.ZipFile(export) as zf:
                names = set(zf.namelist())
                self.assertIn("runs/energy_nnunet/nnunet_energy_trials.csv", names)
                self.assertIn("nnunet3d_all.42.out", names)
                self.assertIn("COLLECTION_MANIFEST.json", names)
                self.assertNotIn("runs/patient.nii.gz", names)
                self.assertNotIn("runs/model.pt", names)
                manifest = json.loads(zf.read("COLLECTION_MANIFEST.json"))
                self.assertEqual(manifest["included_count"], 2)


if __name__ == "__main__":
    unittest.main()

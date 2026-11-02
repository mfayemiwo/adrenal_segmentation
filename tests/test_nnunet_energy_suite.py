"""CPU-only tests for the optional nnU-Net whole-volume energy suite."""
import csv
import tempfile
import unittest
from pathlib import Path

from scripts.run_nnunet_energy_suite import (
    dataset_id_from_path, read_full_scan_metrics, trial_row,
)


class NnunetEnergyTests(unittest.TestCase):
    def test_dataset_id(self):
        self.assertEqual(dataset_id_from_path(Path("/tmp/Dataset701_AMOSAdrenalCT")), "701")
        with self.assertRaises(ValueError):
            dataset_id_from_path(Path("/tmp/NotAnNnunetDataset"))

    def test_trial_row_full_scan_energy(self):
        row = trial_row(2, {
            "energy_j": 1200.0, "elapsed_s": 30.0,
            "energy_source": "amd_smi_energy_counter",
            "mean_sample_power_w": 40.0,
            "peak_sampled_gpu_vram_gib": 7.5,
            "gpu_mapping": "hip_pci_bdf_match",
            "device_smi_index": 3,
        }, 10, {"mean_dice_left": 0.8, "mean_dice_right": 0.75})
        self.assertEqual(row["gpu_energy_j_per_ct"], 120.0)
        self.assertEqual(row["seconds_per_ct"], 3.0)
        self.assertEqual(row["peak_sampled_vram_gib"], 7.5)
        self.assertEqual(row["mean_dice_left"], 0.8)

    def test_invalid_energy_rejected(self):
        with self.assertRaises(ValueError):
            trial_row(1, {"energy_j": None, "elapsed_s": 1}, 2, {})

    def test_per_case_validation_quality_and_ids(self):
        with tempfile.TemporaryDirectory() as d:
            metrics = Path(d) / "scores.csv"
            with metrics.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "case_id", "dice_left", "dice_right",
                    "hd95_left_mm", "hd95_right_mm",
                    "nsd2mm_left", "nsd2mm_right",
                ])
                writer.writeheader()
                for case, score in (("amos_0001", 0.6), ("amos_0002", 0.8)):
                    writer.writerow({
                        "case_id": case,
                        "dice_left": score, "dice_right": 0.5,
                        "hd95_left_mm": 8.0, "hd95_right_mm": 10.0,
                        "nsd2mm_left": 0.7, "nsd2mm_right": 0.8,
                    })
            names, scores = read_full_scan_metrics(metrics)
            self.assertEqual(names, ("amos_0001", "amos_0002"))
            self.assertAlmostEqual(scores["mean_dice_left"], 0.7)


if __name__ == "__main__":
    unittest.main()

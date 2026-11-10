"""Synthetic, CPU-only tests: blinded QA, external reference mapping and metrics.

No AMOS or TotalSegmentator patient images are required by CI.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from scripts.plan_totalseg_v3_adrenal_qc import (
    build_inventory, add_volume_features, select_stratified,
)
from scripts.prepare_totalseg_external_adrenal_test import (
    compose_left_right, prepare_one, read_test_inventory,
)
from scripts.evaluate_totalseg_external_adrenal import (
    score_case, bootstrap_mean_ci,
)
from scripts.render_totalseg_v3_adrenal_qc import montage


def mock_case(root, case_id="s0001"):
    folder = root / case_id
    (folder / "segmentations").mkdir(parents=True)
    shape = (24, 25, 26)
    affine = np.diag([1.5, 1.5, 1.5, 1.])
    ct = np.random.default_rng(12).normal(50, 45, shape).astype(np.float32)
    left = np.zeros(shape, np.uint8)
    right = np.zeros(shape, np.uint8)
    left[3:7, 4:8, 8:11] = 1
    right[16:20, 15:19, 10:13] = 1
    files = [
        (folder / "ct.nii.gz", ct),
        (folder / "segmentations" / "adrenal_gland_left.nii.gz", left),
        (folder / "segmentations" / "adrenal_gland_right.nii.gz", right),
    ]
    for path, data in files:
        nib.save(nib.Nifti1Image(data, affine), str(path))
    return folder


class NextStageTests(unittest.TestCase):
    def test_external_label_id_mapping_and_perfect_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = mock_case(root)
            ct = source / "ct.nii.gz"
            left = source / "segmentations/adrenal_gland_left.nii.gz"
            right = source / "segmentations/adrenal_gland_right.nii.gz"
            label_img = compose_left_right(ct, left, right)
            merged = np.asarray(label_img.dataobj)
            self.assertEqual(int((merged == 1).sum()), 4*4*3)
            self.assertEqual(int((merged == 2).sum()), 4*4*3)
            out = root / "external"
            (out / "imagesTs").mkdir(parents=True)
            (out / "labelsTs").mkdir(parents=True)
            pid = prepare_one({"case_id": "s0001"}, root, out, resume=False)
            self.assertEqual(pid, "tsv3_s0001")
            saved = out / "labelsTs" / (pid + ".nii.gz")
            self.assertTrue(saved.is_file())
            self.assertTrue((out / "imagesTs" / (pid + "_0000.nii.gz")).is_symlink())
            predict = root / "pred.nii.gz"
            nib.save(label_img, str(predict))
            scores = score_case(saved, predict, ct)
            self.assertAlmostEqual(scores["dice_left"], 1.0)
            self.assertAlmostEqual(scores["dice_right"], 1.0)
            self.assertAlmostEqual(scores["dice_bilateral"], 1.0)
            self.assertAlmostEqual(scores["hd95_left_mm"], 0.0)
            self.assertAlmostEqual(scores["nsd2mm_right"], 1.0)
            self.assertEqual(prepare_one({"case_id": "s0001"}, root, out, resume=True),
                             pid)
            with self.assertRaises(FileExistsError):
                prepare_one({"case_id": "s0001"}, root, out, resume=False)

    def test_reference_review_montage_source_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = mock_case(root)
            row = {
                "case_id": "s0001", "split": "test",
                "image_path": str(source / "ct.nii.gz"),
                "left_label_path": str(source / "segmentations/adrenal_gland_left.nii.gz"),
                "right_label_path": str(source / "segmentations/adrenal_gland_right.nii.gz"),
                "original_slice_thickness_mm": 2.5,
            }
            png = montage(row, root / "qc", zoom_radius=18)
            self.assertTrue(png.exists())
            self.assertGreater(png.stat().st_size, 10000)

    def test_review_sample_is_split_safe_and_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for cid in ("s0001", "s0002", "s0003"):
                mock_case(root, cid)
            audit = []
            meta = {}
            for i, cid in enumerate(("s0001", "s0002", "s0003")):
                audit.append({
                    "case_id": cid, "split": "train" if i < 2 else "test",
                    "both_present": "True", "left_aligned": "True",
                    "right_aligned": "True", "left_binary": "True",
                    "right_binary": "True", "left_mm3": 2000+i*100,
                    "right_mm3": 1600+i*100, "left_voxels": "500",
                    "right_voxels": "550", "error": "",
                })
                meta[cid] = {"age": 10+i*30, "slice_thickness": 1.0+i,
                             "institute": "A" if i < 2 else "B",
                             "study_type": "ct abdomen"}
            inventory = build_inventory(audit, meta, root)
            train = [x for x in inventory if x["split"] == "train"]
            test = [x for x in inventory if x["split"] == "test"]
            add_volume_features(train)
            add_volume_features(test)
            a = select_stratified(train, 1, 20261110)[0]["case_id"]
            b = select_stratified(train, 1, 20261110)[0]["case_id"]
            self.assertEqual(a, b)
            self.assertNotIn(a, {row["case_id"] for row in test})

    def test_bootstrap_ci_on_perfect_scores(self):
        rows = [{"dice_bilateral": 1.0} for _ in range(3)]
        result = bootstrap_mean_ci(rows, "dice_bilateral", draws=200)
        self.assertEqual(result["ci95"], [1.0, 1.0])
        self.assertEqual(result["n"], 3)

    def test_excludes_train_cases_from_external_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "cohort.csv"
            with csv_path.open("w", newline="") as fp:
                writer = csv.DictWriter(fp, fieldnames=["case_id", "split"])
                writer.writeheader()
                writer.writerow({"case_id": "s0001", "split": "train"})
            with self.assertRaises(ValueError):
                read_test_inventory(csv_path)

if __name__ == "__main__":
    unittest.main()

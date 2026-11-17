"""Synthetic CPU-only tests for nnU-Net Dataset702 time-limit checkpoints.

No patient CT images, GPU, nnunetv2 package or cluster filesystem required.
"""
from __future__ import annotations

import json
from pathlib import Path
import pickle
import tempfile
import unittest

from scripts.resume_nnunet702_preprocessing import (
    commit_staged_case, required_case_files, validate_case_files,
    write_json_once,
)
from scripts.finalize_nnunet702_preprocessing import complete_ground_truth


class AtomicResumeTests(unittest.TestCase):
    def test_blosc2_staging_commits_metadata_last(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stage = root / "stage"
            target = root / "target"
            stage.mkdir()
            case = "tsv3_s0001"
            self.assertFalse(validate_case_files(target, case))
            # A previous interrupted job may have data + seg but no .pkl.
            target.mkdir()
            (target / (case + ".b2nd")).write_bytes(b"partial")
            (target / (case + "_seg.b2nd")).write_bytes(b"partial")
            self.assertIsNone(required_case_files(target, case))
            (stage / (case + ".b2nd")).write_bytes(b"complete data")
            (stage / (case + "_seg.b2nd")).write_bytes(b"complete seg")
            with (stage / (case + ".pkl")).open("wb") as fp:
                pickle.dump({"spacing": [1.5, 1.5, 1.5]}, fp)
            commit_staged_case(stage, target, case)
            self.assertTrue(validate_case_files(target, case))
            self.assertFalse((stage / (case + ".pkl")).exists())
            self.assertEqual((target / (case + ".b2nd")).read_bytes(),
                             b"complete data")
            self.assertEqual(len(required_case_files(target, case)), 3)

    def test_npz_case_outputs_compatible_with_older_nnunet(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = "am_amos_0001"
            (root / (case + ".npz")).write_bytes(b"npz dummy")
            with (root / (case + ".pkl")).open("wb") as fp:
                pickle.dump({"class_locations": {}}, fp)
            self.assertTrue(validate_case_files(root, case))
            self.assertEqual(len(required_case_files(root, case)), 2)

    def test_no_overwriting_preprocessing_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "method.json"
            write_json_once(path, {"method": "unchanged", "plan_sha": "abc"})
            write_json_once(path, {"method": "unchanged", "plan_sha": "abc"})
            with self.assertRaises(ValueError):
                write_json_once(path, {"method": "changed", "plan_sha": "xyz"})

    def test_finalization_requires_1253_cases_and_copies_gt_atomically(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "Dataset702_AMOSTotalSegAdrenalCT"
            pre = root / "pre"
            images = raw / "labelsTr"
            cases = [f"tsv3_s{i:04d}" for i in range(1253)]
            folder = pre / "nnUNetPlans_3d_fullres"
            images.mkdir(parents=True)
            folder.mkdir(parents=True)
            manifest = root / "plan.json"
            manifest.write_text(json.dumps({
                "dataset_id": 702,
                "fold0_train_ids": cases[:1095],
                "fold0_val_ids": cases[1095:],
            }))
            for c in cases:
                (images / (c + ".nii.gz")).write_bytes(b"fake-label-" + c.encode())
                (folder / (c + ".b2nd")).write_bytes(b"data")
                (folder / (c + "_seg.b2nd")).write_bytes(b"seg")
                with (folder / (c + ".pkl")).open("wb") as fp:
                    pickle.dump({"spacing": [1.0, 1.0, 1.0]}, fp)

            # One missing segmentation or .pkl prevents an incomplete
            # dataset from appearing to be finished.
            broken = folder / (cases[10] + "_seg.b2nd")
            broken.unlink()
            with self.assertRaises(ValueError):
                complete_ground_truth(raw, pre, manifest)
            broken.write_bytes(b"seg")

            complete_ground_truth(raw, pre, manifest)
            self.assertEqual(len(list((pre / "gt_segmentations").glob("*.nii.gz"))),
                             1253)
            complete_ground_truth(raw, pre, manifest)  # exact restart
            target = pre / "gt_segmentations" / (cases[10] + ".nii.gz")
            target.write_bytes(b"modified")
            with self.assertRaises(ValueError):
                complete_ground_truth(raw, pre, manifest)


if __name__ == "__main__":
    unittest.main()

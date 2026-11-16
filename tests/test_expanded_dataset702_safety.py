"""Synthetic CPU tests of expanded AMOS + TotalSegmentator v3 fold0 freeze.

All synthetic images are tiny and anonymous. No patient files needed.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import nibabel as nib
import numpy as np

from scripts.freeze_expanded_amos_tsv3_manifest import generate_plan
from scripts.prepare_expanded_amos_tsv3_dataset702 import run, frozen_plan


def save(path, data, affine):
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine), str(path))


def tscase(root, cid, *, shear=0.0):
    source = root / cid
    affine = np.diag([1.5, 1.5, 1.5, 1.0])
    affine[0, 1] = shear
    affine[:3, 3] = [-70, 40, 20]
    arr = np.zeros((24, 28, 22), dtype=np.float32)
    arr[3:16, 5:20, 2:15] = 60.0
    la = np.zeros_like(arr, dtype=np.uint8)
    ra = la.copy()
    la[5:10, 8:14, 10:14] = 1
    ra[15:20, 8:14, 10:14] = 1
    save(source / "ct.nii.gz", arr, affine)
    save(source / "segmentations/adrenal_gland_left.nii.gz", la, affine)
    save(source / "segmentations/adrenal_gland_right.nii.gz", ra, affine)


def write_csv(path, columns, records, delimiter=","):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(records)


class Dataset702SafetyTests(unittest.TestCase):
    def build(self, tmp):
        root = Path(tmp)
        amos = root / "Dataset701_AMOSAdrenalCT"
        ts = root / "total_seg"
        report_dir = root / "report"
        for cid in ("amos_0001", "amos_0002"):
            aff = np.diag([1., 1., 1.5, 1.])
            ct = np.zeros((20, 24, 26), dtype=np.float32)
            mask = np.zeros((20, 24, 26), dtype=np.uint8)
            mask[2:5, 5:8, 4:7] = 1
            mask[12:15, 5:8, 4:7] = 2
            save(amos / "imagesTr" / (cid + "_0000.nii.gz"), ct, aff)
            save(amos / "labelsTr" / (cid + ".nii.gz"), mask, aff)
        amos.mkdir(exist_ok=True)
        (amos / "split_manifest.json").write_text(json.dumps({
            "train_cases": ["amos_0001", "amos_0002"],
            "heldout_cases": ["amos_0999"],
        }))
        names = [f"s{i:04d}" for i in range(1, 5)]
        for i, cid in enumerate(names):
            tscase(ts, cid, shear=0.65 if i == 1 else 0.)
        tscase(ts, "s9999")
        rows = []
        for cid in names + ["s9999"]:
            istrain = cid != "s9999"
            rows.append({
                "case_id": cid, "split": "train" if istrain else "test",
                "both_present": "True",
                "candidate_train_only": "True" if istrain else "False",
                "left_aligned": "True", "right_aligned": "True",
                "left_binary": "True", "right_binary": "True", "error": "",
            })
        csv_fields = list(rows[0])
        write_csv(report_dir / "case_audit.csv", csv_fields, rows)
        write_csv(report_dir / "candidates_not_train_ready.csv", csv_fields, rows[:-1])
        write_csv(ts / "meta.csv",
                  ["image_id", "split", "institute", "study_type", "slice_thickness"],
                  [
                      {
                          "image_id": cid,
                          "split": "test" if cid == "s9999" else "train",
                          "institute": "site_A" if i % 2 else "site_B",
                          "study_type": "Abdomen", "slice_thickness": "2.5"
                      } for i, cid in enumerate(names + ["s9999"])
                  ], delimiter=";")
        external = root / "external_test_manifest.json"
        external.write_text(json.dumps({
            "source_partition": "test", "case_ids": ["tsv3_s9999"]
        }))
        plan = generate_plan(
            amos, ts, report_dir / "candidates_not_train_ready.csv",
            report_dir / "case_audit.csv", external,
            seed=20261116, val_fraction=.25,
            expected_amos=2, expected_amos_holdout=1,
            expected_ts=4, expected_external=1
        )
        plan_path = root / "plan.json"
        plan_path.write_text(json.dumps(plan, indent=2) + "\n")
        return plan_path, plan, root / "Dataset702_AMOSTotalSegAdrenalCT"

    def test_frozen_dev_split_never_includes_amos_holdout_or_tsv3_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, plan, _ = self.build(tmp)
            self.assertEqual(plan["amos_train_count"], 2)
            self.assertEqual(plan["ts_train_count"], 3)
            self.assertEqual(plan["ts_validation_count"], 1)
            self.assertEqual(len(plan["fold0_train_ids"]), 5)
            self.assertEqual(len(plan["fold0_val_ids"]), 1)
            self.assertEqual(plan["full_imagesTr_count"], 6)
            self.assertEqual(
                set(plan["fold0_train_ids"]) & set(plan["fold0_val_ids"]), set()
            )
            self.assertNotIn("tsv3_s9999", plan["fold0_train_ids"])
            self.assertNotIn("tsv3_s9999", plan["fold0_val_ids"])
            self.assertNotIn("am_amos_0999", plan["fold0_train_ids"])
            self.assertEqual(frozen_plan(path)["dataset_id"], 702)
            # Repeated selection with identical pre-declared seed is stable.
            path2, plan2, _ = self.build(tmp + "2")
            self.assertEqual(plan["fold0_val_ids"], plan2["fold0_val_ids"])

    def test_prepare_mixed_dataset_uses_identical_ct_mask_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, plan, output = self.build(tmp)
            options = SimpleNamespace(
                manifest=path, out=output, resume=False,
                orthogonality_tolerance=1e-4,
                max_output_voxels=10_000_000
            )
            run(options)
            summary = json.loads((output / "geometry_manifest.json").read_text())
            self.assertEqual(summary["num_total_TS_cases"], 4)
            self.assertEqual(summary["num_regridded_TS_cases"], 1)
            self.assertEqual(
                json.loads((output / "dataset.json").read_text())["numTraining"], 6
            )
            self.assertFalse((output / "imagesTs").exists())
            self.assertFalse((output / "labelsTs").exists())
            self.assertFalse((output / "imagesTr" / "tsv3_s9999_0000.nii.gz").exists())
            for cid in plan["ts_source_train_case_ids"]:
                data = nib.load(str(output / "imagesTr" /
                                    ("tsv3_" + cid + "_0000.nii.gz")))
                label = nib.load(str(output / "labelsTr" / ("tsv3_" + cid + ".nii.gz")))
                self.assertEqual(data.shape, label.shape)
                np.testing.assert_allclose(data.affine, label.affine, atol=1e-4)
                self.assertTrue(np.isin(np.asarray(label.dataobj), (0, 1, 2)).all())
                self.assertGreater((np.asarray(label.dataobj) == 1).sum(), 0)
                self.assertGreater((np.asarray(label.dataobj) == 2).sum(), 0)
            self.assertFalse((output / "imagesTr" / "tsv3_s0002_0000.nii.gz").is_symlink())
            self.assertTrue((output / "imagesTr" / "tsv3_s0001_0000.nii.gz").is_symlink())
            options.resume = True
            run(options)


if __name__ == "__main__":
    unittest.main()

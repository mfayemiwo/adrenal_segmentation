#!/usr/bin/env python3
"""Create a traceable local evidence snapshot without exporting any patient CT data.

Copies small ORIGINAL metrics, postprocessor audits, telemetry JSON/CSV, dataset
QC and overlap-screen summaries to a versioned local folder; computes SHA-256
both at source and snapshot. Records model/prediction inventory, git commit,
Python version and experiment identity. Never modifies source files, deletes
results, exports NIfTI/CT, or pushes data to GitHub. Source raw imaging data and
checkpoints must be backed up separately to approved institutional storage.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import shutil
import subprocess

SMALL_FILES = [
    "nnunet_raw/Dataset701_AMOSAdrenalCT/dataset.json",
    "nnunet_raw/Dataset701_AMOSAdrenalCT/split_manifest.json",
    "runs/nnunet3d_holdout_metrics_fold0.csv",
    "runs/nnunet3d_holdout_metrics_cc_largest.csv",
    "runs/nnunet3d_holdout_metrics_cc_relative15.csv",
    "runs/nnunet3d_holdout_predictions_cc_largest/postprocessing_audit.csv",
    "runs/nnunet3d_holdout_predictions_cc_relative15/postprocessing_audit.csv",
    "runs/totalsegmentator_v3_audit/summary.json",
    "runs/totalsegmentator_v3_audit/case_audit.csv",
    "runs/totalsegmentator_v3_audit/candidates_not_train_ready.csv",
    "runs/ct_overlap_screen/summary.json",
    "runs/ct_overlap_screen/candidate_overlap_pairs.csv",
    "runs/energy_nnunet_fullvolume_repro/telemetry_probe.json",
    "runs/energy_nnunet_fullvolume_repro/nnunet_energy_summary.json",
    "runs/energy_nnunet_fullvolume_repro/nnunet_energy_trials.csv",
    *[f"runs/energy_nnunet_fullvolume_repro/nnunet_fullvolume_r{i}.{ext}"
      for i in (1, 2, 3) for ext in ("json", "csv")],
]
CHECKPOINT = (
    "nnunet_results/Dataset701_AMOSAdrenalCT/"
    "nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth"
)
PRED_DIRS = (
    "runs/nnunet3d_holdout_predictions_fold0",
    "runs/nnunet3d_holdout_predictions_cc_largest",
    "runs/nnunet3d_holdout_predictions_cc_relative15",
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def commit(root: Path) -> dict:
    def run(*args):
        try:
            return subprocess.check_output(
                ["git", "-C", str(root), *args],
                text=True, stderr=subprocess.DEVNULL, timeout=10
            ).strip()
        except (OSError, subprocess.SubprocessError):
            return None
    return {
        "branch": run("branch", "--show-current"),
        "commit": run("rev-parse", "HEAD"),
        "working_tree_status": run("status", "--porcelain"),
    }


def inventory_prediction_dir(path: Path, hash_nifti: bool) -> dict:
    if not path.is_dir():
        return {"exists": False}
    images = sorted(
        p for p in path.glob("*.nii.gz")
        if not p.name.endswith(".partial.nii.gz")
    )
    data = {
        "exists": True,
        "nifti_count": len(images),
        "file_names": [p.name for p in images],
        "total_bytes": sum(p.stat().st_size for p in images),
    }
    if hash_nifti:
        data["sha256_by_filename"] = {p.name: digest(p) for p in images}
    return data


def write_snapshot(root: Path, out: Path, hash_checkpoint: bool,
                   hash_nifti: bool) -> dict:
    if out.resolve() == root.resolve():
        raise ValueError("Evidence snapshot cannot replace the project root")
    out.mkdir(parents=True, exist_ok=False)
    copied = []
    missing = []
    for name in SMALL_FILES:
        source = root / name
        if not source.is_file():
            missing.append(name)
            continue
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        original_sha = digest(source)
        shutil.copy2(source, target)
        copied_sha = digest(target)
        if original_sha != copied_sha:
            raise IOError(f"Checksum mismatch during archival copy: {name}")
        copied.append({
            "path": name, "sha256": original_sha, "bytes": source.stat().st_size,
        })

    model = root / CHECKPOINT
    checkpoint_info = {
        "path": CHECKPOINT, "exists": model.is_file(),
        "bytes": model.stat().st_size if model.is_file() else None,
        "sha256": digest(model) if model.is_file() and hash_checkpoint else None,
        "note": "Checkpoint is NOT copied by this script; back up separately.",
    }
    listing = {
        path: inventory_prediction_dir(root / path, hash_nifti)
        for path in PRED_DIRS
    }
    rows = None
    metrics = root / "runs/nnunet3d_holdout_metrics_fold0.csv"
    if metrics.is_file():
        with metrics.open(newline="", encoding="utf-8") as fp:
            rows = list(csv.DictReader(fp))
        if len(rows) != 100 or len({r["case_id"] for r in rows}) != 100:
            raise ValueError("Refusing to freeze: AMOS metrics not 100 unique cases")

    report = {
        "snapshot_format": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "research_branch": "research/adrenaspike3d",
        "source_project_root": str(root.resolve()),
        "git": commit(root),
        "python_version": platform.python_version(),
        "copied_source_metrics_and_logs": copied,
        "missing_expected_files": missing,
        "model_checkpoint": checkpoint_info,
        "prediction_inventories": listing,
        "metrics_case_count": len(rows) if rows is not None else None,
        "guardrails": [
            "Only original de-identified aggregate research metrics/logs are copied.",
            "This snapshot does not include source CT scans, label NIfTIs, "
            "image predictions or model weights.",
            "Do not push source patient-derived metadata, original imaging data "
            "or potentially sensitive per-case metrics to a public GitHub repository.",
            "Back up snapshot and original large checkpoint/NIfTIs to approved "
            "institutional storage; record archive checksum/location separately.",
        ],
    }
    (out / "manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "snapshot": str(out.resolve()),
        "small_files_archived": len(copied),
        "missing_optional_or_expected_files": missing,
        "checkpoint_exists": checkpoint_info["exists"],
        "metrics_case_count": report["metrics_case_count"],
        "prediction_counts": {
            name: data.get("nifti_count") for name, data in listing.items()
        },
        "git_commit": report["git"]["commit"],
    }, indent=2), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--hash-checkpoint", action="store_true")
    p.add_argument("--hash-nifti", action="store_true",
                   help="Hash each of 300 large prediction files (slow)")
    args = p.parse_args()
    root = args.root.resolve()
    dest = (args.out or
            (root / "runs" / "research_evidence_snapshots" /
             datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")))
    if dest.exists():
        p.error(f"Refusing to overwrite existing evidence snapshot: {dest}")
    write_snapshot(root, dest, args.hash_checkpoint, args.hash_nifti)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Package adrenal-segmentation experiment results for scientific review.

No GPU, third-party packages, or training required. Collects text results,
metrics, plots, and SLURM logs. Deliberately excludes patient CT images,
label volumes, prediction NIfTIs, raw caches, and model checkpoints.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ALLOWED = {".csv", ".json", ".txt", ".log", ".md", ".yaml", ".yml", ".png", ".svg", ".pdf"}
EXCLUDED_TOKENS = ("secret", "password", "credential", "access_token", "private_key", "api_key")
CHECK_PATHS = (
    "runs/nnunet3d_holdout_metrics_fold0.csv",
    "runs/nnunet3d_holdout_predictions_fold0",
    "runs/energy_nnunet/telemetry_probe.json",
    "runs/energy_nnunet/nnunet_energy_trials.csv",
    "runs/energy_nnunet/nnunet_energy_summary.json",
    "runs/energy_nnunet/hook.log",
    "runs/energy_phase2/telemetry_probe.json",
    "runs/energy_phase2/phase2_energy_trials.csv",
    "runs/energy_phase2/phase2_energy_paired.csv",
    "runs/energy_phase2/phase2_energy_summary.json",
    "runs/energy_phase2/hook.log",
    "runs/phase3_diagnostics",
    "runs/phase2_anatomy_snn_s1/best_model.pt",
    "runs/phase2_anatomy_analog_s1/best_model.pt",
    "runs/enable_energy_after_nnunet",
    "nnunet_results",
    "nnunet_preprocessed",
    "nnunet_raw/Dataset701_AMOSAdrenalCT/split_manifest.json",
)


def safe_command(args, max_chars=10000):
    try:
        result = subprocess.run(args, capture_output=True, text=True,
                                timeout=12, check=False)
        if result.returncode:
            return f"unavailable (exit {result.returncode}): " + result.stderr.strip()[:300]
        return result.stdout[:max_chars]
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
        return f"unavailable ({type(exc).__name__})"


def status(root):
    outputs = {}
    for rel in CHECK_PATHS:
        path = root / rel
        if path.is_file():
            outputs[rel] = "present (file)"
        elif path.is_dir():
            outputs[rel] = "present (directory)"
        else:
            outputs[rel] = "not found yet"
    fold_parent = root / "nnunet_results" / "Dataset701_AMOSAdrenalCT"
    checkpoints = []
    if fold_parent.exists():
        checkpoints = sorted(
            str(p.relative_to(root)) for name in ("checkpoint_latest.pth", "checkpoint_final.pth")
            for p in fold_parent.rglob(name) if p.is_file()
        )
    outputs["nnunet_checkpoint_paths_only_not_included"] = checkpoints
    return outputs


def candidates(root):
    seen = set()
    def add(path):
        if path in seen or path.is_symlink() or not path.is_file():
            return
        if any(token in path.name.lower() for token in EXCLUDED_TOKENS):
            return
        seen.add(path)
        return path

    # Metrics, research reports and small diagnostic figures from all runs.
    runs = root / "runs"
    if runs.is_dir():
        for p in runs.rglob("*"):
            if p.suffix.lower() in ALLOWED:
                item = add(p)
                if item:
                    yield item

    # nnU-Net training logs, curves, and configuration (no checkpoint weights).
    results = root / "nnunet_results"
    if results.is_dir():
        for p in results.rglob("*"):
            if p.suffix.lower() in ALLOWED:
                item = add(p)
                if item:
                    yield item

    # Metadata ONLY from raw/preprocessed directories; never include volumes.
    for folder, names in (
        ("nnunet_preprocessed", {"dataset.json", "dataset_fingerprint.json", "nnUNetPlans.json"}),
        ("nnunet_raw", {"dataset.json", "split_manifest.json"}),
    ):
        base = root / folder
        if base.is_dir():
            for p in base.rglob("*"):
                if p.name in names:
                    item = add(p)
                    if item:
                        yield item

    # SLURM stdout and stderr may be in repository root.
    for p in root.iterdir():
        if p.is_file() and (p.suffix in (".out", ".err", ".log") or
                            (p.name.startswith("slurm-") and p.suffix == ".txt")):
            item = add(p)
            if item:
                yield item

    # Optional scheduler logs subdirectory.
    logs = root / "logs"
    if logs.is_dir():
        for p in logs.rglob("*"):
            if p.suffix.lower() in (".out", ".err", ".log", ".txt"):
                item = add(p)
                if item:
                    yield item


def collect(root, output, max_file_mb, max_total_mb):
    output.parent.mkdir(parents=True, exist_ok=True)
    file_cap = int(max_file_mb * 1024 * 1024)
    total_cap = int(max_total_mb * 1024 * 1024)
    files = sorted(candidates(root), key=lambda p: (
        0 if any(k in str(p) for k in ("nnunet3d_holdout_metrics",
                                       "energy_nnunet", "energy_phase2",
                                       "phase3_diagnostics",
                                       "nnunet_results")) else 1,
        str(p),
    ))
    added = []
    skipped = []
    total = 0
    scheduler = {
        "squeue_my_jobs": safe_command(["squeue", "--me", "-o",
                                        "%.18i %.20j %.12T %.10M %.12l %R"]),
        "sacct_recent": safe_command(["sacct", "-S",
                                    (dt.date.today() - dt.timedelta(days=14)).isoformat(),
                                    "--format", "JobID,JobName,State,Elapsed,ExitCode,Timelimit",
                                    "--parsable2", "--noheader"]),
    }
    manifest = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "purpose": "Adrenal segmentation results archive; safe to transfer for review after inspection",
        "git_commit": safe_command(["git", "-C", str(root), "rev-parse", "HEAD"]).strip(),
        "git_branch": safe_command(["git", "-C", str(root), "branch", "--show-current"]).strip(),
        "important_result_status": status(root),
        "slurm_scheduler_status": scheduler,
        "limits": {"max_file_mb": max_file_mb, "max_total_uncompressed_mb": max_total_mb},
        "notes": [
            "Absence of an output may mean training/evaluation has not finished.",
            "Dataset images, segmentation masks, predictions and checkpoints are NOT included.",
            "nnU-Net full-volume and Phase 2 cropped-case metrics have different evaluation scopes.",
            "Log files can contain local cluster paths; inspect before sharing.",
        ],
    }

    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6, allowZip64=True) as archive:
        for path in files:
            try:
                size = path.stat().st_size
                rel = path.relative_to(root).as_posix()
                if size > file_cap:
                    skipped.append({"file": rel, "reason": "exceeds per-file limit"})
                    continue
                if total + size > total_cap:
                    skipped.append({"file": rel, "reason": "exceeds total-size limit"})
                    continue
                archive.write(path, arcname=rel)
                added.append(rel)
                total += size
            except (PermissionError, OSError) as exc:
                skipped.append({"file": str(path), "reason": type(exc).__name__})

        manifest["included_files"] = added
        manifest["skipped_files"] = skipped
        manifest["included_count"] = len(added)
        manifest["total_uncompressed_bytes"] = total
        archive.writestr("COLLECTION_MANIFEST.json",
                         json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        archive.writestr("SLURM_JOB_STATUS.txt",
                         "squeue --me:\n" + scheduler["squeue_my_jobs"] +
                         "\n\nsacct (last 14 days):\n" + scheduler["sacct_recent"] + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(),
                        help="repository root (default: current directory)")
    parser.add_argument("--output", type=Path, default=None,
                        help="destination ZIP; defaults to results_exports/")
    parser.add_argument("--max-file-mb", type=float, default=25)
    parser.add_argument("--max-total-mb", type=float, default=250)
    args = parser.parse_args()
    root = args.root.resolve()
    if not (root / "runs").is_dir():
        parser.error(f"Expected 'runs/' in repository root: {root}")
    if args.max_file_mb <= 0 or args.max_total_mb <= 0:
        parser.error("size limits must be positive")
    filename = "Adrenal_All_Results_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S") + ".zip"
    output = (args.output or root / "results_exports" / filename).resolve()
    try:
        if root not in output.parents:
            print("Note: writing archive outside project root", file=sys.stderr)
        manifest = collect(root, output, args.max_file_mb, args.max_total_mb)
    except Exception:
        if output.exists():
            output.unlink()
        raise
    print(f"\nArchive created: {output}")
    print(f"Included result files: {manifest['included_count']}")
    print(f"Skipped (size/permissions): {len(manifest['skipped_files'])}")
    print("Critical results:")
    for name, availability in manifest["important_result_status"].items():
        if "energy_" in name or "holdout_metrics" in name:
            print(f"  {name}: {availability}")
    print("\nUpload this ZIP file to ChatGPT when ready. No GPU allocation required.")


if __name__ == "__main__":
    main()

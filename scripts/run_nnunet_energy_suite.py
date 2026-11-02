#!/usr/bin/env python3
"""Benchmark nnU-Net 3d_fullres full-volume prediction within an existing GPU job.

Reruns the *same held-out images* with the *same fold* to a temporary output
directory, with AMD-SMI energy monitoring. Original nnU-Net predictions and
segmentation metrics are never touched. This measures complete nnU-Net
predict-command GPU-socket energy (I/O and CPU phases included), NOT just
neural-network forward kernels. Runs only as an explicitly enabled optional
post-evaluation stage.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

from scripts.run_phase2_energy_suite import gpu_time_left_s


def dataset_id_from_path(directory: Path) -> str:
    match = re.fullmatch(r"Dataset([0-9]{3})_.+", directory.name)
    if match is None:
        raise ValueError(f"Cannot identify nnU-Net dataset ID from {directory}")
    return str(int(match.group(1)))


def read_full_scan_metrics(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("Empty nnU-Net evaluation metrics")
    names = [row["case_id"] for row in rows]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate CT case IDs in nnU-Net metrics")
    summary = {}
    for key in ("dice_left", "dice_right", "hd95_left_mm",
                "hd95_right_mm", "nsd2mm_left", "nsd2mm_right"):
        values = [float(row[key]) for row in rows]
        values = [v for v in values if math.isfinite(v)]
        summary["mean_" + key] = statistics.mean(values) if values else None
    return tuple(sorted(names)), summary


def save(out_dir: Path, trials: list, status: str, metadata: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "nnunet_energy_trials.csv"
    if trials:
        with path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(trials[0]))
            writer.writeheader()
            writer.writerows(trials)
    def summary(key):
        values = [float(row[key]) for row in trials
                  if row.get(key) is not None and math.isfinite(float(row[key]))]
        return {
            "mean": statistics.mean(values) if values else None,
            "std": statistics.stdev(values) if len(values) > 1 else None,
        }
    report = {
        "status": status,
        "trials_completed": len(trials),
        **metadata,
        "gpu_joules_per_ct": summary("gpu_energy_j_per_ct"),
        "wall_seconds_per_ct": summary("seconds_per_ct"),
        "peak_sampled_vram_gib": summary("peak_sampled_vram_gib"),
        "measurement_scope": "Full nnUNetv2_predict subprocess on complete CT images; GPU socket "
                             "energy includes idle periods while preprocessing, resampling and writing",
        "limits": "Not directly comparable to Phase 2 label-derived z-cropped evaluation; "
                  "not a direct network-only forward-pass energy measurement",
    }
    (out_dir / "nnunet_energy_summary.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(f"nnU-Net energy benchmark: {status}; completed {len(trials)} repeats", flush=True)


def run(args):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset_id = dataset_id_from_path(args.dataset_dir)
    image_files = sorted((args.dataset_dir / "imagesTs").glob("*_0000.nii.gz"))
    names = tuple(sorted(p.name[:-12] for p in image_files))
    if not names:
        raise FileNotFoundError(f"No full-volume holdout images in {args.dataset_dir / 'imagesTs'}")
    metric_names, scores = read_full_scan_metrics(args.metrics)
    if names != metric_names:
        raise ValueError("Heldout dataset and evaluated metrics have different CT case IDs")
    metadata = {"dataset_id": dataset_id, "fold": args.fold, "ct_cases": len(names),
                "baseline_full_volume_metrics": scores}
    if args.repeats < 1 or args.min_remaining_minutes < 0:
        raise ValueError("Invalid repetitions or remaining-time threshold")

    wrapper = [sys.executable, "scripts/measure_gpu_energy.py", "--output-dir", str(args.out_dir)]
    if args.gpu_index is not None:
        wrapper += ["--gpu-index", str(args.gpu_index)]
    check = subprocess.run(wrapper + ["--probe"], check=False)
    if check.returncode:
        save(args.out_dir, [], "skipped: AMD-SMI unavailable, energy not readable or GPU identity unclear", metadata)
        return 0

    trials = []
    for number in range(1, args.repeats + 1):
        label = f"nnunet_fullvolume_r{number}"
        outfile = args.out_dir / (label + ".json")
        # Allow safe resume without recomputing completed experiments.
        if outfile.is_file():
            try:
                old = json.loads(outfile.read_text())
                if old.get("return_code") == 0 and old.get("energy_j") is not None and old.get("elapsed_s", 0) > 0:
                    trials.append(trial_row(number, old, len(names), scores))
                    save(args.out_dir, trials, "in_progress", metadata)
                    continue
            except (OSError, TypeError, KeyError, ValueError):
                pass

        time_left = gpu_time_left_s()
        if time_left is not None and time_left < args.min_remaining_minutes * 60:
            save(args.out_dir, trials, "partial: too little SLURM time remaining to start nnU-Net prediction", metadata)
            return 0

        # Temporary output avoids overwriting the original model predictions.
        # On completion or failure, these large NIfTI volumes are deleted.
        with tempfile.TemporaryDirectory(dir=str(args.out_dir),
                                         prefix=f"nnunet_predictions_r{number}_") as temp:
            output_dir = Path(temp)
            cmd = wrapper + ["--label", label, "--require-energy", "--",
                "nnUNetv2_predict", "-i", str(args.dataset_dir / "imagesTs"),
                "-o", str(output_dir), "-d", dataset_id,
                "-c", "3d_fullres", "-f", str(args.fold), "-device", "cuda"]
            print(f"Measuring full-volume nnU-Net inference repeat {number}/{args.repeats}",
                  flush=True)
            result = subprocess.run(cmd, check=False)
            if result.returncode:
                save(args.out_dir, trials,
                     f"partial: nnU-Net energy repeat {number} failed (exit {result.returncode})", metadata)
                return 0
            missing = [name for name in names if not (output_dir / (name + ".nii.gz")).is_file()]
            if missing:
                save(args.out_dir, trials,
                     f"partial: nnU-Net inference repeat {number} missing {len(missing)} outputs", metadata)
                return 0

        energy = json.loads(outfile.read_text())
        trials.append(trial_row(number, energy, len(names), scores))
        save(args.out_dir, trials, "in_progress", metadata)

    save(args.out_dir, trials, "complete", metadata)
    return 0


def trial_row(number, energy, n, scores):
    j, sec = energy.get("energy_j"), energy.get("elapsed_s")
    if j is None or not math.isfinite(float(j)) or float(j) <= 0:
        raise ValueError("Unusable nnU-Net measured GPU energy")
    if sec is None or float(sec) <= 0:
        raise ValueError("Unusable nnU-Net measured wall time")
    return {
        "model": "nnunet3d_fullres", "repeat": number, "full_ct_volumes": n,
        "gpu_energy_j_total": float(j),
        "gpu_energy_j_per_ct": float(j) / n,
        "seconds_total": float(sec),
        "seconds_per_ct": float(sec) / n,
        "mean_gpu_power_w": energy.get("mean_sample_power_w"),
        "peak_sampled_vram_gib": energy.get("peak_sampled_gpu_vram_gib"),
        "energy_measurement_source": energy.get("energy_source"),
        "device_smi_index": energy.get("device_smi_index"),
        "gpu_device_mapping": energy.get("gpu_mapping"),
        **scores,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/energy_nnunet"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--min-remaining-minutes", type=int, default=120)
    parser.add_argument("--gpu-index", type=int, default=None)
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:
        save(args.out_dir, [], f"skipped or failed: {type(exc).__name__}: {exc}",
             {"dataset_dir": str(args.dataset_dir), "fold": args.fold})
        return 0  # Optional study never invalidates already-written nnU-Net metrics.


if __name__ == "__main__":
    sys.exit(main())

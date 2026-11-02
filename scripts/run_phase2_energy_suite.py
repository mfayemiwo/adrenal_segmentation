#!/usr/bin/env python3
"""Matched Phase 2 SNN / analog energy on the *same* evaluation workload.

Run after nnU-Net training, within its existing GPU allocation, via the
explicit optional hook in evaluate_nnunet_adrenal.py. No new sbatch needed.
Whole-evaluation subprocess energy is NOT isolated neural-network inference
energy and is NOT directly comparable to nnU-Net full-scan energy.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path


def gpu_time_left_s():
    """SLURM remaining time, or None if not parseable."""
    import os
    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        return None
    try:
        output = subprocess.check_output(
            ["squeue", "--noheader", "--jobs", job, "--format", "%L"],
            stderr=subprocess.DEVNULL, text=True, timeout=8).strip().splitlines()
        if not output:
            return None
        value = output[0].strip()
        if value in ("UNLIMITED", "NOT_SET", "INVALID", "N/A"):
            return None
        days, sep, hms = value.rpartition("-")
        seconds = 0
        if sep:
            seconds = 86400 * int(days)
        else:
            hms = value
        parts = [int(x) for x in hms.split(":")]
        if len(parts) == 3:
            seconds += parts[0] * 3600 + parts[1] * 60 + parts[2]
        elif len(parts) == 2:
            seconds += parts[0] * 60 + parts[1]
        else:
            return None
        return seconds
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def record_for_trial(method, repeat, energy, per_case):
    with per_case.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Empty per-case evaluation")
    case_ids = tuple(sorted(row["case_id"] for row in rows))
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Duplicate patient IDs")
    n = len(rows)
    j, sec = energy.get("energy_j"), energy.get("elapsed_s")
    if j is None or not math.isfinite(float(j)) or float(j) <= 0:
        raise ValueError("GPU energy not measured")
    if sec is None or float(sec) <= 0:
        raise ValueError("Nonpositive elapsed time")
    def average(key):
        values = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
        return statistics.mean(values) if values else None
    return {
        "method": method, "repeat": repeat, "patients": n,
        "energy_j": float(j), "gpu_energy_j_per_patient": float(j) / n,
        "seconds_per_patient": float(sec) / n,
        "mean_dice_left": average("dice_left_postprocessed"),
        "mean_dice_right": average("dice_right_postprocessed"),
        "mean_hd95_left_mm": average("hd95_left_mm"),
        "mean_hd95_right_mm": average("hd95_right_mm"),
        "mean_nsd_left": average("nsd2mm_left"),
        "mean_nsd_right": average("nsd2mm_right"),
        "energy_source": energy.get("energy_source"),
        "gpu_mapping": energy.get("gpu_mapping"),
        "gpu_index": energy.get("device_smi_index"),
        "case_ids": case_ids,
    }


def save_reports(out, records, status):
    out.mkdir(parents=True, exist_ok=True)
    tabular = [{k: v for k, v in rec.items() if k != "case_ids"} for rec in records]
    if tabular:
        with (out / "phase2_energy_trials.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(tabular[0]))
            writer.writeheader()
            writer.writerows(tabular)
    pairs = []
    by_key = {(rec["repeat"], rec["method"]): rec for rec in records}
    for repeat in range(1, 1000):
        snn, analog = by_key.get((repeat, "snn")), by_key.get((repeat, "analog"))
        if not snn or not analog:
            if repeat > max([0] + [r["repeat"] for r in records]):
                break
            continue
        if snn["case_ids"] != analog["case_ids"]:
            raise ValueError(f"Different patient IDs in paired repeat {repeat}")
        ratio = snn["energy_j"] / analog["energy_j"]
        pairs.append({
            "repeat": repeat, "patients": snn["patients"],
            "snn_gpu_j_per_patient": snn["gpu_energy_j_per_patient"],
            "analog_gpu_j_per_patient": analog["gpu_energy_j_per_patient"],
            "snn_to_analog_energy_ratio": ratio,
            "snn_energy_saving_percent": 100 * (1 - ratio),
            "snn_sec_per_patient": snn["seconds_per_patient"],
            "analog_sec_per_patient": analog["seconds_per_patient"],
        })
    if pairs:
        with (out / "phase2_energy_paired.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(pairs[0]))
            writer.writeheader()
            writer.writerows(pairs)
    report = {
        "status": status, "trials": len(records), "paired_repeats": len(pairs),
        "mean_energy_ratio_snn_to_analog": statistics.mean(
            x["snn_to_analog_energy_ratio"] for x in pairs) if pairs else None,
        "mean_snn_energy_saving_percent": statistics.mean(
            x["snn_energy_saving_percent"] for x in pairs) if pairs else None,
        "energy_scope": "AMD GPU socket across full Phase 2 evaluation subprocess, idle included",
        "dataset_scope": "Same historically mask-derived z-cropped validation cases, NOT unrestricted abdomen",
        "caveat": "CPU preprocessing and metric computations included in wall time and GPU idle energy.",
    }
    (out / "phase2_energy_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Energy suite status:", status, "paired repeats:", len(pairs), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=Path("../data/amos22"))
    p.add_argument("--cache-dir", type=Path, default=Path("../cache"))
    p.add_argument("--checkpoint-root", type=Path, default=Path("runs"))
    p.add_argument("--output-dir", type=Path, default=Path("runs/energy_phase2"))
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--min-remaining-minutes", type=int, default=180,
                   help="Skip starting a trial below this SLURM time-left threshold")
    p.add_argument("--gpu-index", type=int, default=None)
    p.add_argument("--max-val-cases", type=int, default=0)
    a = p.parse_args()
    if a.repeats <= 0 or a.min_remaining_minutes < 0:
        p.error("invalid repeat count or time threshold")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = {
        "snn": a.checkpoint_root / "phase2_anatomy_snn_s1" / "best_model.pt",
        "analog": a.checkpoint_root / "phase2_anatomy_analog_s1" / "best_model.pt",
    }
    absent = [str(f) for f in checkpoints.values() if not f.is_file()]
    if absent:
        save_reports(a.output_dir, [], "skipped: missing checkpoints: " + ", ".join(absent))
        return 0
    wrapper = [sys.executable, "scripts/measure_gpu_energy.py", "--output-dir",
               str(a.output_dir)]
    if a.gpu_index is not None:
        wrapper += ["--gpu-index", str(a.gpu_index)]
    probe = subprocess.run(wrapper + ["--probe"], check=False)
    if probe.returncode:
        save_reports(a.output_dir, [], "skipped: no working AMD-SMI hardware telemetry or safe GPU mapping")
        return 0

    results = []
    try:
        for repeat in range(1, a.repeats + 1):
            order = ("snn", "analog") if repeat % 2 else ("analog", "snn")
            for variant in order:
                label = f"phase2_{variant}_r{repeat}"
                energy_path = a.output_dir / (label + ".json")
                csv_path = a.output_dir / label / "per_case.csv"
                if energy_path.is_file() and csv_path.is_file():
                    try:
                        energy = json.loads(energy_path.read_text())
                        if energy.get("return_code") == 0:
                            results.append(record_for_trial(variant, repeat, energy, csv_path))
                            save_reports(a.output_dir, results, "in_progress")
                            continue
                    except (KeyError, ValueError, OSError):
                        pass
                remaining = gpu_time_left_s()
                if remaining is not None and remaining < a.min_remaining_minutes * 60:
                    save_reports(a.output_dir, results, "partial: insufficient SLURM time remaining")
                    return 0
                cmd = wrapper + ["--label", label, "--require-energy", "--",
                     sys.executable, "-u", "scripts/evaluate_segmenter.py",
                     "--checkpoint", str(checkpoints[variant]),
                     "--data-root", str(a.data_root),
                     "--cache-dir", str(a.cache_dir),
                     "--output-dir", str(a.output_dir / label),
                     "--keep-largest", "2",
                     "--min-component-voxels", "30",
                     "--modality", "ct",
                     "--device", "cuda"]
                if a.max_val_cases:
                    cmd += ["--max-val-cases", str(a.max_val_cases)]
                print("Running:", label, flush=True)
                proc = subprocess.run(cmd, check=False)
                if proc.returncode:
                    save_reports(a.output_dir, results, f"partial: trial {label} failed (exit {proc.returncode})")
                    return 0
                energy = json.loads(energy_path.read_text())
                results.append(record_for_trial(variant, repeat, energy, csv_path))
                save_reports(a.output_dir, results, "in_progress")
    except Exception as exc:
        save_reports(a.output_dir, results, f"partial: {type(exc).__name__}: {exc}")
        return 0
    save_reports(a.output_dir, results, "complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())

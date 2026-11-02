#!/usr/bin/env python3
"""Measure physical AMD GPU energy while running one subprocess.

Run inside an already allocated ROCm GPU job. Compares *full commands*,
including program loading, preprocessing and inference. A second controlled
model-only benchmark is needed to isolate the refiner kernel costs.

Example:
 python scripts/measure_gpu_energy.py --label phase2_snn --gpu-index 0 \
   --output-dir runs/energy -- python -u scripts/evaluate_segmenter.py ...
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import subprocess
import sys
import threading
import time
from pathlib import Path


def numeric(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def get_energy_joules(amdsmi, handle):
    """AMD SMI reports accumulator * resolution in microjoules."""
    data = amdsmi.amdsmi_get_energy_count(handle)
    counter = numeric(data.get("energy_accumulator", data.get("power")))
    resolution = numeric(data.get("counter_resolution"))
    if counter is None or resolution is None or resolution <= 0:
        return None
    return counter * resolution / 1_000_000.0


def get_power_watts(amdsmi, handle):
    data = amdsmi.amdsmi_get_power_info(handle)
    for field in ("current_socket_power", "socket_power", "average_socket_power"):
        value = numeric(data.get(field))
        if value is not None and 0 < value < 2000:
            return value
    return None


def sample(sm, handle):
    now = time.perf_counter()
    try:
        power = get_power_watts(sm, handle)
    except Exception:
        power = None
    try:
        energy = get_energy_joules(sm, handle)
    except Exception:
        energy = None
    return {"t_seconds": now, "power_w": power, "energy_j_counter": energy}


def integrate_power(samples):
    total = 0.0
    segments = 0
    for left, right in zip(samples, samples[1:]):
        w1, w2 = left["power_w"], right["power_w"]
        dt = right["t_seconds"] - left["t_seconds"]
        if w1 is not None and w2 is not None and dt > 0:
            total += 0.5 * (w1 + w2) * dt
            segments += 1
    return total if segments else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="Unique simple filename label")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/energy"))
    parser.add_argument("--gpu-index", type=int, default=None,
                        help="PHYSICAL AMD-SMI index (not necessarily torch cuda:0)")
    parser.add_argument("--interval", type=float, default=0.5, help="sample interval in seconds")
    parser.add_argument("--idle-seconds", type=float, default=3.0)
    parser.add_argument("--require-energy", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER,
                        help="command to run after a '--' separator")
    args = parser.parse_args()
    if not args.label.replace("_", "").replace("-", "").isalnum():
        parser.error("label must contain only letters, digits, - and _")
    if args.interval < 0.1 or args.idle_seconds < 0:
        parser.error("interval must be >=0.1 and idle-seconds must be >=0")
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        parser.error("provide a command after --")

    try:
        import amdsmi
    except ImportError as exc:
        raise SystemExit(
            "AMD-SMI Python binding unavailable. Load the ROCm module or activate "
            "an environment with matching amdsmi bindings; no energy claims possible."
        ) from exc
    amdsmi.amdsmi_init()
    try:
        handles = amdsmi.amdsmi_get_processor_handles()
        if args.gpu_index is None and len(handles) != 1:
            raise SystemExit(
                f"AMD-SMI sees {len(handles)} devices: supply --gpu-index explicitly "
                "after mapping the assigned HIP GPU to the correct AMD-SMI device."
            )
        idx = args.gpu_index if args.gpu_index is not None else 0
        if idx < 0 or idx >= len(handles):
            raise SystemExit(f"gpu-index {idx} outside available 0..{len(handles)-1}")
        handle = handles[idx]
        args.output_dir.mkdir(parents=True, exist_ok=True)
        idle = []
        end_idle = time.monotonic() + args.idle_seconds
        while time.monotonic() < end_idle:
            idle.append(sample(amdsmi, handle))
            time.sleep(min(args.interval, max(0.0, end_idle - time.monotonic())))
        idle_power = [x["power_w"] for x in idle if x["power_w"] is not None]
        idle_w = sum(idle_power) / len(idle_power) if idle_power else None

        readings = []
        stop = threading.Event()
        def poll():
            while not stop.is_set():
                readings.append(sample(amdsmi, handle))
                stop.wait(args.interval)

        start = sample(amdsmi, handle)
        readings.append(start)
        thread = threading.Thread(target=poll, daemon=True)
        thread.start()
        rc = 1
        t0 = time.perf_counter()
        try:
            rc = subprocess.call(command)
        finally:
            elapsed = time.perf_counter() - t0
            stop.set()
            thread.join(timeout=5)
            end = sample(amdsmi, handle)
            readings.append(end)

        first_j, last_j = start["energy_j_counter"], end["energy_j_counter"]
        direct_j = (last_j - first_j if first_j is not None and last_j is not None
                    and last_j >= first_j else None)
        estimated_j = integrate_power(readings)
        actual_j = direct_j if direct_j is not None else estimated_j
        source = ("amd_smi_energy_counter" if direct_j is not None
                  else "trapezoid_power_samples" if estimated_j is not None
                  else "unavailable")
        watts = [x["power_w"] for x in readings if x["power_w"] is not None]
        mean_w = sum(watts) / len(watts) if watts else None
        net_estimate = (max(actual_j - idle_w * elapsed, 0.0)
                        if actual_j is not None and idle_w is not None else None)
        report = {
            "label": args.label, "device_smi_index": idx, "command": command,
            "return_code": rc, "elapsed_s": elapsed,
            "energy_j": actual_j, "energy_source": source,
            "energy_counter_j": direct_j, "power_integrated_j": estimated_j,
            "mean_sample_power_w": mean_w, "idle_mean_power_w": idle_w,
            "net_energy_after_idle_j_estimate": net_estimate,
            "sample_interval_s": args.interval, "sample_count": len(readings),
            "scope": "GPU socket telemetry over complete child process; includes idle component; CPU/node energy excluded.",
        }
        target = args.output_dir / args.label
        (target.with_suffix(".json")).write_text(json.dumps(report, indent=2) + "\n")
        with target.with_suffix(".csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["t_seconds", "power_w", "energy_j_counter"])
            writer.writeheader()
            writer.writerows(readings)
        print(json.dumps(report, indent=2), flush=True)
        if args.require_energy and actual_j is None:
            raise SystemExit("No usable AMD GPU energy telemetry; do not infer savings from spike rates.")
        if rc:
            raise SystemExit(rc)
    finally:
        amdsmi.amdsmi_shut_down()


if __name__ == "__main__":
    main()

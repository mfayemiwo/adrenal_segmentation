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
import ctypes
import ctypes.util
import os
import re
import csv
import json
import math
import subprocess
import sys
import threading
import time
from pathlib import Path


def normalize_bdf(value):
    """Canonicalise PCI bus addresses for exact HIP <-> AMD-SMI matching."""
    value = str(value).strip().lower()
    if re.fullmatch(r"[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", value):
        value = "0000:" + value
    match = re.fullmatch(r"([0-9a-f]{1,4}):([0-9a-f]{2}):([0-9a-f]{2})\.([0-7])", value)
    if not match:
        return None
    domain, bus, dev, function = match.groups()
    return f"{int(domain,16):04x}:{bus}:{dev}.{function}"


def hip_device_zero_bdf():
    """Ask HIP runtime for the physical BDF of logical GPU 0; no GPU work."""
    guesses = [ctypes.util.find_library("amdhip64"), "libamdhip64.so"]
    rocm = os.environ.get("ROCM_PATH", "/opt/rocm")
    guesses.append(os.path.join(rocm, "lib", "libamdhip64.so"))
    for library in filter(None, guesses):
        try:
            hip = ctypes.CDLL(library)
            get_bdf = hip.hipDeviceGetPCIBusId
            get_bdf.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
            get_bdf.restype = ctypes.c_int
            buff = ctypes.create_string_buffer(64)
            if get_bdf(buff, len(buff), 0) == 0:
                return normalize_bdf(buff.value.decode())
        except (OSError, AttributeError, ValueError):
            continue
    return None


def select_gpu(sm, handles, gpu_index):
    """Never assume AMD-SMI index zero is the assigned HIP device."""
    if gpu_index is not None:
        if not 0 <= gpu_index < len(handles):
            raise ValueError(f"AMD-SMI index {gpu_index} is not in 0..{len(handles)-1}")
        return gpu_index, "explicit_user_index"
    if len(handles) == 1:
        return 0, "only_one_amdsmi_device_visible"
    hip_bdf = hip_device_zero_bdf()
    if hip_bdf is None:
        raise ValueError(f"AMD-SMI sees {len(handles)} devices but HIP bus ID could not be read. "
                         "Set ENERGY_GPU_INDEX to the *verified* physical AMD-SMI index.")
    matches = []
    for index, handle in enumerate(handles):
        try:
            bdf = normalize_bdf(sm.amdsmi_get_gpu_device_bdf(handle))
        except Exception:
            continue
        if bdf == hip_bdf:
            matches.append(index)
    if len(matches) != 1:
        raise ValueError(f"HIP GPU 0 BDF={hip_bdf}; AMD-SMI match count={len(matches)}. "
                         "Cannot safely attribute device energy. Set ENERGY_GPU_INDEX "
                         "only after verifying the physical mapping.")
    return matches[0], "hip_pci_bdf_match"


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


def get_vram_bytes(sm, handle):
    data = sm.amdsmi_get_gpu_vram_usage(handle)
    used = numeric(data.get("vram_used"))
    return int(used) if used is not None and used >= 0 else None


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
    try:
        vram = get_vram_bytes(sm, handle)
    except Exception:
        vram = None
    return {"t_seconds": now, "power_w": power, "energy_j_counter": energy,
            "vram_used_bytes": vram}


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



def load_amdsmi():
    """Load AMD-SMI from .venv or ROCm's matching Python installation.

    ROCm cluster modules commonly ship the amdsmi wrapper under
    $ROCM_PATH/share/amd_smi rather than the user's Python virtualenv.
    Do not install or mix a mismatched PyPI wrapper/ROCm native library.
    """
    import importlib
    try:
        return importlib.import_module("amdsmi")
    except ModuleNotFoundError as exc:
        if exc.name != "amdsmi":
            raise

    roots = [
        os.environ.get("ROCM_PATH"),
        os.environ.get("ROCM_HOME"),
        os.environ.get("ROCM_ROOT"),
        "/opt/rocm",
    ]
    root_paths = [Path(p).expanduser() for p in roots if p]
    candidate_paths = []
    for root in root_paths:
        candidate_paths.extend((
            root / "share" / "amd_smi",
            root / "libexec" / "amd_smi",
            root / "share" / "amd_smi" / "py-interface",
        ))
    override = os.environ.get("AMDSMI_PYTHONPATH")
    if override:
        candidate_paths.insert(0, Path(override).expanduser())

    attempted = []
    for directory in candidate_paths:
        directory = directory.resolve()
        if str(directory) in attempted:
            continue
        attempted.append(str(directory))
        if not ((directory / "amdsmi" / "__init__.py").is_file() or
                (directory / "amdsmi.py").is_file()):
            continue
        sys.path.insert(0, str(directory))
        try:
            imported = importlib.import_module("amdsmi")
            print(f"AMD-SMI Python bindings loaded from {imported.__file__}", flush=True)
            return imported
        except ModuleNotFoundError as exc:
            if exc.name != "amdsmi":
                raise
            sys.path.remove(str(directory))

    raise ModuleNotFoundError(
        "AMD-SMI Python binding unavailable in .venv or the matching ROCm "
        "installation. Checked: " + ", ".join(attempted) +
        ". Check ROCM_PATH and the ROCm module or ask cluster support for "
        "the matching amdsmi Python wrapper. Do not install an arbitrary "
        "newer ROCm binding into the existing torch .venv.",
        name="amdsmi",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="probe", help="Unique simple filename label")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/energy"))
    parser.add_argument("--gpu-index", type=int, default=None,
                        help="PHYSICAL AMD-SMI index (not necessarily torch cuda:0)")
    parser.add_argument("--interval", type=float, default=0.5, help="sample interval in seconds")
    parser.add_argument("--idle-seconds", type=float, default=3.0)
    parser.add_argument("--require-energy", action="store_true")
    parser.add_argument("--probe", action="store_true", help="Check usable telemetry without launching workload")
    parser.add_argument("command", nargs=argparse.REMAINDER,
                        help="command to run after a '--' separator")
    args = parser.parse_args()
    if not args.label.replace("_", "").replace("-", "").isalnum():
        parser.error("label must contain only letters, digits, - and _")
    if args.interval < 0.1 or args.idle_seconds < 0:
        parser.error("interval must be >=0.1 and idle-seconds must be >=0")
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command and not args.probe:
        parser.error("provide a command after --")

    try:
        amdsmi = load_amdsmi()
    except (ImportError, OSError) as exc:
        raise SystemExit(
            f"AMD-SMI Python binding unavailable or could not load its native "
            f"library: {exc}; no energy claims possible."
        ) from exc
    amdsmi.amdsmi_init()
    try:
        handles = amdsmi.amdsmi_get_processor_handles()
        try:
            idx, mapping = select_gpu(amdsmi, handles, args.gpu_index)
        except ValueError as exc:
            raise SystemExit(f"GPU energy mapping unavailable: {exc}") from exc
        handle = handles[idx]
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if args.probe:
            measured = sample(amdsmi, handle)
            available = measured["power_w"] is not None or measured["energy_j_counter"] is not None
            report = {"telemetry_available": available, "physical_amdsmi_index": idx,
                      "mapping": mapping, "hip_gpu_0_bdf": hip_device_zero_bdf(),
                      "smi_bdf": str(amdsmi.amdsmi_get_gpu_device_bdf(handle)),
                      "sample": measured}
            (args.output_dir / "telemetry_probe.json").write_text(
                json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2), flush=True)
            if not available:
                raise SystemExit("No power/energy counter exposed for this GPU")
            return
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
        used = [p["vram_used_bytes"] for p in readings if p.get("vram_used_bytes") is not None]
        peak_vram = max(used) if used else None
        net_estimate = (max(actual_j - idle_w * elapsed, 0.0)
                        if actual_j is not None and idle_w is not None else None)
        report = {
            "label": args.label, "device_smi_index": idx, "gpu_mapping": mapping, "command": command,
            "return_code": rc, "elapsed_s": elapsed,
            "energy_j": actual_j, "energy_source": source,
            "energy_counter_j": direct_j, "power_integrated_j": estimated_j,
            "mean_sample_power_w": mean_w, "idle_mean_power_w": idle_w,
            "net_energy_after_idle_j_estimate": net_estimate,
            "sample_interval_s": args.interval, "sample_count": len(readings),
            "peak_sampled_gpu_vram_bytes": peak_vram,
            "peak_sampled_gpu_vram_gib": peak_vram / (1024 ** 3) if peak_vram is not None else None,
            "scope": "GPU socket telemetry over complete child process; includes idle component; CPU/node energy excluded.",
        }
        target = args.output_dir / args.label
        (target.with_suffix(".json")).write_text(json.dumps(report, indent=2) + "\n")
        with target.with_suffix(".csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["t_seconds", "power_w", "energy_j_counter", "vram_used_bytes"])
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

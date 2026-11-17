#!/usr/bin/env python3
"""Safely resume nnU-Net v2 3d_fullres preprocessing for Dataset702.

Why this exists: nnU-Net's DefaultPreprocessor.run() deletes an existing
preprocessed configuration folder on invocation. For 1,253 3D CTs, rerunning
that standard CLI after a 24-hour SLURM timeout could discard hours of prior
work. This wrapper calls THE INSTALLED nnU-Net preprocessor's run_case_save()
unchanged for each missing case, stages all output files under .staging/,
commits the .pkl metadata file last as a completion marker, and resumes safely.

The data transform, image reader, resampling and nnU-Net plans are not changed.
A separate nnUNetv2_plan_and_preprocess --no_pp invocation must have written
the default plan. Foreground-sampling indexing and GT file copies happen
AFTER all per-case outputs are ready, via the SLURM caller.

No GPU, no test data, no hyperparameter tuning, no deletion of completed
preprocessed cases. All outputs remain under /mnt/userapps/isrcdl_apps/mfayemiwo/.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import shutil
import sys
import time


DATASET = "Dataset702_AMOSTotalSegAdrenalCT"
EXPECTED = 1253


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for block in iter(lambda: fp.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json_once(path: Path, data: dict):
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != data:
            raise ValueError(f"Previous preprocessing configuration differs: {path}")
        return
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def required_case_files(directory: Path, case: str):
    """Return one full nnU-Net case output set or None.

    Supports both newer Blosc2 (data, seg, pkl) and older NumPy (npz, pkl).
    The .pkl file is the commit marker, written/moved LAST.
    """
    pkl = directory / (case + ".pkl")
    if not pkl.is_file() or pkl.stat().st_size == 0:
        return None
    b2nd = [
        directory / (case + ".b2nd"),
        directory / (case + "_seg.b2nd"),
        pkl
    ]
    npz = [directory / (case + ".npz"), pkl]
    for files in (b2nd, npz):
        if all(p.is_file() and p.stat().st_size > 0 for p in files):
            return files
    return None


def validate_case_files(directory: Path, case: str, *, read_pickle: bool = True):
    files = required_case_files(directory, case)
    if files is None:
        return False
    if read_pickle:
        with (directory / (case + ".pkl")).open("rb") as fp:
            record = pickle.load(fp)
        if not isinstance(record, dict):
            raise ValueError(f"Invalid nnU-Net case property metadata: {case}")
    return True


def commit_staged_case(staging: Path, destination: Path, case: str):
    """Write all nnU-Net case files into output, properties LAST.

    If SLURM kills the job between these renames, the absence of a valid .pkl
    marks the case incomplete and the entire case is safely regenerated.
    Stage and destination MUST live on the same filesystem for atomic replace.
    """
    if not validate_case_files(staging, case):
        raise ValueError(f"Preprocessor did not write a complete staged case: {case}")
    expected = required_case_files(staging, case)
    if expected is None:
        raise AssertionError("Preprocessor completion changed unexpectedly")
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / (case + ".pkl")
    if marker.exists():
        marker.unlink()
    others = sorted((p for p in expected if p.suffix != ".pkl"),
                    key=lambda x: x.name)
    for source in others:
        os.replace(source, destination / source.name)
    # This last single-file rename is the completion/commit boundary.
    os.replace(staging / (case + ".pkl"), marker)
    if not validate_case_files(destination, case):
        raise IOError(f"Committed output invalid: {case}")


def worker(case: str, data: dict, output_dir: str,
           plans_file: str, dataset_file: str, staging_root: str):
    """Executed in a spawned CPU process; one CT at a time."""
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    out = Path(output_dir)
    staging = Path(staging_root) / case
    if staging.exists():
        # Only ever remove this case's *incomplete* temporary outputs.
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=False)
    try:
        plans = PlansManager(json.loads(Path(plans_file).read_text(encoding="utf-8")))
        configuration = plans.get_configuration("3d_fullres")
        preprocessor = configuration.preprocessor_class(verbose=False)
        preprocessor.run_case_save(
            str(staging / case), data["images"], data["label"],
            plans, configuration, json.loads(
                Path(dataset_file).read_text(encoding="utf-8")
            )
        )
        commit_staged_case(staging, out, case)
        return case
    finally:
        # Leaves committed result intact; removes unfinished staging only.
        if staging.exists():
            shutil.rmtree(staging)


def progress(existing: int, total: int, completed_now: int, started: float):
    elapsed = time.monotonic() - started
    print(
        f"[resume-preprocess] complete={existing + completed_now}/{total}; "
        f"new_this_job={completed_now}; elapsed_seconds={elapsed:.0f}",
        flush=True,
    )


def run(args):
    raw = args.nnunet_raw.resolve() / DATASET
    pre = args.nnunet_preprocessed.resolve() / DATASET
    plan_path = args.frozen_plan.resolve()
    if args.workers < 1 or args.workers > 16:
        raise ValueError("workers must be 1..16; monitor RAM on 3D full-volume CT")
    if args.batch_factor < 1 or args.batch_factor > 8:
        raise ValueError("batch-factor must be 1..8")

    frozen = json.loads(plan_path.read_text(encoding="utf-8"))
    expected = sorted(set(frozen["fold0_train_ids"]) |
                      set(frozen["fold0_val_ids"]))
    if (frozen.get("dataset_id") != 702 or
        frozen.get("status") != "FROZEN_BEFORE_EXPANDED_CNN_TRAINING" or
        len(expected) != EXPECTED or
        len(frozen["fold0_train_ids"]) != 1095 or
        len(frozen["fold0_val_ids"]) != 158 or
        set(frozen["fold0_train_ids"]) & set(frozen["fold0_val_ids"])):
        raise ValueError("Dataset702 frozen 1,095/158 development split changed")
    raw_geom = raw / "geometry_manifest.json"
    geom = json.loads(raw_geom.read_text(encoding="utf-8"))
    if (geom.get("status") != "COMPLETE" or
        geom.get("original_split_plan_sha256") != sha256(plan_path)):
        raise ValueError("Raw Dataset702 is not prepared from the frozen split")

    plans = pre / "nnUNetPlans.json"
    dataset = pre / "dataset.json"
    fingerprint = pre / "dataset_fingerprint.json"
    if not all(p.is_file() for p in (plans, dataset, fingerprint)):
        raise FileNotFoundError(
            "First finish nnUNetv2_plan_and_preprocess -d 702 "
            "--verify_dataset_integrity --no_pp -c 3d_fullres"
        )
    try:
        import nnunetv2
        from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
        from nnunetv2.utilities.utils import get_filenames_of_train_images_and_targets
        package_version = importlib.metadata.version("nnunetv2")
    except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
        raise RuntimeError("Installed nnunetv2 is required") from exc
    plan_manager = PlansManager(json.loads(plans.read_text(encoding="utf-8")))
    config = plan_manager.get_configuration("3d_fullres")
    target = pre / config.data_identifier
    if target != pre / "nnUNetPlans_3d_fullres":
        raise ValueError(
            f"Expected default data identifier nnUNetPlans_3d_fullres; got {config.data_identifier}")
    ground_truth = raw / "labelsTr"
    info = json.loads(dataset.read_text(encoding="utf-8"))
    mapping = get_filenames_of_train_images_and_targets(str(raw), info)
    if set(mapping) != set(expected):
        raise ValueError("nnU-Net source identifiers differ from frozen Dataset702 manifest")
    for cid in expected:
        if not (ground_truth / (cid + ".nii.gz")).is_file():
            raise FileNotFoundError(f"Missing frozen case label: {cid}")

    target.mkdir(parents=True, exist_ok=True)
    run_meta = {
        "dataset": DATASET,
        "fullres_data_identifier": config.data_identifier,
        "frozen_plan_sha256": sha256(plan_path),
        "raw_geometry_manifest_sha256": sha256(raw_geom),
        "nnUNetPlans_sha256": sha256(plans),
        "dataset_json_sha256": sha256(dataset),
        "dataset_fingerprint_sha256": sha256(fingerprint),
        "installed_nnunetv2_version": package_version,
        "total_development_cases": EXPECTED,
        "method": "unchanged installed nnU-Net run_case_save, atomic per-case staging and resume",
        "source_data_quality": "automated CT/mask checks only; not human validated",
        "expected_output": "nnUNetPlans_3d_fullres per-case data, seg and pkl",
    }
    marker_path = pre / "dataset702_resume_method.json"
    with (pre / ".dataset702_resume.lock").open("a+") as lock_fp:
        try:
            fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "Another Dataset702 preprocessing job is running; do not overlap jobs"
            ) from exc
        if not marker_path.exists() and any(
            target.glob("*.npz")
        ) or (not marker_path.exists() and any(target.glob("*.b2nd"))):
            raise RuntimeError(
                "Unmanaged pre-existing preprocessed outputs found. "
                "Do not mix standard nnU-Net runs and checkpointed runs.")
        write_json_once(marker_path, run_meta)

        existing = []
        absent = []
        for cid in expected:
            if validate_case_files(target, cid):
                existing.append(cid)
            else:
                absent.append(cid)
        print(json.dumps({
            "status": "INITIAL_INVENTORY", "total_cases": EXPECTED,
            "complete_existing": len(existing), "remaining": len(absent),
            "workers": args.workers, "target": str(target),
            "stop_at_epoch": args.stop_at_epoch or None
        }, indent=2), flush=True)
        if not absent:
            print("SUCCESS: all 1,253 preprocessed cases are present and complete.", flush=True)
            return 0

        # Stage only a small finite batch at a time. Deadline is checked
        # BETWEEN batches, allowing natural SLURM-safe, no-restart resumes.
        batch_size = args.workers * args.batch_factor
        started = time.monotonic()
        completed = 0
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=context,
            max_tasks_per_child=8
        ) as pool:
            for offset in range(0, len(absent), batch_size):
                if args.stop_at_epoch and time.time() >= args.stop_at_epoch:
                    print(
                        "CHECKPOINT: CPU allocation deadline approaching; "
                        f"{len(absent)-completed} cases remain. "
                        "Resubmit SAME script; completed cases are retained.",
                        flush=True,
                    )
                    return 3
                batch = absent[offset:offset + batch_size]
                futures = {
                    pool.submit(
                        worker, case, mapping[case], str(target), str(plans),
                        str(dataset), str(target / ".staging")
                    ): case for case in batch
                }
                for future in as_completed(futures):
                    cid = future.result()  # propagate errors, never silent skip
                    completed += 1
                    if completed % 20 == 0 or completed + len(existing) == EXPECTED:
                        progress(len(existing), EXPECTED, completed, started)
        leftover = [cid for cid in expected if not validate_case_files(target, cid)]
        if leftover:
            raise RuntimeError(
                f"Preprocessing reported done but {len(leftover)} cases incomplete: "
                f"{leftover[:5]}"
            )
        print(f"SUCCESS: all {EXPECTED} full-volume CT cases preprocessed; "
              "original nnU-Net method and frozen plans preserved.", flush=True)
        return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nnunet-raw", type=Path, default=Path("nnunet_raw"))
    p.add_argument("--nnunet-preprocessed", type=Path,
                   default=Path("nnunet_preprocessed"))
    p.add_argument("--frozen-plan", type=Path,
                   default=Path("runs/expanded_dataset702/plan.json"))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--batch-factor", type=int, default=2)
    p.add_argument("--stop-at-epoch", type=int, default=0,
                   help="Epoch seconds. Gracefully checkpoint between batches before SLURM walltime.")
    args = p.parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()

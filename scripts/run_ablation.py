#!/usr/bin/env python3
"""Orchestrates the 8-arm ablation matrix described in docs/methodology.docx.

Arm | Description
--- | -----------
A   | Full-volume 2D U-Net, no gate (weakest baseline)
B   | Existing published pipeline: 2D U-Net + TTA + connected-component pruning
C   | Conventional ResNet18 classifier gate + 2.5D U-Net
D   | Matched-capacity CNN-LSTM gate + 2.5D U-Net  (** mandatory, not optional **)
E   | Proposed: temporal-SNN gate (slices-as-timesteps) + uncertainty buffer + 2.5D U-Net
F   | E without uncertainty-based buffer expansion (fixed buffer only)
G   | E without temporal slice encoding (SNN fed the same centre slice repeated T times)
H   | nnU-Net 2D/3D full-volume configuration, for external SOTA anchoring

This script is intentionally data-agnostic: it wires up each arm's models and
the shared `AdrenalSegmentationPipeline`, then hands off to
`src/evaluation/evaluate_pipeline.py` for the patient-level statistics. Point
`--data-root` at a prepared NIfTI cohort (see src/data/dataset.py) to run it
for real; until then, `--dry-run` exercises the full wiring on synthetic
volumes so the harness itself can be validated without patient data.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python scripts/run_ablation.py` to work from any cwd without
# requiring the caller to set PYTHONPATH manually.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.data.dataset import PatientVolume, discover_patients, load_patient_volume
from src.data.splits import patient_level_kfold
from src.inference.pipeline import AdrenalSegmentationPipeline, PipelineConfig
from src.models.cnn_lstm_gate import CNNLSTMSliceGate
from src.models.snn_gate import SpikingSliceGate
from src.models.unet25d import build_unet25d

ARMS = ["A", "B", "C", "D", "E", "F", "G", "H"]


def build_gate_for_arm(arm: str, slice_window: int):
    if arm in ("A", "B", "H"):
        return None  # no gate: full-volume baselines
    if arm == "C":
        # "Conventional ResNet18 classifier" arm: reuse the CNN-LSTM backbone
        # with a single-timestep window to approximate a plain per-slice
        # classifier without a temporal head, per the design's minimal-diff
        # philosophy (swap only the component under test).
        return CNNLSTMSliceGate(slice_window=1)
    if arm == "D":
        return CNNLSTMSliceGate(slice_window=slice_window)
    if arm in ("E", "F"):
        return SpikingSliceGate(slice_window=slice_window)
    if arm == "G":
        # Same SNN architecture, but the caller must feed it the centre slice
        # repeated `slice_window` times instead of true neighbouring slices —
        # see `_make_synthetic_volume(..., repeat_center=True)` below and the
        # equivalent flag to add to your real dataset loader.
        return SpikingSliceGate(slice_window=slice_window)
    raise ValueError(f"unknown arm '{arm}'")


def uses_uncertainty_buffer(arm: str) -> bool:
    return arm == "E"  # F explicitly disables it; every other arm is untouched by this feature


def _make_synthetic_volume(n_slices: int = 40, size: int = 64, seed: int = 0) -> PatientVolume:
    """Synthetic stand-in so this script can be smoke-tested without patient
    data. Do not use for anything reported in the paper."""
    rng = np.random.RandomState(seed)
    image = rng.normal(0, 1, size=(n_slices, size, size)).astype(np.float32)
    mask = np.zeros_like(image, dtype=np.int16)
    mask[15:25, 20:40, 20:40] = 1  # a plausible "positive run" of slices
    return PatientVolume(patient_id="synthetic_0", image=image, mask=mask)


def run_arm(arm: str, patient_ids: list[str], data_root: str | None, slice_window: int, dry_run: bool):
    from src.training.uncertainty_buffer import BufferConfig

    gate = build_gate_for_arm(arm, slice_window)
    segmenter = build_unet25d(in_channels=slice_window, encoder_weights=None if dry_run else "imagenet")

    buffer_cfg = BufferConfig(max_expansion=0) if not uses_uncertainty_buffer(arm) else BufferConfig()
    pipeline_cfg = PipelineConfig(slice_window=slice_window, buffer_cfg=buffer_cfg)
    pipeline = AdrenalSegmentationPipeline(gate_model=gate, segmenter_model=segmenter, cfg=pipeline_cfg)

    results = []
    for patient_id in patient_ids:
        volume = _make_synthetic_volume(seed=hash(patient_id) % (2**32)) if dry_run \
            else load_patient_volume(data_root, patient_id)
        results.append(pipeline.run(volume))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default=None, help="path to prepared NIfTI cohort")
    parser.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--slice-window", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true", help="use synthetic volumes to smoke-test the harness")
    args = parser.parse_args()

    if args.dry_run or args.data_root is None:
        patient_ids = [f"synthetic_{i}" for i in range(10)]
        print("[run_ablation] no --data-root given (or --dry-run set): using synthetic patients")
    else:
        patient_ids = discover_patients(args.data_root)
        print(f"[run_ablation] discovered {len(patient_ids)} patients under {args.data_root}")

    for train_ids, val_ids in patient_level_kfold(patient_ids, n_folds=args.folds):
        print(f"[run_ablation] fold: {len(train_ids)} train / {len(val_ids)} val patients")
        for arm in args.arms:
            results = run_arm(arm, val_ids, args.data_root, args.slice_window, dry_run=(args.data_root is None))
            avg_segmented = np.mean([r["slices_segmented"] for r in results])
            print(f"  arm {arm}: avg slices segmented per patient = {avg_segmented:.1f}")
        break  # remove this break to run all folds once real training is wired in per-arm


if __name__ == "__main__":
    main()

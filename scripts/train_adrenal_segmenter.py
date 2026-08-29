#!/usr/bin/env python3
"""Standalone training script for the 2.5D adrenal gland segmenter (Stage B).

Run it, walk away, watch the log file. Everything it produces goes into one
run directory:

    <output-dir>/<run-name>/
        train.log             human-readable, line-flushed  -> tail -f this
        metrics.csv           one row per epoch, flushed    -> plot/inspect
        config.json           the exact settings used
        best_model.pt         best checkpoint by validation Dice
        last_model.pt         latest epoch (for --resume)

Why this exists, and what it changes versus the exploratory notebook:

  * Pretrained encoder. Training a U-Net from random init on a structure that
    occupies ~0.3% of pixels does not work in any reasonable number of steps.
  * Physical resampling. AMOS22 mixes 1.25/2/5 mm slice thickness. Without
    resampling, a k-slice window spans a different number of millimetres for
    every patient, which is a confound for a slice-sequence model and shrinks
    the usable training set on thick-slice cases.
  * Threshold sweep. Dice at a fixed 0.5 cutoff reads exactly 0.0 until the
    model's outputs happen to cross 0.5, so it cannot show early progress.
    Validation Dice is reported at the best threshold as well as at 0.5.
  * Volume cache. Decode + resample each case once, then reuse it every epoch.
  * Real training length. Max-epoch cap plus early stopping on validation Dice.

Metrics. Validation reports MEAN PER-CASE Dice (score each patient, then
average) as the headline figure, because that is what the literature reports.
The pooled "aggregate" Dice - every pixel in the cohort treated as one image -
is logged alongside it, and the two are not interchangeable: pooling weights
patients by gland size. Compare like with like before claiming an improvement.

Scope. Each case is cropped to the gland extent +/- --z-margin slices, and
validation samples come only from inside that window. The reported Dice is
therefore the segmenter's ability on the region where the gland actually is,
NOT full-volume performance - slices elsewhere in the abdomen never enter the
score. That gap is what Stage A (the gate) exists to close, and it means these
numbers sit above what the full pipeline will achieve on whole scans.

Also absent here: test-time augmentation and connected-component pruning (see
src/postprocessing/). Both are applied downstream, and in the prior published
pipeline they were worth a large margin - so a raw number from this script is
not comparable with a post-processed one.

Usage
-----
    python scripts/train_adrenal_segmenter.py --data-root /path/to/amos22

    # quick end-to-end check (a few cases, 2 epochs, small images)
    python scripts/train_adrenal_segmenter.py --data-root ... --smoke-test

    # watch it from another shell
    tail -f runs/<run-name>/train.log
"""
from __future__ import annotations

# --- ROCm / MIOpen -----------------------------------------------------------
# MIOpen caches tuned kernels in a SQLite DB. In many ROCm images the default
# location is read-only, which surfaces as
#   "attempt to write a readonly database" -> RuntimeError: miopenStatusInternalError
# from an innocent-looking conv/BatchNorm call. These must be set before torch
# initialises the HIP runtime, so they come before the torch import.
import os as _os
import tempfile as _tempfile

for _var, _sub in (("MIOPEN_USER_DB_PATH", "miopen-db"), ("MIOPEN_CUSTOM_CACHE_DIR", "miopen-cache")):
    _os.environ.setdefault(_var, _os.path.join(_tempfile.gettempdir(), _sub))
    _os.makedirs(_os.environ[_var], exist_ok=True)
# -----------------------------------------------------------------------------

import argparse
import csv
import json
import logging
import math
import random
import re
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train the 2.5D adrenal gland segmenter on AMOS22.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = p.add_argument_group("data")
    data.add_argument("--data-root", type=Path, required=True,
                      help="AMOS22 root containing imagesTr/labelsTr (and optionally imagesVa/labelsVa)")
    data.add_argument("--cache-dir", type=Path, default=None,
                      help="Where to cache resampled volumes (default: <output-dir>/_volume_cache)")
    data.add_argument("--max-train-cases", type=int, default=0, help="0 = use all")
    data.add_argument("--max-val-cases", type=int, default=0, help="0 = use all")
    data.add_argument("--val-fraction", type=float, default=0.2,
                      help="Only used when the dataset has no imagesVa/ split")
    data.add_argument("--right-label", type=int, default=11, help="AMOS22 right adrenal gland label id")
    data.add_argument("--left-label", type=int, default=12, help="AMOS22 left adrenal gland label id")
    data.add_argument("--rebuild-cache", action="store_true")
    data.add_argument("--combine-glands", action="store_true",
                      help="predict one merged gland mask instead of separate left/right "
                           "channels. Separate is the default: the prior published pipeline "
                           "reports L.A.G and R.A.G individually, and the two differ in "
                           "difficulty, so a merged mask cannot be compared with it.")

    geom = p.add_argument_group("geometry")
    geom.add_argument("--target-spacing-z", type=float, default=2.5,
                      help="mm between slices after resampling; one gate/segmenter time step")
    geom.add_argument("--target-spacing-xy", type=float, default=1.0, help="in-plane mm after resampling")
    geom.add_argument("--image-size", type=int, default=384, help="in-plane pixels after pad/crop (multiple of 32)")
    geom.add_argument("--slice-window", type=int, default=5, help="odd; neighbouring slices stacked as channels")
    geom.add_argument("--z-margin", type=int, default=32,
                      help="slices kept above/below the gland extent; the hard-negative neighbourhood")

    sampling = p.add_argument_group("sampling")
    sampling.add_argument("--negative-ratio", type=float, default=0.7,
                          help="negative slices sampled per positive slice")
    sampling.add_argument("--max-positive-per-case", type=int, default=0, help="0 = keep all positive slices")

    model = p.add_argument_group("model")
    model.add_argument("--encoder", type=str, default="resnet34")
    model.add_argument("--encoder-weights", type=str, default="imagenet",
                       help='"imagenet" or "none" (from scratch - not recommended)')
    model.add_argument("--decoder-attention", type=str, default="scse", choices=["none", "scse"])

    optim = p.add_argument_group("optimisation")
    optim.add_argument("--max-epochs", type=int, default=200, help="hard cap on epochs")
    optim.add_argument("--patience", type=int, default=30, help="early stop after N epochs with no val-Dice gain")
    optim.add_argument("--batch-size", type=int, default=8)
    optim.add_argument("--lr", type=float, default=3e-4)
    optim.add_argument("--min-lr", type=float, default=1e-6)
    optim.add_argument("--weight-decay", type=float, default=1e-4)
    optim.add_argument("--warmup-epochs", type=int, default=3)
    optim.add_argument("--grad-clip", type=float, default=1.0)
    optim.add_argument("--batch-dice", dest="batch_dice", action="store_true", default=True,
                       help="pool Dice over the batch rather than scoring each sample alone. "
                            "Essential with separate left/right channels: many slices contain "
                            "one gland but not the other, and per-sample Dice on an empty "
                            "target is ~1.0 for any non-zero prediction, so it drives the "
                            "network to predict nothing at all.")
    optim.add_argument("--no-batch-dice", dest="batch_dice", action="store_false")
    optim.add_argument("--amp", dest="amp", action="store_true", default=True)
    optim.add_argument("--no-amp", dest="amp", action="store_false")
    optim.add_argument("--num-workers", type=int, default=4)
    optim.add_argument("--augment", dest="augment", action="store_true", default=True)
    optim.add_argument("--no-augment", dest="augment", action="store_false")
    optim.add_argument("--augment-level", choices=["light", "full"], default="full",
                       help='"light" = affine + simple intensity jitter (reproduces earlier runs). '
                            '"full" = the nnU-Net transform set minus every flip: adds Gaussian '
                            'blur, brightness, contrast, simulated low resolution and gamma. '
                            'No flip of any axis is ever applied - see the augmentation notes in '
                            'the dataset class for why.')

    run = p.add_argument_group("run")
    run.add_argument("--output-dir", type=Path, default=Path("runs"))
    run.add_argument("--run-name", type=str, default=None, help="default: adrenal_<encoder>_<timestamp>")
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    run.add_argument("--resume", type=Path, default=None, help="path to last_model.pt")
    run.add_argument("--smoke-test", action="store_true",
                     help="tiny everything: proves the pipeline runs end to end")
    return p


SMOKE_TEST_SETTINGS = {
    "max_train_cases": 3, "max_val_cases": 2,
    "max_epochs": 2, "patience": 2, "warmup_epochs": 0,
    "image_size": 128, "batch_size": 2, "num_workers": 0,
    "target_spacing_z": 5.0, "target_spacing_xy": 3.0, "z_margin": 8,
}


def explicitly_provided(argv) -> set[str]:
    """Which options the caller actually typed, as opposed to defaults."""
    probe = build_parser()
    for action in probe._actions:
        action.default = argparse.SUPPRESS
    try:
        return set(vars(probe.parse_args(argv)))
    except SystemExit:  # pragma: no cover - the real parse already validated argv
        return set()


def apply_smoke_test(args: argparse.Namespace, provided: set[str]) -> None:
    """Shrink everything for a fast end-to-end check, but never override a
    value the caller passed explicitly (so `--smoke-test --max-epochs 4`
    honours the 4)."""
    for key, value in SMOKE_TEST_SETTINGS.items():
        if key not in provided:
            setattr(args, key, value)


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

class MetricsWriter:
    """Append-only CSV, flushed every row so `tail -f` and pandas both work."""

    FIELDS = [
        "epoch", "lr", "train_loss", "train_dice", "val_loss",
        "val_dice_at_0.5", "val_dice_best", "val_best_threshold",
        "val_dice_aggregate", "val_dice_case_std", "val_dice_case_worst",
        "val_dice_left", "val_dice_right", "val_dice_gland",
        "val_precision", "val_recall", "epoch_seconds", "is_best",
    ]

    def __init__(self, path: Path):
        self.path = path
        new = not path.exists()
        self._fh = path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.FIELDS)
        if new:
            self._writer.writeheader()
            self._fh.flush()

    def write(self, row: dict) -> None:
        self._writer.writerow({k: row.get(k, "") for k in self.FIELDS})
        self._fh.flush()
        _os.fsync(self._fh.fileno())

    def close(self) -> None:
        self._fh.close()


def format_duration(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    return f"{m}m {s:02d}s" if m else f"{s}s"


def write_progress(path: Path, *, run_name: str, epoch: int, max_epochs: int, history: list[dict],
                   best: float, best_epoch: int, patience: int, elapsed_total: float,
                   avg_epoch_seconds: float, latest: dict) -> None:
    """Rewrite a small human-readable status page every epoch.

    Unlike train.log (append-only, chronological) this file is meant to be
    *opened*, glanced at, and closed: it always shows the current state, the
    trend, and how close early stopping is. Written atomically so a reader
    never catches it half-written.
    """
    since_best = epoch - best_epoch if best_epoch else 0
    if since_best == 0:
        status = "IMPROVING"
    elif since_best < max(2, patience // 3):
        status = "flat - still within normal variation"
    elif since_best < patience:
        status = f"PLATEAUED - will stop in {patience - since_best} epochs without a gain"
    else:
        status = "STOPPING - patience exhausted"

    recent = history[-20:]
    scale = max((h["dice"] for h in history), default=0.0)
    scale = max(scale, 0.02)

    lines = [
        "=" * 72,
        f" Adrenal gland segmenter  -  run: {run_name}",
        f" Updated {time.strftime('%Y-%m-%d %H:%M:%S')}   |   epoch {epoch} of {max_epochs}",
        "=" * 72,
        "",
        f"  STATUS          {status}",
        "",
        f"  Best val Dice   {best:.4f}   (epoch {best_epoch})   [mean per-case]",
        f"  Latest val Dice {latest['dice']:.4f}   at threshold {latest['threshold']:.2f}",
        "",
        *[f"  {name.capitalize():<14}{value:.4f}" for name, value in latest["per_gland"].items()],
        "",
        f"  Aggregate Dice  {latest['aggregate']:.4f}   (pooled over all pixels - not "
        f"the figure papers report)",
        f"  Worst case      {latest['worst']:.4f}   |  spread (sd) {latest['std']:.4f}",
        f"  Epochs since best improvement: {since_best}   (early stop at {patience})",
        "",
        f"  Precision       {latest['precision']:.3f}",
        f"  Recall          {latest['recall']:.3f}",
        f"  Train loss      {latest['train_loss']:.4f}      Val loss  {latest['val_loss']:.4f}",
        f"  Train Dice      {latest['train_dice']:.4f}",
        "",
        f"  Elapsed         {format_duration(elapsed_total)}"
        f"   ({avg_epoch_seconds:.0f}s per epoch)",
        f"  Remaining       ~{format_duration(avg_epoch_seconds * max(0, max_epochs - epoch))}"
        f" if it runs all {max_epochs} epochs (early stopping may end it sooner)",
        "",
        f"  Validation Dice (mean per-case), last {len(recent)} epochs   (bars scaled to best so far = {scale:.4f})",
        "  " + "-" * 68,
    ]
    for h in recent:
        bar = "#" * int(round(40 * h["dice"] / scale))
        mark = "  <- best" if h["epoch"] == best_epoch else ""
        lines.append(f"   epoch {h['epoch']:4d}   {h['dice']:.4f}  {bar}{mark}")

    lines += [
        "  " + "-" * 68,
        "",
        "  Reading this: Dice near 0.00 for the first few epochs is expected.",
        "  Adrenal glands are small and hard - 0.60-0.75 is a good final result.",
        "  Full per-epoch history is in metrics.csv; full log in train.log.",
        "=" * 72,
        "",
    ]

    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("\n".join(lines), encoding="utf-8")
    tmp.replace(path)


def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("adrenal")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    class _FlushingFileHandler(logging.FileHandler):
        def emit(self, record):
            super().emit(record)
            self.flush()
            try:
                _os.fsync(self.stream.fileno())
            except (OSError, ValueError):
                pass

    fh = _FlushingFileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


# --------------------------------------------------------------------------- #
# Case discovery
# --------------------------------------------------------------------------- #

_CASE_RE = re.compile(r"(amos_\d+)", re.IGNORECASE)


def _case_id(path: Path) -> str:
    m = _CASE_RE.search(path.name)
    return m.group(1).lower() if m else path.name.split(".")[0]


def discover_split(image_dir: Path, label_dir: Path) -> list[dict]:
    if not image_dir.is_dir() or not label_dir.is_dir():
        return []
    labels = {_case_id(p): p for p in sorted(label_dir.glob("*.nii*"))}
    records = []
    for image_path in sorted(image_dir.glob("*.nii*")):
        cid = _case_id(image_path)
        if cid in labels:
            records.append({"case_id": cid, "image_path": image_path, "label_path": labels[cid]})
    return records


def discover_cases(args, logger) -> tuple[list[dict], list[dict]]:
    root = args.data_root
    train = discover_split(root / "imagesTr", root / "labelsTr")
    val = discover_split(root / "imagesVa", root / "labelsVa")

    if not train:
        raise FileNotFoundError(
            f"No image/label pairs under {root}/imagesTr + {root}/labelsTr. "
            "Expected the standard AMOS22 layout."
        )

    if not val:
        # No official validation split present - hold out whole patients.
        rng = random.Random(args.seed)
        shuffled = list(train)
        rng.shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * args.val_fraction)))
        val, train = shuffled[:n_val], shuffled[n_val:]
        logger.info("No imagesVa/ found - held out %d of %d cases as validation.", len(val), len(val) + len(train))

    if args.max_train_cases:
        train = train[: args.max_train_cases]
    if args.max_val_cases:
        val = val[: args.max_val_cases]

    overlap = {r["case_id"] for r in train} & {r["case_id"] for r in val}
    if overlap:
        raise RuntimeError(f"Patient leakage between train and validation: {sorted(overlap)[:5]}")

    return train, val


# --------------------------------------------------------------------------- #
# Volume preparation + cache
# --------------------------------------------------------------------------- #

LEFT_CHANNEL_VALUE = 1
RIGHT_CHANNEL_VALUE = 2


@dataclass
class GeometryConfig:
    spacing_z: float
    spacing_xy: float
    image_size: int
    z_margin: int
    hu_window: tuple[float, float] = (-135.0, 215.0)
    right_label: int = 11
    left_label: int = 12

    def cache_key(self) -> str:
        # "v2" marks the switch from a merged binary mask to a label-coded one
        # (1 = left, 2 = right). A v1 cache holds different semantics and must
        # not be silently reused.
        return (f"v2_z{self.spacing_z:g}_xy{self.spacing_xy:g}_s{self.image_size}"
                f"_m{self.z_margin}_hu{self.hu_window[0]:g},{self.hu_window[1]:g}"
                f"_l{self.right_label}-{self.left_label}")


def _resample(volume: np.ndarray, zoom_factors: tuple[float, float, float], is_mask: bool) -> np.ndarray:
    from scipy import ndimage

    if all(abs(f - 1.0) < 1e-3 for f in zoom_factors):
        return volume
    order = 0 if is_mask else 1
    out = ndimage.zoom(volume, zoom_factors, order=order, mode="nearest", prefilter=False)
    return out


def _pad_or_crop_inplane(volume: np.ndarray, size: int, pad_value: float) -> np.ndarray:
    """Centre pad/crop axes 1 and 2 to (size, size)."""
    out = volume
    for axis in (1, 2):
        current = out.shape[axis]
        if current == size:
            continue
        if current < size:
            total = size - current
            before, after = total // 2, total - total // 2
            pad = [(0, 0)] * 3
            pad[axis] = (before, after)
            out = np.pad(out, pad, mode="constant", constant_values=pad_value)
        else:
            start = (current - size) // 2
            sl = [slice(None)] * 3
            sl[axis] = slice(start, start + size)
            out = out[tuple(sl)]
    return out


def prepare_case(record: dict, geom: GeometryConfig) -> dict | None:
    """Load one case, resample to physical target spacing, HU-window, normalise,
    crop to the gland neighbourhood. Returns None if the case has no adrenal."""
    import nibabel as nib

    image_nifti = nib.load(str(record["image_path"]))
    label_nifti = nib.load(str(record["label_path"]))
    if image_nifti.shape != label_nifti.shape:
        raise ValueError(f"{record['case_id']}: image {image_nifti.shape} != label {label_nifti.shape}")

    # nibabel gives (X, Y, Z); the project convention is slice-first (Z, X, Y).
    image = np.transpose(np.asarray(image_nifti.dataobj, dtype=np.float32), (2, 0, 1))
    label = np.transpose(np.asarray(label_nifti.dataobj, dtype=np.int16), (2, 0, 1))

    sx, sy, sz = (float(v) for v in image_nifti.header.get_zooms()[:3])
    zoom_factors = (sz / geom.spacing_z, sx / geom.spacing_xy, sy / geom.spacing_xy)

    # Label-coded rather than merged: 1 = left, 2 = right. Storing the sides
    # separately here means the merged/separate choice is a training-time
    # decision, not baked into the cache.
    adrenal = np.zeros(label.shape, dtype=np.uint8)
    adrenal[label == geom.left_label] = LEFT_CHANNEL_VALUE
    adrenal[label == geom.right_label] = RIGHT_CHANNEL_VALUE
    if not adrenal.any():
        return None

    image = _resample(image, zoom_factors, is_mask=False)
    adrenal = _resample(adrenal, zoom_factors, is_mask=True)

    if not adrenal.any():
        # Resampling can erase a structure only 1 slice thick at a coarse target.
        return None

    # HU window -> [0, 1], then z-score over the volume (matches src/data/preprocessing).
    lo, hi = geom.hu_window
    image = (np.clip(image, lo, hi) - lo) / max(hi - lo, 1e-6)
    mean, std = float(image.mean()), float(image.std())
    image = (image - mean) / std if std > 1e-6 else image - mean

    image = _pad_or_crop_inplane(image, geom.image_size, pad_value=float(image.min()))
    adrenal = _pad_or_crop_inplane(adrenal, geom.image_size, pad_value=0)

    positive = np.flatnonzero((adrenal > 0).any(axis=(1, 2)))
    if positive.size == 0:
        return None
    z0 = max(0, int(positive.min()) - geom.z_margin)
    z1 = min(image.shape[0], int(positive.max()) + geom.z_margin + 1)
    image, adrenal = image[z0:z1], adrenal[z0:z1]

    return {
        "case_id": record["case_id"],
        "image": image.astype(np.float16),
        "mask": adrenal.astype(np.uint8),
        "n_positive": int((adrenal > 0).any(axis=(1, 2)).sum()),
        "n_slices": int(image.shape[0]),
    }


def load_or_build_cache(records, geom, cache_dir: Path, rebuild: bool, logger) -> list[dict]:
    cache_dir = cache_dir / geom.cache_key()
    cache_dir.mkdir(parents=True, exist_ok=True)

    prepared, skipped = [], []
    for i, record in enumerate(records, 1):
        path = cache_dir / f"{record['case_id']}.npz"
        if path.exists() and not rebuild:
            try:
                with np.load(path) as z:
                    prepared.append({
                        "case_id": record["case_id"],
                        "image": z["image"], "mask": z["mask"],
                        "n_positive": int(z["n_positive"]), "n_slices": int(z["n_slices"]),
                    })
                continue
            except Exception as exc:  # corrupt/partial cache entry - rebuild it
                logger.warning("Cache entry %s unreadable (%s); rebuilding.", path.name, exc)

        try:
            case = prepare_case(record, geom)
        except Exception as exc:
            logger.warning("Skipping %s: %s", record["case_id"], exc)
            skipped.append(record["case_id"])
            continue
        if case is None:
            logger.warning("Skipping %s: no adrenal voxels after resampling.", record["case_id"])
            skipped.append(record["case_id"])
            continue

        # Write via an explicit handle: np.savez_compressed appends ".npz" to
        # any path that does not already end in it, which would break the
        # atomic rename below.
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("wb") as fh:
            np.savez_compressed(fh, image=case["image"], mask=case["mask"],
                                n_positive=case["n_positive"], n_slices=case["n_slices"])
        tmp.replace(path)
        prepared.append(case)

        if i % 10 == 0 or i == len(records):
            logger.info("  prepared %d/%d cases", i, len(records))

    if skipped:
        logger.warning("Skipped %d case(s): %s", len(skipped), ", ".join(skipped[:10]))
    if not prepared:
        raise RuntimeError("No usable cases. Check --right-label/--left-label against your dataset.json.")
    return prepared


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

def build_sample_index(cases, negative_ratio: float, max_positive: int, seed: int):
    """(case_index, centre_slice) pairs: every positive slice, plus negatives
    drawn from the surrounding neighbourhood (the useful hard negatives)."""
    rng = random.Random(seed)
    samples = []
    for ci, case in enumerate(cases):
        positive = np.flatnonzero((case["mask"] > 0).any(axis=(1, 2))).tolist()
        if max_positive and len(positive) > max_positive:
            idx = np.linspace(0, len(positive) - 1, max_positive).round().astype(int)
            positive = [positive[i] for i in idx]
        negatives_pool = [i for i in range(case["n_slices"]) if i not in set(positive)]
        rng.shuffle(negatives_pool)
        n_neg = int(round(len(positive) * negative_ratio))
        samples.extend((ci, c) for c in positive)
        samples.extend((ci, c) for c in negatives_pool[:n_neg])
    rng.shuffle(samples)
    return samples


def _make_dataset_class():
    import torch
    import torch.nn.functional as F
    from torch.utils.data import Dataset

    class AdrenalSliceDataset(Dataset):
        def __init__(self, cases, samples, slice_window: int, augment: bool,
                     combine_glands: bool = False, level: str = "full", seed: int = 0):
            self.cases = cases
            self.samples = samples
            self.slice_window = slice_window
            self.augment = augment
            self.combine_glands = combine_glands
            self.level = level
            self.seed = seed

        def __len__(self) -> int:
            return len(self.samples)

        def _window(self, image, centre):
            half = self.slice_window // 2
            idx = np.clip(np.arange(centre - half, centre + half + 1), 0, image.shape[0] - 1)
            return np.ascontiguousarray(image[idx])

        # -- augmentation ---------------------------------------------- #
        #
        # NO FLIP OF ANY AXIS IS APPLIED, and each exclusion is deliberate:
        #
        #   left-right   The abdomen is not laterally symmetric (liver right,
        #                spleen left, IVC right of midline), so a mirrored scan
        #                is anatomically impossible. Decisively, left and right
        #                are separate output channels here: flipping the image
        #                without swapping the channels corrupts the target.
        #   ant-post     Same asymmetry argument, more obviously.
        #   craniocaudal Looks like free data for a slice-sequence model, and is
        #                the most damaging of the three. Stage A's whole premise
        #                is that consecutive slices carry directional
        #                progression through the body; teaching the network that
        #                superior-to-inferior and inferior-to-superior are
        #                equivalent destroys the signal the SNN integrates.
        #
        # Everything else follows nnU-Net's validated transform set, whose
        # intensity operations target the variation that actually exists in this
        # cohort: multiple scanners, contrast and non-contrast studies, and
        # slice thickness from 1.25 to 5 mm.

        @staticmethod
        def _gaussian_blur(x, sigma):
            radius = max(1, int(round(3.0 * sigma)))
            coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
            kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
            kernel = kernel / kernel.sum()
            c = x.shape[0]
            kx = kernel.view(1, 1, 1, -1).expand(c, 1, 1, -1)
            ky = kernel.view(1, 1, -1, 1).expand(c, 1, -1, 1)
            x = F.conv2d(x.unsqueeze(0), kx, padding=(0, radius), groups=c)
            x = F.conv2d(x, ky, padding=(radius, 0), groups=c)
            return x.squeeze(0)

        def _augment_spatial(self, x, y, rng):
            wide = self.level == "full"
            angle = math.radians(rng.uniform(-20, 20) if wide else rng.uniform(-12, 12))
            scale = rng.uniform(0.85, 1.25) if wide else rng.uniform(0.9, 1.1)
            shift = 0.08 if wide else 0.06
            tx, ty = rng.uniform(-shift, shift), rng.uniform(-shift, shift)
            cos, sin = math.cos(angle) / scale, math.sin(angle) / scale
            theta = torch.tensor([[cos, -sin, tx], [sin, cos, ty]], dtype=torch.float32).unsqueeze(0)

            grid = F.affine_grid(theta, (1, x.shape[0], x.shape[1], x.shape[2]), align_corners=False)
            x = F.grid_sample(x.unsqueeze(0), grid, mode="bilinear",
                              padding_mode="border", align_corners=False).squeeze(0)
            y = F.grid_sample(y.unsqueeze(0), grid, mode="nearest",
                              padding_mode="zeros", align_corners=False).squeeze(0)
            return x, y

        def _augment_intensity(self, x, rng):
            if self.level != "full":
                if rng.random() < 0.3:
                    x = x * rng.uniform(0.9, 1.1) + rng.uniform(-0.1, 0.1)
                if rng.random() < 0.2:
                    x = x + torch.randn_like(x) * rng.uniform(0.01, 0.05)
                return x

            if rng.random() < 0.10:                       # additive noise
                x = x + torch.randn_like(x) * rng.uniform(0.0, 0.1)

            if rng.random() < 0.20:                       # blur
                x = self._gaussian_blur(x, rng.uniform(0.5, 1.5))

            if rng.random() < 0.15:                       # brightness
                x = x * rng.uniform(0.75, 1.25)

            if rng.random() < 0.15:                       # contrast about the mean
                mean = x.mean()
                lo, hi = x.min(), x.max()
                x = ((x - mean) * rng.uniform(0.75, 1.25) + mean).clamp(lo, hi)

            if rng.random() < 0.25:
                # Simulated low resolution: downsample with nearest, restore with
                # bicubic. Directly targets this cohort's thick-slice cases -
                # resampling 5 mm data onto a 2.5 mm grid interpolates it up but
                # cannot make it genuinely sharp, so the model should expect it.
                h, w = x.shape[-2:]
                f = rng.uniform(1.0, 2.0)
                small = (max(8, int(round(h / f))), max(8, int(round(w / f))))
                x = F.interpolate(x.unsqueeze(0), size=small, mode="nearest")
                x = F.interpolate(x, size=(h, w), mode="bicubic", align_corners=False).squeeze(0)

            for invert, prob in ((True, 0.10), (False, 0.30)):   # gamma
                if rng.random() >= prob:
                    continue
                if invert:
                    x = -x
                lo = x.min()
                rng_span = x.max() - lo
                if float(rng_span) > 1e-6:
                    gamma = rng.uniform(0.7, 1.5)
                    x = ((x - lo) / rng_span).clamp(min=0) ** gamma * rng_span + lo
                if invert:
                    x = -x
            return x

        def __getitem__(self, index):
            case_index, centre = self.samples[index]
            case = self.cases[case_index]

            x = torch.from_numpy(self._window(case["image"], centre).astype(np.float32))
            coded = case["mask"][centre]
            if self.combine_glands:
                y = torch.from_numpy((coded > 0).astype(np.float32)).unsqueeze(0)
            else:
                # channel 0 = left, channel 1 = right, matching CHANNEL_NAMES
                y = torch.from_numpy(np.stack([
                    coded == LEFT_CHANNEL_VALUE, coded == RIGHT_CHANNEL_VALUE,
                ]).astype(np.float32))

            if self.augment:
                rng = random.Random((self.seed, index, torch.initial_seed() & 0xFFFF).__hash__())
                x, y = self._augment_spatial(x, y, rng)
                y = (y > 0.5).float()
                x = self._augment_intensity(x, rng)

            # case_index travels with the sample so validation can score each
            # patient separately (mean per-case Dice), which is what the
            # literature reports - not one Dice pooled over every pixel.
            return {"slices": x, "mask": y, "case_index": case_index}

    return AdrenalSliceDataset


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def dice_from_counts(intersection: float, pred_sum: float, target_sum: float) -> float:
    denom = pred_sum + target_sum
    if denom == 0:
        return 1.0  # both empty: a correct empty prediction
    return float(2.0 * intersection / denom)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

@dataclass
class EarlyStopper:
    patience: int
    best: float = -1.0
    best_epoch: int = 0
    _bad: int = field(default=0, repr=False)

    def update(self, value: float, epoch: int) -> bool:
        if value > self.best + 1e-5:
            self.best, self.best_epoch, self._bad = value, epoch, 0
            return True
        self._bad += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self._bad >= self.patience


def lr_at(epoch: int, args) -> float:
    if args.warmup_epochs and epoch <= args.warmup_epochs:
        return args.lr * epoch / max(1, args.warmup_epochs)
    span = max(1, args.max_epochs - args.warmup_epochs)
    progress = min(1.0, (epoch - args.warmup_epochs) / span)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * progress))


# Fine spacing below 0.10, coarse above. A Dice loss under heavy class
# imbalance pushes probabilities down, so the optimum routinely lands in the
# low tail: the first real run peaked at 0.05, which was the old floor of this
# sweep, leaving no way to tell whether something lower was better. An optimum
# reported at the edge of its own search range is not a chosen operating point.
THRESHOLDS = np.round(
    np.concatenate([np.arange(0.01, 0.10, 0.01), np.arange(0.10, 0.96, 0.05)]), 2
)


def main(argv=None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    if args.smoke_test:
        apply_smoke_test(args, explicitly_provided(raw_argv))

    if args.slice_window % 2 == 0 or args.slice_window < 1:
        raise SystemExit("--slice-window must be a positive odd integer")
    if args.image_size % 32:
        raise SystemExit("--image-size must be a multiple of 32 (U-Net downsamples 5 times)")

    run_name = args.run_name or f"adrenal_{args.encoder}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(run_dir / "train.log")
    logger.info("=" * 78)
    logger.info("Adrenal gland segmenter (Stage B) - run %s", run_name)
    logger.info("=" * 78)

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from src.models.losses import DiceFocalLoss
    except ImportError as exc:
        raise SystemExit(
            f"Could not import the project's loss from {repo_root}. "
            "Run this script from inside the repository.\n"
            f"  {exc}"
        )
    import segmentation_models_pytorch as smp

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu"
    )
    use_amp = args.amp and device.type == "cuda"
    logger.info("Device: %s | AMP: %s | torch %s", device, use_amp, torch.__version__)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    (run_dir / "config.json").write_text(
        json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2),
        encoding="utf-8",
    )

    # -- data ---------------------------------------------------------------
    train_records, val_records = discover_cases(args, logger)
    logger.info("Cases discovered: %d train / %d validation", len(train_records), len(val_records))

    geom = GeometryConfig(
        spacing_z=args.target_spacing_z, spacing_xy=args.target_spacing_xy,
        image_size=args.image_size, z_margin=args.z_margin,
        right_label=args.right_label, left_label=args.left_label,
    )
    cache_dir = args.cache_dir or (args.output_dir / "_volume_cache")
    logger.info("Resampling to %.2f mm in-plane / %.2f mm between slices; cache: %s",
                geom.spacing_xy, geom.spacing_z, cache_dir / geom.cache_key())

    t0 = time.perf_counter()
    logger.info("Preparing training volumes ...")
    train_cases = load_or_build_cache(train_records, geom, cache_dir, args.rebuild_cache, logger)
    logger.info("Preparing validation volumes ...")
    val_cases = load_or_build_cache(val_records, geom, cache_dir, args.rebuild_cache, logger)
    logger.info("Volume preparation took %.1fs", time.perf_counter() - t0)

    train_samples = build_sample_index(train_cases, args.negative_ratio, args.max_positive_per_case, args.seed)
    val_samples = build_sample_index(val_cases, args.negative_ratio, args.max_positive_per_case, args.seed + 1)

    total_pos = sum(c["n_positive"] for c in train_cases)
    logger.info("Training slices: %d (%d positive across %d cases, ratio %.2f negatives/positive)",
                len(train_samples), total_pos, len(train_cases), args.negative_ratio)
    logger.info("Validation slices: %d across %d cases", len(val_samples), len(val_cases))
    if len(train_samples) < 200:
        logger.warning("Only %d training slices - too few for a meaningful run. "
                       "Add cases (--max-train-cases 0 uses all).", len(train_samples))

    channel_names = ("gland",) if args.combine_glands else ("left", "right")
    n_ch = len(channel_names)
    logger.info("Predicting %d channel(s): %s", n_ch, ", ".join(channel_names))
    logger.info("Augmentation: %s (no flips on any axis - see dataset class)",
                f"{args.augment_level}" if args.augment else "disabled")
    logger.info("Dice pooling: %s", "batch" if args.batch_dice else "per-sample")

    Dataset = _make_dataset_class()
    train_ds = Dataset(train_cases, train_samples, args.slice_window, augment=args.augment,
                       combine_glands=args.combine_glands, level=args.augment_level, seed=args.seed)
    val_ds = Dataset(val_cases, val_samples, args.slice_window, augment=False,
                     combine_glands=args.combine_glands)

    loader_kwargs = dict(num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=len(train_ds) > args.batch_size, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    # -- model --------------------------------------------------------------
    weights = None if args.encoder_weights.lower() in ("none", "") else args.encoder_weights
    try:
        model = smp.Unet(
            encoder_name=args.encoder,
            encoder_weights=weights,
            in_channels=args.slice_window,
            classes=n_ch,
            decoder_attention_type=None if args.decoder_attention == "none" else args.decoder_attention,
        ).to(device)
    except Exception as exc:
        if weights is None:
            raise
        logger.error("Could not obtain '%s' weights for encoder '%s': %s", weights, args.encoder, exc)
        raise SystemExit(
            "\nPretrained encoder weights could not be downloaded. This machine is probably "
            "offline or behind a proxy.\n"
            "Options, best first:\n"
            "  1. Give the machine network access and rerun - pretrained weights matter a lot here.\n"
            "  2. Pre-download the weights somewhere with access, copy the torch cache over, and set\n"
            "     TORCH_HOME=/path/to/cache (weights land in $TORCH_HOME/hub/checkpoints).\n"
            "  3. --encoder-weights none  (trains from scratch; expect far worse results on a\n"
            "     structure this small unless you have a lot of data and epochs).\n"
        ) from exc
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model: U-Net/%s (weights=%s, attention=%s) - %.1fM trainable parameters",
                args.encoder, weights, args.decoder_attention, n_params / 1e6)
    if weights is None:
        logger.warning("Training the encoder from scratch. On a structure this small that usually "
                       "needs far more data and epochs; --encoder-weights imagenet is strongly advised.")

    criterion = DiceFocalLoss(batch_dice=args.batch_dice)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 1
    stopper = EarlyStopper(patience=args.patience)
    if args.resume and args.resume.is_file():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scaler_state_dict") and use_amp:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        stopper.best = float(ckpt.get("best_dice", -1.0))
        stopper.best_epoch = int(ckpt.get("best_epoch", 0))
        logger.info("Resumed from %s at epoch %d (best Dice %.4f)", args.resume, start_epoch, stopper.best)

    metrics = MetricsWriter(run_dir / "metrics.csv")
    progress_path = run_dir / "progress.txt"

    # Seed the trend view from any earlier epochs so --resume keeps its history.
    history: list[dict] = []
    csv_path = run_dir / "metrics.csv"
    if csv_path.exists():
        try:
            with csv_path.open(encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    history.append({"epoch": int(row["epoch"]), "dice": float(row["val_dice_best"])})
        except Exception as exc:
            logger.warning("Could not read prior metrics for the progress view: %s", exc)

    # A collapse to the all-zero prediction is recoverable information, not a
    # result: catch it in a few epochs rather than waiting out full patience.
    collapse_streak = 0
    ever_learned = False

    interrupted = {"flag": False}

    def _handle_sigint(signum, frame):
        if interrupted["flag"]:
            raise KeyboardInterrupt
        interrupted["flag"] = True
        logger.warning("Interrupt received - finishing this epoch, then saving and stopping. "
                       "Press Ctrl-C again to stop immediately.")

    signal.signal(signal.SIGINT, _handle_sigint)

    logger.info("-" * 78)
    logger.info("Training for at most %d epochs (early stop after %d without improvement)",
                args.max_epochs, args.patience)
    logger.info("Live progress summary: %s", run_dir / "progress.txt")
    logger.info("-" * 78)
    run_started = time.perf_counter()

    # -- loop ---------------------------------------------------------------
    for epoch in range(start_epoch, args.max_epochs + 1):
        epoch_start = time.perf_counter()
        lr = lr_at(epoch, args)
        for group in optimizer.param_groups:
            group["lr"] = lr

        model.train()
        train_loss, n_batches = 0.0, 0
        train_inter = np.zeros(n_ch); train_pred = np.zeros(n_ch); train_target = np.zeros(n_ch)
        for batch in train_loader:
            x = batch["slices"].to(device, non_blocking=True)
            y = batch["mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)

            if not torch.isfinite(loss):
                logger.warning("Non-finite loss at epoch %d - skipping this batch.", epoch)
                continue

            scaler.scale(loss).backward()
            if args.grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            train_loss += float(loss.item())
            n_batches += 1
            with torch.no_grad():
                pred = (torch.sigmoid(logits.float()) >= 0.5)
                truth = y > 0.5
                dims = (0,) + tuple(range(2, truth.ndim))
                train_inter += (pred & truth).sum(dim=dims).cpu().numpy()
                train_pred += pred.sum(dim=dims).cpu().numpy()
                train_target += truth.sum(dim=dims).cpu().numpy()

        train_loss /= max(n_batches, 1)
        train_dice = float(np.mean([
            dice_from_counts(train_inter[c], train_pred[c], train_target[c]) for c in range(n_ch)
        ]))

        # -- validation: accumulate counts at every threshold in one pass ---
        model.eval()
        val_loss, n_val_batches = 0.0, 0
        n_cases = len(val_cases)
        n_thr = len(THRESHOLDS)
        # (threshold, case, gland) so each patient is scored separately for each
        # gland - the prior published pipeline reports L.A.G and R.A.G apart,
        # and they differ in difficulty, so a merged figure hides the gap.
        inter_c = np.zeros((n_thr, n_cases, n_ch))
        pred_c = np.zeros((n_thr, n_cases, n_ch))
        target_c = np.zeros((n_cases, n_ch))
        with torch.inference_mode():
            for batch in val_loader:
                x = batch["slices"].to(device, non_blocking=True)
                y = batch["mask"].to(device, non_blocking=True)
                case_ids = batch["case_index"].numpy()
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(x)
                    loss = criterion(logits, y)
                if torch.isfinite(loss):
                    val_loss += float(loss.item())
                    n_val_batches += 1

                probs = torch.sigmoid(logits.float())
                truth = y > 0.5
                dims = tuple(range(2, truth.ndim))          # keep (batch, gland)
                np.add.at(target_c, case_ids, truth.sum(dim=dims).cpu().numpy())
                for ti, threshold in enumerate(THRESHOLDS):
                    pred = probs >= float(threshold)
                    np.add.at(inter_c[ti], case_ids, (pred & truth).sum(dim=dims).cpu().numpy())
                    np.add.at(pred_c[ti], case_ids, pred.sum(dim=dims).cpu().numpy())

        val_loss /= max(n_val_batches, 1)

        # Mean per-case Dice per gland: score every patient, then average over
        # the patients that actually contain that gland.
        denom = pred_c + target_c[None, :, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            per_case = np.where(denom > 0, 2.0 * inter_c / denom, 1.0)   # (thr, case, gland)
        seen = target_c > 0                                              # (case, gland)
        dice_case = np.zeros((n_thr, n_ch))
        for c in range(n_ch):
            if seen[:, c].any():
                dice_case[:, c] = per_case[:, seen[:, c], c].mean(axis=1)

        # One shared operating threshold, chosen on the mean across glands, so
        # the model has a single deployable cutoff rather than one per output.
        dice_case_mean = dice_case.mean(axis=1)
        best_i = int(np.argmax(dice_case_mean))
        best_threshold = float(THRESHOLDS[best_i])
        val_dice_best = float(dice_case_mean[best_i])
        per_gland = {name: float(dice_case[best_i, c]) for c, name in enumerate(channel_names)}

        half_i = int(np.argmin(np.abs(THRESHOLDS - 0.5)))
        val_dice_half = float(dice_case_mean[half_i])

        # Pooled ("aggregate") Dice, logged for reference only - it treats the
        # whole cohort as one image and is not what the literature reports.
        agg_inter = inter_c[best_i].sum(axis=0)
        agg_pred = pred_c[best_i].sum(axis=0)
        agg_target = target_c.sum(axis=0)
        val_dice_agg = float(np.mean([
            dice_from_counts(agg_inter[c], agg_pred[c], agg_target[c]) for c in range(n_ch)
        ]))
        precision = float(agg_inter.sum() / agg_pred.sum()) if agg_pred.sum() > 0 else 0.0
        recall = float(agg_inter.sum() / agg_target.sum()) if agg_target.sum() > 0 else 0.0

        scores_at_best = per_case[best_i][seen.any(axis=1)]
        dice_case_std = float(scores_at_best.mean(axis=1).std()) if scores_at_best.size else 0.0
        dice_case_worst = float(scores_at_best.mean(axis=1).min()) if scores_at_best.size else 0.0

        improved = stopper.update(val_dice_best, epoch)
        elapsed = time.perf_counter() - epoch_start

        logger.info(
            "epoch %3d/%d | lr %.2e | loss %.4f/%.4f | dice tr %.4f | "
            "val/case %.4f@%.2f [%s] | worst %.3f | P %.3f R %.3f | %5.1fs%s",
            epoch, args.max_epochs, lr, train_loss, val_loss, train_dice,
            val_dice_best, best_threshold,
            "  ".join(f"{n} {v:.4f}" for n, v in per_gland.items()),
            dice_case_worst, precision, recall, elapsed,
            "  <- best" if improved else "",
        )
        metrics.write({
            "epoch": epoch, "lr": f"{lr:.6e}",
            "train_loss": f"{train_loss:.6f}", "train_dice": f"{train_dice:.6f}",
            "val_loss": f"{val_loss:.6f}", "val_dice_at_0.5": f"{val_dice_half:.6f}",
            "val_dice_best": f"{val_dice_best:.6f}", "val_best_threshold": f"{best_threshold:.2f}",
            "val_dice_aggregate": f"{val_dice_agg:.6f}",
            "val_dice_case_std": f"{dice_case_std:.6f}",
            "val_dice_case_worst": f"{dice_case_worst:.6f}",
            **{f"val_dice_{name}": f"{value:.6f}" for name, value in per_gland.items()},
            "val_precision": f"{precision:.6f}", "val_recall": f"{recall:.6f}",
            "epoch_seconds": f"{elapsed:.2f}", "is_best": int(improved),
        })

        history.append({"epoch": epoch, "dice": val_dice_best})
        epochs_done = max(1, epoch - start_epoch + 1)
        write_progress(
            progress_path, run_name=run_name, epoch=epoch, max_epochs=args.max_epochs,
            history=history, best=stopper.best, best_epoch=stopper.best_epoch,
            patience=args.patience,
            elapsed_total=time.perf_counter() - run_started,
            avg_epoch_seconds=(time.perf_counter() - run_started) / epochs_done,
            latest={
                "dice": val_dice_best, "threshold": best_threshold,
                "aggregate": val_dice_agg, "std": dice_case_std, "worst": dice_case_worst,
                "per_gland": per_gland,
                "precision": precision, "recall": recall,
                "train_loss": train_loss, "val_loss": val_loss, "train_dice": train_dice,
            },
        )

        config_blob = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        # last_model.pt carries optimiser state so --resume can continue exactly.
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if use_amp else None,
            "best_dice": stopper.best, "best_epoch": stopper.best_epoch,
            "best_threshold": best_threshold,
            "config": config_blob,
        }, run_dir / "last_model.pt")
        if improved:
            # best_model.pt is for inference: weights + the threshold they were
            # scored at, without the (3x larger) optimiser state.
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_dice": val_dice_best,
                "threshold": best_threshold,
                "config": config_blob,
            }, run_dir / "best_model.pt")

        if val_dice_best > 0.05:
            ever_learned = True
        collapse_streak = collapse_streak + 1 if val_dice_best < 1e-6 else 0
        if ever_learned and collapse_streak >= 5:
            logger.error(
                "COLLAPSED: validation Dice has been exactly 0 for %d consecutive epochs "
                "after reaching %.4f at epoch %d. The network is predicting empty masks "
                "everywhere and will not recover on its own.", collapse_streak,
                stopper.best, stopper.best_epoch)
            logger.error(
                "Most likely causes, in order: (1) per-sample Dice with empty target "
                "channels - use --batch-dice (currently %s); (2) learning rate too high - "
                "it collapsed just as warmup reached its peak in previous runs; "
                "(3) fp16 instability. Best checkpoint is preserved at %s.",
                "on" if args.batch_dice else "OFF", run_dir / "best_model.pt")
            break

        if interrupted["flag"]:
            logger.warning("Stopping after epoch %d at user request.", epoch)
            break
        if stopper.should_stop:
            logger.info("Early stopping: no improvement for %d epochs.", args.patience)
            break

    logger.info("-" * 78)
    logger.info("Best validation Dice %.4f at epoch %d. Checkpoint: %s",
                stopper.best, stopper.best_epoch, run_dir / "best_model.pt")
    logger.info("Metrics: %s", run_dir / "metrics.csv")
    logger.info("-" * 78)
    metrics.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Stage A: train the slice-selection gate (spiking, or the CNN-LSTM baseline).

Stage A decides which axial slices are worth segmenting. Stage B
(`train_adrenal_segmenter.py`) then segments only those. The research claim is
that presenting k consecutive slices to a spiking network as k discrete time
steps - so LIF membrane potential integrates evidence along the craniocaudal
axis - beats a matched-capacity CNN-LSTM given the same windows. That claim is
only testable if both models are trained under an identical protocol, which is
what this script exists to guarantee: one sampler, one augmentation path, one
schedule, one metric, selected by `--model`.

Three arms live here, matching docs/methodology.docx:

    --model snn                 arm E   sequence-aware spiking gate
    --model cnn_lstm            arm D   matched-capacity recurrent baseline
    --model snn --static-repeat arm G   the centre slice repeated k times,
                                        i.e. conventional SNN rate coding with
                                        no anatomical sequence. If arm E does
                                        not beat arm G, the "sequence-aware"
                                        framing is not earning its place.

Preprocessing is imported from the Stage B trainer and reads the SAME volume
cache, so the two stages cannot drift apart and no cache needs rebuilding.

What is being optimised
-----------------------
Not accuracy. A missed positive slice permanently removes adrenal tissue from
the segmenter's input and cannot be recovered downstream; a false positive only
costs one wasted segmentation. So the operating threshold is chosen post hoc on
the validation set as the highest threshold that still retains
`--target-sensitivity` of gland-bearing slices, and the headline number is how
many slices the gate removes at that operating point.

Usage
-----
    # ~2 minute end-to-end check
    python scripts/train_slice_gate.py --data-root ../data/amos22 --smoke-test

    # the two arms that matter, identical but for --model
    python scripts/train_slice_gate.py --data-root ../data/amos22 \
        --cache-dir ../cache --model snn      --run-name gate_snn
    python scripts/train_slice_gate.py --data-root ../data/amos22 \
        --cache-dir ../cache --model cnn_lstm --run-name gate_cnnlstm

Each run writes runs/<run-name>/: progress.txt (open this), train.log,
metrics.csv, config.json, best_model.pt, last_model.pt.
"""
from __future__ import annotations

import os as _os
import tempfile as _tempfile

# MIOpen builds its kernel database on the first convolution and the packaged
# location is read-only on the cluster (-> miopenStatusInternalError). Must
# happen before torch is imported anywhere in the process.
for _var, _sub in (("MIOPEN_USER_DB_PATH", "miopen-db"), ("MIOPEN_CUSTOM_CACHE_DIR", "miopen-cache")):
    _os.environ.setdefault(_var, _os.path.join(_tempfile.gettempdir(), _sub))
    _os.makedirs(_os.environ[_var], exist_ok=True)

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_adrenal_segmenter import (          # noqa: E402  (path set above)
    GeometryConfig,
    LEFT_CHANNEL_VALUE,
    RIGHT_CHANNEL_VALUE,
    discover_cases,
    format_duration,
    load_or_build_cache,
    setup_logging,
)

HEADS = ("left_present", "right_present")
SHORT = {"left_present": "left", "right_present": "right"}


# --------------------------------------------------------------------------- #
# Arguments
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    g = p.add_argument_group("data")
    g.add_argument("--data-root", type=Path, required=True, help="AMOS22 root (imagesTr/, labelsTr/, ...)")
    g.add_argument("--cache-dir", type=Path, default=Path("runs/_volume_cache"),
                   help="the SAME cache the segmenter uses; nothing is rebuilt if it is populated")
    g.add_argument("--rebuild-cache", action="store_true")
    g.add_argument("--modality", choices=["ct", "mri", "all"], default="ct")
    g.add_argument("--mri-id-threshold", type=int, default=500)
    g.add_argument("--val-fraction", type=float, default=0.2)
    g.add_argument("--max-train-cases", type=int, default=0)
    g.add_argument("--max-val-cases", type=int, default=0)

    g = p.add_argument_group("geometry (must match the segmenter's cache key)")
    g.add_argument("--target-spacing-z", type=float, default=2.5)
    g.add_argument("--target-spacing-xy", type=float, default=1.0)
    g.add_argument("--cache-image-size", type=int, default=384,
                   help="the cache's in-plane size; leave at the segmenter's value")
    g.add_argument("--z-margin", type=int, default=32,
                   help="slices kept either side of the gland in the cache. This also "
                        "defines what a 'negative' slice is for the gate - see the note "
                        "printed at startup. Pass a large value (with a matching cache) "
                        "for whole-volume negatives.")
    g.add_argument("--right-label", type=int, default=11)
    g.add_argument("--left-label", type=int, default=12)

    g = p.add_argument_group("model")
    g.add_argument("--model", choices=["snn", "cnn_lstm"], default="snn")
    g.add_argument("--static-repeat", action="store_true",
                   help="arm G: repeat the centre slice k times instead of using k "
                        "different slices, removing the anatomical sequence")
    g.add_argument("--slice-window", type=int, default=5, help="k: slices == SNN time steps")
    g.add_argument("--image-size", type=int, default=192,
                   help="gate input size; the cache is downsampled to it by an integer "
                        "factor. Presence detection does not need delineation resolution.")
    g.add_argument("--hidden-dim", type=int, default=256)
    g.add_argument("--beta", type=float, default=0.9, help="LIF membrane decay (snn only)")
    g.add_argument("--lif-threshold", type=float, default=1.0, help="LIF firing threshold (snn only)")
    g.add_argument("--lstm-layers", type=int, default=1, help="cnn_lstm only")
    g.add_argument("--lstm-hidden", type=int, default=128,
                   help="cnn_lstm only. 128 reproduces the model as originally written, "
                        "which is NOT capacity-matched to the spiking gate - the LSTM's four "
                        "gate matrices add ~132k parameters that LIF integration does not "
                        "have. 32 matches the two within ~2%. The startup check prints both "
                        "counts either way.")

    g = p.add_argument_group("optimisation")
    g.add_argument("--max-epochs", type=int, default=60)
    g.add_argument("--batch-size", type=int, default=32)
    g.add_argument("--lr", type=float, default=3e-4)
    g.add_argument("--min-lr", type=float, default=1e-6)
    g.add_argument("--warmup-epochs", type=int, default=3)
    g.add_argument("--weight-decay", type=float, default=1e-4)
    g.add_argument("--grad-clip", type=float, default=1.0,
                   help="surrogate gradients through k time steps can spike; 0 disables")
    g.add_argument("--patience", type=int, default=15)
    g.add_argument("--focal-gamma", type=float, default=2.0)
    g.add_argument("--max-pos-weight", type=float, default=20.0,
                   help="cap on the per-head positive class weight")
    g.add_argument("--negative-ratio", type=float, default=0.0,
                   help="negatives per positive in the training index; 0 keeps every slice")
    g.add_argument("--augment", action="store_true", default=True)
    g.add_argument("--no-augment", dest="augment", action="store_false")
    g.add_argument("--amp", action="store_true",
                   help="off by default: fp16 through surrogate gradients is not reliably "
                        "stable, and an identical protocol across arms matters more than speed")

    g = p.add_argument_group("evaluation")
    g.add_argument("--target-sensitivity", type=float, default=0.99,
                   help="the gate's contract: retain this fraction of gland-bearing slices")
    g.add_argument("--select-by", choices=["prauc", "f2", "reduction"], default="prauc",
                   help="checkpoint selection metric. prauc is threshold-free and stable "
                        "early; reduction is the deployment number but degenerate until "
                        "the model separates the classes at all.")

    g = p.add_argument_group("run")
    g.add_argument("--run-name", default=None, help="default: gate_<model>[_static]")
    g.add_argument("--runs-dir", type=Path, default=Path("runs"))
    g.add_argument("--resume", type=Path, default=None)
    g.add_argument("--num-workers", type=int, default=8)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    g.add_argument("--smoke-test", action="store_true")
    return p


SMOKE_TEST_SETTINGS = {
    "max_epochs": 2, "batch_size": 4, "max_train_cases": 3, "max_val_cases": 2,
    "num_workers": 0, "image_size": 96, "patience": 2, "warmup_epochs": 1,
}


def explicitly_provided(argv) -> set[str]:
    """Which options the caller actually typed, as opposed to defaults, so that
    `--smoke-test --max-epochs 4` honours the 4."""
    probe = build_parser()
    for action in probe._actions:
        action.default = argparse.SUPPRESS
    try:
        return set(vars(probe.parse_args(argv)))
    except SystemExit:  # pragma: no cover - the real parse already validated argv
        return set()


def apply_smoke_test(args: argparse.Namespace, provided: set[str]) -> None:
    for key, value in SMOKE_TEST_SETTINGS.items():
        if key not in provided:
            setattr(args, key, value)


# --------------------------------------------------------------------------- #
# Run artefacts
# --------------------------------------------------------------------------- #

class GateMetricsWriter:
    """Append-only CSV, flushed and fsynced every row so `tail -f` and pandas
    both see a complete file at any moment."""

    FIELDS = [
        "epoch", "lr", "train_loss", "val_loss",
        "prauc_left", "prauc_right", "prauc_mean",
        "thr_left", "thr_right",
        "sens_left", "sens_right", "spec_left", "spec_right",
        "npv_left", "npv_right", "f2_left", "f2_right",
        "any_retention", "slice_reduction",
        "spikes_per_window", "epoch_seconds", "is_best",
    ]

    def __init__(self, path: Path):
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


def write_progress(path: Path, *, run_name: str, model_name: str, epoch: int, max_epochs: int,
                   history: list[dict], best: float, best_epoch: int, patience: int,
                   select_by: str, target_sensitivity: float,
                   elapsed_total: float, avg_epoch_seconds: float, latest: dict) -> None:
    """Rewritten every epoch, atomically. This is the file to keep open while a
    job runs: it answers 'is this working' without reading a log."""
    lines: list[str] = []
    add = lines.append

    add("=" * 78)
    add(f" STAGE A GATE - {run_name}  ({model_name})")
    add("=" * 78)
    add(f" epoch {epoch}/{max_epochs}"
        f"   elapsed {format_duration(elapsed_total)}"
        f"   ~{format_duration(avg_epoch_seconds)}/epoch")
    remaining = max(0, max_epochs - epoch) * avg_epoch_seconds
    add(f" estimated remaining {format_duration(remaining)} (if it runs to the epoch limit)")
    add("")

    add("-" * 78)
    add(f" SELECTION METRIC ({select_by})")
    add("-" * 78)
    add(f"   best {best:.4f} at epoch {best_epoch}"
        f"   |   {epoch - best_epoch} epoch(s) since improvement"
        f"   |   stops at {patience}")
    add("")

    add("-" * 78)
    add(f" THE GATE'S CONTRACT  (retain {target_sensitivity:.0%} of gland-bearing slices)")
    add("-" * 78)
    add(f"   slices removed             {latest['slice_reduction']:.1%}"
        "      <- the whole point of Stage A")
    add(f"   gland slices retained      {latest['any_retention']:.1%}")
    add("")
    add(f"   {'head':<8}{'PR-AUC':>9}{'thr':>8}{'sens':>9}{'spec':>9}{'NPV':>9}{'F2':>9}")
    for head in HEADS:
        s = SHORT[head]
        add(f"   {s:<8}{latest['prauc'][head]:>9.4f}{latest['thr'][head]:>8.3f}"
            f"{latest['sens'][head]:>9.4f}{latest['spec'][head]:>9.4f}"
            f"{latest['npv'][head]:>9.4f}{latest['f2'][head]:>9.4f}")
    add("")
    add(f"   train loss {latest['train_loss']:.4f}   val loss {latest['val_loss']:.4f}")
    if latest.get("spikes_per_window"):
        add(f"   mean spikes per window {latest['spikes_per_window']:.0f}"
            "   (energy proxy only - meaningless on GPU, see the methodology)")
    add("")

    if len(history) > 1:
        add("-" * 78)
        add(f" {select_by.upper()} BY EPOCH")
        add("-" * 78)
        values = [h["score"] for h in history]
        lo, hi = min(values), max(values)
        span = max(hi - lo, 1e-6)
        rows = 10
        recent = history[-64:]
        for r in range(rows, 0, -1):
            level = lo + span * r / rows
            line = "".join("#" if h["score"] >= level - span / (2 * rows) else " " for h in recent)
            add(f"   {level:6.3f} |{line}")
        add(f"          +{'-' * len(recent)}")
        add(f"           epoch {recent[0]['epoch']} -> {recent[-1]['epoch']}")
        add("")

    add("=" * 78)

    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(path)


# --------------------------------------------------------------------------- #
# Cases -> per-slice labels
# --------------------------------------------------------------------------- #

def _downsample(volume: np.ndarray, factor: int) -> np.ndarray:
    """Mean-pool axes 1 and 2 by an integer factor. Exact and fast, unlike a
    generic zoom, and the gate does not need interpolation quality."""
    if factor == 1:
        return volume
    z, h, w = volume.shape
    h2, w2 = (h // factor) * factor, (w // factor) * factor
    cropped = volume[:, :h2, :w2].astype(np.float32)
    return cropped.reshape(z, h2 // factor, factor, w2 // factor, factor).mean(axis=(2, 4))


def prepare_gate_cases(cases: list[dict], image_size: int, cache_image_size: int, logger):
    """Downsample once and reduce each mask to per-slice presence flags.

    Keeping full-resolution masks in memory for 300 cases is pointless when the
    gate's target is two booleans per slice, and downsampling here rather than
    per-window removes it from the inner loop entirely.
    """
    if cache_image_size % image_size != 0:
        raise SystemExit(
            f"--image-size {image_size} must divide the cache's {cache_image_size} exactly "
            f"(try {', '.join(str(cache_image_size // f) for f in (1, 2, 3, 4) if cache_image_size % f == 0)})."
        )
    factor = cache_image_size // image_size

    prepared = []
    n_pos = n_slices = 0
    for case in cases:
        image = _downsample(np.asarray(case["image"], dtype=np.float32), factor)
        mask = case["mask"]
        labels = np.stack([
            (mask == LEFT_CHANNEL_VALUE).any(axis=(1, 2)),
            (mask == RIGHT_CHANNEL_VALUE).any(axis=(1, 2)),
        ], axis=1).astype(np.uint8)
        prepared.append({"case_id": case["case_id"],
                         "image": image.astype(np.float16), "labels": labels})
        n_pos += int(labels.any(axis=1).sum())
        n_slices += labels.shape[0]
    logger.info("  %d cases | %d slices | %d gland-bearing (%.1f%%) | downsampled %dx to %dpx",
                len(prepared), n_slices, n_pos, 100 * n_pos / max(n_slices, 1), factor, image_size)
    return prepared


def build_sample_index(cases, negative_ratio: float, seed: int):
    """(case_index, centre_slice) pairs. Every gland-bearing slice, plus
    negatives - all of them by default, since the gate's job is precisely to
    reject negatives and subsampling them flatters its specificity."""
    positives, negatives = [], []
    for ci, case in enumerate(cases):
        any_gland = case["labels"].any(axis=1)
        for z in range(case["labels"].shape[0]):
            (positives if any_gland[z] else negatives).append((ci, z))
    if negative_ratio > 0:
        rng = random.Random(seed)
        keep = min(len(negatives), int(round(len(positives) * negative_ratio)))
        negatives = rng.sample(negatives, keep)
    index = positives + negatives
    random.Random(seed).shuffle(index)
    return index, len(positives), len(negatives)


def _make_dataset_class():
    """Defined lazily so importing this module does not require torch."""
    import torch
    from torch.utils.data import Dataset

    class SliceWindowDataset(Dataset):
        def __init__(self, cases, index, slice_window: int, static_repeat: bool):
            self.cases = cases
            self.index = index
            self.half = slice_window // 2
            self.slice_window = slice_window
            self.static_repeat = static_repeat

        def __len__(self):
            return len(self.index)

        def __getitem__(self, i):
            ci, z = self.index[i]
            case = self.cases[ci]
            image, labels = case["image"], case["labels"]
            n = image.shape[0]

            if self.static_repeat:
                # Arm G: the same slice k times. Rate coding with no anatomy in
                # the time axis - the control the sequence claim is measured against.
                window = np.repeat(image[z][None], self.slice_window, axis=0)
            else:
                idx = np.clip(np.arange(z - self.half, z + self.half + 1), 0, n - 1)
                window = image[idx]

            x = torch.from_numpy(np.ascontiguousarray(window, dtype=np.float32)).unsqueeze(1)
            y = torch.from_numpy(labels[z].astype(np.float32))
            return x, y

    return SliceWindowDataset


# --------------------------------------------------------------------------- #
# Augmentation (on device, identical across the k slices of a window)
# --------------------------------------------------------------------------- #

def augment_batch(x, generator=None):
    """x: (B, T, 1, H, W). One affine and one intensity transform per SAMPLE,
    shared by all T slices.

    Augmenting slices independently would be a bug rather than a variation: the
    whole premise of Stage A is that the k slices form a coherent craniocaudal
    sequence, and per-slice jitter destroys exactly the signal the model is
    supposed to integrate.

    No flips on any axis. A left-right flip swaps which gland is which while the
    targets stay put, and a craniocaudal flip reverses the anatomical direction
    the sequence encodes.
    """
    import torch
    import torch.nn.functional as F

    B, T, C, H, W = x.shape
    device = x.device
    flat = x.reshape(B * T, C, H, W)

    angle = (torch.rand(B, device=device) * 2 - 1) * (15 * math.pi / 180)
    scale = 1.0 + (torch.rand(B, device=device) * 2 - 1) * 0.10
    tx = (torch.rand(B, device=device) * 2 - 1) * 0.05
    ty = (torch.rand(B, device=device) * 2 - 1) * 0.05
    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale

    theta = torch.zeros(B, 2, 3, device=device, dtype=flat.dtype)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = cos, -sin, tx
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = sin, cos, ty
    theta = theta.repeat_interleave(T, dim=0)

    grid = F.affine_grid(theta, flat.shape, align_corners=False)
    flat = F.grid_sample(flat, grid, mode="bilinear", padding_mode="border", align_corners=False)

    out = flat.reshape(B, T, C, H, W)

    # Intensity: per sample, so the sequence stays internally consistent.
    noise = torch.randn_like(out) * (torch.rand(B, 1, 1, 1, 1, device=device) * 0.06)
    brightness = 1.0 + (torch.rand(B, 1, 1, 1, 1, device=device) * 2 - 1) * 0.12
    out = out * brightness + noise
    return out


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def threshold_for_sensitivity(scores: np.ndarray, targets: np.ndarray, target: float) -> float:
    """Highest threshold that still recalls `target` of the positives.

    Chosen by construction rather than by scanning a grid: with n_pos positives,
    recalling ceil(target * n_pos) of them means the cutoff is the k-th largest
    positive score. No grid resolution to argue about, and it cannot land on the
    edge of its own search range.
    """
    pos = scores[targets > 0]
    if pos.size == 0:
        return 1.0
    k = max(1, int(math.ceil(target * pos.size)))
    k = min(k, pos.size)
    return float(np.sort(pos)[::-1][k - 1])


def binary_scores(scores: np.ndarray, targets: np.ndarray, threshold: float) -> dict:
    pred = scores >= threshold
    truth = targets > 0
    tp = int(np.logical_and(pred, truth).sum())
    fp = int(np.logical_and(pred, ~truth).sum())
    tn = int(np.logical_and(~pred, ~truth).sum())
    fn = int(np.logical_and(~pred, truth).sum())
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    ppv = tp / (tp + fp) if tp + fp else float("nan")
    npv = tn / (tn + fn) if tn + fn else float("nan")
    if ppv == ppv and sens == sens and (4 * ppv + sens) > 0:
        f2 = 5 * ppv * sens / (4 * ppv + sens)
    else:
        f2 = float("nan")
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "sensitivity": sens, "specificity": spec, "ppv": ppv, "npv": npv, "f2": f2}


def average_precision(scores: np.ndarray, targets: np.ndarray) -> float:
    try:
        from sklearn.metrics import average_precision_score
    except ImportError:                       # pragma: no cover
        return float("nan")
    if targets.max() == targets.min():
        return float("nan")
    return float(average_precision_score(targets, scores))


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

def build_models(args, logger):
    """Build the requested arm, and count the other one's parameters too.

    'Matched capacity' is a claim in the paper, so it is measured here and
    printed rather than asserted in prose.
    """
    from src.models.cnn_lstm_gate import CNNLSTMSliceGate
    from src.models.snn_gate import SpikingSliceGate

    def _snn():
        return SpikingSliceGate(slice_window=args.slice_window, in_channels=1,
                                hidden_dim=args.hidden_dim, beta=args.beta,
                                threshold=args.lif_threshold, heads=HEADS)

    def _lstm():
        return CNNLSTMSliceGate(slice_window=args.slice_window, in_channels=1,
                                hidden_dim=args.hidden_dim, lstm_layers=args.lstm_layers,
                                lstm_hidden=args.lstm_hidden, heads=HEADS)

    def _count(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    n_snn, n_lstm = _count(_snn()), _count(_lstm())
    ratio = n_lstm / max(n_snn, 1)
    logger.info("Parameter counts: snn %s | cnn_lstm %s | ratio %.2f",
                f"{n_snn:,}", f"{n_lstm:,}", ratio)
    if not 0.8 <= ratio <= 1.25:
        logger.warning(
            "The two gates are NOT matched in capacity (ratio %.2f: cnn_lstm/snn). An "
            "advantage either way is then confounded by size, and 'matched-capacity "
            "baseline' would not be an accurate description in the paper. The difference "
            "is the LSTM's four gate matrices, which LIF integration has no equivalent of; "
            "--lstm-hidden 32 brings the two within a few percent. Either match them, or "
            "report the counts and the confound explicitly.", ratio)

    model = _snn() if args.model == "snn" else _lstm()
    return model, {"snn": n_snn, "cnn_lstm": n_lstm}


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


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    if args.smoke_test:
        apply_smoke_test(args, explicitly_provided(argv))

    # A smoke test defaults to its own run directory. Sharing one with the real
    # run is harmless for the checkpoints (overwritten on epoch 1, and there is
    # no auto-resume) but metrics.csv and train.log both APPEND, so the throwaway
    # epochs would sit under the real ones in every plot you make afterwards.
    run_name = args.run_name or ("gate_" + args.model
                                 + ("_static" if args.static_repeat else "")
                                 + ("_smoke" if args.smoke_test else ""))
    run_dir = args.runs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    progress_path = run_dir / "progress.txt"
    logger = setup_logging(run_dir / "train.log")

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    from src.models.losses import MultiHeadFocalBCELoss

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if (args.device in ("auto", "cuda") and torch.cuda.is_available())
                          else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")

    logger.info("=" * 78)
    logger.info("STAGE A - slice gate | run '%s' | arm: %s%s", run_name, args.model,
                " + static-repeat (arm G)" if args.static_repeat else "")
    logger.info("=" * 78)
    logger.info("Device %s | AMP %s | torch %s", device, use_amp, torch.__version__)
    if device.type == "cuda":
        logger.info("GPU %s", torch.cuda.get_device_name(0))

    # ---- data ----------------------------------------------------------- #
    geom = GeometryConfig(
        spacing_z=args.target_spacing_z, spacing_xy=args.target_spacing_xy,
        image_size=args.cache_image_size, z_margin=args.z_margin,
        right_label=args.right_label, left_label=args.left_label,
    )
    logger.info("Volume cache key: %s", geom.cache_key())
    logger.info("NOTE ON NEGATIVES: the cache keeps %d slices either side of the gland, so a "
                "'negative' here is a near-gland slice, not a random abdominal one. That is the "
                "hard-negative regime and the right one for discrimination, but the specificity "
                "below is NOT a whole-volume deployment number. Rebuild with a large --z-margin "
                "for that.", args.z_margin)

    train_records, val_records = discover_cases(args, logger)
    logger.info("Preparing training volumes...")
    train_cases = prepare_gate_cases(
        load_or_build_cache(train_records, geom, args.cache_dir, args.rebuild_cache, logger),
        args.image_size, args.cache_image_size, logger)
    logger.info("Preparing validation volumes...")
    val_cases = prepare_gate_cases(
        load_or_build_cache(val_records, geom, args.cache_dir, False, logger),
        args.image_size, args.cache_image_size, logger)

    train_index, n_pos, n_neg = build_sample_index(train_cases, args.negative_ratio, args.seed)
    val_index, v_pos, v_neg = build_sample_index(val_cases, 0.0, args.seed)
    logger.info("Training windows   %d  (%d gland-bearing, %d not)", len(train_index), n_pos, n_neg)
    logger.info("Validation windows %d  (%d gland-bearing, %d not)", len(val_index), v_pos, v_neg)

    Dataset = _make_dataset_class()
    train_ds = Dataset(train_cases, train_index, args.slice_window, args.static_repeat)
    val_ds = Dataset(val_cases, val_index, args.slice_window, args.static_repeat)
    loader_kw = dict(num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
                     persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=max(args.batch_size, 8), shuffle=False, **loader_kw)

    # ---- model, loss, optimiser ----------------------------------------- #
    model, param_counts = build_models(args, logger)
    model.to(device)

    # Positive class weights from the actual training index. A missed positive
    # slice is unrecoverable downstream, so the rarer class is weighted up.
    pos_weights = {}
    for h, head in enumerate(HEADS):
        pos = sum(int(train_cases[ci]["labels"][z, h]) for ci, z in train_index)
        neg = len(train_index) - pos
        w = min(args.max_pos_weight, neg / pos) if pos else 1.0
        pos_weights[head] = float(w)
        logger.info("  %-14s positives %6d / %6d  ->  pos_weight %.2f",
                    head, pos, len(train_index), w)

    criterion = MultiHeadFocalBCELoss(heads=HEADS, gamma=args.focal_gamma, pos_weights=pos_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 1
    stopper = EarlyStopper(patience=args.patience)
    history: list[dict] = []
    if args.resume and args.resume.exists():
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scaler_state_dict") and use_amp:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        stopper.best = float(ckpt.get("best_score", -1.0))
        stopper.best_epoch = int(ckpt.get("best_epoch", 0))
        history = list(ckpt.get("history", []))
        logger.info("Resumed from %s at epoch %d (best %.4f)", args.resume, start_epoch, stopper.best)

    config_blob = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config_blob["param_counts"] = param_counts
    config_blob["heads"] = list(HEADS)
    (run_dir / "config.json").write_text(json.dumps(config_blob, indent=2), encoding="utf-8")

    metrics = GateMetricsWriter(run_dir / "metrics.csv")
    logger.info("Artefacts in %s (open progress.txt)", run_dir)
    logger.info("-" * 78)

    def split_targets(y):
        return {head: y[:, h] for h, head in enumerate(HEADS)}

    run_started = time.perf_counter()
    degenerate_streak = 0

    for epoch in range(start_epoch, args.max_epochs + 1):
        epoch_started = time.perf_counter()
        lr = lr_at(epoch, args)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # -- train -- #
        model.train()
        train_loss, n_batches, spike_total, spike_windows = 0.0, 0, 0.0, 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if args.augment:
                with torch.no_grad():
                    x = augment_batch(x)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits, stats = model(x)
                loss, _ = criterion(logits, split_targets(y))

            scaler.scale(loss).backward()
            if args.grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            train_loss += float(loss.detach())
            n_batches += 1
            if stats.get("total_spike_count"):
                spike_total += float(stats["total_spike_count"])
                spike_windows += x.shape[0]

        train_loss /= max(n_batches, 1)

        # -- validate -- #
        model.eval()
        val_loss, v_batches = 0.0, 0
        scores = {head: [] for head in HEADS}
        truths = {head: [] for head in HEADS}
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    logits, _ = model(x)
                    loss, _ = criterion(logits, split_targets(y))
                val_loss += float(loss.detach())
                v_batches += 1
                for h, head in enumerate(HEADS):
                    scores[head].append(torch.sigmoid(logits[head].float()).cpu().numpy())
                    truths[head].append(y[:, h].cpu().numpy())
        val_loss /= max(v_batches, 1)

        scores = {h: np.concatenate(v) for h, v in scores.items()}
        truths = {h: np.concatenate(v) for h, v in truths.items()}

        prauc, thr, per_head = {}, {}, {}
        for head in HEADS:
            prauc[head] = average_precision(scores[head], truths[head])
            thr[head] = threshold_for_sensitivity(scores[head], truths[head], args.target_sensitivity)
            per_head[head] = binary_scores(scores[head], truths[head], thr[head])

        # The gate's actual decision: keep a slice if EITHER head fires. This,
        # not the per-head numbers, is what Stage B receives.
        keep = np.zeros_like(truths[HEADS[0]], dtype=bool)
        any_truth = np.zeros_like(keep)
        for head in HEADS:
            keep |= scores[head] >= thr[head]
            any_truth |= truths[head] > 0
        any_retention = (float(np.logical_and(keep, any_truth).sum()) / max(int(any_truth.sum()), 1))
        slice_reduction = 1.0 - float(keep.sum()) / max(keep.size, 1)

        finite = [prauc[h] for h in HEADS if prauc[h] == prauc[h]]
        prauc_mean = float(np.mean(finite)) if finite else float("nan")
        if args.select_by == "prauc":
            score = prauc_mean
        elif args.select_by == "f2":
            f2s = [per_head[h]["f2"] for h in HEADS if per_head[h]["f2"] == per_head[h]["f2"]]
            score = float(np.mean(f2s)) if f2s else 0.0
        else:
            score = slice_reduction
        if score != score:
            score = 0.0

        improved = stopper.update(score, epoch)
        elapsed = time.perf_counter() - epoch_started
        spikes_per_window = (spike_total / spike_windows) if spike_windows else 0.0

        logger.info(
            "epoch %3d/%d | lr %.2e | train %.4f | val %.4f | PR-AUC L %.4f R %.4f | "
            "keeps %.1f%% of slices at %.1f%% retention%s",
            epoch, args.max_epochs, lr, train_loss, val_loss,
            prauc[HEADS[0]], prauc[HEADS[1]], 100 * (1 - slice_reduction), 100 * any_retention,
            "  <- best" if improved else "")

        metrics.write({
            "epoch": epoch, "lr": f"{lr:.6e}",
            "train_loss": f"{train_loss:.6f}", "val_loss": f"{val_loss:.6f}",
            "prauc_left": f"{prauc[HEADS[0]]:.6f}", "prauc_right": f"{prauc[HEADS[1]]:.6f}",
            "prauc_mean": f"{prauc_mean:.6f}",
            "thr_left": f"{thr[HEADS[0]]:.6f}", "thr_right": f"{thr[HEADS[1]]:.6f}",
            "sens_left": f"{per_head[HEADS[0]]['sensitivity']:.6f}",
            "sens_right": f"{per_head[HEADS[1]]['sensitivity']:.6f}",
            "spec_left": f"{per_head[HEADS[0]]['specificity']:.6f}",
            "spec_right": f"{per_head[HEADS[1]]['specificity']:.6f}",
            "npv_left": f"{per_head[HEADS[0]]['npv']:.6f}",
            "npv_right": f"{per_head[HEADS[1]]['npv']:.6f}",
            "f2_left": f"{per_head[HEADS[0]]['f2']:.6f}",
            "f2_right": f"{per_head[HEADS[1]]['f2']:.6f}",
            "any_retention": f"{any_retention:.6f}",
            "slice_reduction": f"{slice_reduction:.6f}",
            "spikes_per_window": f"{spikes_per_window:.1f}",
            "epoch_seconds": f"{elapsed:.2f}", "is_best": int(improved),
        })

        history.append({"epoch": epoch, "score": score})
        epochs_done = max(1, epoch - start_epoch + 1)
        write_progress(
            progress_path, run_name=run_name,
            model_name=args.model + (" (static repeat)" if args.static_repeat else ""),
            epoch=epoch, max_epochs=args.max_epochs, history=history,
            best=stopper.best, best_epoch=stopper.best_epoch, patience=args.patience,
            select_by=args.select_by, target_sensitivity=args.target_sensitivity,
            elapsed_total=time.perf_counter() - run_started,
            avg_epoch_seconds=(time.perf_counter() - run_started) / epochs_done,
            latest={
                "prauc": prauc, "thr": thr,
                "sens": {h: per_head[h]["sensitivity"] for h in HEADS},
                "spec": {h: per_head[h]["specificity"] for h in HEADS},
                "npv": {h: per_head[h]["npv"] for h in HEADS},
                "f2": {h: per_head[h]["f2"] for h in HEADS},
                "any_retention": any_retention, "slice_reduction": slice_reduction,
                "train_loss": train_loss, "val_loss": val_loss,
                "spikes_per_window": spikes_per_window,
            },
        )

        def _save(path: Path, payload: dict) -> None:
            tmp = path.with_name(path.name + ".tmp")
            torch.save(payload, tmp)
            tmp.replace(path)          # atomic: a killed job cannot leave a torn checkpoint

        _save(run_dir / "last_model.pt", {
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if use_amp else None,
            "best_score": stopper.best, "best_epoch": stopper.best_epoch,
            "history": history, "config": config_blob,
        })
        if improved:
            _save(run_dir / "best_model.pt", {
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "score": score, "select_by": args.select_by,
                "thresholds": {h: float(thr[h]) for h in HEADS},
                "target_sensitivity": args.target_sensitivity,
                "prauc": {h: float(prauc[h]) for h in HEADS},
                "any_retention": any_retention, "slice_reduction": slice_reduction,
                "config": config_blob,
            })

        # A gate that has collapsed onto one class produces a constant score and
        # a meaningless threshold; catch it rather than burning the allocation.
        spread = min(float(np.std(scores[h])) for h in HEADS)
        degenerate_streak = degenerate_streak + 1 if spread < 1e-4 else 0
        if degenerate_streak >= 5:
            logger.error("COLLAPSED: the gate has produced an almost constant score for %d "
                         "consecutive epochs - it is predicting one class for every slice. "
                         "Lower --lr, or raise --max-pos-weight if it has collapsed onto the "
                         "negative class.", degenerate_streak)
            break

        if stopper.should_stop:
            logger.info("STOPPING: no improvement for %d epochs (best %.4f at epoch %d).",
                        args.patience, stopper.best, stopper.best_epoch)
            break

    metrics.close()
    logger.info("-" * 78)
    logger.info("Best %s %.4f at epoch %d", args.select_by, stopper.best, stopper.best_epoch)
    logger.info("Best checkpoint: %s", run_dir / "best_model.pt")
    logger.info("Compare arms by putting the two runs' best_model.pt metrics side by side; "
                "the number that matters is slice reduction at %.0f%% retention.",
                100 * args.target_sensitivity)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

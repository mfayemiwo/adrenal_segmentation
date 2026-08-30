#!/usr/bin/env python3
"""Diagnose the validation cases that score near zero - headless.

`evaluate_segmenter.py` tells you a mean of 0.75 hides a tail; this tells you
why the tail is there. It is the batch equivalent of
`notebooks/inspect_failure_cases.ipynb`: no Jupyter, no desktop session, no X
server. It writes a text report you can `cat` over SSH and a directory of PNG
panels you can copy down afterwards.

For each failing (case, gland) it separates the five explanations that a mean
cannot:

  annotation error      the mask is not on adrenal tissue, or is on the wrong side
  anatomical variant    a real gland shaped or placed unlike the training data
  outside the crop      the gland is cut off before the network ever sees it
  lateralisation        the gland was found but written to the other channel
  genuine miss          low probability everywhere near the gland

The geometry audit answers the cheapest question first - could the model have
seen the gland at all - by mapping each gland's native bounding box through the
same resample and centre crop that training applies, and counting how many
voxels survive into the cached volume.

Usage
-----
    python scripts/inspect_failures.py --checkpoint runs/run5/best_model.pt \
        --data-root ../data/amos22 --cache-dir ../cache

    # specific cases, more slices per panel
    python scripts/inspect_failures.py --checkpoint runs/run5/best_model.pt \
        --data-root ../data/amos22 --cases amos_0346 amos_0333 --n-slices 8

Runs on CPU in seconds per case; no GPU allocation needed.
"""
from __future__ import annotations

import os as _os
import tempfile as _tempfile

# MIOpen builds its kernel database on the first convolution and the default
# location is read-only on the cluster. Must precede any torch import.
for _var, _sub in (("MIOPEN_USER_DB_PATH", "miopen-db"), ("MIOPEN_CUSTOM_CACHE_DIR", "miopen-cache")):
    _os.environ.setdefault(_var, _os.path.join(_tempfile.gettempdir(), _sub))
    _os.makedirs(_os.environ[_var], exist_ok=True)

# Matplotlib must not look for a display: this script is meant for a batch job.
_os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_adrenal_segmenter import (          # noqa: E402  (path set above)
    GeometryConfig,
    LEFT_CHANNEL_VALUE,
    RIGHT_CHANNEL_VALUE,
    THRESHOLDS,
    discover_cases,
    prepare_case,
    setup_logging,
)

COLOUR = {"left": "#ffb000", "right": "#00b4d8", "gland": "#ffb000", "pred": "#ff2d95"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True, help="best_model.pt from a training run")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="the same --cache-dir used for training; rebuilt on the fly if absent")
    p.add_argument("--per-case", type=Path, default=None,
                   help="per_case.csv from evaluate_segmenter.py (default: beside the checkpoint)")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="where the report and PNGs go (default: <checkpoint dir>/inspection)")
    p.add_argument("--cases", nargs="*", default=None,
                   help="explicit case ids; default is the worst --top cases in per_case.csv")
    p.add_argument("--top", type=int, default=4, help="how many failing cases to inspect")
    p.add_argument("--fail-below", type=float, default=0.5,
                   help="a (case, gland) Dice under this counts as a failure, not variation")
    p.add_argument("--stage", choices=["raw", "postprocessed"], default="postprocessed")
    p.add_argument("--n-slices", type=int, default=6, help="axial slices per panel")
    p.add_argument("--threshold", type=float, default=None,
                   help="override the checkpoint's operating threshold")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--no-figures", action="store_true",
                   help="report only - skip the PNGs (fastest, if you only want the tables)")
    return p


class Report:
    """Accumulates the report and mirrors it to the log as it goes, so a long
    run is readable while it is still running."""

    def __init__(self, logger):
        self.lines: list[str] = []
        self.logger = logger

    def __call__(self, text: str = "") -> None:
        self.lines.append(text)
        self.logger.info(text)

    def rule(self, char: str = "-", width: int = 78) -> None:
        self(char * width)

    def write(self, path: Path) -> None:
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def dice(pred: np.ndarray, truth: np.ndarray) -> float:
    pred, truth = pred.astype(bool), truth.astype(bool)
    denom = pred.sum() + truth.sum()
    return float("nan") if denom == 0 else 2.0 * float(np.logical_and(pred, truth).sum()) / denom


def crop_bounds(native_len: int, zoom: float, size: int) -> tuple[float, float, bool]:
    """Where the centre crop lands, expressed in NATIVE voxel indices."""
    resampled = int(round(native_len * zoom))
    if resampled <= size:                      # padded rather than cropped: nothing lost
        return 0.0, float(native_len), False
    start = (resampled - size) // 2
    return start / zoom, (start + size) / zoom, True


class Inspector:
    def __init__(self, args, logger):
        import torch
        import segmentation_models_pytorch as smp

        self.args, self.logger, self.torch = args, logger, torch

        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        self.cfg = cfg = ckpt.get("config", {})
        self.threshold = (args.threshold if args.threshold is not None
                          else float(ckpt.get("threshold", ckpt.get("best_threshold", 0.5))))
        self.slice_window = int(cfg.get("slice_window", 5))
        self.combined = bool(cfg.get("combine_glands", False))
        self.channels = ("gland",) if self.combined else ("left", "right")
        self.mri_threshold = int(cfg.get("mri_id_threshold", 500))

        self.geom = GeometryConfig(
            spacing_z=float(cfg.get("target_spacing_z", 2.5)),
            spacing_xy=float(cfg.get("target_spacing_xy", 1.0)),
            image_size=int(cfg.get("image_size", 384)),
            z_margin=int(cfg.get("z_margin", 32)),
            right_label=int(cfg.get("right_label", 11)),
            left_label=int(cfg.get("left_label", 12)),
        )
        self.channel_value = {"left": LEFT_CHANNEL_VALUE, "right": RIGHT_CHANNEL_VALUE}

        disc = SimpleNamespace(
            data_root=args.data_root, seed=int(cfg.get("seed", 42)),
            val_fraction=float(cfg.get("val_fraction", 0.2)),
            max_train_cases=0, max_val_cases=0,
            modality=str(cfg.get("modality", "all")),
            mri_id_threshold=self.mri_threshold,
        )
        train_records, val_records = discover_cases(disc, logger)
        self.records = {r["case_id"]: r for r in val_records + train_records}
        self.n_val, self.n_train = len(val_records), len(train_records)

        device = args.device
        self.device = torch.device("cuda" if (device in ("auto", "cuda") and torch.cuda.is_available())
                                   else "cpu")
        self.model = smp.Unet(
            encoder_name=str(cfg.get("encoder", "resnet34")), encoder_weights=None,
            in_channels=self.slice_window, classes=len(self.channels),
            decoder_attention_type=None if cfg.get("decoder_attention") == "none" else "scse",
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device).eval()
        self.ckpt_meta = {"epoch": ckpt.get("epoch"), "val_dice": ckpt.get("val_dice")}
        self._pred_cache: dict[str, tuple[dict, np.ndarray]] = {}

    # -- data ------------------------------------------------------------- #

    def truth_mask(self, coded: np.ndarray, gland: str) -> np.ndarray:
        if self.combined or gland == "gland":
            return coded > 0
        return coded == self.channel_value[gland]

    def native_label_value(self, gland: str):
        if gland == "gland":
            return None
        return self.geom.left_label if gland == "left" else self.geom.right_label

    def gland_mask(self, label: np.ndarray, gland: str) -> np.ndarray:
        if gland == "gland":
            return (label == self.geom.left_label) | (label == self.geom.right_label)
        return label == self.native_label_value(gland)

    def native_volume(self, case_id: str):
        """Native image and label as (Z, X, Y) - the transpose prepare_case
        applies - plus (sz, sx, sy) spacing, so indices stay comparable."""
        rec = self.records[case_id]
        img_nii, lab_nii = nib.load(str(rec["image_path"])), nib.load(str(rec["label_path"]))
        image = np.transpose(np.asarray(img_nii.dataobj, dtype=np.float32), (2, 0, 1))
        label = np.transpose(np.asarray(lab_nii.dataobj, dtype=np.int16), (2, 0, 1))
        sx, sy, sz = (float(v) for v in img_nii.header.get_zooms()[:3])
        return image, label, (sz, sx, sy)

    def load_cached(self, case_id: str):
        """The volume the model actually saw: from the training cache when
        present, otherwise rebuilt with identical preprocessing."""
        if self.args.cache_dir is not None:
            path = self.args.cache_dir / self.geom.cache_key() / f"{case_id}.npz"
            if path.exists():
                with np.load(path) as z:
                    return {"case_id": case_id, "image": z["image"], "mask": z["mask"],
                            "source": "cache"}
        case = prepare_case(self.records[case_id], self.geom)
        if case is not None:
            case["source"] = "rebuilt"
        return case

    def is_mri(self, case_id: str) -> bool:
        digits = "".join(c for c in case_id if c.isdigit())
        return bool(digits) and int(digits) >= self.mri_threshold

    def display_window(self, image: np.ndarray, case_id: str) -> np.ndarray:
        """HU window for CT; MRI has no HU scale, so use a percentile window."""
        if self.is_mri(case_id):
            lo, hi = np.percentile(image, [1, 99])
        else:
            lo, hi = self.geom.hu_window
        return (np.clip(image, lo, hi) - lo) / max(hi - lo, 1e-6)

    # -- analysis --------------------------------------------------------- #

    def audit(self, case_id: str, glands) -> dict:
        image, label, (sz, sx, sy) = self.native_volume(case_id)
        zoom = (sz / self.geom.spacing_z, sx / self.geom.spacing_xy, sy / self.geom.spacing_xy)
        voxel_mm3 = sx * sy * sz
        cached = self.load_cached(case_id)

        out = {"case_id": case_id, "shape": image.shape, "spacing": (sz, sx, sy),
               "zoom": zoom, "cached": cached is not None, "glands": {}}
        for gland in glands:
            m = self.gland_mask(label, gland)
            n = int(m.sum())
            entry = {"native_voxels": n, "mm3": n * voxel_mm3}
            if n:
                zs = np.flatnonzero(m.any(axis=(1, 2)))
                xs = np.flatnonzero(m.any(axis=(0, 2)))
                ys = np.flatnonzero(m.any(axis=(0, 1)))
                entry["z_range"] = (int(zs.min()), int(zs.max()))
                entry["z_extent_mm"] = float((zs.max() - zs.min() + 1) * sz)
                entry["bbox_x"] = (int(xs.min()), int(xs.max()))
                entry["bbox_y"] = (int(ys.min()), int(ys.max()))

                inside = True
                for ax, key in ((1, "bbox_x"), (2, "bbox_y")):
                    lo, hi = entry[key]
                    c0, c1, cropped = crop_bounds(image.shape[ax], zoom[ax], self.geom.image_size)
                    if cropped and (lo < c0 or hi > c1):
                        inside = False
                    entry["crop_" + ("x" if ax == 1 else "y")] = (round(c0), round(c1))
                entry["inside_crop"] = inside

                if cached is not None:
                    kept = int(self.truth_mask(cached["mask"], gland).sum())
                    expected = n * zoom[0] * zoom[1] * zoom[2]
                    entry["cached_voxels"] = kept
                    entry["retention"] = kept / expected if expected else float("nan")
            out["glands"][gland] = entry
        return out

    def probabilities(self, case_id: str):
        if case_id not in self._pred_cache:
            case = self.load_cached(case_id)
            if case is None:
                self._pred_cache[case_id] = (None, None)
            else:
                self._pred_cache[case_id] = (case, self._predict(case["image"].astype(np.float32)))
        return self._pred_cache[case_id]

    def _predict(self, image: np.ndarray) -> np.ndarray:
        torch = self.torch
        half, n = self.slice_window // 2, image.shape[0]
        out = np.zeros((n, len(self.channels)) + image.shape[1:], dtype=np.float32)
        with torch.no_grad():
            for start in range(0, n, self.args.batch_size):
                centres = range(start, min(start + self.args.batch_size, n))
                windows = np.stack([image[np.clip(np.arange(c - half, c + half + 1), 0, n - 1)]
                                    for c in centres])
                batch = torch.from_numpy(windows.astype(np.float32)).to(self.device)
                out[start:start + len(windows)] = torch.sigmoid(self.model(batch)).float().cpu().numpy()
        return out

    def diagnose(self, case_id: str) -> list[dict]:
        case, probs = self.probabilities(case_id)
        if case is None:
            return []
        coded = case["mask"]
        spacing = np.array([self.geom.spacing_z, self.geom.spacing_xy, self.geom.spacing_xy])
        rows = []
        for c, gland in enumerate(self.channels):
            truth = self.truth_mask(coded, gland)
            p = probs[:, c]
            pred = p >= self.threshold
            sweep = np.array([dice(p >= t, truth) for t in THRESHOLDS], dtype=float)
            best_i = int(np.nanargmax(sweep)) if np.any(~np.isnan(sweep)) else 0

            row = {
                "gland": gland,
                "truth_voxels": int(truth.sum()),
                "pred_voxels": int(pred.sum()),
                "dice": dice(pred, truth),
                "best_dice": float(sweep[best_i]),
                "best_threshold": float(THRESHOLDS[best_i]),
                "max_p_in_truth": float(p[truth].max()) if truth.any() else float("nan"),
                "mean_p_in_truth": float(p[truth].mean()) if truth.any() else float("nan"),
                "max_p_anywhere": float(p.max()),
                "swapped_dice": float("nan"),
                "centroid_mm": float("nan"),
            }
            if not self.combined and gland in ("left", "right"):
                other = self.truth_mask(coded, "right" if gland == "left" else "left")
                if other.any():
                    row["swapped_dice"] = dice(pred, other)
            if pred.any() and truth.any():
                cp = np.argwhere(pred).mean(axis=0) * spacing
                ct = np.argwhere(truth).mean(axis=0) * spacing
                row["centroid_mm"] = float(np.linalg.norm(cp - ct))
            rows.append(row)
        return rows

    @staticmethod
    def verdict(row: dict, audit_entry: dict, fail_below: float) -> str:
        """A first-pass label. The images decide; this narrows where to look."""
        # Order matters. The geometry verdicts must come before anything derived
        # from the cached volume: a gland cropped out of the field of view has
        # ZERO voxels in the model's input, which would otherwise be misread as
        # "this scan has no label for that side".
        audit_entry = audit_entry or {}
        native = audit_entry.get("native_voxels")
        if native is not None and not native:
            return "no label for this gland in this scan"

        cropped = not audit_entry.get("inside_crop", True)
        ret = audit_entry.get("retention")
        lost = ret is not None and ret == ret and ret < 0.5

        if row["dice"] >= fail_below:
            return "ok" + ("  (note: partly outside the crop)" if cropped else "")
        if cropped:
            return "OUTSIDE CROP - gland is cut off before the network ever sees it"
        if lost:
            return f"LOST IN RESAMPLING - only {ret:.0%} survives; try a finer --target-spacing-z"
        if not row["truth_voxels"]:
            return ("GONE FROM THE MODEL'S INPUT - the native label exists but nothing "
                    "survives preprocessing")
        if row["swapped_dice"] == row["swapped_dice"] and row["swapped_dice"] > 0.3:
            return "LATERALISATION - found the gland, wrote it to the other channel"
        if not row["pred_voxels"]:
            return "SILENT - predicted nothing anywhere in this scan"
        if row["max_p_in_truth"] < 0.05:
            return "GENUINE MISS - almost no response inside the true gland"
        if row["best_dice"] >= fail_below:
            return f"THRESHOLD - {row['best_dice']:.3f} reachable at {row['best_threshold']:.2f}"
        if row["pred_voxels"] > 10 * max(row["truth_voxels"], 1):
            return (f"OVER-SEGMENTING - {row['pred_voxels']} predicted voxels against "
                    f"{row['truth_voxels']} true; the threshold is far too low for this case")
        if row["centroid_mm"] == row["centroid_mm"] and row["centroid_mm"] > 40:
            return "WRONG STRUCTURE - prediction is far from the gland"
        if row["truth_voxels"] < 400:
            return "very small gland - a few boundary voxels dominate the score"
        return "partial overlap - inspect the images"

    # -- figures ---------------------------------------------------------- #

    @staticmethod
    def _overlay(ax, base, masks, box=None):
        ax.imshow(base, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        for m, colour, alpha in masks:
            if m is None or not m.any():
                continue
            rgba = np.zeros(m.shape + (4,), dtype=np.float32)
            rgb = tuple(int(colour[i:i + 2], 16) / 255 for i in (1, 3, 5))
            rgba[m] = (*rgb, alpha)
            ax.imshow(rgba, interpolation="nearest")
        if box:
            x0, x1, y0, y1 = box
            ax.set_xlim(y0, y1)
            ax.set_ylim(x1, x0)
        ax.set_xticks([])
        ax.set_yticks([])

    def figure_native(self, case_id: str, gland: str, entry: dict, out_path: Path) -> None:
        image, label, _ = self.native_volume(case_id)
        base = self.display_window(image, case_id)
        z0, z1 = entry["z_range"]
        zs = np.unique(np.linspace(z0, z1, self.args.n_slices).round().astype(int))
        pad = 45
        box = (max(0, entry["bbox_x"][0] - pad), min(image.shape[1], entry["bbox_x"][1] + pad),
               max(0, entry["bbox_y"][0] - pad), min(image.shape[2], entry["bbox_y"][1] + pad))

        fig, axes = plt.subplots(2, len(zs), figsize=(2.1 * len(zs), 5.0))
        axes = axes.reshape(2, -1)
        masks = [(label == self.geom.left_label, COLOUR["left"], 0.75),
                 (label == self.geom.right_label, COLOUR["right"], 0.75)]
        for j, z in enumerate(zs):
            slice_masks = [(m[z], c, a) for m, c, a in masks]
            self._overlay(axes[0, j], base[z], slice_masks, box=box)
            self._overlay(axes[1, j], base[z], slice_masks)
            axes[0, j].set_title(f"z={z}")
        axes[0, 0].set_ylabel("zoomed", fontsize=8)
        axes[1, 0].set_ylabel("whole slice", fontsize=8)
        fig.suptitle(f"{case_id} - {gland} gland, ground truth on the native scan - "
                     f"{entry['native_voxels']} voxels, {entry['z_extent_mm']:.0f} mm "
                     f"(amber = left / label {self.geom.left_label}, "
                     f"cyan = right / label {self.geom.right_label})", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight", dpi=120)
        plt.close(fig)

    def figure_prediction(self, case_id: str, gland: str, row: dict, out_path: Path) -> bool:
        case, probs = self.probabilities(case_id)
        if case is None:
            return False
        c = self.channels.index(gland)
        coded, image = case["mask"], case["image"].astype(np.float32)
        truth = self.truth_mask(coded, gland)
        if not truth.any():
            return False
        lo, hi = np.percentile(image, [1, 99])
        base = (np.clip(image, lo, hi) - lo) / max(hi - lo, 1e-6)
        p = probs[:, c]
        pred = p >= self.threshold

        zs_present = np.flatnonzero(truth.any(axis=(1, 2)))
        zs = np.unique(np.linspace(zs_present.min(), zs_present.max(),
                                   self.args.n_slices).round().astype(int))
        pad = 45
        xs, ys = np.flatnonzero(truth.any(axis=(0, 2))), np.flatnonzero(truth.any(axis=(0, 1)))
        x0, x1 = max(0, xs.min() - pad), min(image.shape[1], xs.max() + pad)
        y0, y1 = max(0, ys.min() - pad), min(image.shape[2], ys.max() + pad)
        box = (x0, x1, y0, y1)

        fig, axes = plt.subplots(3, len(zs), figsize=(2.1 * len(zs), 6.6))
        axes = axes.reshape(3, -1)
        for j, z in enumerate(zs):
            self._overlay(axes[0, j], base[z], [(truth[z], COLOUR[gland], 0.75)], box=box)
            self._overlay(axes[1, j], base[z], [(truth[z], COLOUR[gland], 0.45),
                                                (pred[z], COLOUR["pred"], 0.55)], box=box)
            heat = np.ma.masked_less(p[z], 0.02)
            axes[2, j].imshow(base[z], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            axes[2, j].imshow(heat, cmap="inferno", vmin=0, vmax=1, alpha=0.75,
                              interpolation="nearest")
            axes[2, j].set_xlim(y0, y1)
            axes[2, j].set_ylim(x1, x0)
            axes[2, j].set_xticks([])
            axes[2, j].set_yticks([])
            axes[0, j].set_title(f"z={z}   max p over slice {p[z].max():.2f}")
        for r, name in enumerate(("truth", "truth + prediction", "probability")):
            axes[r, 0].set_ylabel(name, fontsize=8)
        fig.suptitle(f"{case_id} - {gland} gland - Dice {row['dice']:.3f} at threshold "
                     f"{self.threshold:.2f} - max probability inside the gland "
                     f"{row['max_p_in_truth']:.3f}", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        return True

    def figure_coronal(self, case_id: str, gland: str, out_path: Path) -> bool:
        case, probs = self.probabilities(case_id)
        if case is None:
            return False
        c = self.channels.index(gland)
        coded, image = case["mask"], case["image"].astype(np.float32)
        truth = self.truth_mask(coded, gland)
        if not truth.any():
            return False
        lo, hi = np.percentile(image, [1, 99])
        base = (np.clip(image, lo, hi) - lo) / max(hi - lo, 1e-6)
        pred = probs[:, c] >= self.threshold
        aspect = self.geom.spacing_z / self.geom.spacing_xy
        x = int(round(np.argwhere(truth)[:, 1].mean()))
        ys = np.flatnonzero(truth.any(axis=(0, 1)))
        y0, y1 = max(0, ys.min() - 60), min(image.shape[2], ys.max() + 60)

        fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
        panels = ((axes[0], [(truth[:, x], COLOUR[gland], 0.7)], "truth"),
                  (axes[1], [(truth[:, x], COLOUR[gland], 0.4), (pred[:, x], COLOUR["pred"], 0.55)],
                   "truth + prediction"))
        for ax, masks, title in panels:
            ax.imshow(base[:, x], cmap="gray", vmin=0, vmax=1, aspect=aspect,
                      interpolation="nearest")
            for m, colour, alpha in masks:
                rgba = np.zeros(m.shape + (4,), dtype=np.float32)
                rgb = tuple(int(colour[i:i + 2], 16) / 255 for i in (1, 3, 5))
                rgba[m] = (*rgb, alpha)
                ax.imshow(rgba, aspect=aspect, interpolation="nearest")
            ax.set_xlim(y0, y1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title)
        fig.suptitle(f"{case_id} - {gland} gland, coronal reslice at x={x}", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        return True


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = args.output_dir or (args.checkpoint.parent / "inspection")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    if not args.no_figures:
        fig_dir.mkdir(exist_ok=True)

    logger = setup_logging(out_dir / "inspect.log")
    R = Report(logger)

    per_case_path = args.per_case or (args.checkpoint.parent / "per_case.csv")
    if not per_case_path.exists():
        logger.error("No per_case.csv at %s. Run scripts/evaluate_segmenter.py first, or pass "
                     "--per-case.", per_case_path)
        return 2
    with per_case_path.open(newline="", encoding="utf-8") as fh:
        per_case = list(csv.DictReader(fh))
    if not per_case:
        logger.error("%s is empty.", per_case_path)
        return 2

    stage = args.stage
    glands = [g for g in ("left", "right", "gland") if f"dice_{g}_{stage}" in per_case[0]]
    if not glands:
        logger.error("No dice_*_%s columns in %s.", stage, per_case_path)
        return 2
    by_case = {r["case_id"]: r for r in per_case}

    inspector = Inspector(args, logger)

    R.rule("=")
    R("FAILURE CASE INSPECTION")
    R.rule("=")
    R(f"Checkpoint      {args.checkpoint}  (epoch {inspector.ckpt_meta['epoch']}, "
      f"val dice {inspector.ckpt_meta['val_dice']})")
    R(f"Per-case table  {per_case_path}  ({len(per_case)} cases, stage '{stage}')")
    R(f"Device          {inspector.device}")
    R(f"Threshold       {inspector.threshold:.2f}"
      + ("  (overridden on the command line)" if args.threshold is not None else "  (from checkpoint)"))
    R(f"Channels        {', '.join(inspector.channels)} | slice window {inspector.slice_window} "
      f"| encoder {inspector.cfg.get('encoder')}")
    R(f"Geometry        {inspector.geom.spacing_xy} mm in-plane / {inspector.geom.spacing_z} mm z, "
      f"{inspector.geom.image_size}px centre crop, labels R={inspector.geom.right_label} "
      f"L={inspector.geom.left_label}")
    R(f"Split resolved  {inspector.n_val} validation / {inspector.n_train} training cases "
      f"(modality '{inspector.cfg.get('modality', 'all')}')")
    R()

    # -- 1. rank ---------------------------------------------------------- #
    pairs = []
    for r in per_case:
        for g in glands:
            pairs.append({"case_id": r["case_id"], "gland": g,
                          "dice": float(r[f"dice_{g}_{stage}"]),
                          "truth_voxels": int(r.get(f"truth_voxels_{g}", 0) or 0)})
    pairs.sort(key=lambda d: d["dice"])
    failures = [p for p in pairs if p["dice"] < args.fail_below]
    scores = np.array([p["dice"] for p in pairs])

    R.rule("=")
    R("1. WHERE THE DEFICIT LIVES")
    R.rule("=")
    R(f"mean {scores.mean():.4f}   median {np.median(scores):.4f}   min {scores.min():.4f}   "
      f"max {scores.max():.4f}")
    R(f"{len(failures)} of {len(pairs)} (case, gland) pairs below {args.fail_below:.2f} "
      f"({100 * len(failures) / len(pairs):.0f}%)")
    R()
    for g in glands:
        vals = np.array([p["dice"] for p in pairs if p["gland"] == g])
        n_fail = int((vals < args.fail_below).sum())
        fixed = np.where(vals < args.fail_below, np.median(vals), vals)
        R(f"  {g:<6} mean {vals.mean():.4f}  ->  {fixed.mean():.4f} if its {n_fail} failing "
          f"case(s) merely reached the median")
    R()
    R(f"  {'case':<12} {'gland':<6} {'dice':>7} {'truth vox':>10}")
    R("  " + "-" * 38)
    for p in pairs[:15]:
        R(f"  {p['case_id']:<12} {p['gland']:<6} {p['dice']:>7.3f} {p['truth_voxels']:>10d}")
    R()

    # -- pick cases ------------------------------------------------------- #
    if args.cases:
        selected = list(args.cases)
    else:
        selected = []
        for p in failures:
            if p["case_id"] not in selected:
                selected.append(p["case_id"])
            if len(selected) >= args.top:
                break
    missing = [c for c in selected if c not in inspector.records]
    selected = [c for c in selected if c in inspector.records]
    if missing:
        R(f"NOT FOUND in {args.data_root}: {', '.join(missing)}")
    if not selected:
        R("Nothing to inspect: no case fell below the failure threshold.")
        R.write(out_dir / "inspection_report.txt")
        logger.info("Report: %s", out_dir / "inspection_report.txt")
        return 0
    R(f"Inspecting {len(selected)} case(s): {', '.join(selected)}")
    R()

    # -- 2. geometry ------------------------------------------------------ #
    R.rule("=")
    R("2. GEOMETRY AUDIT - could the model have seen the gland at all?")
    R.rule("=")
    R("'in crop' maps the gland's native bounding box through the same resample and centre")
    R("crop training applies. 'kept' is the fraction of the gland that survives into the")
    R("model's input. Either one failing is a preprocessing bug, not a modelling problem.")
    R()
    R(f"  {'case':<12} {'gland':<6} {'dice':>6} {'native vox':>11} {'mm^3':>9} "
      f"{'z slices':>9} {'z mm':>7} {'in crop':>8} {'kept':>7}")
    R("  " + "-" * 82)
    audits = {}
    for cid in selected:
        audits[cid] = inspector.audit(cid, glands)
        a = audits[cid]
        for g in glands:
            e = a["glands"][g]
            d = float(by_case[cid][f"dice_{g}_{stage}"]) if cid in by_case else float("nan")
            if not e["native_voxels"]:
                R(f"  {cid:<12} {g:<6} {d:>6.3f}   (no {g} label in this scan)")
                continue
            ret = e.get("retention")
            ret_s = "-" if ret is None else f"{ret:.0%}"
            R(f"  {cid:<12} {g:<6} {d:>6.3f} {e['native_voxels']:>11d} {e['mm3']:>9.0f} "
              f"{e['z_range'][1] - e['z_range'][0] + 1:>9d} {e['z_extent_mm']:>7.1f} "
              f"{'yes' if e['inside_crop'] else 'NO':>8} {ret_s:>7}")
        R(f"  {'':<12} shape {a['shape']}  spacing (z,x,y) "
          f"({a['spacing'][0]:.2f}, {a['spacing'][1]:.2f}, {a['spacing'][2]:.2f}) mm  "
          f"[{'cached' if a['cached'] else 'PREPROCESSING RETURNED NOTHING'}]")
    R()

    # -- 3. diagnosis ----------------------------------------------------- #
    R.rule("=")
    R("3. WHAT THE MODEL DID")
    R.rule("=")
    R("max p in truth  highest probability anywhere inside the true gland")
    R("best dice       best reachable over the whole threshold sweep")
    R("swap            prediction scored against the OTHER gland - high means lateralisation")
    R("centroid mm     distance from the predicted centre to the true centre")
    R()
    R(f"  {'case':<12} {'gland':<6} {'dice':>6} {'best':>6} {'@thr':>5} {'truth':>7} {'pred':>8} "
      f"{'max p':>7} {'mean p':>7} {'swap':>6} {'cent mm':>8}")
    R("  " + "-" * 92)
    diagnoses = {}
    for cid in selected:
        diagnoses[cid] = inspector.diagnose(cid)
        for row in diagnoses[cid]:
            R(f"  {cid:<12} {row['gland']:<6} {row['dice']:>6.3f} {row['best_dice']:>6.3f} "
              f"{row['best_threshold']:>5.2f} {row['truth_voxels']:>7d} {row['pred_voxels']:>8d} "
              f"{row['max_p_in_truth']:>7.3f} {row['mean_p_in_truth']:>7.3f} "
              f"{row['swapped_dice']:>6.3f} {row['centroid_mm']:>8.1f}")
    R()

    # -- 4. verdicts ------------------------------------------------------ #
    R.rule("=")
    R("4. FIRST-PASS VERDICTS")
    R.rule("=")
    R("Automatic, and only as good as the rules behind it - the figures decide. Use these")
    R("to choose which figure to open first.")
    R()
    for cid in selected:
        for row in diagnoses[cid]:
            e = audits[cid]["glands"].get(row["gland"], {})
            R(f"  {cid:<12} {row['gland']:<6} {row['dice']:>6.3f}  "
              f"{inspector.verdict(row, e, args.fail_below)}")
    R()

    # -- 5. figures ------------------------------------------------------- #
    written = []
    if not args.no_figures:
        R.rule("=")
        R("5. FIGURES")
        R.rule("=")
        for cid in selected:
            failing = [r["gland"] for r in diagnoses[cid] if r["dice"] < args.fail_below]
            targets = failing or [r["gland"] for r in diagnoses[cid]]
            for row in diagnoses[cid]:
                g = row["gland"]
                if g not in targets:
                    continue
                e = audits[cid]["glands"].get(g, {})
                try:
                    if e.get("native_voxels"):
                        p = fig_dir / f"{cid}_{g}_1_truth_native.png"
                        inspector.figure_native(cid, g, e, p)
                        written.append(p)
                    p = fig_dir / f"{cid}_{g}_2_prediction.png"
                    if inspector.figure_prediction(cid, g, row, p):
                        written.append(p)
                    p = fig_dir / f"{cid}_{g}_3_coronal.png"
                    if inspector.figure_coronal(cid, g, p):
                        written.append(p)
                except Exception as exc:                       # a bad case must not kill the run
                    logger.warning("Figures for %s %s failed: %s", cid, g, exc)

        # A good case for comparison - a failure only means something against one.
        best = max(per_case, key=lambda r: float(np.mean([float(r[f"dice_{g}_{stage}"])
                                                          for g in glands])))
        if best["case_id"] in inspector.records:
            for row in inspector.diagnose(best["case_id"]):
                p = fig_dir / f"REFERENCE_{best['case_id']}_{row['gland']}_prediction.png"
                try:
                    if inspector.figure_prediction(best["case_id"], row["gland"], row, p):
                        written.append(p)
                except Exception as exc:
                    logger.warning("Reference figure failed: %s", exc)
            R(f"  reference (best) case: {best['case_id']}")
        for p in written:
            R(f"  {p}")
        R()

    R.rule("=")
    R("6. WHAT EACH VERDICT MEANS FOR THE NEXT RUN")
    R.rule("=")
    R("  OUTSIDE CROP      raise --image-size, or crop around the body rather than the image")
    R("                    centre. No amount of training recovers a gland that was deleted.")
    R("  LOST IN RESAMPLING  --target-spacing-z 1.5 (already the planned run 6).")
    R("  LATERALISATION    the two sigmoid channels are being confused; a softmax over")
    R("                    {background, left, right} makes them mutually exclusive.")
    R("  THRESHOLD         calibrate per modality, or per case, rather than one global cut.")
    R("  GENUINE MISS      open the native figure: if the mask is not on adrenal tissue this")
    R("                    is an annotation error and the label is the ceiling, not the model.")
    R("  OVER-SEGMENTING   the model floods the slice at this threshold; usually pairs with a")
    R("                    THRESHOLD verdict elsewhere and points at poor calibration.")
    R("  SILENT            no prediction at all; check the geometry rows above first.")
    R("  GONE FROM THE MODEL'S INPUT  the label exists in the scan but not in the network's")
    R("                    input - a preprocessing bug, and the most important one to fix.")
    R("  WRONG STRUCTURE   add a position prior to post-processing (reject components on the")
    R("                    wrong side of the midline or far from the kidney).")
    R("  very small gland  expected; report it rather than tuning for it.")
    R()

    report_path = out_dir / "inspection_report.txt"
    R.write(report_path)
    logger.info("Report:  %s", report_path)
    if not args.no_figures:
        logger.info("Figures: %s (%d files)", fig_dir, len(written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

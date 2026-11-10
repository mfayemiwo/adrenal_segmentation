#!/usr/bin/env python3
"""Render blinded human-review adrenal CT montages from a pre-registered QC plan.

This visualises ONLY source CT and reference adrenal labels; NO model prediction
or Dice score enters the clinical/annotation-quality review. Axes are nibabel's
RAS+ canonical world orientation; display uses image-array axis conventions and
must not be mistaken for a diagnostic PACS viewport. Generated images stay on
the institutional HPC storage and must never be committed to public GitHub.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import nibabel as nib
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_case(row):
    files = [Path(row[key]) for key in
             ("image_path", "left_label_path", "right_label_path")]
    ct, left, right = [nib.load(str(path)) for path in files]
    for name, label in (("left", left), ("right", right)):
        if (label.shape != ct.shape or not np.allclose(
                label.affine, ct.affine, atol=1e-3, rtol=0)):
            raise ValueError(f"{row['case_id']}: {name} mask misaligned with CT")
    c, l, r = (nib.as_closest_canonical(n) for n in (ct, left, right))
    data = np.asarray(c.dataobj, dtype=np.float32)
    masks = [np.asarray(x.dataobj, dtype=np.uint8) > 0 for x in (l, r)]
    if np.logical_and(masks[0], masks[1]).any():
        raise ValueError(f"{row['case_id']}: left and right reference masks overlap")
    if not masks[0].any() or not masks[1].any():
        raise ValueError(f"{row['case_id']}: reference gland missing")
    return data, masks


def middle_idx(mask, axis):
    # Median of all gland voxels gives a representative central region.
    return int(np.median(np.nonzero(mask)[axis]))


def plane(volume, axis, idx):
    return np.take(volume, idx, axis=axis).T


def draw(ax, volume, masks, axis, idx, title, crop=None):
    img = plane(volume, axis, idx)
    ax.imshow(img, origin="lower", interpolation="nearest",
              cmap="gray", vmin=-160, vmax=250)
    for mask, color in zip(masks, ("#00d7c2", "#ff9d2e")):
        overlay = plane(mask, axis, idx).astype(float)
        masked = np.ma.masked_where(overlay < 0.5, overlay)
        ax.imshow(masked, origin="lower", interpolation="nearest",
                  cmap=matplotlib.colors.ListedColormap([color]),
                  alpha=0.64, vmin=0, vmax=1)
    if crop:
        center_i, center_j, radius = crop
        # Plane T reverses displayed axes, so first display coordinate is
        # the 1st in-plane original axis, then the 2nd.
        ax.set_xlim(center_i - radius, center_i + radius)
        ax.set_ylim(center_j - radius, center_j + radius)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def in_plane_center(mask, axis):
    med = [int(np.median(v)) for v in np.nonzero(mask)]
    return tuple(med[a] for a in range(3) if a != axis)


def montage(row, out_dir: Path, zoom_radius=48):
    ct, masks = load_case(row)
    left, right = masks
    zl, zr = middle_idx(left, 2), middle_idx(right, 2)
    mid_y = (middle_idx(left, 1) + middle_idx(right, 1)) // 2
    fig, axs = plt.subplots(3, 3, figsize=(15, 15), constrained_layout=True)
    draw(axs[0, 0], ct, masks, 2, zl, "Full axial at left adrenal z")
    draw(axs[0, 1], ct, masks, 2, zr, "Full axial at right adrenal z")
    draw(axs[0, 2], ct, masks, 1, mid_y, "Full coronal context")
    for row_idx, (side, mask) in enumerate(
        (("Left adrenal label", left), ("Right adrenal label", right)), start=1
    ):
        for col, (axis, title) in enumerate(((2, "Axial"), (1, "Coronal"), (0, "Sagittal"))):
            idx = middle_idx(mask, axis)
            crop = (*in_plane_center(mask, axis), zoom_radius)
            draw(axs[row_idx, col], ct, masks, axis, idx,
                 f"{side} — {title} (zoom)", crop=crop)
    fig.suptitle(
        f"Reference-only annotation QC • {row['case_id']} • official split={row['split']}\n"
        f"Cyan=left, orange=right • RAS+ canonical; x increases toward patient's right; "
        f"original thickness={row.get('original_slice_thickness_mm', '?')} mm",
        fontsize=11,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{row['case_id']}_reference_qc.png"
    fig.savefig(target, dpi=130)
    plt.close(fig)
    return target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--zoom-radius", type=int, default=48)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    if a.zoom_radius < 16 or a.limit < 0:
        p.error("zoom-radius must be >=16; limit must be >=0")
    with a.plan.open(newline="", encoding="utf-8") as fp:
        cases = list(csv.DictReader(fp))
    if not cases:
        raise ValueError("Empty review sample")
    if a.limit:
        cases = cases[:a.limit]
    for i, row in enumerate(cases, 1):
        target = a.out_dir / f"{row['case_id']}_reference_qc.png"
        if a.resume and target.is_file() and target.stat().st_size > 0:
            print(f"{i}/{len(cases)} reused {target}", flush=True)
            continue
        print(f"{i}/{len(cases)} rendering {row['case_id']}", flush=True)
        print(montage(row, a.out_dir, zoom_radius=a.zoom_radius), flush=True)
    print(f"SUCCESS: {len(cases)} source-annotation review montages.", flush=True)


if __name__ == "__main__":
    main()

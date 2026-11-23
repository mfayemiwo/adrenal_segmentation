#!/usr/bin/env python3
"""Probability-level postprocessing, chosen on the 158 DEVELOPMENT cases only.

Mask-only postprocessing (component removal) can only delete voxels, and the
failures on this model are missing voxels. These rules can ADD voxels:

  threshold t : a voxel is gland c if p_c >= t and p_c is the larger of the two
                gland probabilities (t = 0.5 reproduces argmax closely, not exactly)
  rescue r    : if a gland is still empty, keep the connected component that
                contains that gland's maximum probability, thresholded at r

Every combination is scored on dev158; the best is written to a frozen JSON
together with its dev score and the argmax baseline. Apply that JSON unchanged
to amos100 / ts76 later - never re-select on a test cohort.

Probabilities come from nnU-Net --save_probabilities (.npz, array order of the
original image as read by SimpleITK: z, y, x). They are checked against the
nnU-Net segmentation written alongside, so an orientation mistake stops the run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

THRESHOLDS = (0.5, 0.4, 0.3, 0.2, 0.15, 0.1)
RESCUE = (None, 0.3, 0.2, 0.1, 0.05)
SMALL_MM3 = 1500.0
MIN_GAIN = 0.002      # predeclared: smaller dev gains are treated as no gain


def dice(a, b):
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2.0 * np.logical_and(a, b).sum() / s


def segment(prob, t, r):
    p1, p2 = prob[1], prob[2]
    out = np.zeros(p1.shape, np.uint8)
    out[(p1 >= t) & (p1 >= p2)] = 1
    out[(p2 >= t) & (p2 > p1)] = 2
    if r is not None:
        for c, pc in ((1, p1), (2, p2)):
            if not (out == c).any():
                mask = pc >= r
                if mask.any():
                    lab, _ = ndimage.label(mask)
                    peak = lab[np.unravel_index(np.argmax(np.where(mask, pc, -1)), pc.shape)]
                    out[(lab == peak) & (out == 0)] = c
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred-dir", type=Path, required=True, help="dev158 folder with .npz + .nii.gz")
    ap.add_argument("--labels", type=Path, required=True, help="dev158 labelsTs")
    ap.add_argument("--output", type=Path, required=True, help="frozen choice JSON")
    a = ap.parse_args()
    cases = sorted(x.name[:-7] for x in a.labels.glob("*.nii.gz"))
    if len(cases) != 158:
        raise SystemExit(f"expected the 158 development cases, found {len(cases)}")
    grid = [(t, r) for t in THRESHOLDS for r in RESCUE]
    scores = {g: [] for g in grid}
    small = []
    for i, cid in enumerate(cases, 1):
        ref_img = nib.load(str(a.labels / f"{cid}.nii.gz"))
        ref = np.asarray(ref_img.dataobj).astype(np.uint8)
        vox = abs(np.linalg.det(ref_img.affine[:3, :3]))
        small.append(min((ref == 1).sum(), (ref == 2).sum()) * vox < SMALL_MM3)
        raw = np.load(a.pred_dir / f"{cid}.npz")["probabilities"].astype(np.float32)
        nn_seg = np.asarray(nib.load(str(a.pred_dir / f"{cid}.nii.gz")).dataobj).astype(np.uint8)
        if nn_seg.shape != ref.shape:
            raise SystemExit(f"{cid}: segmentation shape {nn_seg.shape} != reference {ref.shape}")
        # SimpleITK order (c, z, y, x) -> nibabel (c, x, y, z); accept whichever
        # orientation reproduces nnU-Net's own segmentation exactly.
        prob = None
        for cand in (raw.transpose(0, 3, 2, 1), raw):
            if cand.shape[1:] == ref.shape and \
                    (np.argmax(cand, axis=0).astype(np.uint8) == nn_seg).mean() >= 0.999:
                prob = cand
                break
        if prob is None:
            raise SystemExit(f"{cid}: probabilities do not reproduce the nnU-Net segmentation "
                             f"in any orientation - stopping")
        base = (dice(nn_seg == 1, ref == 1) + dice(nn_seg == 2, ref == 2)) / 2
        scores.setdefault("argmax", []).append(base)
        for t, r in grid:
            s = segment(prob, t, r)
            scores[(t, r)].append((dice(s == 1, ref == 1) + dice(s == 2, ref == 2)) / 2)
        if i % 20 == 0:
            print(f"  {i}/158", flush=True)
    small = np.array(small)
    base = np.array(scores.pop("argmax"))
    print(f"\nargmax (nnU-Net output): mean {base.mean():.4f} | small-gland {base[small].mean():.4f} (n={small.sum()})")
    print(f"{'threshold':>9s} {'rescue':>7s} {'mean':>7s} {'delta':>7s} {'small':>7s} {'better':>6s} {'worse':>6s}")
    ranked = []
    for (t, r), v in scores.items():
        v = np.array(v); d = v - base
        ranked.append((v.mean(), t, r))
        print(f"{t:9.2f} {str(r):>7s} {v.mean():7.4f} {d.mean():+7.4f} {v[small].mean():7.4f} "
              f"{int((d > 1e-9).sum()):6d} {int((d < -1e-9).sum()):6d}")
    # Ties go to the least aggressive rule: highest threshold, then no rescue.
    best_mean, t, r = max(ranked, key=lambda x: (round(x[0], 6), x[1], 1.0 if x[2] is None else x[2]))
    if best_mean - base.mean() < MIN_GAIN:
        print(f"\nNo rule beats argmax by >= {MIN_GAIN} on dev158: keep nnU-Net output unchanged.")
        t, r, best_mean = None, None, float(base.mean())
    choice = {"threshold": t, "rescue": r, "dev158_mean": best_mean, "dev158_argmax_mean": float(base.mean()),
              "selected_on": "dev158 only", "source_predictions": str(a.pred_dir)}
    a.output.write_text(json.dumps(choice, indent=2) + "\n")
    print(f"\nCHOSEN on dev158: threshold {t}, rescue {r}: {best_mean:.4f} vs argmax {base.mean():.4f}")
    print(f"frozen choice written to {a.output} - apply unchanged to test cohorts")


if __name__ == "__main__":
    main()

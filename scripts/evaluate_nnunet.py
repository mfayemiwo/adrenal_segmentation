#!/usr/bin/env python3
"""Score nnU-Net predictions with THIS project's metric, in native space.

Letting nnU-Net report its own Dice would reintroduce exactly the problem the
comparison with the 2025 paper was meant to settle: two numbers computed
differently are not comparable. This scores a folder of prediction NIfTIs the
way the rest of the project does - mean per-case Dice, each patient scored over
the whole volume and then averaged, per gland.

    python scripts/evaluate_nnunet.py \
        --predictions ../nnunet/predictions \
        --labels ../nnunet/nnUNet_raw/Dataset501_AdrenalAMOS/labelsTs

A NOTE ON MEASUREMENT SPACE, which matters for the headline comparison
----------------------------------------------------------------------
This evaluates in NATIVE voxel space: nnU-Net predicts at each scan's original
resolution, and the labels here are the original AMOS labels relabelled in
place, so no resampling is involved on either side.

Our own 0.746 / 0.747 is NOT measured that way. `evaluate_segmenter.py` scores
in the resampled cache grid (1.0 mm in-plane, 2.5 mm through-plane). Resampling
changes voxel counts and smooths boundaries, so the two numbers are not strictly
comparable, and the direction of the bias is not obvious a priori. Reporting
them side by side without saying so would repeat the mistake this project has
already had to correct once.

The fix is to write our own predictions back into native space (inverse
resample, then undo the in-plane crop) and score them here too. Until that
exists, treat any nnU-Net-versus-ours difference smaller than a few points as
uninterpretable, and say which space each number came from.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

GLANDS = {"left": 1, "right": 2}          # matches prepare_nnunet_dataset.py


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", type=Path, required=True,
                   help="folder of nnU-Net output NIfTIs, one per case")
    p.add_argument("--labels", type=Path, required=True,
                   help="matching ground-truth folder (labelsTs)")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="default: alongside the predictions")
    p.add_argument("--left-value", type=int, default=1)
    p.add_argument("--right-value", type=int, default=2)
    return p


def dice(pred: np.ndarray, truth: np.ndarray) -> float:
    denom = int(pred.sum()) + int(truth.sum())
    return float("nan") if denom == 0 else 2.0 * float(np.logical_and(pred, truth).sum()) / denom


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    import nibabel as nib

    out_dir = args.output_dir or args.predictions.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    values = {"left": args.left_value, "right": args.right_value}

    truths = {p.name.split(".")[0]: p for p in sorted(args.labels.glob("*.nii*"))}
    preds = {p.name.split(".")[0]: p for p in sorted(args.predictions.glob("*.nii*"))}
    shared = sorted(set(truths) & set(preds))
    if not shared:
        print(f"No matching case ids between {args.predictions} and {args.labels}.",
              file=sys.stderr)
        return 2
    missing = sorted(set(truths) - set(preds))

    lines: list[str] = []
    def emit(t: str = "") -> None:
        lines.append(t); print(t, flush=True)

    rows = []
    for cid in shared:
        t = np.asarray(nib.load(str(truths[cid])).dataobj)
        q = np.asarray(nib.load(str(preds[cid])).dataobj)
        if t.shape != q.shape:
            print(f"  {cid}: shape mismatch {t.shape} vs {q.shape} - skipped", file=sys.stderr)
            continue
        row = {"case_id": cid}
        for gland, v in values.items():
            tm, qm = (t == v), (q == v)
            row[f"dice_{gland}"] = dice(qm, tm)
            row[f"truth_voxels_{gland}"] = int(tm.sum())
            row[f"pred_voxels_{gland}"] = int(qm.sum())
        rows.append(row)

    emit("=" * 74)
    emit(" nnU-NET - SCORED WITH THIS PROJECT'S METRIC (mean per-case Dice)")
    emit("=" * 74)
    emit(f" predictions  {args.predictions}")
    emit(f" ground truth {args.labels}")
    emit(f" cases        {len(rows)} scored"
         + (f", {len(missing)} missing a prediction" if missing else ""))
    emit(" space        NATIVE voxel grid (no resampling on either side)")
    emit("")
    emit(f"  {'gland':<8}{'n':>5}{'mean':>9}{'median':>9}{'std':>8}{'min':>8}{'max':>8}{'<0.5':>7}")
    emit("  " + "-" * 62)
    summary = {}
    for gland in values:
        vals = np.array([r[f"dice_{gland}"] for r in rows
                         if r[f"dice_{gland}"] == r[f"dice_{gland}"]], dtype=float)
        if vals.size == 0:
            emit(f"  {gland:<8}{'0':>5}   (no scoreable cases)")
            continue
        summary[gland] = float(vals.mean())
        emit(f"  {gland:<8}{vals.size:>5}{vals.mean():>9.4f}{np.median(vals):>9.4f}"
             f"{vals.std(ddof=1) if vals.size > 1 else float('nan'):>8.4f}"
             f"{vals.min():>8.4f}{vals.max():>8.4f}{int((vals < 0.5).sum()):>7}")
    emit("")
    if len(summary) == 2:
        emit(f"  mean over both glands: {np.mean(list(summary.values())):.4f}")
        emit("")
    emit("  Reference points, all mean per-case Dice on AMOS:")
    emit("    this project's 2.5D pipeline   0.746 left / 0.745 right   (RESAMPLED space)")
    emit("    2025 paper, 2D nnU-Net         0.850 left / 0.820 right")
    emit("    2025 paper, TransUNet          0.880 left / 0.830 right")
    emit("")
    emit("  The 2.5D figure is measured on the resampled cache grid, not the native")
    emit("  grid used here. Say so when reporting the two together, and treat small")
    emit("  differences between them as uninterpretable until our predictions are")
    emit("  written back into native space and rescored with this script.")
    emit("=" * 74)

    csv_path = out_dir / "nnunet_per_case.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    report = out_dir / "nnunet_evaluation.txt"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nPer-case: {csv_path}\nReport:   {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

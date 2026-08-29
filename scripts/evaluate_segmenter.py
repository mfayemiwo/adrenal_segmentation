#!/usr/bin/env python3
"""Full-volume evaluation of a trained Stage B checkpoint, with post-processing.

Training reports Dice over *sampled slices*. This script is different in three
ways that matter for a number you would publish:

  * It runs the segmenter over EVERY slice of each validation case and assembles
    a 3D prediction volume, rather than scoring a sampled subset.
  * It scores each patient separately (mean per-case Dice, per gland), matching
    how the prior published pipeline reports L.A.G and R.A.G.
  * It applies test-time augmentation and connected-component pruning, and
    reports Dice at each stage, so the contribution of each step is measured
    rather than assumed.

It also writes a per-case CSV. A mean of 0.72 built from mostly-0.78 cases plus
a handful near 0.00 needs a different fix from a uniformly mediocre 0.72, and
only the per-case table tells you which you have.

A note on TTA for this model. The prior pipeline used horizontal flip, vertical
flip and 90-degree rotation. None is safe here unmodified: left and right
adrenal are separate output channels, so a horizontal flip must also swap the
channels or it silently scores the left gland against the right one, and the
abdomen is not laterally symmetric, so a mirrored scan is anatomically
impossible input. Rotations of 90 degrees are equally out of distribution. The
options below are therefore small rotations (which training augmentation
already covers, so they are in-distribution) and an explicit flip-with-swap,
with the measurement left to decide.

Usage
-----
    python scripts/evaluate_segmenter.py --checkpoint runs/run4/best_model.pt \
        --data-root ../data/amos22 --cache-dir ../cache

    # measure each post-processing step in turn
    python scripts/evaluate_segmenter.py --checkpoint ... --tta rot --min-component-voxels 30
"""
from __future__ import annotations

import os as _os
import tempfile as _tempfile

for _var, _sub in (("MIOPEN_USER_DB_PATH", "miopen-db"), ("MIOPEN_CUSTOM_CACHE_DIR", "miopen-cache")):
    _os.environ.setdefault(_var, _os.path.join(_tempfile.gettempdir(), _sub))
    _os.makedirs(_os.environ[_var], exist_ok=True)

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_adrenal_segmenter import (          # noqa: E402  (path set above)
    GeometryConfig, LEFT_CHANNEL_VALUE, RIGHT_CHANNEL_VALUE,
    discover_cases, load_or_build_cache, setup_logging,
)
from src.postprocessing.connected_components import (  # noqa: E402
    count_components, keep_largest_k_components, remove_small_components,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True, help="best_model.pt from a training run")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None,
                   help="where to write per_case.csv (default: alongside the checkpoint)")
    p.add_argument("--threshold", type=float, default=None,
                   help="override the operating threshold stored in the checkpoint")
    p.add_argument("--tta", choices=["none", "rot", "flip", "all"], default="none",
                   help="'rot' = small rotations (in-distribution). 'flip' = horizontal flip "
                        "with left/right channel swap. 'all' = both.")
    p.add_argument("--min-component-voxels", type=int, default=30,
                   help="drop 3D components smaller than this; 0 disables")
    p.add_argument("--keep-largest", type=int, default=1,
                   help="per gland channel, keep only the N largest components; 0 disables")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-val-cases", type=int, default=0)
    p.add_argument("--modality", choices=["ct", "mri", "all"], default=None,
                   help="default: whatever the checkpoint was trained with. Evaluating a "
                        "CT-trained model on MRI cases mixes an unusable modality back into "
                        "the score, which is exactly what made run4 read 0.71 instead of 0.75.")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    return p


def dice(pred: np.ndarray, truth: np.ndarray) -> float:
    denom = pred.sum() + truth.sum()
    if denom == 0:
        return 1.0            # correctly predicted nothing where there is nothing
    return float(2.0 * np.logical_and(pred, truth).sum() / denom)



# Prior published pipeline (Fayemiwo et al., 2025), mean per-case Dice on AMOS.
PRIOR_WORK = {"raw": {"left": 0.82, "right": 0.75}, "postprocessed": {"left": 0.91, "right": 0.90}}


def _stats(values):
    a = np.asarray(values, dtype=float)
    q1, med, q3 = np.percentile(a, [25, 50, 75])
    return {"n": len(a), "mean": a.mean(), "std": a.std(), "min": a.min(), "max": a.max(),
            "q1": q1, "median": med, "q3": q3,
            "below_0.5": int((a < 0.5).sum()), "below_0.3": int((a < 0.3).sum())}


def write_report(path: Path, *, checkpoint, settings, channel_names, stage_scores, rows, mri_threshold):
    """A self-contained results file, for reading after a batch job rather than
    watching a console. evaluate.log is the chronological record; this is the
    result."""
    import datetime

    L = []
    add = L.append
    add("=" * 78)
    add(" ADRENAL GLAND SEGMENTER - FULL-VOLUME EVALUATION")
    add("=" * 78)
    add(f" checkpoint   {checkpoint}")
    add(f" generated    {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    for k, v in settings.items():
        add(f" {k:<12} {v}")
    add("")

    add("-" * 78)
    add(" HEADLINE  (mean per-case Dice - each patient scored, then averaged)")
    add("-" * 78)
    header = "".join(f"{n:>12}" for n in channel_names) + f"{'mean':>12}"
    add(f" {'stage':<16}{header}")
    stage_means = {}
    for stage in ("raw", "postprocessed"):
        per = [float(np.mean(stage_scores[stage][n])) for n in channel_names]
        stage_means[stage] = per
        add(f" {stage:<16}" + "".join(f"{v:12.4f}" for v in per) + f"{float(np.mean(per)):12.4f}")
    delta = [b - a for a, b in zip(stage_means["raw"], stage_means["postprocessed"])]
    add(f" {'post-proc gain':<16}" + "".join(f"{v:+12.4f}" for v in delta) +
        f"{float(np.mean(delta)):+12.4f}")
    add("")

    if set(channel_names) == {"left", "right"}:
        add("-" * 78)
        add(" AGAINST THE PRIOR PUBLISHED PIPELINE (Fayemiwo et al., 2025)")
        add("-" * 78)
        add(f" {'':<26}{'L.A.G':>10}{'R.A.G':>10}")
        for stage in ("raw", "postprocessed"):
            tgt = PRIOR_WORK[stage]
            ours = dict(zip(channel_names, stage_means[stage]))
            add(f" prior work, {stage:<14}" + f"{tgt['left']:10.3f}{tgt['right']:10.3f}")
            add(f" this work,  {stage:<14}" + f"{ours['left']:10.3f}{ours['right']:10.3f}")
            add(f" {'difference':<26}" +
                f"{ours['left']-tgt['left']:+10.3f}{ours['right']-tgt['right']:+10.3f}")
            add("")

    add("-" * 78)
    add(" DISTRIBUTION ACROSS PATIENTS (after post-processing)")
    add("-" * 78)
    add(f" {'gland':<8}{'n':>5}{'mean':>9}{'median':>9}{'std':>8}{'min':>8}{'q1':>8}"
        f"{'q3':>8}{'max':>8}{'<0.5':>7}{'<0.3':>7}")
    for name in channel_names:
        st = _stats(stage_scores["postprocessed"][name])
        add(f" {name:<8}{st['n']:5d}{st['mean']:9.4f}{st['median']:9.4f}{st['std']:8.4f}"
            f"{st['min']:8.4f}{st['q1']:8.4f}{st['q3']:8.4f}{st['max']:8.4f}"
            f"{st['below_0.5']:7d}{st['below_0.3']:7d}")
    add("")
    add(" A mean well below the median means a small number of failing patients is")
    add(" carrying the result. Those cases are listed below - fix them before tuning.")
    add("")

    add("-" * 78)
    add(" CONNECTED COMPONENTS (does pruning have anything to remove?)")
    add("-" * 78)
    for name in channel_names:
        before = np.mean([r[f"components_{name}_before"] for r in rows])
        removed = np.mean([r[f"components_{name}_removed"] for r in rows])
        add(f" {name:<8} mean components predicted {before:8.1f}   removed by pruning {removed:8.1f}")
    add("")
    add(" Few predicted components means the masks are already spatially coherent and")
    add(" pruning cannot help much - a property of the 2.5D formulation, which sees")
    add(" neighbouring slices, rather than a failure of the post-processing.")
    add("")

    by_modality = {"CT": [], "MRI": []}
    for r in rows:
        digits = "".join(ch for ch in r["case_id"] if ch.isdigit())
        key = "MRI" if digits and int(digits) >= mri_threshold else "CT"
        by_modality[key].append(r)
    if all(by_modality.values()):
        add("-" * 78)
        add(" BY MODALITY  (AMOS22 Task 2 mixes CT and MRI; MRI has no Hounsfield scale,")
        add(" so a CT intensity window makes those volumes unusable)")
        add("-" * 78)
        for mod, rs in by_modality.items():
            per = [float(np.mean([r[f"dice_{n}_postprocessed"] for r in rs])) for n in channel_names]
            add(f" {mod:<5} n={len(rs):<4}" + "".join(f"{v:12.4f}" for v in per))
        add("")

    scored = sorted(rows, key=lambda r: np.mean([r[f"dice_{n}_postprocessed"] for n in channel_names]))
    add("-" * 78)
    add(" WORST 15 PATIENTS")
    add("-" * 78)
    add(f" {'case':<12}{'mean':>8}" +
        "".join(f"{n + ' dice':>12}{n + ' vox':>11}" for n in channel_names))
    for r in scored[:15]:
        mean = np.mean([r[f"dice_{n}_postprocessed"] for n in channel_names])
        add(f" {r['case_id']:<12}{mean:8.3f}" +
            "".join(f"{r[f'dice_{n}_postprocessed']:12.3f}{r[f'truth_voxels_{n}']:11d}"
                    for n in channel_names))
    add("")
    add(" Large truth voxel counts alongside a near-zero Dice mean the model missed an")
    add(" obvious structure - not a 'small organ is hard' failure. Check those scans.")
    add("")

    add("-" * 78)
    add(" BEST 5 PATIENTS (for contrast)")
    add("-" * 78)
    for r in scored[-5:][::-1]:
        mean = np.mean([r[f"dice_{n}_postprocessed"] for n in channel_names])
        add(f" {r['case_id']:<12}{mean:8.3f}" +
            "".join(f"{r[f'dice_{n}_postprocessed']:12.3f}" for n in channel_names))
    add("")
    add("=" * 78)
    add(f" Per-case scores: {path.parent / 'per_case.csv'}")
    add("=" * 78)
    add("")

    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("\n".join(L), encoding="utf-8")
    tmp.replace(path)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = args.output_dir or args.checkpoint.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(out_dir / "evaluate.log")

    import torch
    import torch.nn.functional as F
    import segmentation_models_pytorch as smp

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    threshold = args.threshold if args.threshold is not None else float(ckpt.get("threshold", 0.5))
    combine = bool(cfg.get("combine_glands", False))
    channel_names = ("gland",) if combine else ("left", "right")
    n_ch = len(channel_names)
    slice_window = int(cfg.get("slice_window", 5))

    device = torch.device("cuda" if (args.device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu")
    logger.info("=" * 78)
    logger.info("Full-volume evaluation of %s", args.checkpoint)
    logger.info("Device %s | channels %s | operating threshold %.3f (from checkpoint: %s)",
                device, ", ".join(channel_names), threshold, "no" if args.threshold is not None else "yes")
    logger.info("TTA: %s | min component voxels: %d | keep largest: %d",
                args.tta, args.min_component_voxels, args.keep_largest)

    geom = GeometryConfig(
        spacing_z=float(cfg.get("target_spacing_z", 2.5)),
        spacing_xy=float(cfg.get("target_spacing_xy", 1.0)),
        image_size=int(cfg.get("image_size", 384)),
        z_margin=int(cfg.get("z_margin", 32)),
        right_label=int(cfg.get("right_label", 11)),
        left_label=int(cfg.get("left_label", 12)),
    )

    modality = args.modality or str(cfg.get("modality", "all"))
    disc = SimpleNamespace(
        data_root=args.data_root, seed=int(cfg.get("seed", 42)),
        val_fraction=float(cfg.get("val_fraction", 0.2)),
        max_train_cases=0, max_val_cases=args.max_val_cases,
        modality=modality, mri_id_threshold=int(cfg.get("mri_id_threshold", 500)),
    )
    logger.info("Modality: %s (%s)", modality,
                "from checkpoint" if args.modality is None else "overridden on the command line")
    _, val_records = discover_cases(disc, logger)
    cache_dir = args.cache_dir or Path(cfg.get("cache_dir") or "runs/_volume_cache")
    val_cases = load_or_build_cache(val_records, geom, cache_dir, False, logger)
    logger.info("Evaluating %d validation cases", len(val_cases))

    model = smp.Unet(
        encoder_name=str(cfg.get("encoder", "resnet34")), encoder_weights=None,
        in_channels=slice_window, classes=n_ch,
        decoder_attention_type=None if cfg.get("decoder_attention") == "none" else "scse",
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    def rotate(x, degrees):
        rad = math.radians(degrees)
        cos, sin = math.cos(rad), math.sin(rad)
        theta = torch.tensor([[cos, -sin, 0.0], [sin, cos, 0.0]], dtype=torch.float32,
                             device=x.device).unsqueeze(0).expand(x.shape[0], -1, -1)
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)

    @torch.no_grad()
    def predict(batch):
        """batch: (B, slice_window, H, W) -> fused probabilities (B, n_ch, H, W)."""
        acc = torch.sigmoid(model(batch))
        n = 1
        if args.tta in ("rot", "all"):
            for deg in (-16, -8, 8, 16):
                pred = torch.sigmoid(model(rotate(batch, deg)))
                acc = acc + rotate(pred, -deg)
                n += 1
        if args.tta in ("flip", "all"):
            flipped = torch.flip(batch, dims=[-1])
            pred = torch.flip(torch.sigmoid(model(flipped)), dims=[-1])
            if n_ch == 2:
                # A mirrored abdomen puts the left gland where the right one
                # belongs, so the channels must swap back or this scores each
                # gland against the other.
                pred = pred[:, [1, 0]]
            acc = acc + pred
            n += 1
        return acc / n

    half = slice_window // 2
    rows = []
    stage_scores = {stage: {c: [] for c in channel_names} for stage in ("raw", "postprocessed")}

    for i, case in enumerate(val_cases, 1):
        image = case["image"].astype(np.float32)
        coded = case["mask"]
        n_slices = image.shape[0]

        probs = np.zeros((n_slices, n_ch) + image.shape[1:], dtype=np.float32)
        for start in range(0, n_slices, args.batch_size):
            centres = range(start, min(start + args.batch_size, n_slices))
            windows = np.stack([
                image[np.clip(np.arange(c - half, c + half + 1), 0, n_slices - 1)] for c in centres
            ])
            batch = torch.from_numpy(windows).to(device)
            probs[start:start + len(windows)] = predict(batch).float().cpu().numpy()

        truths = ([(coded > 0)] if combine
                  else [coded == LEFT_CHANNEL_VALUE, coded == RIGHT_CHANNEL_VALUE])

        row = {"case_id": case["case_id"], "n_slices": n_slices}
        for c, name in enumerate(channel_names):
            truth = truths[c]
            raw = probs[:, c] >= threshold
            d_raw = dice(raw, truth)

            pruned = raw.copy()
            removed = 0
            if args.min_component_voxels > 0:
                pruned, removed = remove_small_components(pruned, args.min_component_voxels)
            if args.keep_largest > 0 and pruned.any():
                pruned = keep_largest_k_components(pruned, args.keep_largest)
            d_pp = dice(pruned, truth)

            stage_scores["raw"][name].append(d_raw)
            stage_scores["postprocessed"][name].append(d_pp)
            row.update({
                f"dice_{name}_raw": round(d_raw, 6),
                f"dice_{name}_postprocessed": round(d_pp, 6),
                f"components_{name}_before": count_components(raw),
                f"components_{name}_removed": removed,
                f"truth_voxels_{name}": int(truth.sum()),
            })
        rows.append(row)
        if i % 20 == 0 or i == len(val_cases):
            logger.info("  evaluated %d/%d cases", i, len(val_cases))

    csv_path = out_dir / "per_case.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    logger.info("-" * 78)
    logger.info("%-16s %s", "", "  ".join(f"{n:>12}" for n in channel_names) + f"{'mean':>12}")
    for stage in ("raw", "postprocessed"):
        per = [float(np.mean(stage_scores[stage][n])) for n in channel_names]
        logger.info("%-16s %s", stage,
                    "  ".join(f"{v:12.4f}" for v in per) + f"{float(np.mean(per)):12.4f}")

    worst = sorted(rows, key=lambda r: np.mean([r[f"dice_{n}_postprocessed"] for n in channel_names]))[:10]
    logger.info("-" * 78)
    logger.info("Worst 10 cases after post-processing (investigate these before tuning anything):")
    for r in worst:
        detail = "  ".join(f"{n} {r[f'dice_{n}_postprocessed']:.3f} "
                           f"(truth {r[f'truth_voxels_{n}']:>6d} vox)" for n in channel_names)
        logger.info("   %-12s %s", r["case_id"], detail)

    report_path = out_dir / "evaluation_report.txt"
    write_report(
        report_path, checkpoint=args.checkpoint,
        settings={
            "modality": modality, "tta": args.tta, "threshold": f"{threshold:.3f}",
            "pruning": f"min {args.min_component_voxels} vox, keep largest {args.keep_largest}",
            "cases": len(val_cases),
        },
        channel_names=channel_names, stage_scores=stage_scores, rows=rows,
        mri_threshold=int(cfg.get("mri_id_threshold", 500)),
    )

    logger.info("-" * 78)
    logger.info("Per-case scores: %s", csv_path)
    logger.info("Detailed report: %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

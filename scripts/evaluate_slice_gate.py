#!/usr/bin/env python3
"""Full-volume evaluation of a trained Stage A gate - the deployment number.

Training scores the gate on the shared volume cache, which keeps only
`z_margin` slices either side of the gland. Every negative it has ever seen is
therefore a NEAR-GLAND slice. That is the right regime for learning to
discriminate, but it is the wrong denominator for the claim the paper makes: in
use, the gate runs over a whole abdominal CT, where most slices are nowhere near
the adrenals and are trivially rejectable. Measured on the cache, the gate
understates its own value.

This script streams each validation case through the SAME preprocessing at full
z-extent, one case at a time, discarding it afterwards. No new cache is built
and nothing is stored, so it costs compute and no disk.

Why the two are directly comparable: `prepare_case` computes its z-score over
the whole resampled volume BEFORE cropping in z, so a given slice is normalised
identically whether it arrives via the narrow cache or the full volume. Changing
z_margin changes which slices you see, not what any one of them looks like.

What it reports
---------------
  * slice reduction over whole volumes at the target retention - the headline
  * the same metrics at the checkpoint's own threshold and at one recalibrated
    on full volumes, because the score distribution is not the same
  * the buffer trade-off: what keeping b slices either side of every positive
    decision buys in retention and costs in reduction
  * where lost gland slices sit - at the ends of the gland or in its interior.
    Losing an end slice trims the gland; losing an interior slice splits it, and
    Stage B can never recover either.

Usage
-----
    python scripts/evaluate_slice_gate.py --checkpoint runs/gate_snn/best_model.pt \
        --data-root ../data/amos22
"""
from __future__ import annotations

import os as _os
import tempfile as _tempfile

for _var, _sub in (("MIOPEN_USER_DB_PATH", "miopen-db"), ("MIOPEN_CUSTOM_CACHE_DIR", "miopen-cache")):
    _os.environ.setdefault(_var, _os.path.join(_tempfile.gettempdir(), _sub))
    _os.makedirs(_os.environ[_var], exist_ok=True)

import argparse
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_adrenal_segmenter import (          # noqa: E402
    GeometryConfig, LEFT_CHANNEL_VALUE, RIGHT_CHANNEL_VALUE,
    discover_cases, prepare_case, setup_logging,
)
from train_slice_gate import (                 # noqa: E402
    HEADS, SHORT, _downsample, average_precision, binary_scores,
    threshold_for_sensitivity,
)

# z_margin large enough that prepare_case's crop never bites.
FULL_VOLUME_MARGIN = 10 ** 6


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True, help="best_model.pt from a gate run")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=None,
                   help="default: <checkpoint dir>/full_volume")
    p.add_argument("--target-sensitivity", type=float, default=None,
                   help="override the checkpoint's retention target")
    p.add_argument("--max-cases", type=int, default=0, help="cap for a quick look")
    p.add_argument("--modality", choices=["ct", "mri", "all"], default=None,
                   help="default: whatever the checkpoint trained on")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--buffers", type=int, nargs="*", default=[0, 1, 2, 3],
                   help="slice buffer widths to tabulate")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    return p


class Report:
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


def dilate_1d(keep: np.ndarray, buffer: int) -> np.ndarray:
    """Widen every kept run by `buffer` slices in each direction - the fixed-width
    version of the adaptive buffer, used here to price the trade-off."""
    if buffer <= 0:
        return keep
    out = keep.copy()
    for shift in range(1, buffer + 1):
        out[shift:] |= keep[:-shift]
        out[:-shift] |= keep[shift:]
    return out


def build_gate(cfg, device):
    import torch

    model_name = str(cfg.get("model", "snn"))
    if model_name == "snn":
        from src.models.snn_gate import SpikingSliceGate
        model = SpikingSliceGate(
            slice_window=int(cfg.get("slice_window", 5)), in_channels=1,
            hidden_dim=int(cfg.get("hidden_dim", 256)),
            beta=float(cfg.get("beta", 0.9)),
            threshold=float(cfg.get("lif_threshold", 1.0)), heads=HEADS)
    else:
        from src.models.cnn_lstm_gate import CNNLSTMSliceGate
        model = CNNLSTMSliceGate(
            slice_window=int(cfg.get("slice_window", 5)), in_channels=1,
            hidden_dim=int(cfg.get("hidden_dim", 256)),
            lstm_layers=int(cfg.get("lstm_layers", 1)),
            lstm_hidden=int(cfg.get("lstm_hidden", 128)), heads=HEADS)
    return model, model_name


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = args.output_dir or (args.checkpoint.parent / "full_volume")
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(out_dir / "evaluate_gate.log")
    R = Report(logger)

    import torch

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    slice_window = int(cfg.get("slice_window", 5))
    static_repeat = bool(cfg.get("static_repeat", False))
    gate_size = int(cfg.get("image_size", 192))
    cache_size = int(cfg.get("cache_image_size", 384))
    target = args.target_sensitivity or float(ckpt.get("target_sensitivity", 0.99))
    train_thr = {h: float(v) for h, v in (ckpt.get("thresholds") or {}).items()}

    device = torch.device("cuda" if (args.device in ("auto", "cuda") and torch.cuda.is_available())
                          else "cpu")
    model, model_name = build_gate(cfg, device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    geom = GeometryConfig(
        spacing_z=float(cfg.get("target_spacing_z", 2.5)),
        spacing_xy=float(cfg.get("target_spacing_xy", 1.0)),
        image_size=cache_size, z_margin=FULL_VOLUME_MARGIN,
        right_label=int(cfg.get("right_label", 11)),
        left_label=int(cfg.get("left_label", 12)),
    )
    factor = cache_size // gate_size

    modality = args.modality or str(cfg.get("modality", "ct"))
    disc = SimpleNamespace(
        data_root=args.data_root, seed=int(cfg.get("seed", 42)),
        val_fraction=float(cfg.get("val_fraction", 0.2)),
        max_train_cases=0, max_val_cases=args.max_cases,
        modality=modality, mri_id_threshold=int(cfg.get("mri_id_threshold", 500)),
    )
    _, val_records = discover_cases(disc, logger)

    R.rule("=")
    R(" STAGE A GATE - FULL-VOLUME EVALUATION")
    R.rule("=")
    R(f" checkpoint       {args.checkpoint}  (epoch {ckpt.get('epoch')}, "
      f"{ckpt.get('select_by', 'score')} {ckpt.get('score', float('nan')):.4f})")
    R(f" arm              {model_name}{' + static-repeat' if static_repeat else ''}")
    R(f" device           {device}   |  modality {modality}   |  {len(val_records)} cases")
    R(f" retention target {target:.1%}")
    R(f" training z-margin {cfg.get('z_margin')} slices  ->  here: whole volume")
    R()

    half = slice_window // 2
    per_case, all_scores, all_truth = [], {h: [] for h in HEADS}, {h: [] for h in HEADS}
    case_arrays = []
    trained_slices = 0

    @torch.no_grad()
    def predict(image: np.ndarray) -> np.ndarray:
        n = image.shape[0]
        out = np.zeros((n, len(HEADS)), dtype=np.float32)
        for start in range(0, n, args.batch_size):
            centres = range(start, min(start + args.batch_size, n))
            if static_repeat:
                windows = np.stack([np.repeat(image[c][None], slice_window, axis=0)
                                    for c in centres])
            else:
                windows = np.stack([image[np.clip(np.arange(c - half, c + half + 1), 0, n - 1)]
                                    for c in centres])
            x = torch.from_numpy(windows.astype(np.float32)).unsqueeze(2).to(device)
            logits, _ = model(x)
            for h, head in enumerate(HEADS):
                out[start:start + len(windows), h] = torch.sigmoid(
                    logits[head].float()).cpu().numpy()
        return out

    for i, record in enumerate(val_records, 1):
        try:
            case = prepare_case(record, geom)
        except Exception as exc:
            logger.warning("Skipping %s: %s", record["case_id"], exc)
            continue
        if case is None:
            continue

        image = _downsample(np.asarray(case["image"], dtype=np.float32), factor)
        mask = case["mask"]
        labels = np.stack([
            (mask == LEFT_CHANNEL_VALUE).any(axis=(1, 2)),
            (mask == RIGHT_CHANNEL_VALUE).any(axis=(1, 2)),
        ], axis=1).astype(np.uint8)

        scores = predict(image)
        case_arrays.append({"case_id": case["case_id"], "scores": scores, "labels": labels})
        for h, head in enumerate(HEADS):
            all_scores[head].append(scores[:, h])
            all_truth[head].append(labels[:, h])

        any_lab = labels.any(axis=1)
        trained_slices += int(any_lab.sum()) + 2 * int(cfg.get("z_margin", 32))
        if i % 10 == 0 or i == len(val_records):
            logger.info("  %d/%d cases", i, len(val_records))

    if not case_arrays:
        logger.error("No usable validation cases.")
        return 2

    scores = {h: np.concatenate(all_scores[h]) for h in HEADS}
    truth = {h: np.concatenate(all_truth[h]) for h in HEADS}
    n_slices = scores[HEADS[0]].size
    any_truth = np.zeros(n_slices, dtype=bool)
    for h in HEADS:
        any_truth |= truth[h] > 0

    R.rule("=")
    R(" 1. WHAT CHANGED FROM THE TRAINING DENOMINATOR")
    R.rule("=")
    R(f"  slices seen here                 {n_slices:>8d}   (whole volumes)")
    R(f"  gland-bearing                    {int(any_truth.sum()):>8d}   "
      f"({100 * any_truth.mean():.1f}% of slices)")
    R(f"  approx. slices in the training cache {trained_slices:>5d}   "
      f"(gland +/- {cfg.get('z_margin')})")
    R("  The prevalence gap between these two is exactly why the training-time")
    R("  slice reduction understates the gate. PR-AUC's no-skill baseline is the")
    R("  prevalence, so it is not comparable between the two settings either.")
    R()

    prauc = {h: average_precision(scores[h], truth[h]) for h in HEADS}
    recal_thr = {h: threshold_for_sensitivity(scores[h], truth[h], target) for h in HEADS}

    R.rule("=")
    R(" 2. PER-HEAD METRICS")
    R.rule("=")
    R(f"  {'head':<8}{'PR-AUC':>9}{'prevalence':>12}"
      f"{'thr(train)':>12}{'sens':>9}{'spec':>9}"
      f"{'thr(here)':>12}{'sens':>9}{'spec':>9}{'NPV':>9}")
    for head in HEADS:
        prev = float((truth[head] > 0).mean())
        a = binary_scores(scores[head], truth[head], train_thr.get(head, 0.5))
        b = binary_scores(scores[head], truth[head], recal_thr[head])
        R(f"  {SHORT[head]:<8}{prauc[head]:>9.4f}{prev:>12.4f}"
          f"{train_thr.get(head, float('nan')):>12.3f}{a['sensitivity']:>9.4f}{a['specificity']:>9.4f}"
          f"{recal_thr[head]:>12.3f}{b['sensitivity']:>9.4f}{b['specificity']:>9.4f}{b['npv']:>9.4f}")
    R()
    R("  thr(train) is the threshold chosen during training, on near-gland negatives.")
    R("  thr(here) is recalibrated on whole volumes. If the two differ materially the")
    R("  gate must be calibrated on the distribution it will actually run on - report")
    R("  the recalibrated one, and say where it came from.")
    R()

    def summarise(thr_map, buffer):
        keeps, retained, gland = [], 0, 0
        for c in case_arrays:
            keep = np.zeros(c["labels"].shape[0], dtype=bool)
            for h, head in enumerate(HEADS):
                keep |= c["scores"][:, h] >= thr_map[head]
            keep = dilate_1d(keep, buffer)
            any_lab = c["labels"].any(axis=1)
            keeps.append(keep)
            retained += int(np.logical_and(keep, any_lab).sum())
            gland += int(any_lab.sum())
        kept = int(sum(int(k.sum()) for k in keeps))
        total = int(sum(k.size for k in keeps))
        return {"keeps": keeps, "retention": retained / max(gland, 1),
                "reduction": 1 - kept / max(total, 1), "kept": kept, "total": total}

    base = summarise(recal_thr, 0)

    R.rule("=")
    R(" 3. DEPLOYMENT HEADLINE")
    R.rule("=")
    R(f"  Over whole volumes, at a recalibrated {target:.0%} retention target:")
    R()
    R(f"     slices removed          {base['reduction']:>7.1%}   "
      f"({base['total'] - base['kept']:,} of {base['total']:,})")
    R(f"     gland slices retained   {base['retention']:>7.1%}")
    R()
    R("  This is the number for the paper. Stage B runs on the remainder, so the")
    R("  reduction is also the segmentation cost saved.")
    R()

    R.rule("=")
    R(" 4. SLICE BUFFER TRADE-OFF")
    R.rule("=")
    R("  Keeping b slices either side of every positive decision. This is the fixed")
    R("  -width control that the uncertainty-adaptive buffer has to beat.")
    R()
    R(f"  {'buffer':>8}{'retention':>12}{'slices removed':>17}")
    buffer_rows = []
    for b in args.buffers:
        s = summarise(recal_thr, b)
        buffer_rows.append({"buffer": b, "retention": s["retention"], "reduction": s["reduction"]})
        R(f"  {b:>8}{s['retention']:>12.4f}{s['reduction']:>17.1%}")
    R()

    R.rule("=")
    R(" 5. WHERE THE LOST GLAND SLICES ARE")
    R.rule("=")
    R("  Distance, in slices, from the nearest end of the gland's extent. An end")
    R("  slice trims the gland; an interior one splits it, and Stage B cannot")
    R("  recover either.")
    R()
    bins = {0: 0, 1: 0, 2: 0, "interior": 0}
    lost_cases = []
    for c, s in zip(case_arrays, base["keeps"]):
        any_lab = c["labels"].any(axis=1)
        idx = np.flatnonzero(any_lab)
        if idx.size == 0:
            continue
        lo, hi = int(idx.min()), int(idx.max())
        missed = [int(z) for z in idx if not s[z]]
        for z in missed:
            d = min(z - lo, hi - z)
            bins[d if d in (0, 1, 2) else "interior"] += 1
        if missed:
            lost_cases.append({"case_id": c["case_id"], "lost": len(missed),
                               "gland_slices": int(idx.size)})
    total_lost = sum(bins.values())
    for key in (0, 1, 2, "interior"):
        label = {0: "end slice", 1: "1 in from an end", 2: "2 in from an end",
                 "interior": "interior (>=3 in)"}[key]
        share = bins[key] / total_lost if total_lost else 0.0
        R(f"    {label:<22}{bins[key]:>6}  {share:>7.1%}")
    R(f"    {'total lost':<22}{total_lost:>6}")
    R()
    if lost_cases:
        lost_cases.sort(key=lambda r: -r["lost"])
        R(f"  Cases losing any gland slice: {len(lost_cases)} of {len(case_arrays)}")
        R(f"  {'case':<14}{'lost':>6}{'of':>6}")
        for r in lost_cases[:10]:
            R(f"  {r['case_id']:<14}{r['lost']:>6}{r['gland_slices']:>6}")
    else:
        R("  No gland slice was lost in any case.")
    R()

    csv_path = out_dir / "per_case.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["case_id", "slices", "gland_slices",
                                           "kept", "retained", "reduction"])
        w.writeheader()
        for c, s in zip(case_arrays, base["keeps"]):
            any_lab = c["labels"].any(axis=1)
            w.writerow({
                "case_id": c["case_id"], "slices": int(s.size),
                "gland_slices": int(any_lab.sum()), "kept": int(s.sum()),
                "retained": int(np.logical_and(s, any_lab).sum()),
                "reduction": round(1 - int(s.sum()) / max(int(s.size), 1), 6),
            })

    report_path = out_dir / "gate_evaluation_report.txt"
    R.rule("=")
    R(f" Per-case scores: {csv_path}")
    R.rule("=")
    R.write(report_path)
    logger.info("Report: %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Statistical comparison for the delta-SNN refiner experiment.

The unit of inference is the patient, not the individual slice and not the
patient-seed pair. For each model variant, this script first averages each
patient's metric across training seeds, then performs paired tests between
variants on those patient-level values. This avoids pseudoreplication.

Expected run layout from run_delta_snn_core.sbatch:
    runs/delta_refiner_<variant>_s<seed>/full_volume/per_case.csv

The baseline CSV should be generated with the same branch/evaluator so it
contains Dice, HD95, NSD@2 mm and through-plane consistency metrics.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

VARIANTS = ("delta_snn", "raw_snn", "delta_analog")
METRICS = {
    "dice": ("higher", ["dice_left_postprocessed", "dice_right_postprocessed"]),
    "hd95_mm": ("lower", ["hd95_left_mm", "hd95_right_mm"]),
    "nsd2mm": ("higher", ["nsd2mm_left", "nsd2mm_right"]),
    "z_presence_dice": (
        "higher",
        ["z_presence_dice_left", "z_presence_dice_right"],
    ),
    "z_fragment_error": (
        "lower",
        ["z_fragment_error_left", "z_fragment_error_right"],
    ),
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--baseline-csv", type=Path, required=True)
    p.add_argument("--runs-dir", type=Path, default=Path("runs"))
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--output", type=Path, default=Path("runs/delta_snn_statistics.txt"))
    return p


def load_metric(path: Path, columns: list[str]) -> pd.Series:
    df = pd.read_csv(path)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path} is missing {missing}. Re-run evaluate_segmenter.py on "
            "research/delta-snn-refiner so all publication metrics are present."
        )
    values = df[columns].mean(axis=1, skipna=True)
    return pd.Series(values.to_numpy(), index=df["case_id"].astype(str), name=path.stem)


def load_variant(
    runs_dir: Path, variant: str, seeds: list[int], columns: list[str]
) -> tuple[pd.Series, list[float], list[int]]:
    patient_series = []
    seed_means = []
    found = []
    for seed in seeds:
        path = (
            runs_dir
            / f"delta_refiner_{variant}_s{seed}"
            / "full_volume"
            / "per_case.csv"
        )
        if not path.is_file():
            continue
        s = load_metric(path, columns)
        patient_series.append(s.rename(str(seed)))
        seed_means.append(float(s.mean()))
        found.append(seed)

    if not patient_series:
        raise FileNotFoundError(f"no completed full-volume runs found for {variant}")

    aligned = pd.concat(patient_series, axis=1, join="inner")
    return aligned.mean(axis=1), seed_means, found


def bootstrap_mean_ci(diff: np.ndarray, n_boot: int, seed: int = 2026):
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(diff, size=(n_boot, diff.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def paired_test(a: pd.Series, b: pd.Series, n_boot: int):
    pair = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    diff = (pair["a"] - pair["b"]).to_numpy(dtype=float)
    if len(diff) < 2:
        return None

    try:
        w = stats.wilcoxon(diff, alternative="two-sided", zero_method="wilcox")
        p = float(w.pvalue)
        stat = float(w.statistic)
    except ValueError:
        p = 1.0
        stat = 0.0

    lo, hi = bootstrap_mean_ci(diff, n_boot)
    # Rank-biserial effect size from paired differences.
    ranks = stats.rankdata(np.abs(diff))
    pos = float(ranks[diff > 0].sum())
    neg = float(ranks[diff < 0].sum())
    denom = pos + neg
    rbc = (pos - neg) / denom if denom else 0.0

    return {
        "n": len(diff),
        "mean_diff": float(np.mean(diff)),
        "median_diff": float(np.median(diff)),
        "ci": (lo, hi),
        "p": p,
        "stat": stat,
        "rbc": float(rbc),
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        value = (m - rank) * p_values[idx]
        running = max(running, value)
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def fmt_mean_sd(values: list[float]) -> str:
    a = np.asarray(values, dtype=float)
    if len(a) == 1:
        return f"{a.mean():.4f} (n=1)"
    return f"{a.mean():.4f} +/- {a.std(ddof=1):.4f} (n={len(a)})"


def main() -> int:
    args = build_parser().parse_args()
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit("=" * 86)
    emit(" DELTA-SNN REFINER - PATIENT-LEVEL STATISTICAL ANALYSIS")
    emit("=" * 86)
    emit(f" baseline: {args.baseline_csv}")
    emit(f" seeds requested: {args.seeds}")
    emit("")

    all_tests = []

    for metric, (direction, columns) in METRICS.items():
        emit("-" * 86)
        emit(f" {metric} ({direction} is better)")
        emit("-" * 86)

        baseline = load_metric(args.baseline_csv, columns)
        data = {"baseline": baseline}
        seed_summary = {}

        emit(f" baseline patient mean: {baseline.mean():.4f} (n={len(baseline)})")
        for variant in VARIANTS:
            patient_mean, seed_means, found = load_variant(
                args.runs_dir, variant, args.seeds, columns
            )
            data[variant] = patient_mean
            seed_summary[variant] = seed_means
            emit(
                f" {variant:<14} seed-level means {fmt_mean_sd(seed_means)} "
                f"[seeds {found}]"
            )

        comparisons = (
            ("delta_snn", "baseline", "Does the proposed refiner improve the base model?"),
            ("delta_snn", "raw_snn", "Does delta encoding beat raw-slice SNN input?"),
            (
                "delta_snn",
                "delta_analog",
                "Do spiking threshold/reset dynamics add beyond delta representation?",
            ),
        )

        for a_name, b_name, question in comparisons:
            result = paired_test(data[a_name], data[b_name], args.bootstrap)
            if result is None:
                emit(f" {a_name} vs {b_name}: insufficient aligned patients")
                continue
            all_tests.append((metric, a_name, b_name, question, result))
            sign = 1.0 if direction == "higher" else -1.0
            benefit = sign * result["mean_diff"]
            verdict = "favours proposed" if benefit > 0 else "does not favour proposed"
            emit(
                f" {a_name} - {b_name}: diff {result['mean_diff']:+.4f}, "
                f"95% bootstrap CI [{result['ci'][0]:+.4f}, {result['ci'][1]:+.4f}], "
                f"Wilcoxon p={result['p']:.4g}, rank-biserial={result['rbc']:+.3f} "
                f"(n={result['n']}) -> {verdict}"
            )
            emit(f"   question: {question}")
        emit("")

    if all_tests:
        adjusted = holm_adjust([x[4]["p"] for x in all_tests])
        emit("-" * 86)
        emit(" HOLM-CORRECTED P VALUES ACROSS ALL PRE-SPECIFIED COMPARISONS")
        emit("-" * 86)
        for item, p_adj in zip(all_tests, adjusted):
            metric, a_name, b_name, question, result = item
            emit(
                f" {metric:<18} {a_name:<12} vs {b_name:<12} "
                f"raw p={result['p']:.4g}  Holm p={p_adj:.4g}"
            )
        emit("")

    emit("-" * 86)
    emit(" INTERPRETATION RULE")
    emit("-" * 86)
    emit(
        " The central claim requires delta_snn to improve the baseline and to beat "
        "both controls by an effect larger than seed variability, with patient-level "
        "confidence intervals that exclude clinically trivial changes. A significant "
        "delta_snn-vs-raw_snn result supports the event representation; a significant "
        "delta_snn-vs-delta_analog result supports a specifically spiking contribution."
    )
    emit(
        " If only delta_snn-vs-baseline is positive, the correct paper claim is a "
        "temporal residual refinement result, not evidence that spiking dynamics are "
        "responsible."
    )
    emit("=" * 86)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWritten to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

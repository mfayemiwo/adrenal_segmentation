#!/usr/bin/env python3
"""Aggregate the seed-replication runs into results with error bars.

Every Stage A number before this was n=1, and two of the three conclusions are
claims about DIFFERENCES between configurations. A difference from one run each
is not a measurement - especially the null result (arm E vs arm G), where the
whole claim is that a gap is indistinguishable from zero. You cannot say that
without knowing how big zero looks.

This reads runs/seed_<config>_s<seed>/best_model.pt for every configuration and
seed it can find, reports mean +/- sd, and runs the three comparisons that
matter with Welch's t-test (unequal variances, which is the right default for
small independent samples), Cohen's d, and a 95% confidence interval on the
difference.

    python scripts/aggregate_seeds.py
    python scripts/aggregate_seeds.py --output runs/seed_summary.txt

Works on partial results - it reports whatever has finished.

A caveat it prints, and which matters for how the numbers are written up: the
two "combined" configurations were selected as best-of-nine on the validation
set. Seeding tells you the variance OF THAT CONFIGURATION; it does not undo the
selection. The defensible claim is "configuration X against configuration Y at
n seeds", not "best-of-nine against best-of-nine".
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

CONFIGS = [
    ("seed_snn",          "SNN, arm E (sequence)"),
    ("seed_snn_static",   "SNN, arm G (static repeat)"),
    ("seed_snn_combined", "SNN, tuned"),
    ("seed_cnnlstm",      "CNN-LSTM, arm D"),
    ("seed_lstm_combined","CNN-LSTM, tuned"),
]

COMPARISONS = [
    ("seed_snn", "seed_snn_static", "prauc",
     "THE SEQUENCE CLAIM - does true slice order beat the static-repeat control?",
     "A difference indistinguishable from zero here means the craniocaudal order of "
     "adjacent slices carries no usable signal for gland presence. The synthetic probe "
     "already showed the architecture CAN represent order, so this is about the data."),
    ("seed_lstm_combined", "seed_snn_combined", "prauc",
     "THE HEADLINE - tuned baseline against tuned spiking gate",
     "Both had nine configurations and capacity matched within 2%, so a gap here is "
     "architectural rather than a difference in effort."),
    ("seed_snn", "seed_snn_combined", "prauc",
     "DID TUNING HELP THE SPIKING GATE?",
     "n=1 suggested +0.080. Confirm it exceeds seed variance."),
    ("seed_cnnlstm", "seed_lstm_combined", "prauc",
     "DID TUNING HELP THE BASELINE?",
     "n=1 suggested +0.0006 - i.e. it was already at its ceiling. Confirm."),
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", type=Path, default=Path("runs"))
    p.add_argument("--seeds", type=int, nargs="*", default=[1, 2, 3, 4, 5])
    p.add_argument("--output", type=Path, default=None, help="also write the report here")
    return p


def load(runs_dir: Path, prefix: str, seeds) -> list[dict]:
    import torch
    out = []
    for s in seeds:
        p = runs_dir / f"{prefix}_s{s}" / "best_model.pt"
        if not p.exists():
            continue
        c = torch.load(p, map_location="cpu", weights_only=False)
        out.append({
            "seed": s,
            "epoch": int(c.get("epoch", 0)),
            "prauc": float(np.mean(list(c["prauc"].values()))),
            "reduction": float(c["slice_reduction"]),
            "retention": float(c["any_retention"]),
        })
    return out


def welch(a: np.ndarray, b: np.ndarray):
    """Welch's t-test plus Cohen's d and a 95% CI on the mean difference.

    Unequal-variance t is the right default: the two configurations have no
    reason to share a variance, and Welch costs nothing when they do.
    """
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return {"diff": ma - mb, "t": float("inf"), "df": float("nan"),
                "p": 0.0, "d": float("inf"), "ci": (ma - mb, ma - mb)}
    t = (ma - mb) / se
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    try:
        from scipy import stats
        p = 2 * stats.t.sf(abs(t), df)
        crit = stats.t.ppf(0.975, df)
    except ImportError:                      # pragma: no cover
        p, crit = float("nan"), 2.0
    pooled = math.sqrt(((na - 1) * va + (nb - 1) * vb) / max(na + nb - 2, 1))
    d = (ma - mb) / pooled if pooled > 0 else float("inf")
    return {"diff": ma - mb, "t": t, "df": df, "p": p, "d": d,
            "ci": (ma - mb - crit * se, ma - mb + crit * se)}


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    lines: list[str] = []

    def emit(text: str = "") -> None:
        lines.append(text)
        print(text, flush=True)

    data = {prefix: load(args.runs_dir, prefix, args.seeds) for prefix, _ in CONFIGS}
    if not any(data.values()):
        print(f"No seed runs found under {args.runs_dir}. Expected "
              f"{args.runs_dir}/seed_<config>_s<seed>/best_model.pt", file=sys.stderr)
        return 2

    emit("=" * 78)
    emit(" STAGE A - SEED REPLICATION")
    emit("=" * 78)
    emit(f" seeds requested: {', '.join(map(str, args.seeds))}")
    emit("")

    emit("-" * 78)
    emit(" 1. PER-CONFIGURATION")
    emit("-" * 78)
    emit(f"  {'configuration':<30}{'n':>3}{'PR-AUC mean':>13}{'sd':>9}"
         f"{'removed':>10}{'sd':>8}")
    emit("  " + "-" * 74)
    for prefix, label in CONFIGS:
        rows = data[prefix]
        if not rows:
            emit(f"  {label:<30}{'-':>3}   (no runs found)")
            continue
        pr = np.array([r["prauc"] for r in rows])
        rd = np.array([r["reduction"] for r in rows])
        sd_pr = pr.std(ddof=1) if len(pr) > 1 else float("nan")
        sd_rd = rd.std(ddof=1) if len(rd) > 1 else float("nan")
        emit(f"  {label:<30}{len(rows):>3}{pr.mean():>13.4f}{sd_pr:>9.4f}"
             f"{rd.mean():>10.1%}{sd_rd:>8.1%}")
    emit("")
    emit("  Seed-to-seed sd is the number to compare every difference in this project")
    emit("  against. Anything smaller than roughly 2 sd is not a finding.")
    emit("")

    emit("-" * 78)
    emit(" 2. COMPARISONS")
    emit("-" * 78)
    for a_key, b_key, metric, heading, note in COMPARISONS:
        ra, rb = data[a_key], data[b_key]
        la = dict(CONFIGS)[a_key]
        lb = dict(CONFIGS)[b_key]
        emit("")
        emit(f"  {heading}")
        emit(f"    {la}  vs  {lb}")
        if len(ra) < 2 or len(rb) < 2:
            emit(f"    not enough runs yet (n={len(ra)} and n={len(rb)}; need 2+ each)")
            continue
        a = np.array([r[metric] for r in ra])
        b = np.array([r[metric] for r in rb])
        st = welch(a, b)
        emit(f"    {a.mean():.4f} +/- {a.std(ddof=1):.4f} (n={len(a)})   vs   "
             f"{b.mean():.4f} +/- {b.std(ddof=1):.4f} (n={len(b)})")
        emit(f"    difference {st['diff']:+.4f}   95% CI [{st['ci'][0]:+.4f}, {st['ci'][1]:+.4f}]"
             f"   Welch t={st['t']:.2f}, df={st['df']:.1f}, p={st['p']:.4f}   d={st['d']:+.2f}")
        crosses_zero = st["ci"][0] <= 0 <= st["ci"][1]
        if crosses_zero:
            half = max(abs(st["ci"][0]), abs(st["ci"][1]))
            emit(f"    -> the interval includes zero: no difference detectable at "
                 f"n={len(a)}/{len(b)}. An effect larger than {half:.4f} would have shown.")
        else:
            emit(f"    -> the interval excludes zero: the difference is real at this sample size.")
        emit(f"    {note}")
    emit("")

    emit("-" * 78)
    emit(" 3. HOW TO WRITE THIS UP")
    emit("-" * 78)
    emit("  * Report mean +/- sd over seeds, with n, for every number in the paper.")
    emit("  * A confidence interval that includes zero supports 'no difference detected")
    emit("    at this sample size'. It does NOT support 'the two are equal' - state the")
    emit("    interval so a reader can see what size of effect you could have missed.")
    emit("  * The two 'tuned' configurations were chosen as best-of-nine on validation")
    emit("    data. Seeding gives the variance OF THAT CONFIGURATION; it does not undo")
    emit("    the selection. Write it as 'configuration X vs configuration Y at n seeds',")
    emit("    not 'best-of-nine vs best-of-nine', and say how the configurations were")
    emit("    chosen.")
    emit("  * Seeds vary initialisation, sampling and augmentation, with the train/")
    emit("    validation split held fixed. That is training variance, not split variance;")
    emit("    cross-validation would be needed for the latter.")
    emit("=" * 78)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nWritten to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

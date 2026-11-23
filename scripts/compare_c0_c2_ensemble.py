#!/usr/bin/env python3
"""One report for C0 vs C2 vs equal-weight ensemble on dev158, amos100, ts76.

Reads runs/c0_c2_ensemble/metrics_<cohort>_<model>.csv (written by the
existing evaluators) and the reference labels, and prints for every cohort:
mean bilateral Dice with a 95% patient bootstrap CI, median, tail counts,
the both-glands-complete subset, the small-gland subset (smaller gland
< 1500 mm3) and paired differences with paired bootstrap CIs.

Gland volume and scan-edge truncation are computed from the REFERENCE label
only, never from a prediction. Nothing here selects or tunes anything.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np

MODELS = ("C0", "C2", "ENS")
SEED = 20261110
N_BOOT = 10_000
SMALL_MM3 = 1500.0


def reference_facts(label_dir: Path) -> dict:
    facts = {}
    for path in sorted(label_dir.glob("*.nii.gz")):
        img = nib.load(str(path))
        y = np.asarray(img.dataobj).astype(np.uint8)
        vox = abs(np.linalg.det(img.affine[:3, :3]))
        si = int(np.argmax(np.abs(img.affine[2, :3])))
        others = tuple(a for a in range(3) if a != si)
        row = {}
        for side, lab in (("left", 1), ("right", 2)):
            m = y == lab
            idx = np.nonzero(m.any(axis=others))[0]
            row[f"mm3_{side}"] = float(m.sum() * vox)
            row[f"trunc_{side}"] = bool(len(idx)) and (idx[0] == 0 or idx[-1] == y.shape[si] - 1)
        facts[path.name[:-7]] = row
    return facts


def load_scores(csv_path: Path) -> dict:
    with csv_path.open() as f:
        return {r["case_id"]: (float(r["dice_left"]), float(r["dice_right"])) for r in csv.DictReader(f)}


def boot_ci(values: np.ndarray, rng) -> tuple:
    idx = rng.integers(0, len(values), size=(N_BOOT, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--labels", nargs="+", required=True, help="cohort=label_dir pairs")
    a = p.parse_args()
    for spec in a.labels:
        cohort, label_dir = spec.split("=", 1)
        facts = reference_facts(Path(label_dir))
        scores = {}
        for m in MODELS:
            f = a.runs / f"metrics_{cohort}_{m}.csv"
            if f.is_file():
                scores[m] = load_scores(f)
        if not scores:
            print(f"\n### {cohort}: no metrics yet")
            continue
        ids = sorted(set.intersection(*(set(s) for s in scores.values())))
        missing = set(facts) - set(ids)
        if missing:
            raise SystemExit(f"{cohort}: {len(missing)} labelled cases lack a score in some model, e.g. {sorted(missing)[:3]}")
        small = np.array([min(facts[c]["mm3_left"], facts[c]["mm3_right"]) < SMALL_MM3 for c in ids])
        complete = np.array([not (facts[c]["trunc_left"] or facts[c]["trunc_right"]) for c in ids])
        print(f"\n### {cohort}: n = {len(ids)} | small-gland cases (smaller gland < {SMALL_MM3:.0f} mm3): "
              f"{small.sum()} | both glands complete in scan: {complete.sum()}")
        print(f"{'model':5s} {'bilateral':>9s} {'95% CI':>17s} {'median':>7s} {'left':>6s} {'right':>6s} "
              f"{'<0.5':>5s} {'>=0.9':>6s} {'complete':>9s} {'small':>7s} {'not small':>10s}")
        bil = {}
        for m, s in scores.items():
            L = np.array([s[c][0] for c in ids]); R = np.array([s[c][1] for c in ids])
            b = (L + R) / 2
            bil[m] = b
            lo, hi = boot_ci(b, np.random.default_rng(SEED))
            print(f"{m:5s} {b.mean():9.4f} [{lo:.4f}, {hi:.4f}] {np.median(b):7.4f} {L.mean():6.4f} {R.mean():6.4f} "
                  f"{(b < .5).sum():5d} {(b >= .9).sum():6d} {b[complete].mean():9.4f} "
                  f"{b[small].mean() if small.any() else float('nan'):7.4f} {b[~small].mean():10.4f}")
        pairs = [(x, y) for x, y in (("C2", "C0"), ("ENS", "C0"), ("ENS", "C2")) if x in bil and y in bil]
        for x, y in pairs:
            d = bil[x] - bil[y]
            lo, hi = boot_ci(d, np.random.default_rng(SEED))
            ds = d[small] if small.any() else np.array([np.nan])
            print(f"  paired {x}-{y}: {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] | better {int((d > 0).sum())} "
                  f"worse {int((d < 0).sum())} | small-gland subset {np.nanmean(ds):+.4f}")


if __name__ == "__main__":
    main()

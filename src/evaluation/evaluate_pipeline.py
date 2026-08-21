"""Patient-level statistical validation: bootstrap CIs, Wilcoxon signed-rank
comparisons between ablation arms, and Holm-Bonferroni correction across the
many pairwise comparisons the ablation matrix implies (see
scripts/run_ablation.py — 8 arms means up to 28 pairwise comparisons, which
without correction would inflate the false-discovery rate substantially).
"""
from __future__ import annotations

import numpy as np

try:
    from scipy import stats
except ImportError as exc:  # pragma: no cover
    raise ImportError("scipy is required for statistical validation; pip install scipy") from exc


def bootstrap_ci(values: np.ndarray, n_resamples: int = 2000, ci: float = 0.95, seed: int = 42) -> tuple[float, float, float]:
    """Patient-level bootstrap: `values` must already be one scalar per
    patient (e.g. mean Dice across that patient's scan), never per-slice,
    or the resample will treat correlated slices as independent evidence."""
    rng = np.random.RandomState(seed)
    values = np.asarray(values)
    n = len(values)
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = rng.choice(values, size=n, replace=True)
        means[i] = sample.mean()
    lo = (1 - ci) / 2 * 100
    hi = (1 - (1 - ci) / 2) * 100
    return float(values.mean()), float(np.percentile(means, lo)), float(np.percentile(means, hi))


def wilcoxon_signed_rank(values_a: np.ndarray, values_b: np.ndarray) -> tuple[float, float]:
    """Paired test between two arms' per-patient scores (same patients, same
    fold assignment). Returns (statistic, p_value)."""
    values_a, values_b = np.asarray(values_a), np.asarray(values_b)
    statistic, p_value = stats.wilcoxon(values_a, values_b)
    return float(statistic), float(p_value)


def holm_bonferroni(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni step-down correction. Returns adjusted p-values keyed
    by the same comparison names passed in, so callers can report which
    pairwise differences remain significant after correcting for the full
    ablation matrix."""
    names = list(p_values.keys())
    raw = np.array([p_values[name] for name in names])
    order = np.argsort(raw)
    m = len(raw)

    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        corrected = raw[idx] * (m - rank)
        running_max = max(running_max, corrected)
        adjusted[idx] = min(running_max, 1.0)

    return {names[i]: float(adjusted[i]) for i in range(m)}


def compare_arms(per_patient_scores: dict[str, np.ndarray]) -> dict:
    """Given per-patient Dice (or any scalar metric) for each ablation arm,
    compute bootstrap CIs per arm and Holm-corrected Wilcoxon p-values for
    every pairwise comparison against the proposed arm ("E")."""
    if "E" not in per_patient_scores:
        raise KeyError("compare_arms expects the proposed method under key 'E'")

    report = {"per_arm_ci": {}, "pairwise_vs_E": {}}
    for arm, scores in per_patient_scores.items():
        report["per_arm_ci"][arm] = bootstrap_ci(scores)

    raw_p_values = {}
    for arm, scores in per_patient_scores.items():
        if arm == "E":
            continue
        _, p_value = wilcoxon_signed_rank(per_patient_scores["E"], scores)
        raw_p_values[f"E_vs_{arm}"] = p_value

    report["pairwise_vs_E"] = holm_bonferroni(raw_p_values)
    return report

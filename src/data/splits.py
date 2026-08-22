"""Patient-level cross-validation splitting.

Slice-level random splitting leaks information between train/val/test because
adjacent slices from the same patient are highly correlated. Every split in
this project must be computed at the patient level, per the design review.
"""
from __future__ import annotations

import numpy as np


def patient_level_kfold(patient_ids: list[str], n_folds: int = 5, seed: int = 42):
    """Yield (train_ids, val_ids) patient-id lists for `n_folds` folds.

    A thin, dependency-free stand-in for sklearn.model_selection.KFold that
    operates explicitly on patient identifiers rather than array indices, so
    it is impossible to accidentally split at the slice level upstream.
    """
    ids = np.array(sorted(patient_ids))
    rng = np.random.RandomState(seed)
    rng.shuffle(ids)

    fold_sizes = np.full(n_folds, len(ids) // n_folds, dtype=int)
    fold_sizes[: len(ids) % n_folds] += 1

    folds = []
    start = 0
    for size in fold_sizes:
        folds.append(ids[start : start + size])
        start += size

    for i in range(n_folds):
        val_ids = folds[i]
        train_ids = np.concatenate([folds[j] for j in range(n_folds) if j != i])
        yield sorted(train_ids.tolist()), sorted(val_ids.tolist())


def assert_no_patient_leakage(train_ids, val_ids, test_ids=None):
    train_set, val_set = set(train_ids), set(val_ids)
    assert train_set.isdisjoint(val_set), "Patient leakage between train and val splits"
    if test_ids is not None:
        test_set = set(test_ids)
        assert train_set.isdisjoint(test_set), "Patient leakage between train and test splits"
        assert val_set.isdisjoint(test_set), "Patient leakage between val and test splits"

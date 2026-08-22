"""Tests for the src/data package.

Deliberately free of torch/snntorch imports so the data contracts stay
testable on a machine without the deep-learning stack installed.
"""
import numpy as np
import pytest

from src.data.dataset import (
    LEFT_ADRENAL_LABEL,
    RIGHT_ADRENAL_LABEL,
    PatientVolume,
    discover_patients,
    load_patient_volume,
)
from src.data.preprocessing import (
    PreprocessConfig,
    apply_hu_window,
    normalise_intensity,
    preprocess_volume,
)
from src.data.splits import assert_no_patient_leakage, patient_level_kfold

nib = pytest.importorskip("nibabel", reason="nibabel is only needed for the NIfTI loading tests")


def _volume(n_slices=20, size=16):
    rng = np.random.RandomState(0)
    image = rng.normal(0, 1, size=(n_slices, size, size)).astype(np.float32)
    mask = np.zeros_like(image, dtype=np.int16)
    mask[8:12, 5:9, 5:9] = 1
    return PatientVolume(patient_id="p0", image=image, mask=mask)


# --- preprocessing ---------------------------------------------------------

def test_hu_window_clips_to_unit_range():
    out = apply_hu_window(np.array([[[-1000.0, -135.0, 40.0, 215.0, 3000.0]]]), (-135.0, 215.0))
    assert out.min() == 0.0 and out.max() == 1.0
    assert out[0, 0, 0] == 0.0   # air clipped to the window floor
    assert out[0, 0, 4] == 1.0   # bone clipped to the window ceiling
    assert 0.0 < out[0, 0, 2] < 1.0


def test_normalise_intensity_is_zero_mean_unit_variance():
    out = normalise_intensity(np.random.RandomState(1).normal(5, 3, (8, 16, 16)))
    assert abs(float(out.mean())) < 1e-5
    assert abs(float(out.std()) - 1.0) < 1e-5


def test_normalise_intensity_handles_constant_volume():
    """A constant volume carries no signal and must not be amplified by a
    near-zero divisor."""
    out = normalise_intensity(np.full((4, 4, 4), 7.0, dtype=np.float32))
    assert np.all(np.isfinite(out)) and float(np.abs(out).max()) == 0.0


def test_preprocess_volume_windows_then_normalises():
    volume = np.random.RandomState(2).uniform(-1000, 1000, (6, 8, 8))
    cfg = PreprocessConfig()
    assert np.allclose(
        preprocess_volume(volume, cfg),
        normalise_intensity(apply_hu_window(volume, cfg.hu_window)),
    )


def test_preprocess_config_defaults_match_configs_default_yaml():
    cfg = PreprocessConfig()
    assert cfg.hu_window == (-135.0, 215.0)
    assert cfg.spacing == (1.0, 1.0, 2.0)


# --- slice windows ---------------------------------------------------------

def test_window_is_centred_and_correctly_sized():
    volume = _volume()
    assert volume.get_window(10, 5).shape == (5, 16, 16)
    assert np.array_equal(volume.get_window(10, 1)[0], volume.get_window(10, 5)[2])


def test_window_clamps_at_volume_edges_instead_of_zero_padding():
    """A zero-padded edge window would read to the gate like "the organ ended
    here" rather than "the scan ended here"."""
    volume = _volume()
    bottom = volume.get_window(0, 5)
    top = volume.get_window(volume.num_slices - 1, 5)
    assert bottom.shape == top.shape == (5, 16, 16)
    for i in range(3):
        assert np.array_equal(bottom[i], bottom[2])   # first real slice repeated below index 0
    for i in range(2, 5):
        assert np.array_equal(top[i], top[2])         # last real slice repeated above the end
    assert float(np.abs(bottom).sum()) > 0.0


def test_num_slices_matches_the_volume():
    assert _volume(n_slices=17).num_slices == 17


# --- gate labels -----------------------------------------------------------

def test_slice_labels_distinguishes_left_from_right():
    """Regression test: a single shared gland label made right_present a copy
    of left_present, so the two gate heads trained on the same target."""
    mask = np.zeros((6, 8, 8), dtype=np.int16)
    mask[1, 2, 2] = RIGHT_ADRENAL_LABEL
    mask[2, 3, 3] = LEFT_ADRENAL_LABEL
    volume = PatientVolume(patient_id="lr", image=np.zeros((6, 8, 8), np.float32), mask=mask)

    right_only = volume.slice_labels(1)
    assert right_only["right_present"] == 1 and right_only["left_present"] == 0

    left_only = volume.slice_labels(2)
    assert left_only["left_present"] == 1 and left_only["right_present"] == 0

    empty = volume.slice_labels(4)
    assert empty == {"left_present": 0, "right_present": 0}


def test_slice_labels_returns_none_without_a_mask():
    volume = PatientVolume(patient_id="nomask", image=np.zeros((4, 8, 8), np.float32))
    assert volume.slice_labels(0) is None
    assert volume.get_seg_target(0) is None


def test_get_seg_target_returns_the_centre_slice_mask():
    volume = _volume()
    assert volume.get_seg_target(9).shape == (16, 16)
    assert int(volume.get_seg_target(9).sum()) == 16   # the 4x4 block on a positive slice
    assert int(volume.get_seg_target(0).sum()) == 0


# --- NIfTI loading ---------------------------------------------------------

def _write_case(directory, n_slices=12, size=32, with_mask=True):
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(0)
    image_xyz = rng.uniform(-1000, 1000, (size, size, n_slices)).astype(np.float32)
    affine = np.diag([0.8, 0.8, 5.0, 1.0])
    nib.save(nib.Nifti1Image(image_xyz, affine), directory / "image.nii.gz")
    if with_mask:
        mask_xyz = np.zeros((size, size, n_slices), np.int16)
        mask_xyz[10:14, 10:14, 4:7] = RIGHT_ADRENAL_LABEL
        nib.save(nib.Nifti1Image(mask_xyz, affine), directory / "mask.nii.gz")


def test_nifti_is_reoriented_to_slice_first(tmp_path):
    """NIfTI is (X, Y, Z) on disk; everything downstream indexes slice-first.
    Z is deliberately the smallest axis here — the reorientation must not be
    inferred from axis lengths."""
    _write_case(tmp_path / "pt001", n_slices=12, size=32)
    volume = load_patient_volume(str(tmp_path), "pt001")
    assert volume.num_slices == 12
    assert volume.get_window(0, 3).shape == (3, 32, 32)


def test_loaded_volume_is_preprocessed(tmp_path):
    _write_case(tmp_path / "pt001")
    volume = load_patient_volume(str(tmp_path), "pt001")
    window = volume.get_window(6, 1)
    assert window.dtype == np.float32
    assert abs(float(volume._image.mean())) < 1e-4 and abs(float(volume._image.std()) - 1.0) < 1e-4


def test_discover_patients_is_sorted_and_skips_non_cases(tmp_path):
    for name in ("pt003", "pt001", "pt002"):
        _write_case(tmp_path / name)
    (tmp_path / "notes").mkdir()
    assert discover_patients(str(tmp_path)) == ["pt001", "pt002", "pt003"]


def test_discover_patients_tolerates_a_missing_root(tmp_path):
    assert discover_patients(str(tmp_path / "does_not_exist")) == []


def test_mask_is_optional(tmp_path):
    _write_case(tmp_path / "pt001", with_mask=False)
    volume = load_patient_volume(str(tmp_path), "pt001")
    assert volume.get_seg_target(0) is None


# --- splits ----------------------------------------------------------------

def test_kfold_partitions_patients_exactly_once_with_no_leakage():
    ids = [f"p{i:02d}" for i in range(10)]
    folds = list(patient_level_kfold(ids, n_folds=5, seed=42))
    assert len(folds) == 5
    assert sorted(pid for _, val in folds for pid in val) == sorted(ids)
    for train, val in folds:
        assert_no_patient_leakage(train, val)
        assert sorted(train + val) == sorted(ids)


def test_kfold_handles_a_patient_count_not_divisible_by_folds():
    ids = [f"p{i:02d}" for i in range(11)]
    folds = list(patient_level_kfold(ids, n_folds=5, seed=0))
    assert sorted(len(val) for _, val in folds) == [2, 2, 2, 2, 3]
    assert sorted(pid for _, val in folds for pid in val) == sorted(ids)


def test_kfold_is_deterministic_for_a_seed():
    ids = [f"p{i:02d}" for i in range(12)]
    assert [v for _, v in patient_level_kfold(ids, 4, 7)] == [v for _, v in patient_level_kfold(ids, 4, 7)]


def test_kfold_ignores_input_ordering():
    """Folds are keyed by patient id, not by the order the filesystem
    happened to list them in."""
    ids = [f"p{i:02d}" for i in range(10)]
    assert [v for _, v in patient_level_kfold(ids, 5, 3)] == \
           [v for _, v in patient_level_kfold(list(reversed(ids)), 5, 3)]


def test_leakage_assertion_fires_on_an_overlapping_split():
    with pytest.raises(AssertionError):
        assert_no_patient_leakage(["a", "b"], ["b", "c"])
    with pytest.raises(AssertionError):
        assert_no_patient_leakage(["a"], ["b"], test_ids=["a"])

"""Patient-level slice-window dataset for the gate and the 2.5D segmenter.

Both stages consume the same underlying unit: a window of `slice_window`
consecutive axial slices centred on a candidate slice. The gate treats the
window as a sequence (one slice per SNN/LSTM time step); the segmenter
treats it as stacked input channels (the standard 2.5D formulation).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import nibabel as nib
except ImportError:  # pragma: no cover
    nib = None

from src.data.preprocessing import PreprocessConfig, preprocess_volume

# AMOS22 label IDs, verified against its dataset.json.
RIGHT_ADRENAL_LABEL = 11
LEFT_ADRENAL_LABEL = 12


@dataclass
class SliceWindowSample:
    patient_id: str
    center_index: int
    slices: np.ndarray            # (slice_window, H, W)
    labels: dict[str, int] | None = None
    mask: np.ndarray | None = None  # (H, W) segmentation ground truth for the center slice


class PatientVolume:
    """Lazily-loaded CT volume + optional mask for one patient.

    Accepts either a path to a NIfTI file (production use) or an in-memory
    numpy array (unit tests / synthetic data), so the dataset logic below can
    be exercised without real patient data.
    """

    def __init__(self, patient_id: str, image=None, mask=None, image_path: str | None = None,
                 mask_path: str | None = None, preprocess_cfg: PreprocessConfig | None = None):
        self.patient_id = patient_id
        self.preprocess_cfg = preprocess_cfg or PreprocessConfig()

        if image is not None:
            # In-memory arrays are expected to already be in (Z, H, W) order
            # — this is the convention used everywhere else in this codebase
            # (get_window, slice_labels, etc.) and by the unit tests, so no
            # reorientation is applied here.
            self._image = image
            self._mask = mask
        else:
            if nib is None:
                raise ImportError("nibabel is required to load NIfTI volumes; pip install nibabel")
            # nibabel returns NIfTI volumes in (X, Y, Z) order; reorient to
            # this codebase's (Z, H, W) convention unconditionally (do NOT
            # infer orientation by comparing axis lengths — Z can legitimately
            # be larger than H/W, e.g. thin-slice CT, so that heuristic is
            # unreliable and was a real bug caught by test_pipeline.py).
            self._image = np.transpose(
                np.asarray(nib.load(image_path).get_fdata(), dtype=np.float32), (2, 0, 1)
            )
            self._mask = (
                np.transpose(np.asarray(nib.load(mask_path).get_fdata(), dtype=np.int16), (2, 0, 1))
                if mask_path else None
            )

        self._image = preprocess_volume(self._image, self.preprocess_cfg)

    @property
    def num_slices(self) -> int:
        return self._image.shape[0]

    def get_window(self, center_index: int, slice_window: int) -> np.ndarray:
        """Return `slice_window` consecutive slices centred on `center_index`,
        zero-padding at the volume boundary."""
        half = slice_window // 2
        lo, hi = center_index - half, center_index - half + slice_window
        pad_lo = max(0, -lo)
        pad_hi = max(0, hi - self.num_slices)
        lo_clamped, hi_clamped = max(lo, 0), min(hi, self.num_slices)

        window = self._image[lo_clamped:hi_clamped]
        if pad_lo or pad_hi:
            window = np.pad(window, ((pad_lo, pad_hi), (0, 0), (0, 0)), mode="edge")
        return window

    def slice_labels(self, center_index: int, left_mask_value: int = LEFT_ADRENAL_LABEL,
                     right_mask_value: int = RIGHT_ADRENAL_LABEL):
        """Derive left_present / right_present for one slice.

        Left and right must come from *different* label values. A single
        shared `gland_mask_value` makes the two gate heads numerically
        identical, so `right_present` would train on the left gland's target
        and the left/right sensitivity columns in the ablation table would be
        the same number twice.

        Defaults are AMOS22's own IDs (right adrenal gland = 11, left = 12),
        verified against its dataset.json. If your cohort stores a binarised
        gland mask, left and right are not recoverable from it — pass the same
        value for both and report a single combined head rather than two.
        """
        if self._mask is None:
            return None
        sl = self._mask[center_index]
        return {
            "left_present": int(np.any(sl == left_mask_value)),
            "right_present": int(np.any(sl == right_mask_value)),
        }

    def get_seg_target(self, center_index: int) -> np.ndarray | None:
        if self._mask is None:
            return None
        return self._mask[center_index]


def discover_patients(root_dir: str) -> list[str]:
    """List patient ids from `<root_dir>/<patient_id>/image.nii.gz`."""
    root = Path(root_dir)
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and (p / "image.nii.gz").exists())


def load_patient_volume(root_dir: str, patient_id: str, preprocess_cfg: PreprocessConfig | None = None) -> PatientVolume:
    root = Path(root_dir) / patient_id
    return PatientVolume(
        patient_id=patient_id,
        image_path=str(root / "image.nii.gz"),
        mask_path=str(root / "mask.nii.gz") if (root / "mask.nii.gz").exists() else None,
        preprocess_cfg=preprocess_cfg,
    )

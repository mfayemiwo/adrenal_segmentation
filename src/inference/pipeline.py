"""End-to-end inference: gate -> uncertainty-aware buffer -> 2.5D segment -> TTA + connected-component pruning.

This is the object every ablation arm in `scripts/run_ablation.py` ultimately
instantiates with a different `gate_model` (or none, for the full-volume
baselines), so the full-volume outcome metrics in
`src/evaluation/metrics.py` are computed identically across arms.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from src.data.dataset import PatientVolume
from src.postprocessing.connected_components import remove_small_components
from src.postprocessing.tta import tta_predict
from src.training.uncertainty_buffer import BufferConfig, build_slice_inclusion_mask


@dataclass
class PipelineConfig:
    slice_window: int = 5
    buffer_cfg: BufferConfig = field(default_factory=BufferConfig)
    operating_threshold: float = 0.5
    use_tta: bool = True
    tta_transforms: tuple[str, ...] = ("hflip", "vflip", "rot90")
    remove_small_components: bool = True
    min_component_voxels: int = 30
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class AdrenalSegmentationPipeline:
    def __init__(self, gate_model, segmenter_model, cfg: PipelineConfig, gate_head: str = "left_present"):
        self.gate_model = gate_model
        self.segmenter_model = segmenter_model
        self.cfg = cfg
        self.gate_head = gate_head
        self.device = torch.device(cfg.device)

    @torch.no_grad()
    def _gate_probs_for_volume(self, volume: PatientVolume) -> np.ndarray:
        """Slide the gate window across every slice in the volume and return
        one probability per slice for `self.gate_head`. If `self.gate_model`
        is None, every slice is marked positive (the full-volume, no-gate
        baseline used by ablation arms A/B/H)."""
        n = volume.num_slices
        if self.gate_model is None:
            return np.ones(n, dtype=np.float32)

        self.gate_model.eval()
        # The gate's own window size (e.g. 1 for the plain single-slice
        # ResNet18 baseline, arm C) is independent of the segmenter's window
        # size in `self.cfg.slice_window` — a classifier ablation that looks
        # at fewer/more slices than the segmenter is a legitimate arm and
        # must not be forced onto the segmenter's window size.
        gate_window = getattr(self.gate_model, "slice_window", self.cfg.slice_window)

        probs = np.zeros(n, dtype=np.float32)
        for center in range(n):
            window = volume.get_window(center, gate_window)  # (T, H, W)
            x = torch.from_numpy(window).float().unsqueeze(1).unsqueeze(0).to(self.device)  # (1, T, 1, H, W)
            logits, _ = self.gate_model(x)
            probs[center] = torch.sigmoid(logits[self.gate_head]).item()
        return probs

    @torch.no_grad()
    def _segment_slice(self, window: np.ndarray) -> np.ndarray:
        """window: (in_channels, H, W) stacked neighbouring slices ->
        binary mask for the centre slice."""
        self.segmenter_model.eval()  # inference-only: avoids BatchNorm errors on batch size 1
        x = torch.from_numpy(window).float().unsqueeze(0).to(self.device)  # (1, C, H, W)
        if self.cfg.use_tta:
            probs = tta_predict(self.segmenter_model, x, self.cfg.tta_transforms)
        else:
            probs = torch.sigmoid(self.segmenter_model(x))
        mask = (probs.squeeze(0).squeeze(0).cpu().numpy() >= 0.5).astype(np.uint8)
        return mask

    def run(self, volume: PatientVolume) -> dict:
        """Returns a dict with the full-volume predicted mask stack and the
        diagnostics needed for the outcome metrics (which slices were sent
        to the segmenter, how much the uncertainty buffer expanded, etc.)."""
        gate_probs = self._gate_probs_for_volume(volume)
        inclusion_mask = build_slice_inclusion_mask(gate_probs, self.cfg.buffer_cfg) \
            if self.gate_model is not None else np.ones_like(gate_probs, dtype=bool)

        h, w = volume.get_window(0, self.cfg.slice_window).shape[1:]
        pred_stack = np.zeros((volume.num_slices, h, w), dtype=np.uint8)

        n_components_removed = 0
        for idx in np.where(inclusion_mask)[0]:
            window = volume.get_window(idx, self.cfg.slice_window)
            mask = self._segment_slice(window)
            if self.cfg.remove_small_components:
                mask, removed = remove_small_components(mask, self.cfg.min_component_voxels)
                n_components_removed += removed
            pred_stack[idx] = mask

        return {
            "patient_id": volume.patient_id,
            "pred_stack": pred_stack,
            "gate_probs": gate_probs,
            "inclusion_mask": inclusion_mask,
            "slices_segmented": int(inclusion_mask.sum()),
            "total_slices": volume.num_slices,
            "components_removed": n_components_removed,
        }

"""Phase 2: anatomy-aware spiking temporal refinement.

Prototype design: a shared frozen U-Net predicts masks for the centre 2.5D
window and its immediately adjacent shifted windows. Changes in *predicted
gland probability* (rather than CT intensity) are encoded as signed events.
A small LIF memory refines the centre prediction only where its uncertainty
or disagreement with neighbouring predictions warrants a correction.

The analog control differs only in the recurrent unit. Both branches have
identical input events, decoder, gating and fixed shared U-Net weights.

This is a window-local anatomical memory prototype, not a global 3D tracker.
No claims of energy savings follow from its spike rate alone.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.delta_snn_refiner import (
    DeltaSNNRefinedSegmenter, RefinerStats,
)


class AnatomyMemorySegmenter(DeltaSNNRefinedSegmenter):
    def __init__(self, *, variant: str, **kwargs):
        if variant not in ("anatomy_snn", "anatomy_analog"):
            raise ValueError(f"Unknown anatomy-memory variant: {variant}")
        super().__init__(
            event_mode="delta",
            spiking=variant == "anatomy_snn",
            **kwargs,
        )
        self.variant = variant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"expected (B,{self.in_channels},H,W), got {tuple(x.shape)}"
            )
        # Shift the five-slice window by one along z. Replicate its edge
        # slices, just as a clamped slice sampler does at volume boundaries.
        before = torch.cat((x[:, :1], x[:, :-1]), dim=1)
        after = torch.cat((x[:, 1:], x[:, -1:]), dim=1)

        # Phase 2 assumes a frozen shared baseline for a strict comparison.
        # Predicting neighbouring masks under no_grad also prevents storing
        # large U-Net activations for those auxiliary passes.
        with torch.no_grad():
            left_logits = self.base_model(before)
            right_logits = self.base_model(after)
        centre_logits = self.base_model(x)

        probs = torch.stack(
            (left_logits.sigmoid(), centre_logits.sigmoid(), right_logits.sigmoid()),
            dim=1,
        )  # B,3,classes,H,W

        # Anatomical change signal: difference in foreground confidence.
        # Summing the channel differences preserves signed changes and
        # spatially separated left/right gland trajectories.
        evidence = probs.mean(dim=2)
        deltas = evidence[:, 1:] - evidence[:, :-1]
        tau = self.event_encoder.threshold.to(device=x.device, dtype=x.dtype)
        events = torch.stack(
            (F.relu(deltas - tau), F.relu(-deltas - tau)), dim=2
        )  # B,2,2,H,W

        correction, spike_rate = self.refiner(events)

        # Selective correction where the centre mask is locally uncertain or
        # inconsistent with the immediately neighbouring predictions.
        p = centre_logits.sigmoid()
        uncertainty = (4.0 * p * (1.0 - p)).clamp(0, 1)
        mismatch = (
            (p - left_logits.sigmoid()).abs()
            + (p - right_logits.sigmoid()).abs()
        ).mul(0.5).clamp(0, 1)
        gate = torch.maximum(uncertainty, mismatch)
        scale = self.residual_scale.tanh()

        self.last_stats = RefinerStats(
            spike_rate=spike_rate.detach(),
            event_density=(events > 0).float().mean().detach(),
            event_magnitude=events.mean().detach(),
        )
        return centre_logits + scale * gate * correction


def build_anatomy_memory_segmenter(*, variant: str, **kwargs):
    return AnatomyMemorySegmenter(variant=variant, **kwargs)

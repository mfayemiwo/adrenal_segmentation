"""Loss functions.

The gate is optimised for sensitivity, not accuracy (per the design review):
a missed positive slice permanently removes true adrenal tissue from
the segmenter's input, while a false positive only costs one wasted
segmentation attempt. `FocalBCELoss` with class weighting keeps rare
positive slices from being drowned out by the (numerically larger) negative
class; the operating threshold is chosen post-hoc on the validation PR curve
to hit `target_sensitivity`, not by taking the default 0.5 cutoff.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalBCELoss(nn.Module):
    def __init__(self, gamma: float = 2.0, pos_weight: torch.Tensor | float | None = None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        pos_weight = self.pos_weight
        if pos_weight is not None and not torch.is_tensor(pos_weight):
            pos_weight = torch.tensor(pos_weight, device=logits.device, dtype=logits.dtype)

        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - p_t).clamp(min=1e-6) ** self.gamma
        return (focal_weight * bce).mean()


class MultiHeadFocalBCELoss(nn.Module):
    """Sums FocalBCELoss across the gate's named heads, with optional
    per-head class weights (e.g. if one side's positive slices are rarer
    than the other's in a given cohort, that head can be given a larger
    positive weight)."""

    def __init__(self, heads: tuple[str, ...], gamma: float = 2.0,
                 pos_weights: dict[str, float] | None = None):
        super().__init__()
        self.heads = heads
        self.gamma = gamma
        self.pos_weights = pos_weights or {}
        self.losses = nn.ModuleDict({
            name: FocalBCELoss(gamma=gamma, pos_weight=self.pos_weights.get(name))
            for name in heads
        })

    def forward(self, logits: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]):
        per_head = {name: self.losses[name](logits[name], targets[name]) for name in self.heads}
        total = sum(per_head.values())
        return total, per_head


class DiceLoss(nn.Module):
    """Soft Dice, optionally pooled over the batch.

    `batch_dice=True` accumulates intersection and union across the whole batch
    before dividing, instead of scoring each sample separately and averaging.
    This matters enormously once left and right adrenal are separate output
    channels, because many slices contain one gland but not the other, leaving
    that channel's target completely empty.

    On an empty target, per-sample Dice is ~1.0 for any non-zero prediction and
    decreases only as the prediction goes to zero (measured: p=0.5 -> 0.9995,
    p=0.018 -> 0.9866). Every such sample therefore contributes a near-maximal
    loss whose gradient says "predict nothing", and with two channels roughly
    half the (sample, channel) pairs are in that state. In practice the network
    finds the all-zero solution within a handful of epochs and never leaves it.

    Pooling over the batch removes the pathology: an empty channel in one slice
    only adds to the denominator, while other slices in the batch supply the
    intersection, so the loss stays informative. This is what nnU-Net does for
    small structures, and for the same reason.
    """

    def __init__(self, smooth: float = 1.0, batch_dice: bool = True):
        super().__init__()
        self.smooth = smooth
        self.batch_dice = batch_dice

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets = targets.float()
        spatial = tuple(range(2, probs.ndim))
        dims = (0,) + spatial if self.batch_dice else spatial
        intersection = (probs * targets).sum(dim=dims)
        union = probs.sum(dim=dims) + targets.sum(dim=dims)
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class DiceFocalLoss(nn.Module):
    """Combined loss for the segmenter: Dice for the class-imbalance-robust
    overlap term, focal BCE for hard-negative slices (empty masks on
    neighbouring-organ / boundary slices) so the segmenter is explicitly
    penalised for hallucinating masks where none should exist."""

    def __init__(self, dice_weight: float = 1.0, focal_weight: float = 1.0, gamma: float = 2.0,
                 batch_dice: bool = True):
        super().__init__()
        self.dice = DiceLoss(batch_dice=batch_dice)
        self.focal = FocalBCELoss(gamma=gamma)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * self.dice(logits, targets) + self.focal_weight * self.focal(logits, targets)

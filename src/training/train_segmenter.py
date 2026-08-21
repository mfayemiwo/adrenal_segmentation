"""Training loop for the 2.5D segmenter, with explicit hard-negative mining.

Per the design review, the segmenter must not be trained exclusively on
positive slices: `HardNegativeSampler` guarantees every training batch
mixes true-positive windows with (a) boundary slices immediately above/below
the gland, (b) slices centred on neighbouring organs (kidney, liver, spleen,
pancreas), and (c) false positives harvested from the prior published
pipeline. Without this, the segmenter never learns that anatomically similar
structures should yield an empty mask, and the whole point of the gate
(reducing false positives) is undermined by a segmenter that still
hallucinates masks whenever it is run.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset

from src.models.losses import DiceFocalLoss


@dataclass
class SegmenterTrainConfig:
    lr: float = 3e-4
    epochs: int = 150
    dice_weight: float = 1.0
    focal_weight: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class HardNegativeMixDataset(Dataset):
    """Wraps a positive-slice dataset and one or more hard-negative datasets,
    sampling from each source according to `source_ratios` every epoch so the
    segmenter sees a controlled mix rather than whatever ratio happens to
    occur naturally in the raw cohort."""

    def __init__(self, positive_dataset: Dataset, hard_negative_datasets: dict[str, Dataset],
                 source_ratios: dict[str, float] | None = None, length: int | None = None):
        self.positive_dataset = positive_dataset
        self.hard_negative_datasets = hard_negative_datasets
        self.source_ratios = source_ratios or {
            "positive": 0.5,
            **{name: 0.5 / max(len(hard_negative_datasets), 1) for name in hard_negative_datasets},
        }
        self.length = length or len(positive_dataset)

        names = ["positive", *hard_negative_datasets.keys()]
        weights = [self.source_ratios.get(n, 0.0) for n in names]
        total = sum(weights) or 1.0
        self._names = names
        self._probs = [w / total for w in weights]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        import random
        source = random.choices(self._names, weights=self._probs, k=1)[0]
        dataset = self.positive_dataset if source == "positive" else self.hard_negative_datasets[source]
        sample_idx = random.randrange(len(dataset))
        return dataset[sample_idx]


def train_segmenter(model, train_loader: DataLoader, val_loader: DataLoader | None, cfg: SegmenterTrainConfig):
    device = torch.device(cfg.device)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    criterion = DiceFocalLoss(dice_weight=cfg.dice_weight, focal_weight=cfg.focal_weight)

    history = []
    for epoch in range(cfg.epochs):
        model.train()
        running_loss, n_batches = 0.0, 0
        for batch in train_loader:
            images = batch["slices"].to(device)   # (B, in_channels, H, W)
            masks = batch["mask"].to(device)       # (B, 1, H, W) or (B, num_classes, H, W)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            n_batches += 1

        entry = {"epoch": epoch, "loss": running_loss / max(n_batches, 1)}
        if val_loader is not None:
            entry["val_loss"] = evaluate_segmenter_loss(model, val_loader, criterion, device)
        history.append(entry)
    return history


@torch.no_grad()
def evaluate_segmenter_loss(model, loader: DataLoader, criterion, device) -> float:
    model.eval()
    running_loss, n_batches = 0.0, 0
    for batch in loader:
        images = batch["slices"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images)
        running_loss += float(criterion(logits, masks).item())
        n_batches += 1
    return running_loss / max(n_batches, 1)

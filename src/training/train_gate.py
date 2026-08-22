"""Training loop shared by the SNN gate and the CNN-LSTM baseline gate.

Both models expose the same forward signature — `forward(x) -> (logits_dict,
spike_stats)` — specifically so this loop, and the downstream evaluation
code, can run unmodified across ablation arms C/D/E/G in
`scripts/run_ablation.py`. That symmetry is what makes the SNN-vs-CNN-LSTM
comparison fair rather than incidental.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader

from src.models.losses import MultiHeadFocalBCELoss


@dataclass
class GateTrainConfig:
    heads: tuple[str, ...] = ("left_present", "right_present")
    lr: float = 1e-3
    epochs: int = 60
    focal_gamma: float = 2.0
    pos_weights: dict = field(default_factory=dict)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def train_one_epoch(model, loader: DataLoader, optimizer, criterion, device) -> dict:
    model.train()
    running_total = 0.0
    running_per_head = {name: 0.0 for name in criterion.heads}
    n_batches = 0

    for batch in loader:
        slices = batch["slices"].to(device)          # (B, T, C, H, W)
        targets = {name: batch["labels"][name].to(device) for name in criterion.heads}

        optimizer.zero_grad()
        logits, _ = model(slices)
        loss, per_head = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_total += float(loss.item())
        for name, val in per_head.items():
            running_per_head[name] += float(val.item())
        n_batches += 1

    n_batches = max(n_batches, 1)
    return {
        "loss": running_total / n_batches,
        **{f"loss_{k}": v / n_batches for k, v in running_per_head.items()},
    }


def train_gate(model, train_loader: DataLoader, val_loader: DataLoader | None, cfg: GateTrainConfig):
    device = torch.device(cfg.device)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    criterion = MultiHeadFocalBCELoss(heads=cfg.heads, gamma=cfg.focal_gamma, pos_weights=cfg.pos_weights)

    history = []
    for epoch in range(cfg.epochs):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        entry = {"epoch": epoch, **train_metrics}
        if val_loader is not None:
            entry.update({f"val_{k}": v for k, v in evaluate_loss(model, val_loader, criterion, device).items()})
        history.append(entry)
    return history


@torch.no_grad()
def evaluate_loss(model, loader: DataLoader, criterion, device) -> dict:
    model.eval()
    running_total, n_batches = 0.0, 0
    for batch in loader:
        slices = batch["slices"].to(device)
        targets = {name: batch["labels"][name].to(device) for name in criterion.heads}
        logits, _ = model(slices)
        loss, _ = criterion(logits, targets)
        running_total += float(loss.item())
        n_batches += 1
    return {"loss": running_total / max(n_batches, 1)}

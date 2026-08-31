"""Normalisation options shared by both slice gates.

Both `SpikingSliceGate` and `CNNLSTMSliceGate` apply their normalisation once
per time step, which makes the choice of normaliser a shared design decision
rather than a detail of either model — and it has to be swept identically on
both, or a difference in results is a difference in tuning effort rather than
in architecture.

Living here, rather than being duplicated in each model, also means the two
cannot drift apart, and that the CNN-LSTM baseline stays importable without
snnTorch installed.

    "batch"  one BatchNorm2d applied at every time step. Its running statistics
             are updated T times per forward pass, each on a different
             distribution (slice t), and one set is then applied to every time
             step at eval. That train/eval mismatch is the leading explanation
             for the spiking gate's generalisation gap, and switching it out
             was worth +0.05 PR-AUC on the first sweep.
    "step"   a separate BatchNorm2d per time step. Removes the mismatch at the
             cost of T x the (small) BN parameter count.
    "group"  GroupNorm: no running statistics at all, so train and eval behave
             identically. Cheapest correct option, and the strongest single
             change found so far.
"""
from __future__ import annotations

import torch
import torch.nn as nn

NORMS = ("batch", "step", "group")
GROUP_NORM_GROUPS = 8          # divides 32, 64 and 128


def build_norm(kind: str, channels: int, time_steps: int) -> nn.Module:
    """Returns a module applied directly, or indexed by time step for 'step'."""
    if kind == "batch":
        return nn.BatchNorm2d(channels)
    if kind == "group":
        return nn.GroupNorm(GROUP_NORM_GROUPS, channels)
    if kind == "step":
        return nn.ModuleList([nn.BatchNorm2d(channels) for _ in range(time_steps)])
    raise ValueError(f"unknown norm '{kind}' (expected one of {NORMS})")


def apply_norm(module: nn.Module, x: torch.Tensor, t: int) -> torch.Tensor:
    return module[t](x) if isinstance(module, nn.ModuleList) else module(x)

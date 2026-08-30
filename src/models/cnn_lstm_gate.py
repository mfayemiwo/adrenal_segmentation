"""Matched-capacity CNN+LSTM gate — the mandatory baseline for the SNN gate.

This model is deliberately built with the same conv channel progression as
`SpikingSliceGate` (32 -> 64 -> 128 -> 128) and consumes the same
(B, T, C, H, W) slice window, replacing LIF membrane integration with an
LSTM over the per-slice CNN feature vectors. If `SpikingSliceGate` cannot
beat this baseline on sensitivity / NPV / F2 / PR-AUC, the SNN choice is not
earning its added training complexity and the paper's contribution should be
reframed accordingly — see `scripts/run_ablation.py`, arm D vs. arm E.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.residual = in_ch == out_ch and stride == 1
        self.skip = None if self.residual else nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        cur = self.bn(self.conv(x))
        cur = cur + x if self.residual else cur + self.skip(x)
        return self.act(cur)


class CNNLSTMSliceGate(nn.Module):
    def __init__(self, slice_window: int = 5, in_channels: int = 1, hidden_dim: int = 256,
                 lstm_layers: int = 1, lstm_hidden: int = 128,
                 heads: tuple[str, ...] = ("left_present", "right_present")):
        super().__init__()
        self.slice_window = slice_window
        self.heads = heads

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.block1 = _ConvBlock(32, 64, stride=2)
        self.block2 = _ConvBlock(64, 128, stride=2)
        self.block3 = _ConvBlock(128, 128, stride=1)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # lstm_hidden defaults to 128, reproducing this model as first written.
        # Note that at 128 it is NOT capacity-matched to SpikingSliceGate: the four
        # LSTM gate matrices add ~132k parameters for which LIF membrane integration
        # has no counterpart, so the "matched-capacity baseline" claim only holds
        # once this is tuned (32 matches within a few percent). scripts/
        # train_slice_gate.py prints both counts at startup and warns on a mismatch.
        self.lstm = nn.LSTM(input_size=128, hidden_size=lstm_hidden,
                            num_layers=lstm_layers, batch_first=True)
        self.head = nn.Linear(lstm_hidden, hidden_dim)
        self.act = nn.ReLU(inplace=True)
        self.outputs = nn.ModuleDict({name: nn.Linear(hidden_dim, 1) for name in heads})

    def _encode_slice(self, slice_t: torch.Tensor) -> torch.Tensor:
        cur = self.stem(slice_t)
        cur = self.block1(cur)
        cur = self.block2(cur)
        cur = self.block3(cur)
        return self.pool(cur).flatten(1)

    def forward(self, x: torch.Tensor):
        B, T, C, H, W = x.shape
        assert T == self.slice_window, f"expected {self.slice_window} time steps, got {T}"

        per_slice_feats = [self._encode_slice(x[:, t]) for t in range(T)]
        seq = torch.stack(per_slice_feats, dim=1)  # (B, T, 128)

        _, (h_n, _) = self.lstm(seq)
        feat = self.act(self.head(h_n[-1]))
        logits = {name: layer(feat).squeeze(-1) for name, layer in self.outputs.items()}
        return logits, {"per_timestep_spike_count": None, "total_spike_count": None}


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

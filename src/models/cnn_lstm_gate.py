"""Matched-capacity CNN+LSTM gate — the mandatory baseline for the SNN gate.

This model is deliberately built with the same conv channel progression as
`SpikingSliceGate` (32 -> 64 -> 128 -> 128) and consumes the same
(B, T, C, H, W) slice window, replacing LIF membrane integration with an
LSTM over the per-slice CNN feature vectors. If `SpikingSliceGate` cannot
beat this baseline on sensitivity / NPV / F2 / PR-AUC, the SNN choice is not
earning its added training complexity and the paper's contribution should be
reframed accordingly — see `scripts/run_ablation.py`, arm D vs. arm E.

--------------------------------------------------------------------------
WHY THIS MODEL ALSO HAS OPTIONS
--------------------------------------------------------------------------
After the first sweep the spiking gate had been given nine configurations and
this baseline exactly one, which flipped the comparison from being biased
AGAINST the SNN (the capacity confound) to being biased in its favour. A
difference in results would then be a difference in tuning effort rather than
in architecture, and no reviewer would accept it.

So the knobs below mirror the SNN's, one for one, and `scripts/
sweep_cnnlstm_gate.sbatch` spends the same number of attempts on this model.
Every default reproduces the model as first written, so prior runs stay
reproducible.

`norm` — identical to the spiking gate's, and for the same reason: this model
    also applies its normalisation once per time step, so the BatchNorm
    train/eval mismatch that cost the SNN ~0.05 PR-AUC exists here too. It is
    an open question whether the baseline suffers from it as much; the sweep
    is what answers that.

`readout` — how the T LSTM outputs become one feature vector.
    "last" (default, original): the final hidden state. Order-sensitive by
           construction, and the natural analogue of the spiking gate's
           integrator readout.
    "mean": average of the outputs across time steps. The analogue of the
           spiking gate's rate code — it discards most ordering information,
           so comparing the two here measures the same thing on the baseline
           that `readout=rate` vs `integrator` measures on the SNN.

`lstm_hidden` — 128 reproduces the model as first written, which is NOT
    capacity-matched to the spiking gate: the four LSTM gate matrices add
    ~132k parameters for which LIF membrane integration has no counterpart.
    32 matches the two within ~2%. `scripts/train_slice_gate.py` prints both
    counts at startup and warns on a mismatch.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.norm import NORMS, apply_norm, build_norm   # noqa: F401  (re-exported)

READOUTS = ("last", "mean")


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1,
                 norm: str = "batch", time_steps: int = 5):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = build_norm(norm, out_ch, time_steps)
        self.act = nn.ReLU(inplace=True)
        self.residual = in_ch == out_ch and stride == 1
        self.skip = None if self.residual else nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)

    def forward(self, x, t: int = 0):
        cur = apply_norm(self.bn, self.conv(x), t)
        cur = cur + x if self.residual else cur + self.skip(x)
        return self.act(cur)


class CNNLSTMSliceGate(nn.Module):
    def __init__(self, slice_window: int = 5, in_channels: int = 1, hidden_dim: int = 256,
                 lstm_layers: int = 1, lstm_hidden: int = 128,
                 heads: tuple[str, ...] = ("left_present", "right_present"),
                 norm: str = "batch", readout: str = "last"):
        super().__init__()
        if readout not in READOUTS:
            raise ValueError(f"unknown readout '{readout}' (expected one of {READOUTS})")
        self.slice_window = slice_window
        self.heads = heads
        self.readout = readout
        self.norm_kind = norm

        # A per-step norm needs the time index, which nn.Sequential cannot pass,
        # so it is held separately and applied by hand in _encode_slice.
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False),
            build_norm(norm, 32, slice_window) if norm != "step" else nn.Identity(),
            nn.ReLU(inplace=True),
        )
        self.stem_norm = build_norm(norm, 32, slice_window) if norm == "step" else None

        block_kw = dict(norm=norm, time_steps=slice_window)
        self.block1 = _ConvBlock(32, 64, stride=2, **block_kw)
        self.block2 = _ConvBlock(64, 128, stride=2, **block_kw)
        self.block3 = _ConvBlock(128, 128, stride=1, **block_kw)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.lstm = nn.LSTM(input_size=128, hidden_size=lstm_hidden,
                            num_layers=lstm_layers, batch_first=True)
        self.head = nn.Linear(lstm_hidden, hidden_dim)
        self.act = nn.ReLU(inplace=True)
        self.outputs = nn.ModuleDict({name: nn.Linear(hidden_dim, 1) for name in heads})

    def _encode_slice(self, slice_t: torch.Tensor, t: int = 0) -> torch.Tensor:
        cur = self.stem(slice_t)
        if self.stem_norm is not None:
            cur = self.act(apply_norm(self.stem_norm, cur, t))
        cur = self.block1(cur, t)
        cur = self.block2(cur, t)
        cur = self.block3(cur, t)
        return self.pool(cur).flatten(1)

    def forward(self, x: torch.Tensor):
        B, T, C, H, W = x.shape
        assert T == self.slice_window, f"expected {self.slice_window} time steps, got {T}"

        per_slice_feats = [self._encode_slice(x[:, t], t) for t in range(T)]
        seq = torch.stack(per_slice_feats, dim=1)  # (B, T, 128)

        out, (h_n, _) = self.lstm(seq)
        # "last" is the final hidden state (order-sensitive); "mean" averages
        # across time steps, which discards most ordering information and is
        # the baseline's analogue of the spiking gate's rate code.
        feat_in = h_n[-1] if self.readout == "last" else out.mean(dim=1)
        feat = self.act(self.head(feat_in))
        logits = {name: layer(feat).squeeze(-1) for name, layer in self.outputs.items()}
        return logits, {"per_timestep_spike_count": None, "total_spike_count": None}


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

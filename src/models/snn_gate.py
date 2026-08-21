"""Sequence-aware spiking slice gate — the primary novel contribution.

Design rationale (see docs/methodology.docx, Section 3.1):

Most SNN-for-classification papers convert a static image into a spike train
by repeating it across T identical time steps (rate or latency coding of one
image). That is difficult to motivate scientifically for a single CT slice —
the "time" axis carries no information.

Here, T time steps carry T *different*, spatially adjacent CT slices, in
their true craniocaudal order. Each LIF layer's membrane potential integrates
evidence across the slice neighbourhood exactly as it integrates evidence
across time in a conventional SNN, which gives the temporal dimension a
genuine anatomical meaning and is the thing that must be ablated against
(a) a static-repeated-slice SNN and (b) a matched-capacity CNN-LSTM to be
a defensible claim (see `src/models/cnn_lstm_gate.py` and
`scripts/run_ablation.py`, arms E vs. G vs. D).
"""
from __future__ import annotations

import torch
import torch.nn as nn

try:
    import snntorch as snn
    from snntorch import surrogate
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "snnTorch is required for the spiking gate; pip install snntorch"
    ) from exc


class _SpikingConvBlock(nn.Module):
    """Conv -> BatchNorm -> LIF, with an optional residual (ResNet-style) skip.

    Stateful: `mem` must be threaded across time steps by the caller, which is
    what makes this a *recurrent-in-time* spiking block rather than a static
    per-frame CNN applied T times.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, beta: float = 0.9,
                 threshold: float = 1.0, residual: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.lif = snn.Leaky(beta=beta, threshold=threshold, spike_grad=surrogate.fast_sigmoid())
        self.residual = residual and in_ch == out_ch and stride == 1
        self.skip = None if self.residual else nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)

    def init_mem(self, batch_size: int, spatial_size: tuple[int, int], device):
        return torch.zeros(batch_size, self.conv.out_channels, *spatial_size, device=device)

    def forward(self, x: torch.Tensor, mem: torch.Tensor):
        cur = self.bn(self.conv(x))
        if self.residual:
            cur = cur + x
        elif self.skip is not None:
            cur = cur + self.skip(x)
        spk, mem = self.lif(cur, mem)
        return spk, mem


class SpikingSliceGate(nn.Module):
    """Spiking residual CNN gate over a craniocaudal slice window.

    Parameters
    ----------
    slice_window: number of consecutive CT slices == number of SNN time steps.
    in_channels: channels per slice (1 for a single HU-windowed grayscale slice).
    heads: named binary outputs, e.g. {"left_present": 1, "right_present": 1, "tumour_present": 1}.
    """

    def __init__(self, slice_window: int = 5, in_channels: int = 1, hidden_dim: int = 256,
                 beta: float = 0.9, threshold: float = 1.0,
                 heads: tuple[str, ...] = ("left_present", "right_present", "tumour_present")):
        super().__init__()
        self.slice_window = slice_window
        self.heads = heads

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
        )
        self.stem_lif = snn.Leaky(beta=beta, threshold=threshold, spike_grad=surrogate.fast_sigmoid())

        self.block1 = _SpikingConvBlock(32, 64, stride=2, beta=beta, threshold=threshold)
        self.block2 = _SpikingConvBlock(64, 128, stride=2, beta=beta, threshold=threshold)
        self.block3 = _SpikingConvBlock(128, 128, stride=1, beta=beta, threshold=threshold)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.readout_lif = snn.Leaky(beta=beta, threshold=threshold, spike_grad=surrogate.fast_sigmoid())
        self.head = nn.Linear(128, hidden_dim)
        self.act = nn.ReLU(inplace=True)
        self.outputs = nn.ModuleDict({name: nn.Linear(hidden_dim, 1) for name in heads})

    def forward(self, x: torch.Tensor):
        """x: (B, T, C, H, W) with T == self.slice_window, slices in craniocaudal order.

        Returns
        -------
        logits: dict[str, Tensor] of shape (B,) per head
        spike_stats: dict with per-timestep spike counts, useful for the
                     uncertainty-based buffering logic and for the
                     spike-activity efficiency comparison against the CNN-LSTM
                     baseline.
        """
        B, T, C, H, W = x.shape
        assert T == self.slice_window, f"expected {self.slice_window} time steps, got {T}"
        device = x.device

        stem_h, stem_w = H // 2, W // 2
        mem_stem = torch.zeros(B, 32, stem_h, stem_w, device=device)
        mem1 = self.block1.init_mem(B, (stem_h // 2, stem_w // 2), device)
        mem2 = self.block2.init_mem(B, (stem_h // 4, stem_w // 4), device)
        mem3 = self.block3.init_mem(B, (stem_h // 4, stem_w // 4), device)
        mem_readout = torch.zeros(B, 128, device=device)

        spike_counts = []
        readout_spk_sum = torch.zeros(B, 128, device=device)

        for t in range(T):
            slice_t = x[:, t]  # (B, C, H, W) — the t-th anatomical slice, not a repeat
            cur = self.stem(slice_t)
            spk_stem, mem_stem = self.stem_lif(cur, mem_stem)

            spk1, mem1 = self.block1(spk_stem, mem1)
            spk2, mem2 = self.block2(spk1, mem2)
            spk3, mem3 = self.block3(spk2, mem3)

            pooled = self.pool(spk3).flatten(1)
            spk_ro, mem_readout = self.readout_lif(pooled, mem_readout)
            readout_spk_sum = readout_spk_sum + spk_ro

            spike_counts.append(
                float((spk_stem.sum() + spk1.sum() + spk2.sum() + spk3.sum() + spk_ro.sum()).item())
            )

        rate = readout_spk_sum / T  # rate-coded readout across the slice sequence
        feat = self.act(self.head(rate))
        logits = {name: layer(feat).squeeze(-1) for name, layer in self.outputs.items()}

        spike_stats = {
            "per_timestep_spike_count": spike_counts,
            "total_spike_count": sum(spike_counts),
        }
        return logits, spike_stats

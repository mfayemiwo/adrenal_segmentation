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

--------------------------------------------------------------------------
CONFIGURATION OPTIONS (added after the first ablation, 30 Aug 2026)
--------------------------------------------------------------------------
The first run of that ablation found arm E (true sequence) beating arm G
(centre slice repeated) by 0.0035 PR-AUC — i.e. not at all. Three structural
suspects were identified, and each is now an option. EVERY DEFAULT REPRODUCES
THE ORIGINAL MODEL BIT-FOR-BIT, so prior runs remain reproducible and each
option is an ablation dimension rather than a silent change.

`readout` — how the T time steps are collapsed into a feature vector.
    "rate"       (default, original): the classifier receives a spike COUNT
                 per readout neuron, an integer in 0..T. Counting is very
                 nearly order-blind, so the temporal trajectory the LIF
                 dynamics build is discarded at the last layer. This is the
                 leading explanation for arm E == arm G.
    "integrator": a non-spiking leaky integrator. Membrane accumulates as
                 mem <- beta*mem + input and the FINAL membrane value is the
                 feature: a recency-weighted integral over the sequence,
                 continuous-valued and genuinely order-sensitive.
                 CAVEAT: an analog readout means the network is no longer
                 purely spiking. This is a common and accepted hybrid, but it
                 weakens any neuromorphic-energy argument and must be stated
                 explicitly in the paper.

`norm` — normalisation inside the spiking blocks.
    "batch" (default, original): one BatchNorm2d applied at every time step.
            Its running statistics are updated T times per forward pass, each
            on a different distribution (slice t), then one set is applied to
            all time steps at eval. That train/eval mismatch is a candidate
            explanation for the spiking arms' large generalisation gap
            (train 0.166 / val 0.326, against 0.185 / 0.193 for the CNN-LSTM).
    "step":  a separate BatchNorm2d per time step. Removes the mismatch,
            costs T x the (small) BN parameter count.
    "group": GroupNorm — no running statistics at all, so train and eval
            behave identically. Cheapest correct option.

`learn_beta` / `learn_threshold` — make LIF decay and firing threshold
    learnable (snnTorch supports both). Negligible parameter cost; commonly
    worth a point or two, and lets the network choose its own integration
    time constant rather than inheriting 0.9.

`surrogate_fn` / `surrogate_slope` — the surrogate gradient. The default
    fast_sigmoid(slope=25) is steep, which thins gradients propagated through
    T time steps; atan, or a gentler slope, often trains more stably.
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

READOUTS = ("rate", "integrator")
NORMS = ("batch", "step", "group")
GROUP_NORM_GROUPS = 8          # divides 32, 64 and 128


def build_surrogate(name: str = "fast_sigmoid", slope: float = 25.0):
    if name == "fast_sigmoid":
        return surrogate.fast_sigmoid(slope=slope)
    if name == "atan":
        return surrogate.atan(alpha=slope)
    raise ValueError(f"unknown surrogate '{name}' (expected fast_sigmoid or atan)")


def build_norm(kind: str, channels: int, time_steps: int) -> nn.Module:
    """Returns a module that is either applied directly, or indexed by time
    step when `kind == 'step'`."""
    if kind == "batch":
        return nn.BatchNorm2d(channels)
    if kind == "group":
        return nn.GroupNorm(GROUP_NORM_GROUPS, channels)
    if kind == "step":
        return nn.ModuleList([nn.BatchNorm2d(channels) for _ in range(time_steps)])
    raise ValueError(f"unknown norm '{kind}' (expected one of {NORMS})")


def apply_norm(module: nn.Module, x: torch.Tensor, t: int) -> torch.Tensor:
    return module[t](x) if isinstance(module, nn.ModuleList) else module(x)


class _SpikingConvBlock(nn.Module):
    """Conv -> Norm -> LIF, with an optional residual (ResNet-style) skip.

    Stateful: `mem` must be threaded across time steps by the caller, which is
    what makes this a *recurrent-in-time* spiking block rather than a static
    per-frame CNN applied T times.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, beta: float = 0.9,
                 threshold: float = 1.0, residual: bool = True,
                 norm: str = "batch", time_steps: int = 5, spike_grad=None,
                 learn_beta: bool = False, learn_threshold: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = build_norm(norm, out_ch, time_steps)
        self.lif = snn.Leaky(
            beta=beta, threshold=threshold,
            spike_grad=spike_grad if spike_grad is not None else surrogate.fast_sigmoid(),
            learn_beta=learn_beta, learn_threshold=learn_threshold,
        )
        self.residual = residual and in_ch == out_ch and stride == 1
        self.skip = None if self.residual else nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)

    def init_mem(self, batch_size: int, spatial_size: tuple[int, int], device):
        return torch.zeros(batch_size, self.conv.out_channels, *spatial_size, device=device)

    def forward(self, x: torch.Tensor, mem: torch.Tensor, t: int = 0):
        cur = apply_norm(self.bn, self.conv(x), t)
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
    heads: named binary outputs, e.g. {"left_present": 1, "right_present": 1} —
        left/right adrenal gland presence. This project's scope is gland
        segmentation only; there is no tumour/ACC head.
    readout, norm, learn_beta, learn_threshold, surrogate_fn, surrogate_slope:
        see the module docstring. Defaults reproduce the original model exactly.
    """

    def __init__(self, slice_window: int = 5, in_channels: int = 1, hidden_dim: int = 256,
                 beta: float = 0.9, threshold: float = 1.0,
                 heads: tuple[str, ...] = ("left_present", "right_present"),
                 readout: str = "rate", norm: str = "batch",
                 learn_beta: bool = False, learn_threshold: bool = False,
                 surrogate_fn: str = "fast_sigmoid", surrogate_slope: float = 25.0):
        super().__init__()
        if readout not in READOUTS:
            raise ValueError(f"unknown readout '{readout}' (expected one of {READOUTS})")
        self.slice_window = slice_window
        self.heads = heads
        self.readout = readout
        self.norm_kind = norm

        spike_grad = build_surrogate(surrogate_fn, surrogate_slope)
        lif_kw = dict(spike_grad=spike_grad, learn_beta=learn_beta,
                      learn_threshold=learn_threshold)
        block_kw = dict(beta=beta, threshold=threshold, norm=norm, time_steps=slice_window,
                        spike_grad=spike_grad, learn_beta=learn_beta,
                        learn_threshold=learn_threshold)

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False),
            build_norm(norm, 32, slice_window) if norm != "step" else nn.Identity(),
        )
        # A per-step norm cannot live inside nn.Sequential (it needs the index),
        # so it is held separately and applied by hand below.
        self.stem_norm = build_norm(norm, 32, slice_window) if norm == "step" else None
        self.stem_lif = snn.Leaky(beta=beta, threshold=threshold, **lif_kw)

        self.block1 = _SpikingConvBlock(32, 64, stride=2, **block_kw)
        self.block2 = _SpikingConvBlock(64, 128, stride=2, **block_kw)
        self.block3 = _SpikingConvBlock(128, 128, stride=1, **block_kw)

        self.pool = nn.AdaptiveAvgPool2d(1)
        if readout == "rate":
            self.readout_lif = snn.Leaky(beta=beta, threshold=threshold, **lif_kw)
        else:
            # Non-spiking leaky integrator: no firing, no reset, so the final
            # membrane is an exponentially weighted integral of the sequence.
            self.readout_lif = None
            if learn_beta:
                self.readout_beta = nn.Parameter(torch.tensor(float(beta)))
            else:
                self.register_buffer("readout_beta", torch.tensor(float(beta)))

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
                     baseline. With readout="integrator" the readout layer does
                     not spike, so its contribution is absent by construction.
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

        # Spike counts are accumulated as device tensors and moved to the host
        # ONCE, after the loop. Calling .item() inside the loop forces a full
        # device synchronisation on every time step, which serialises the
        # pipeline: T syncs per forward, thousands of batches per epoch, for a
        # diagnostic number nothing reads until the epoch ends.
        spike_counts_t = []
        readout_spk_sum = torch.zeros(B, 128, device=device)

        for t in range(T):
            slice_t = x[:, t]  # (B, C, H, W) — the t-th anatomical slice, not a repeat
            cur = self.stem(slice_t)
            if self.stem_norm is not None:
                cur = apply_norm(self.stem_norm, cur, t)
            spk_stem, mem_stem = self.stem_lif(cur, mem_stem)

            spk1, mem1 = self.block1(spk_stem, mem1, t)
            spk2, mem2 = self.block2(spk1, mem2, t)
            spk3, mem3 = self.block3(spk2, mem3, t)

            pooled = self.pool(spk3).flatten(1)
            step_spikes = spk_stem.sum() + spk1.sum() + spk2.sum() + spk3.sum()

            if self.readout == "rate":
                spk_ro, mem_readout = self.readout_lif(pooled, mem_readout)
                readout_spk_sum = readout_spk_sum + spk_ro
                step_spikes = step_spikes + spk_ro.sum()
            else:
                # Leaky integration without firing or reset: order matters, and
                # the feature stays continuous rather than a 0..T count.
                mem_readout = self.readout_beta * mem_readout + pooled

            spike_counts_t.append(step_spikes)

        if self.readout == "rate":
            feat_in = readout_spk_sum / T      # rate-coded readout across the sequence
        else:
            feat_in = mem_readout              # final membrane potential

        feat = self.act(self.head(feat_in))
        logits = {name: layer(feat).squeeze(-1) for name, layer in self.outputs.items()}

        spike_counts = torch.stack(spike_counts_t).detach().tolist()   # one sync
        spike_stats = {
            "per_timestep_spike_count": spike_counts,
            "total_spike_count": sum(spike_counts),
        }
        return logits, spike_stats

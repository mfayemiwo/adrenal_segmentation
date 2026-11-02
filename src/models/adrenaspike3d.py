"""AdrenaSpike-3D ROI prototype: independent left/right, bidirectional z-memory.

Consumes predicted gland probabilities and CT intensities from *real consecutive
slices* within a probability-localised 3D ROI, not padded 2.5D shifted windows.
Control mode uses identical convolutional weights/shapes and analog leaky units.
A 3D nnU-Net backbone is a separate study arm; this module is backend-agnostic.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

try:
    import snntorch as snn
    from snntorch import surrogate
except ImportError as exc:
    raise ImportError("AdrenaSpike-3D needs snntorch") from exc


class TemporalCell(nn.Module):
    def __init__(self, cin: int, cout: int, *, spiking: bool, beta: float):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.norm = nn.GroupNorm(min(4, cout), cout)
        self.spiking = spiking
        if spiking:
            self.neuron = snn.Leaky(
                beta=beta, threshold=1.0,
                spike_grad=surrogate.fast_sigmoid(slope=10.0),
                learn_beta=False, learn_threshold=False,
            )
        else:
            self.register_buffer("beta", torch.tensor(beta))

    def forward(self, input, mem):
        current = self.norm(self.conv(input))
        if mem is None:
            mem = torch.zeros_like(current)
        if self.spiking:
            return self.neuron(current, mem)
        mem = self.beta * mem + current
        return F.silu(mem), mem


class TemporalDirection(nn.Module):
    def __init__(self, *, spiking: bool, beta: float):
        super().__init__()
        self.a = TemporalCell(6, 12, spiking=spiking, beta=beta)
        self.b = TemporalCell(12, 16, spiking=spiking, beta=beta)
        self.spiking = spiking

    def forward(self, frames):
        # frames (B,D,6,H,W). Membrane state propagates along real CT z-axis.
        m1 = m2 = None
        features = []
        rate = frames.new_zeros(())
        count = 0
        for t in range(frames.shape[1]):
            h1, m1 = self.a(frames[:, t], m1)
            h2, m2 = self.b(h1, m2)
            features.append(h2)
            if self.spiking:
                rate = rate + h1.detach().sum() + h2.detach().sum()
                count += h1.numel() + h2.numel()
        out = torch.stack(features, dim=2)  # B,C,D,H,W
        return out, rate / max(count, 1)


class AdrenaSpike3D(nn.Module):
    """One shared model is applied independently to the two gland ROIs."""

    def __init__(self, variant="snn", beta=0.9, event_threshold=0.002):
        super().__init__()
        if variant not in ("snn", "analog"):
            raise ValueError("variant must be snn or analog")
        self.variant = variant
        self.event_threshold = float(event_threshold)
        self.forward_memory = TemporalDirection(spiking=(variant == "snn"), beta=beta)
        self.reverse_memory = TemporalDirection(spiking=(variant == "snn"), beta=beta)
        self.decoder = nn.Sequential(
            nn.Conv3d(32, 16, 3, padding=1, bias=False),
            nn.GroupNorm(4, 16), nn.SiLU(),
            nn.Conv3d(16, 1, 1),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        self.last_spike_rate = None
        self.last_event_density = None

    def forward(self, ct, prob, *, operating_threshold=0.5, zero_events=False, use_gate=True):
        # Both arrays (B,1,D,H,W). Probabilities must already be calibrated
        # or produced with a single, pre-specified checkpoint/threshold.
        if ct.ndim != 5 or ct.shape != prob.shape or prob.shape[1] != 1:
            raise ValueError("ct and prob must be same shape (B,1,D,H,W)")
        p = prob.clamp(1e-5, 1 - 1e-5)
        dx = F.pad((p[..., 1:] - p[..., :-1]).abs(), (0, 1, 0, 0))
        dy = F.pad((p[..., 1:, :] - p[..., :-1, :]).abs(), (0, 0, 0, 1))
        boundary = (dx + dy).clamp(0, 1)
        delta = p[:, :, 1:] - p[:, :, :-1]  # actual consecutive z slices
        pos = torch.cat((torch.zeros_like(p[:, :, :1]),
                         F.relu(delta - self.event_threshold)), dim=2)
        neg = torch.cat((torch.zeros_like(p[:, :, :1]),
                         F.relu(-delta - self.event_threshold)), dim=2)
        if zero_events:
            pos = torch.zeros_like(pos)
            neg = torch.zeros_like(neg)
        uncertainty = (4 * p * (1 - p)).clamp(0, 1)
        inputs = torch.cat((ct, p, uncertainty, boundary, pos, neg), dim=1)
        seq = inputs.permute(0, 2, 1, 3, 4)
        fwd, r1 = self.forward_memory(seq)
        rev, r2 = self.reverse_memory(seq.flip(1))
        rev = rev.flip(2)
        correction = self.decoder(torch.cat((fwd, rev), dim=1))
        disagreement = torch.cat((torch.zeros_like(p[:, :, :1]),
                                  delta.abs()), dim=2)
        gate = torch.maximum(uncertainty, disagreement) if use_gate else torch.ones_like(p)
        if not 0 < operating_threshold < 1:
            raise ValueError("operating_threshold must be strictly between zero and one")
        reference_logit = math.log(operating_threshold / (1.0 - operating_threshold))
        logits = torch.logit(p) - reference_logit + self.residual_scale.tanh() * gate * correction
        self.last_spike_rate = (r1 + r2).mul(0.5).detach()
        self.last_event_density = ((pos > 0).float().mean() +
                                   (neg > 0).float().mean()).mul(0.5).detach()
        return logits

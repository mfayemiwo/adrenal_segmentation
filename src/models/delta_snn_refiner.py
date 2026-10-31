"""Inter-slice delta-event spiking refinement for 2.5D adrenal CT segmentation.

The existing project showed that feeding dense adjacent CT slices directly to an
SNN gate produced almost no gain over repeating the centre slice. This module
tests a stronger SNN-native hypothesis: adjacent CT slices are highly redundant,
so the temporal stream should represent change rather than re-present the full
frame at every time step.

A conventional 2.5D U-Net remains responsible for absolute anatomy. A small
temporal refiner receives either delta events (the proposed representation) or
raw intensity polarity channels (a representation control). The temporal branch
can use LIF neurons or a parameter-near-matched analog leaky integrator. Its
dense correction is added residually to the U-Net logits.

The default dense readout uses the final membrane state. This makes the model a
hybrid ANN-SNN rather than a fully spiking network; any efficiency claim must
state that limitation explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import snntorch as snn
    from snntorch import surrogate
except ImportError as exc:  # pragma: no cover
    raise ImportError("snnTorch is required for the delta-SNN refiner") from exc

try:
    import segmentation_models_pytorch as smp
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "segmentation-models-pytorch is required for the base U-Net"
    ) from exc


EVENT_MODES = ("delta", "raw")
READOUTS = ("membrane", "rate")


@dataclass
class RefinerStats:
    spike_rate: torch.Tensor
    event_density: torch.Tensor
    event_magnitude: torch.Tensor


class InterSliceEventEncoder(nn.Module):
    """Convert a stack shaped (B,T,H,W) into a two-polarity temporal stream."""

    def __init__(
        self,
        mode: str = "delta",
        threshold: float = 0.05,
        learn_threshold: bool = False,
    ):
        super().__init__()
        if mode not in EVENT_MODES:
            raise ValueError(f"mode must be one of {EVENT_MODES}, got {mode!r}")
        if threshold < 0:
            raise ValueError("threshold must be non-negative")
        self.mode = mode
        self.learn_threshold = bool(learn_threshold)

        if self.learn_threshold:
            init_value = max(float(threshold), 1e-4)
            init = torch.log(torch.expm1(torch.tensor(init_value)))
            self._threshold_unconstrained = nn.Parameter(init)
        else:
            self.register_buffer("_threshold", torch.tensor(float(threshold)))

    @property
    def threshold(self) -> torch.Tensor:
        if self.learn_threshold:
            return F.softplus(self._threshold_unconstrained)
        return self._threshold

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"expected (B,T,H,W), got shape {tuple(x.shape)}")

        if self.mode == "delta":
            if x.shape[1] < 2:
                raise ValueError("delta encoding needs at least two slices")
            signal = x[:, 1:] - x[:, :-1]
        else:
            signal = x

        tau = self.threshold.to(dtype=x.dtype, device=x.device)
        pos = F.relu(signal - tau)
        neg = F.relu(-signal - tau)
        events = torch.stack((pos, neg), dim=2)  # (B,time,2,H,W)

        event_density = (events > 0).to(x.dtype).mean()
        event_magnitude = events.mean()
        return events, event_density, event_magnitude


class _SpikingBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int,
        beta: float,
        threshold: float,
        learn_beta: bool,
        learn_threshold: bool,
        surrogate_slope: float,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False
        )
        groups = min(8, out_ch)
        while out_ch % groups:
            groups -= 1
        self.norm = nn.GroupNorm(groups, out_ch)
        self.lif = snn.Leaky(
            beta=beta,
            threshold=threshold,
            spike_grad=surrogate.fast_sigmoid(slope=surrogate_slope),
            learn_beta=learn_beta,
            learn_threshold=learn_threshold,
        )

    def forward(
        self, x: torch.Tensor, mem: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cur = self.norm(self.conv(x))
        if mem is None:
            mem = torch.zeros_like(cur)
        spk, mem = self.lif(cur, mem)
        return spk, mem


class _AnalogBlock(nn.Module):
    """Continuous leaky temporal control with the same convolutional pathway."""

    def __init__(
        self, in_ch: int, out_ch: int, stride: int, beta: float
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False
        )
        groups = min(8, out_ch)
        while out_ch % groups:
            groups -= 1
        self.norm = nn.GroupNorm(groups, out_ch)
        self.register_buffer("beta", torch.tensor(float(beta)))

    def forward(
        self, x: torch.Tensor, mem: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cur = self.norm(self.conv(x))
        if mem is None:
            mem = torch.zeros_like(cur)
        mem = self.beta.to(cur.dtype) * mem + cur
        return F.silu(mem), mem


class TemporalContextRefiner(nn.Module):
    """Small temporal encoder-decoder that predicts a residual logit map."""

    def __init__(
        self,
        num_classes: int,
        spiking: bool = True,
        beta: float = 0.9,
        lif_threshold: float = 1.0,
        learn_beta: bool = True,
        learn_lif_threshold: bool = True,
        surrogate_slope: float = 10.0,
        readout: str = "membrane",
    ):
        super().__init__()
        if readout not in READOUTS:
            raise ValueError(f"readout must be one of {READOUTS}, got {readout!r}")
        self.spiking = bool(spiking)
        self.readout = readout

        if self.spiking:
            block_kwargs = dict(
                beta=beta,
                threshold=lif_threshold,
                learn_beta=learn_beta,
                learn_threshold=learn_lif_threshold,
                surrogate_slope=surrogate_slope,
            )
            self.block1 = _SpikingBlock(2, 16, stride=2, **block_kwargs)
            self.block2 = _SpikingBlock(16, 32, stride=2, **block_kwargs)
            self.block3 = _SpikingBlock(32, 32, stride=1, **block_kwargs)
        else:
            self.block1 = _AnalogBlock(2, 16, stride=2, beta=beta)
            self.block2 = _AnalogBlock(16, 32, stride=2, beta=beta)
            self.block3 = _AnalogBlock(32, 32, stride=1, beta=beta)

        self.decode1 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.SiLU(inplace=True),
        )
        self.decode2 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, 8),
            nn.SiLU(inplace=True),
        )
        self.out = nn.Conv2d(8, num_classes, kernel_size=1)

    def forward(
        self, events: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if events.ndim != 5 or events.shape[2] != 2:
            raise ValueError(
                f"expected (B,time,2,H,W), got {tuple(events.shape)}"
            )

        mem1 = mem2 = mem3 = None
        spike_sum = events.new_zeros(())
        spike_opportunities = 0
        rate_accum = None

        for t in range(events.shape[1]):
            h1, mem1 = self.block1(events[:, t], mem1)
            h2, mem2 = self.block2(h1, mem2)
            h3, mem3 = self.block3(h2, mem3)

            if self.spiking:
                spike_sum = spike_sum + h1.sum() + h2.sum() + h3.sum()
                spike_opportunities += h1.numel() + h2.numel() + h3.numel()
                if self.readout == "rate":
                    rate_accum = h3 if rate_accum is None else rate_accum + h3

        if self.spiking and self.readout == "rate":
            feat = rate_accum / max(int(events.shape[1]), 1)
        else:
            feat = mem3

        target_h, target_w = events.shape[-2:]
        feat = F.interpolate(
            feat, scale_factor=2.0, mode="bilinear", align_corners=False
        )
        feat = self.decode1(feat)
        feat = F.interpolate(
            feat,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )
        feat = self.decode2(feat)
        correction = self.out(feat)

        if self.spiking:
            spike_rate = spike_sum / max(spike_opportunities, 1)
        else:
            spike_rate = correction.new_zeros(())
        return correction, spike_rate


class DeltaSNNRefinedSegmenter(nn.Module):
    """2.5D U-Net plus an inter-slice temporal residual refiner."""

    def __init__(
        self,
        encoder: str = "resnet34",
        encoder_weights: str | None = "imagenet",
        in_channels: int = 5,
        num_classes: int = 2,
        decoder_attention: str | None = "scse",
        event_mode: str = "delta",
        event_threshold: float = 0.05,
        learn_event_threshold: bool = False,
        spiking: bool = True,
        beta: float = 0.9,
        lif_threshold: float = 1.0,
        learn_beta: bool = True,
        learn_lif_threshold: bool = True,
        surrogate_slope: float = 10.0,
        readout: str = "membrane",
        residual_scale: float = 0.1,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.event_mode = event_mode
        self.spiking = bool(spiking)

        self.base_model = smp.Unet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=None,
            decoder_attention_type=decoder_attention,
        )
        self.event_encoder = InterSliceEventEncoder(
            mode=event_mode,
            threshold=event_threshold,
            learn_threshold=learn_event_threshold,
        )
        self.refiner = TemporalContextRefiner(
            num_classes=num_classes,
            spiking=spiking,
            beta=beta,
            lif_threshold=lif_threshold,
            learn_beta=learn_beta,
            learn_lif_threshold=learn_lif_threshold,
            surrogate_slope=surrogate_slope,
            readout=readout,
        )
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))
        self.last_stats: RefinerStats | None = None

    def load_base_state_dict(
        self, state_dict: dict[str, torch.Tensor], strict: bool = True
    ):
        return self.base_model.load_state_dict(state_dict, strict=strict)

    def freeze_base(self) -> None:
        self.base_model.eval()
        for parameter in self.base_model.parameters():
            parameter.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if not any(
            parameter.requires_grad for parameter in self.base_model.parameters()
        ):
            self.base_model.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"expected (B,{self.in_channels},H,W), got {tuple(x.shape)}"
            )

        base_logits = self.base_model(x)
        events, density, magnitude = self.event_encoder(x)
        correction, spike_rate = self.refiner(events)
        scale = torch.tanh(self.residual_scale)

        self.last_stats = RefinerStats(
            spike_rate=spike_rate.detach(),
            event_density=density.detach(),
            event_magnitude=magnitude.detach(),
        )
        return base_logits + scale * correction


def build_refined_segmenter(
    *,
    variant: str,
    encoder: str,
    encoder_weights: str | None,
    in_channels: int,
    num_classes: int,
    decoder_attention: str | None,
    event_threshold: float = 0.05,
    learn_event_threshold: bool = False,
    beta: float = 0.9,
    lif_threshold: float = 1.0,
    learn_beta: bool = True,
    learn_lif_threshold: bool = True,
    surrogate_slope: float = 10.0,
    readout: str = "membrane",
) -> DeltaSNNRefinedSegmenter:
    """Build one of the publication ablation variants."""
    if variant not in ("delta_snn", "raw_snn", "delta_analog"):
        raise ValueError(f"unknown refined variant {variant!r}")

    return DeltaSNNRefinedSegmenter(
        encoder=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        num_classes=num_classes,
        decoder_attention=decoder_attention,
        event_mode="raw" if variant == "raw_snn" else "delta",
        event_threshold=event_threshold,
        learn_event_threshold=learn_event_threshold,
        spiking=(variant != "delta_analog"),
        beta=beta,
        lif_threshold=lif_threshold,
        learn_beta=learn_beta,
        learn_lif_threshold=learn_lif_threshold,
        surrogate_slope=surrogate_slope,
        readout=readout,
    )

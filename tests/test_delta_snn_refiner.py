from __future__ import annotations

import pytest
import torch

pytest.importorskip("snntorch")
pytest.importorskip("segmentation_models_pytorch")

from src.models.delta_snn_refiner import (
    InterSliceEventEncoder,
    build_refined_segmenter,
)


def test_delta_encoder_is_silent_for_identical_slices():
    x = torch.ones(2, 5, 32, 32)
    encoder = InterSliceEventEncoder(mode="delta", threshold=0.0)
    events, density, magnitude = encoder(x)

    assert events.shape == (2, 4, 2, 32, 32)
    assert torch.count_nonzero(events) == 0
    assert density.item() == 0.0
    assert magnitude.item() == 0.0


def test_delta_encoder_preserves_change_polarity():
    x = torch.zeros(1, 3, 8, 8)
    x[:, 1] = 1.0
    x[:, 2] = 0.25

    encoder = InterSliceEventEncoder(mode="delta", threshold=0.0)
    events, _, _ = encoder(x)

    # 0 -> 1 is a positive event; 1 -> 0.25 is a negative event.
    assert torch.all(events[:, 0, 0] > 0)
    assert torch.count_nonzero(events[:, 0, 1]) == 0
    assert torch.count_nonzero(events[:, 1, 0]) == 0
    assert torch.all(events[:, 1, 1] > 0)


@pytest.mark.parametrize("variant", ["delta_snn", "raw_snn", "delta_analog"])
def test_refined_variants_return_segmentation_shape(variant):
    model = build_refined_segmenter(
        variant=variant,
        encoder="resnet18",
        encoder_weights=None,
        in_channels=5,
        num_classes=2,
        decoder_attention=None,
        event_threshold=0.05,
        learn_beta=False,
        learn_lif_threshold=False,
    ).eval()

    x = torch.randn(1, 5, 64, 64)
    with torch.no_grad():
        y = model(x)

    assert y.shape == (1, 2, 64, 64)
    assert torch.isfinite(y).all()
    assert model.last_stats is not None
    assert 0.0 <= model.last_stats.event_density.item() <= 1.0
    if variant.endswith("snn"):
        assert model.last_stats.spike_rate.item() >= 0.0
    else:
        assert model.last_stats.spike_rate.item() == 0.0


def test_frozen_base_leaves_only_refiner_trainable():
    model = build_refined_segmenter(
        variant="delta_snn",
        encoder="resnet18",
        encoder_weights=None,
        in_channels=5,
        num_classes=2,
        decoder_attention=None,
        learn_beta=False,
        learn_lif_threshold=False,
    )
    model.freeze_base()

    assert not any(p.requires_grad for p in model.base_model.parameters())
    assert any(p.requires_grad for p in model.refiner.parameters())
    assert model.residual_scale.requires_grad

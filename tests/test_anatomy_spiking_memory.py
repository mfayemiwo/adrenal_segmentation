"""Smoke tests for anatomy-memory model, with tiny synthetic inputs."""
import pytest
import torch

pytest.importorskip("snntorch")
pytest.importorskip("segmentation_models_pytorch")

from src.models.anatomy_spiking_memory import build_anatomy_memory_segmenter


@pytest.mark.parametrize("variant", ["anatomy_snn", "anatomy_analog"])
def test_forward_backward(variant):
    model = build_anatomy_memory_segmenter(
        variant=variant, encoder="resnet18", encoder_weights=None,
        in_channels=5, num_classes=2, decoder_attention=None,
        learn_beta=False, learn_lif_threshold=False,
    )
    model.freeze_base()
    model.train()
    x = torch.rand(1, 5, 64, 64)
    y = model(x)
    assert y.shape == (1, 2, 64, 64)
    assert torch.isfinite(y).all()
    y.square().mean().backward()
    assert any(p.grad is not None for p in model.refiner.parameters())
    assert all(p.grad is None for p in model.base_model.parameters())
    assert torch.isfinite(model.last_stats.event_density)

import torch

from src.models.unet25d import build_unet25d


def test_unet25d_forward_shape():
    model = build_unet25d(encoder="resnet18", in_channels=5, num_classes=1, encoder_weights=None)
    x = torch.randn(2, 5, 64, 64)
    out = model(x)
    assert out.shape == (2, 1, 64, 64)

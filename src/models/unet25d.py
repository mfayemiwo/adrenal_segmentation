"""2.5D U-Net segmenter: neighbouring slices stacked as input channels.

Wraps `segmentation_models_pytorch` so the encoder backbone stays swappable
(InceptionV3/V4, ResNet34, VGG16, ...) to stay consistent with the backbones
compared in the prior published pipeline. Only the slices flagged positive
(plus the uncertainty-expanded buffer) are ever passed to this model at
inference time — see `src/inference/pipeline.py`.
"""
from __future__ import annotations

import torch.nn as nn

try:
    import segmentation_models_pytorch as smp
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "segmentation-models-pytorch is required for the 2.5D U-Net; "
        "pip install segmentation-models-pytorch"
    ) from exc


def build_unet25d(encoder: str = "inceptionv4", in_channels: int = 5, num_classes: int = 1,
                   encoder_weights: str | None = "imagenet") -> nn.Module:
    """Build a 2.5D U-Net.

    `in_channels` == the slice window size: the centre slice plus its
    neighbours are stacked as channels, which is the standard way to give a
    2D architecture partial 3D context without the memory cost of true 3D
    convolutions. `num_classes` defaults to 1 (binary gland mask); set to 2
    for a joint left/right gland formulation with two output channels
    instead of one shared mask.
    """
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        activation=None,  # raw logits; loss functions apply sigmoid/softmax internally
    )

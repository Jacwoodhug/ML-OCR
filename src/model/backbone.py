"""MobileNetV3-Small backbone adapted for text recognition.

Produces feature maps at stride 4 along the width dimension, giving
W/4 timesteps for the sequence model. Height is collapsed to 1 via
adaptive pooling.
"""

import torch
import torch.nn as nn
from torchvision import models


class MobileNetV3Backbone(nn.Module):
    """MobileNetV3-Small feature extractor for CRNN.

    Takes input (B, 3, 32, W) and outputs (B, C, 1, W') where W' ≈ W/4.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        # Load MobileNetV3-Small
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        mobilenet = models.mobilenet_v3_small(weights=weights)

        # We use only the feature layers, not the classifier
        features = mobilenet.features

        # MobileNetV3-Small features structure (by index):
        #   0: Conv2d (stride 2)               -> H/2, W/2, 16ch
        #   1: InvertedResidual (stride 2)     -> H/4, W/4, 16ch
        #   2: InvertedResidual (stride 2)     -> H/8, W/8, 24ch
        #   ...
        # We use features[0:2] -> total stride 4 -> W/4 timesteps

        # features[0]: Conv2d stride 2 -> (B, 16, H/2, W/2)
        # features[1]: InvertedResidual stride 2 -> (B, 16, H/4, W/4)
        # Total stride = 4, output channels = 16
        self.early = features[0:2]  # stride 4, 16 channels out

        # Add conv layers to increase channel capacity without changing spatial dims
        self.extra = nn.Sequential(
            # Block 1: expand -> depthwise -> project
            nn.Conv2d(16, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.Hardswish(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, groups=64, bias=False),
            nn.BatchNorm2d(64),
            nn.Hardswish(inplace=True),
            nn.Conv2d(64, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            # Block 2: expand -> depthwise -> project
            nn.Conv2d(32, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.Hardswish(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.Hardswish(inplace=True),
            nn.Conv2d(128, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            # Block 3: expand -> depthwise -> project
            nn.Conv2d(64, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.Hardswish(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, groups=256, bias=False),
            nn.BatchNorm2d(256),
            nn.Hardswish(inplace=True),
            nn.Conv2d(256, 128, 1, bias=False),
            nn.BatchNorm2d(128),
        )

        self.out_channels = 128

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) input image tensor

        Returns:
            (B, C, 1, W') feature maps where W' = W // 4
        """
        x = self.early(x)             # (B, 16, H/4, W/4)
        x = self.extra(x)             # (B, 128, H/4, W/4)
        x = x.mean(dim=2, keepdim=True)  # (B, 128, 1, W/4) — collapse height
        return x

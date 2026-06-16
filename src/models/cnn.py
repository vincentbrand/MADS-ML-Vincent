"""Configurable CNN for the Flowers hypertuning experiments.

The network is assembled from a stack of conv blocks (``nn.ModuleList``) whose
depth, width, regularisation and block type are all driven by :class:`CNNConfig`.
Two block types are provided so the report can test the skip-connection hypothesis:

- ``ConvBlock``      — plain Conv -> (BatchNorm) -> ReLU
- ``ResidualBlock``  — ResNet-style block with an identity/1x1 shortcut

An ``AdaptiveAvgPool2d`` head makes the model independent of the input image size,
so changing ``img_size`` never breaks the flattened dimension.
"""

import torch
from torch import nn

from src.models.config import CNNConfig


class ConvBlock(nn.Module):
    """Conv -> optional BatchNorm -> ReLU."""

    def __init__(self, in_channels: int, out_channels: int, use_batchnorm: bool) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResidualBlock(nn.Module):
    """Two conv layers with a skip connection (1x1 conv shortcut if channels differ)."""

    def __init__(self, in_channels: int, out_channels: int, use_batchnorm: bool) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        if in_channels != out_channels:
            self.shortcut: nn.Module = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act(out + identity)


class FlowerCNN(nn.Module):
    """Depth/width/regularisation-configurable CNN ending in a dense classifier head."""

    def __init__(self, config: CNNConfig) -> None:
        super().__init__()
        block_cls = ResidualBlock if config.use_skip else ConvBlock

        self.blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        in_channels = config.input_channels
        out_channels = config.filters
        for _ in range(config.num_blocks):
            self.blocks.append(block_cls(in_channels, out_channels, config.use_batchnorm))
            self.pools.append(nn.MaxPool2d(2))
            in_channels = out_channels
            out_channels = out_channels * 2

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(config.dropout),
            nn.Linear(in_channels, config.units),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.units, config.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block, pool in zip(self.blocks, self.pools):
            x = pool(block(x))
        return self.head(x)

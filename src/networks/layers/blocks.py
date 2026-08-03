"""
Convolutional building blocks shared by the segmentation architectures.
"""

import torch
import torch.nn.functional as F
from torch import nn


class DoubleConv(nn.Module):
    """
    Two ``3x3`` convolutions, each followed by batch norm and ReLU.
    """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        """
        :param in_channels: Channels of the input feature map
        :param out_channels: Channels of the output feature map
        :param mid_channels: Channels between the two convolutions, or ``None`` for ``out_channels``
        """
        super().__init__()
        mid_channels = out_channels if mid_channels is None else mid_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        :param x: ``(B, in_channels, H, W)`` feature map
        :return: ``(B, out_channels, H, W)`` feature map
        """
        return self.block(x)


class Down(nn.Module):
    """
    One encoder stage: halve the resolution, then convolve.
    """

    def __init__(self, in_channels, out_channels):
        """
        :param in_channels: Channels of the input feature map
        :param out_channels: Channels of the output feature map
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        """
        :param x: ``(B, in_channels, H, W)`` feature map
        :return: ``(B, out_channels, H // 2, W // 2)`` feature map
        """
        return self.block(x)


class Up(nn.Module):
    """
    One decoder stage: upsample, concatenate the skip connection, then convolve.
    """

    def __init__(self, in_channels, skip_channels, out_channels, bilinear=True):
        """
        :param in_channels: Channels of the feature map coming from below
        :param skip_channels: Channels of the encoder feature map to concatenate
        :param out_channels: Channels of the output feature map
        :param bilinear: Upsample bilinearly instead of with a transposed convolution
        """
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            up_channels = in_channels
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            up_channels = in_channels // 2
        self.conv = DoubleConv(up_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        """
        :param x: ``(B, in_channels, H, W)`` feature map from the stage below
        :param skip: Encoder feature map at the target resolution
        :return: ``(B, out_channels, 2H, 2W)`` feature map
        """
        x = self.up(x)
        diff_y = skip.shape[-2] - x.shape[-2]
        diff_x = skip.shape[-1] - x.shape[-1]
        if diff_x or diff_y:
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        return self.conv(torch.cat([skip, x], dim=1))

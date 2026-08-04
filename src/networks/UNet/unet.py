"""
U-Net for binary segmentation of solar filaments.
"""

from torch import nn

from .layers import DoubleConv, Down, Up


class UNet(nn.Module):
    """
    U-Net over grayscale observations, returning per-pixel logits.

    The output is *not* passed through a sigmoid, so it pairs directly with
    :class:`torch.nn.BCEWithLogitsLoss`.
    """

    def __init__(self, in_channels=1, out_channels=1, base_channels=64, depth=4, bilinear=True):
        """
        :param in_channels: Channels of the input observation
        :param out_channels: Channels of the prediction, ``1`` for a binary mask
        :param base_channels: Channels produced by the first stage; doubled at every stage
        :param depth: Number of downsampling stages
        :param bilinear: Upsample bilinearly instead of with transposed convolutions
        """
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be at least 1, got {depth}")

        self.depth = depth
        # Channels held by the encoder at each resolution, e.g. [64, 128, 256, 512, 1024].
        channels = [base_channels * 2**i for i in range(depth + 1)]

        self.stem = DoubleConv(in_channels, channels[0])
        self.downs = nn.ModuleList(Down(channels[i], channels[i + 1]) for i in range(depth))
        self.ups = nn.ModuleList(Up(channels[i + 1], channels[i], channels[i], bilinear) for i in reversed(range(depth)))
        self.head = nn.Conv2d(channels[0], out_channels, kernel_size=1)

    def forward(self, x):
        """
        :param x: ``(B, in_channels, H, W)`` batch of observations
        :return: ``(B, out_channels, H, W)`` logits
        """
        divisor = 2**self.depth
        height, width = x.shape[-2:]
        if height % divisor or width % divisor:
            raise ValueError(f"input size ({height}, {width}) must be divisible by {divisor} for depth {self.depth}")

        skips = []
        x = self.stem(x)
        for down in self.downs:
            skips.append(x)
            x = down(x)
        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip)
        return self.head(x)

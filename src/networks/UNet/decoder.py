from torch import Tensor, nn

from .blocks import Up


class UNetDecoder(nn.Module):
    """
    Expanding path: upsample, fuse the matching skip, convolve, once per stage.
    """

    def __init__(self, channels, bilinear=True):
        """
        :param channels: Channels held at each encoder resolution, finest first
        :param bilinear: Upsample bilinearly instead of with transposed convolutions
        """
        super().__init__()
        depth = len(channels) - 1
        if depth < 1:
            raise ValueError(f"channels must describe at least two resolutions, got {len(channels)}")

        self.depth = depth
        self.channels: list[int] = list(channels)
        self.ups = nn.ModuleList(
            Up(self.channels[i + 1], self.channels[i], self.channels[i], bilinear) for i in reversed(range(depth))
        )

    @property
    def out_channels(self) -> int:
        """
        :return: Channels of the feature map leaving the decoder, at full resolution
        """
        return self.channels[0]

    def forward(self, x, skips) -> Tensor:
        """
        :param x: ``(B, channels[-1], H, W)`` feature map from the bottleneck
        :param skips: One skip per stage ordered finest to coarsest, as the encoder returns them
        :return: ``(B, channels[0], H * 2**depth, W * 2**depth)`` feature map
        """
        if len(skips) != self.depth:
            raise ValueError(f"expected {self.depth} skip connections, got {len(skips)}")

        out: Tensor = x
        for up, skip in zip(self.ups, reversed(skips)):
            out = up(out, skip)
        return out

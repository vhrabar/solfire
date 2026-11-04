from torch import Tensor, nn

from .blocks import DoubleConv, Down


class UNetEncoder(nn.Module):
    """
    Contracting path: a stem at full resolution followed by ``depth`` halving stages.

    """

    def __init__(self, in_channels=1, base_channels=64, depth=4):
        """
        :param in_channels: Channels of the input observation
        :param base_channels: Channels produced by the stem; doubled at every stage
        :param depth: Number of downsampling stages
        """
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be at least 1, got {depth}")

        self.depth = depth
        # Resolution factor between the input and the coarsest feature map.
        self.divisor = 2**depth
        # Channels held at each resolution, e.g. [64, 128, 256, 512, 1024].
        self.channels: list[int] = [base_channels * 2**i for i in range(depth + 1)]

        self.stem = DoubleConv(in_channels, self.channels[0])
        self.downs = nn.ModuleList(Down(self.channels[i], self.channels[i + 1]) for i in range(depth))

    @property
    def out_channels(self) -> int:
        """
        :return: Channels of the coarsest feature map, the one a bottleneck operates on
        """
        return self.channels[-1]

    def forward(self, x) -> tuple[Tensor, list[Tensor]]:
        """
        :param x: ``(B, in_channels, H, W)`` batch, spatially divisible by :attr:`divisor`
        :return: The ``(B, channels[-1], H / divisor, W / divisor)`` bottleneck feature map,
            and one skip per stage ordered finest to coarsest
        """
        skips: list[Tensor] = []
        out: Tensor = self.stem(x)
        for down in self.downs:
            skips.append(out)
            out = down(out)
        return out, skips

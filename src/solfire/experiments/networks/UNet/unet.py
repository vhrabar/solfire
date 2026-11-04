"""
U-Net for binary segmentation of solar filaments.
"""

from torch import Tensor, nn

from solfire.networks.padding import pad_to_multiple

from .decoder import UNetDecoder
from .encoder import UNetEncoder


class UNet(nn.Module):
    """
    U-Net over grayscale observations, returning per-pixel logits.

    Only wires :class:`UNetEncoder` straight into :class:`UNetDecoder`; a hybrid
    is the same wiring with something inserted at the bottleneck.

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
        self.encoder = UNetEncoder(in_channels, base_channels=base_channels, depth=depth)
        self.decoder = UNetDecoder(self.encoder.channels, bilinear=bilinear)
        self.head = nn.Conv2d(self.decoder.out_channels, out_channels, kernel_size=1)

    def forward(self, x) -> Tensor:
        """
        Accepts any input size; sizes the encoder cannot halve are padded and cropped back.

        :param x: ``(B, in_channels, H, W)`` batch of observations
        :return: ``(B, out_channels, H, W)`` logits
        """
        height, width = x.shape[-2:]
        padded = pad_to_multiple(x, self.encoder.divisor)

        bottleneck, skips = self.encoder(padded)
        out = self.decoder(bottleneck, skips)

        logits: Tensor = self.head(out)
        return logits[..., :height, :width]

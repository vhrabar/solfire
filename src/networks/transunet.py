from torch import Tensor, nn

from .padding import pad_to_multiple
from .Transformer import TransformerBottleneck
from .UNet import UNetDecoder, UNetEncoder


class TransUNet(nn.Module):
    """
    Convolutional encoder, attention over the coarsest grid, convolutional decoder.

    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        base_channels=32,
        depth=4,
        bilinear=True,
        blocks=4,
        num_heads=8,
        embed_dim=None,
        mlp_ratio=2.0,
        dropout=0.0,
        drop_path=0.1,
    ):
        """
        :param in_channels: Channels of the input observation
        :param out_channels: Channels of the prediction, ``1`` for a binary mask
        :param base_channels: Channels produced by the first stage; doubled at every stage
        :param depth: Number of downsampling stages
        :param bilinear: Upsample bilinearly instead of with transposed convolutions
        :param blocks: Number of attention blocks at the bottleneck
        :param num_heads: Attention heads; must divide the token width
        :param embed_dim: Token width, or ``None`` to attend at the bottleneck width
        :param mlp_ratio: Hidden width of every Mix-FFN as a multiple of the token width
        :param dropout: Dropout rate inside the attention blocks
        :param drop_path: Maximum stochastic depth rate, ramped linearly over the blocks
        """
        super().__init__()
        self.encoder = UNetEncoder(in_channels, base_channels=base_channels, depth=depth)
        self.bottleneck = TransformerBottleneck(
            self.encoder.out_channels,
            blocks=blocks,
            num_heads=num_heads,
            embed_dim=embed_dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            drop_path=drop_path,
        )
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
        mixed = self.bottleneck(bottleneck)
        out = self.decoder(mixed, skips)

        logits: Tensor = self.head(out)
        return logits[..., :height, :width]

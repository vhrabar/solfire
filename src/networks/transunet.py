import torch
from torch import Tensor, nn

from .padding import pad_to_multiple
from .Transformer.encoder import EncoderBlock
from .UNet.layers import DoubleConv, Down, Up


class TransUNet(nn.Module):
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
        if depth < 1:
            raise ValueError(f"depth must be at least 1, got {depth}")

        self.depth = depth
        self.divisor = 2**depth
        channels = [base_channels * 2**i for i in range(depth + 1)]

        bottleneck = channels[-1]
        embed_dim = bottleneck if embed_dim is None else embed_dim
        if embed_dim % num_heads:
            raise ValueError(f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}")

        self.stem = DoubleConv(in_channels, channels[0])
        self.downs = nn.ModuleList(Down(channels[i], channels[i + 1]) for i in range(depth))

        self.project_in: nn.Module = nn.Identity()
        self.project_out: nn.Module = nn.Identity()
        if embed_dim != bottleneck:
            self.project_in = nn.Conv2d(bottleneck, embed_dim, kernel_size=1)
            self.project_out = nn.Conv2d(embed_dim, bottleneck, kernel_size=1)

        rates = torch.linspace(0, drop_path, max(1, blocks)).tolist()
        self.blocks = nn.ModuleList(
            EncoderBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout, reduction=1, drop_path=rate)
            for rate in rates[:blocks]
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.ups = nn.ModuleList(Up(channels[i + 1], channels[i], channels[i], bilinear) for i in reversed(range(depth)))
        self.head = nn.Conv2d(channels[0], out_channels, kernel_size=1)

        self._init_transformer()

    def _init_transformer(self) -> None:
        """
        Give the attention stack the small-std init transformers need.

        :return: None
        """
        for module in self.blocks.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _attend(self, x) -> Tensor:
        """
        Mix the bottleneck feature map globally with self-attention.

        :param x: ``(B, bottleneck, H, W)`` coarsest encoder feature map
        :return: ``(B, bottleneck, H, W)`` feature map, globally mixed
        """
        x = self.project_in(x)
        height, width = x.shape[-2:]

        tokens: Tensor = x.flatten(2).transpose(1, 2)
        for block in self.blocks:
            tokens = block(tokens, height, width)
        tokens = self.norm(tokens)

        x = tokens.transpose(1, 2).reshape(tokens.shape[0], -1, height, width)
        mixed: Tensor = self.project_out(x)
        return mixed

    def forward(self, x) -> Tensor:
        """
        FWD pass through TransUNet
        :param x: ``(B, in_channels, H, W)`` batch of observations
        :return: ``(B, out_channels, H, W)`` logits
        """
        height, width = x.shape[-2:]
        padded = pad_to_multiple(x, self.divisor)

        skips = []
        out = self.stem(padded)
        for down in self.downs:
            skips.append(out)
            out = down(out)

        out = self._attend(out)

        for up, skip in zip(self.ups, reversed(skips)):
            out = up(out, skip)

        logits: Tensor = self.head(out)
        return logits[..., :height, :width]

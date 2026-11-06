import torch
from torch import Tensor, nn

from .encoder import EncoderBlock


class TransformerBottleneck(nn.Module):
    """
    Globally mix a convolutional feature map with self-attention, shape in equals shape out.

    Takes and returns ``(B, channels, H, W)``, so it drops into any convolutional
    architecture between an encoder and a decoder without either side knowing.
    """

    def __init__(
        self,
        channels,
        blocks=4,
        num_heads=8,
        embed_dim=None,
        mlp_ratio=2.0,
        dropout=0.0,
        drop_path=0.1,
        reduction=1,
    ):
        """
        :param channels: Channels of the feature map entering and leaving the bottleneck
        :param blocks: Number of attention blocks
        :param num_heads: Attention heads; must divide the token width
        :param embed_dim: Token width, or ``None`` to attend at ``channels``
        :param mlp_ratio: Hidden width of every Mix-FFN as a multiple of the token width
        :param dropout: Dropout rate inside the attention blocks
        :param drop_path: Maximum stochastic depth rate, ramped linearly over the blocks
        :param reduction: Factor the key/value grid is downsampled by, ``1`` for full attention
        """
        super().__init__()
        embed_dim = channels if embed_dim is None else embed_dim
        if embed_dim % num_heads:
            raise ValueError(f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}")

        self.channels = channels
        self.embed_dim = embed_dim

        # Only pay for the projections when attention runs at a different width.
        self.project_in: nn.Module = nn.Identity()
        self.project_out: nn.Module = nn.Identity()
        if embed_dim != channels:
            self.project_in = nn.Conv2d(channels, embed_dim, kernel_size=1)
            self.project_out = nn.Conv2d(embed_dim, channels, kernel_size=1)

        rates = torch.linspace(0, drop_path, max(1, blocks)).tolist()
        self.blocks = nn.ModuleList(
            EncoderBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout, reduction=reduction, drop_path=rate)
            for rate in rates[:blocks]
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module) -> None:
        """
        Give the attention stack the small-std init transformers need.

        :param module: Submodule visited by :meth:`torch.nn.Module.apply`
        :return: None
        """
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x) -> Tensor:
        """
        :param x: ``(B, channels, H, W)`` feature map
        :return: ``(B, channels, H, W)`` feature map, globally mixed
        """
        projected: Tensor = self.project_in(x)
        height, width = projected.shape[-2:]

        tokens: Tensor = projected.flatten(2).transpose(1, 2)
        for block in self.blocks:
            tokens = block(tokens, height, width)
        tokens = self.norm(tokens)

        folded = tokens.transpose(1, 2).reshape(tokens.shape[0], -1, height, width)
        mixed: Tensor = self.project_out(folded)
        return mixed

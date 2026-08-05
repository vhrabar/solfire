from torch import Tensor, nn

from .embeder import PatchEmbeder
from .mlp import MixFFN


class DropPath(nn.Module):
    """
    Stochastic depth: drop a whole residual branch for a random subset of the batch.
    """

    def __init__(self, probability: float = 0.0):
        """
        :param probability: Chance of dropping the branch for a given sample
        """
        super().__init__()
        self.probability = probability

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: Output of a residual branch
        :return: The branch output, zeroed for some samples and rescaled to keep the expectation
        """
        if self.probability == 0.0 or not self.training:
            return x
        keep = 1.0 - self.probability
        # Broadcast one Bernoulli draw per sample over every remaining axis.
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


class EfficientSelfAttention(nn.Module):
    """
    Self-attention over a spatially reduced key/value sequence.
    """

    def __init__(self, dim, num_heads, dropout=0.0, reduction=1):
        """
        :param dim: Channels of the tokens
        :param num_heads: Number of attention heads
        :param dropout: Dropout rate applied to the attention weights
        :param reduction: Factor the key/value grid is downsampled by, ``1`` for full attention
        """
        super().__init__()
        self.reduction = reduction
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        if reduction > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=reduction, stride=reduction)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, height, width) -> Tensor:
        """
        :param x: ``(B, H * W, dim)`` tokens
        :param height: Height of the token grid
        :param width: Width of the token grid
        :return: ``(B, H * W, dim)`` tokens
        """
        context = x
        if self.reduction > 1:
            spatial = x.transpose(1, 2).reshape(x.shape[0], x.shape[2], height, width)
            context = self.norm(self.sr(spatial).flatten(2).transpose(1, 2))

        attended: Tensor
        attended, _ = self.attn(x, context, context, need_weights=False)
        return attended


class EncoderBlock(nn.Module):
    """
    One pre-norm encoder layer: reduced self-attention, then a Mix-FFN.
    """

    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0, reduction=1, drop_path=0.0):
        """
        :param dim: Channels of the tokens
        :param num_heads: Number of attention heads
        :param mlp_ratio: Hidden width of the Mix-FFN as a multiple of ``dim``
        :param dropout: Dropout rate inside attention and the Mix-FFN
        :param reduction: Factor the key/value grid is downsampled by
        :param drop_path: Stochastic depth rate for both residual branches
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention(dim, num_heads, dropout=dropout, reduction=reduction)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFN(dim, hidden_dim=int(dim * mlp_ratio), dropout=dropout)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: Tensor, height, width) -> Tensor:
        """
        :param x: ``(B, H * W, dim)`` tokens
        :param height: Height of the token grid
        :param width: Width of the token grid
        :return: ``(B, H * W, dim)`` tokens
        """
        attended: Tensor = self.drop_path(self.attn(self.norm1(x), height, width))
        x = x + attended
        mixed: Tensor = self.drop_path(self.mlp(self.norm2(x), height, width))
        return x + mixed


class EncoderStage(nn.Module):
    """
    One resolution level of the encoder: tokenise, attend, fold back to a grid.
    """

    def __init__(
        self,
        in_channels,
        embed_dim,
        num_heads,
        patch_size,
        stride,
        mlp_ratio=4.0,
        dropout=0.0,
        reduction=1,
        drop_paths=(0.0,),
    ):
        """
        :param in_channels: Channels of the feature map entering the stage
        :param embed_dim: Token width of this stage
        :param num_heads: Number of attention heads
        :param patch_size: Convolution kernel size of the patch embedding
        :param stride: Factor this stage reduces the resolution by
        :param mlp_ratio: Hidden width of every Mix-FFN as a multiple of ``embed_dim``
        :param dropout: Dropout rate inside attention and the Mix-FFN
        :param reduction: Factor the key/value grid is downsampled by
        :param drop_paths: Stochastic depth rate for each block, one entry per block
        """
        super().__init__()
        self.embed = PatchEmbeder(in_channels, embed_dim, patch_size=patch_size, stride=stride)
        self.blocks = nn.ModuleList(
            EncoderBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout, reduction=reduction, drop_path=rate)
            for rate in drop_paths
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x) -> Tensor:
        """
        :param x: ``(B, in_channels, H, W)`` feature map
        :return: ``(B, embed_dim, H / stride, W / stride)`` feature map
        """
        tokens: Tensor
        tokens, height, width = self.embed(x)
        for block in self.blocks:
            tokens = block(tokens, height, width)
        tokens = self.norm(tokens)
        return tokens.transpose(1, 2).reshape(tokens.shape[0], -1, height, width)

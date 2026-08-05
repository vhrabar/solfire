from torch import Tensor, nn


class PatchEmbeder(nn.Module):
    """
    Tokenise a feature map with overlapping patches.
    """

    def __init__(self, in_channels=1, embed_dim=32, patch_size=7, stride=4):
        """
        :param in_channels: Channels of the input feature map
        :param embed_dim: Channels of the produced tokens
        :param patch_size: Convolution kernel size
        :param stride: Factor the input resolution is reduced by
        """
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=stride, padding=patch_size // 2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x) -> tuple[Tensor, int, int]:
        """
        :param x: ``(B, in_channels, H, W)`` feature map
        :return: ``(B, H', W', ...)`` flattened as ``(B, H' * W', embed_dim)`` tokens, plus the grid height and width
        """
        x = self.proj(x)
        height, width = x.shape[-2:]
        tokens: Tensor = self.norm(x.flatten(2).transpose(1, 2))
        return tokens, height, width

from torch import Tensor, nn


class MixFFN(nn.Module):
    """
    Feed-forward network with a depthwise convolution between its two linear layers.
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        """
        :param dim: Channels of the tokens entering and leaving the block
        :param hidden_dim: Channels of the intermediate representation
        :param dropout: Dropout rate applied after each stage
        """
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, height, width) -> Tensor:
        """
        :param x: ``(B, H * W, dim)`` tokens
        :param height: Height of the token grid
        :param width: Width of the token grid
        :return: ``(B, H * W, dim)`` tokens
        """
        x = self.fc1(x)
        spatial = x.transpose(1, 2).reshape(x.shape[0], x.shape[2], height, width)
        x = self.conv(spatial).flatten(2).transpose(1, 2)
        x = self.drop(self.act(x))
        out: Tensor = self.drop(self.fc2(x))
        return out

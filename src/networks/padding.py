import torch.nn.functional as F
from torch import Tensor


def pad_to_multiple(x: Tensor, divisor: int) -> Tensor:
    """
    Grow a batch so its spatial size is divisible by ``divisor``.

    :param x: ``(B, C, H, W)`` batch
    :param divisor: Factor the padded spatial size must be divisible by
    :return: The batch, padded to a divisible size, or unchanged if it already divides
    """
    height, width = x.shape[-2:]
    pad_height = -height % divisor
    pad_width = -width % divisor
    if not pad_height and not pad_width:
        return x
    return F.pad(x, (0, pad_width, 0, pad_height), mode="replicate")

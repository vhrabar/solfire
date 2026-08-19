import torch
import torch.nn.functional as F
from torch import Tensor

DEFAULT_TILE = 512
DEFAULT_OVERLAP = 0.25


@torch.no_grad()
def predict_probabilities(model, images, tta=True) -> Tensor:
    """
    Foreground probabilities for a batch, optionally averaged over flips.

    :param model: Model in eval mode
    :param images: ``(B, 1, H, W)`` batch of observations
    :param tta: Average over horizontal and vertical flips
    :return: ``(B, 1, H, W)`` probabilities in ``[0, 1]``
    """
    if not tta:
        return torch.sigmoid(model(images).float())

    total = torch.zeros_like(images, dtype=torch.float32)
    for flip_x in (False, True):
        for flip_y in (False, True):
            dims = [d for d, on in ((-1, flip_x), (-2, flip_y)) if on]
            transformed = torch.flip(images, dims) if dims else images
            probabilities = torch.sigmoid(model(transformed).float())
            total += torch.flip(probabilities, dims) if dims else probabilities
    return total / 4.0


def tile_offsets(size, tile, stride) -> list[int]:
    """
    Start coordinates of the windows covering one axis.

    The last window is pulled back to the edge rather than padded, so no tile is
    ever evaluated on invented pixels.
    :param size: Length of the axis
    :param tile: Length of one window
    :param stride: Step between consecutive windows
    :return: Ascending list of start coordinates
    """
    if size <= tile:
        return [0]
    offsets = list(range(0, size - tile + 1, stride))
    if offsets[-1] != size - tile:
        offsets.append(size - tile)
    return offsets


def blend_window(tile, device, floor=0.05) -> Tensor:
    """
    Separable raised cosine weights, so overlapping tiles fade into each other.

    :param tile: Edge length of the window
    :param device: Device the weights are built on
    :param floor: Smallest weight at the border; keeps the normalisation finite
        where a pixel is covered by exactly one tile
    :return: ``(1, 1, tile, tile)`` weights in ``(0, 1]``
    """
    ramp = torch.hann_window(tile + 2, periodic=False, device=device)[1:-1]
    ramp = ramp.clamp_min(floor)
    return (ramp[:, None] * ramp[None, :])[None, None]


@torch.no_grad()
def predict_tiled(model, image, device, tile=DEFAULT_TILE, overlap=DEFAULT_OVERLAP, tta=True, batch_size=8) -> Tensor:
    """
    Probabilities for one whole observation, assembled from overlapping windows.

    :param model: Model in eval mode
    :param image: ``(1, H, W)`` or ``(1, 1, H, W)`` observation at native resolution
    :param device: Device to run on
    :param tile: Edge length of the windows the model is applied to
    :param overlap: Fraction of a window shared with its neighbour
    :param tta: Average predictions over flips
    :param batch_size: Windows evaluated per forward pass
    :return: ``(H, W)`` probabilities on the cpu
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)
    height, width = image.shape[-2:]

    padded = image
    pad_h, pad_w = max(0, tile - height), max(0, tile - width)
    if pad_h or pad_w:
        padded = F.pad(padded, (0, pad_w, 0, pad_h), mode="replicate")
    full_h, full_w = padded.shape[-2:]

    stride = max(1, round(tile * (1.0 - overlap)))
    ys = tile_offsets(full_h, tile, stride)
    xs = tile_offsets(full_w, tile, stride)

    weights = blend_window(tile, device)
    total = torch.zeros((1, 1, full_h, full_w), dtype=torch.float32, device=device)
    counts = torch.zeros_like(total)

    corners = [(y, x) for y in ys for x in xs]
    for start in range(0, len(corners), batch_size):
        chunk = corners[start : start + batch_size]
        batch = torch.cat([padded[..., y : y + tile, x : x + tile] for y, x in chunk]).to(device, non_blocking=True)
        probabilities = predict_probabilities(model, batch, tta=tta)
        for (y, x), probability in zip(chunk, probabilities):
            total[..., y : y + tile, x : x + tile] += probability[None] * weights
            counts[..., y : y + tile, x : x + tile] += weights

    return (total / counts)[0, 0, :height, :width].cpu()

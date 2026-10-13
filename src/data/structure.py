"""
Training-only targets that describe the fine structure of the annotated filaments.
"""

from functools import cache

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from skimage.morphology import disk, skeletonize

# half-width, in pixels of the distance transform, of a typical body
REFERENCE_HALF_WIDTH = 6.0
# radius of the opening that defines "thin", the same as the r=2 column of the diagnostic
THIN_RADIUS = 2
SPINE_WIDTH = 3
SPINE_SIGMA = 1.0
# context added around a crop before the morphology
_PAD = 8


def _padded(mask: np.ndarray) -> np.ndarray:
    """
    Extend a crop by repeating its edge pixels.
    :param mask: ``(H, W)`` boolean mask
    :return: ``(H + 2 * _PAD, W + 2 * _PAD)`` boolean mask
    """
    return np.pad(mask, _PAD, mode="edge")


def _unpadded(array: np.ndarray) -> np.ndarray:
    """
    :param array: Array built on a :func:`_padded` mask
    :return: The window of the original crop
    """
    return array[_PAD:-_PAD, _PAD:-_PAD]


def local_half_width(mask: np.ndarray) -> np.ndarray:
    """
    Half-width of the structure each foreground pixel belongs to.

    The distance transform alone is small on every edge pixel, wide body or not; the
    half-width is instead read off the nearest point of the skeleton, where the distance
    transform peaks across the structure.
    :param mask: ``(H, W)`` boolean mask
    :return: ``(H, W)`` float32 array, ``0`` on background
    """
    out = np.zeros(mask.shape, dtype=np.float32)
    if not mask.any():
        return out
    padded = _padded(mask)
    distance = ndimage.distance_transform_edt(padded)
    skeleton = skeletonize(padded)
    if not skeleton.any():
        return out
    _, (rows, cols) = ndimage.distance_transform_edt(~skeleton, return_indices=True)
    width = distance[rows, cols]
    return _unpadded(np.where(padded, width, 0.0)).astype(np.float32)


def thickness_weight(mask: np.ndarray, cap: float) -> np.ndarray:
    """
    Per-pixel loss weight inversely proportional to the local half-width.

    A pixel of a 3 px barb weighs about three body pixels, a 1 px thread about six, so
    the thin structures stop being a two percent minority of the signal.
    :param mask: ``(H, W)`` boolean mask of the annotated filaments
    :param cap: Upper bound on the weight
    :return: ``(H, W)`` float32 weights in ``[1, cap]``, ``1`` on background
    """
    width = local_half_width(mask)
    weight = np.ones(mask.shape, dtype=np.float32)
    foreground = width > 0
    weight[foreground] = np.clip(REFERENCE_HALF_WIDTH / width[foreground], 1.0, cap)
    return weight


def thin_mask(mask: np.ndarray, radius: int = THIN_RADIUS) -> np.ndarray:
    """
    The annotated pixels an opening removes: barbs, thin ends and thread-like filaments.
    :param mask: ``(H, W)`` boolean mask
    :param radius: Radius of the disk the opening uses
    :return: ``(H, W)`` boolean mask
    """
    if not mask.any():
        return np.zeros(mask.shape, dtype=bool)
    padded = _padded(mask)
    return _unpadded(padded & ~ndimage.binary_opening(padded, structure=disk(radius)))


@cache
def _spine_peak() -> float:
    """
    :return: Value the blur leaves on the axis of a long straight spine, used to rescale to 1
    """
    canvas = Image.new("L", (64, 64), 0)
    ImageDraw.Draw(canvas).line([(0, 32), (63, 32)], fill=255, width=SPINE_WIDTH)
    return float(ndimage.gaussian_filter(np.asarray(canvas, dtype=np.float32) / 255.0, SPINE_SIGMA)[32, 32])


def spine_heatmap(spines, height, width, scale=1.0, origin=(0, 0)) -> np.ndarray:
    """
    Draw annotated centre lines and blur them into a heatmap peaking at 1 on the line.
    :param spines: Flat ``[x0, y0, x1, y1, ...]`` polylines in annotation coordinates
    :param height: Height of the window
    :param width: Width of the window
    :param scale: Factor applied to the coordinates
    :param origin: ``(x, y)`` of the window inside the scaled observation
    :return: ``(height, width)`` float32 heatmap in ``[0, 1]``
    """
    canvas = Image.new("L", (width + 2 * _PAD, height + 2 * _PAD), 0)
    draw = ImageDraw.Draw(canvas)
    shift = np.asarray(origin, dtype=np.float64) - _PAD
    for spine in spines:
        points = np.asarray(spine, dtype=np.float64).reshape(-1, 2) * scale - shift
        if len(points) >= 2:
            draw.line([(float(x), float(y)) for x, y in points], fill=255, width=SPINE_WIDTH, joint="curve")
    blurred = ndimage.gaussian_filter(np.asarray(canvas, dtype=np.float32) / 255.0, SPINE_SIGMA)
    return np.minimum(_unpadded(blurred) / _spine_peak(), 1.0).astype(np.float32)

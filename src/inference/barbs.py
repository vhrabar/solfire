"""
Barbs grown into predicted filaments from the observation, without labels.

"""

import numpy as np
from scipy import ndimage
from skimage.filters import sato
from skimage.morphology import disk

from .postprocess import EIGHT_CONNECTED

# contrast and ridge cut the candidate pixels; a grown piece is kept when its length over
# its mean width is at least min_ratio and that width at most max_width pixels
DEFAULT_BARBS = {"contrast": 0.05, "ridge": 0.025, "band": 20, "min_ratio": 4.0, "max_width": 4.0}
BACKGROUND_FLOOR = 20 / 255
BLOCK = 4


def local_contrast(image) -> np.ndarray:
    """
    How much darker every pixel is than its surroundings.

    The background is a median over about 60 pixels, taken on a 4x smaller copy for speed.
    :param image: ``(H, W)`` observation in ``[0, 1]``
    :return: ``(H, W)`` float32 ``(background - image) / background``, smoothed, 0 off the disk
    """
    height, width = image.shape
    rows, columns = -(-height // BLOCK) * BLOCK, -(-width // BLOCK) * BLOCK
    padded = np.pad(image.astype(np.float32), ((0, rows - height), (0, columns - width)), mode="edge")
    small = padded.reshape(rows // BLOCK, BLOCK, columns // BLOCK, BLOCK).mean((1, 3))
    background = np.kron(ndimage.median_filter(small, 15), np.ones((BLOCK, BLOCK), np.float32))[:height, :width]
    contrast = np.where(background > BACKGROUND_FLOOR, (background - image) / np.maximum(background, 1e-6), 0)
    return ndimage.gaussian_filter(contrast.astype(np.float32), 1)


def thin_pieces(pieces, mask, min_ratio, max_width) -> np.ndarray:
    """
    :param pieces: Boolean candidate pixels around one instance
    :param mask: Boolean instance mask of the same window
    :param min_ratio: Length over mean width a piece needs
    :param max_width: Widest mean width a piece may have, in pixels
    :return: The pieces that are long and narrow, measured from where they leave the mask
    """
    labels, n_labels = ndimage.label(pieces, structure=EIGHT_CONNECTED)
    if n_labels == 0:
        return pieces
    distance = ndimage.distance_transform_edt(~mask)
    keep = np.zeros(n_labels + 1, dtype=bool)
    for number, window in enumerate(ndimage.find_objects(labels), start=1):
        if window is None:
            continue
        points = np.argwhere(labels[window] == number) + np.array([window[0].start, window[1].start])
        base = points[np.argmin(distance[points[:, 0], points[:, 1]])]
        length = np.sqrt(((points - base) ** 2).sum(1).max()) + 1
        width = len(points) / length
        keep[number] = length / width >= min_ratio and width <= max_width
    return keep[labels]


def grow_barbs(labels, n_labels, image, settings=None) -> np.ndarray:
    """
    Add the visible barbs of every instance to it.

    Grown pixels only fill background, so instances never take pixels from each other.
    :param labels: ``(H, W)`` instance labels, 0 for background
    :param n_labels: Number of instances
    :param image: ``(H, W)`` observation in ``[0, 1]``
    :param settings: Overrides of :data:`DEFAULT_BARBS`
    :return: ``(H, W)`` labels with the barbs added
    """
    settings = {**DEFAULT_BARBS, **(settings or {})}
    if n_labels == 0:
        return labels
    contrast = local_contrast(image)
    grown = labels.copy()
    margin = settings["band"] + 2
    height, width = labels.shape
    for number, window in enumerate(ndimage.find_objects(labels), start=1):
        if window is None:
            continue
        window = (
            slice(max(window[0].start - margin, 0), min(window[0].stop + margin, height)),
            slice(max(window[1].start - margin, 0), min(window[1].stop + margin, width)),
        )
        mask = labels[window] == number
        local = contrast[window]
        ridge = sato(local, sigmas=[1, 2], black_ridges=False)
        band = ndimage.binary_dilation(mask, disk(int(settings["band"]))) & ~mask
        candidates = band & (local > settings["contrast"]) & (ridge > settings["ridge"])
        pieces, _ = ndimage.label(candidates, structure=EIGHT_CONNECTED)
        touching = np.unique(pieces[ndimage.binary_dilation(mask, EIGHT_CONNECTED) & candidates])
        barbs = np.isin(pieces, touching[touching > 0])
        barbs = thin_pieces(barbs, mask, settings["min_ratio"], settings["max_width"])
        target = grown[window]
        target[barbs & (target == 0)] = number
    return grown

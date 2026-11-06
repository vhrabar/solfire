"""
CLAHE preprocessing of the observations, registered as ``--preprocess clahe``.
"""

import numpy as np
from skimage import exposure

CLAHE_KERNEL = 256
CLAHE_CLIP = 0.01


def clahe(image: np.ndarray) -> np.ndarray:
    """
    Contrast limited adaptive histogram equalisation of one observation.
    :param image: ``(H, W)`` uint8 observation
    :return: ``(H, W)`` uint8 equalised observation
    """
    equalised = exposure.equalize_adapthist(image, kernel_size=CLAHE_KERNEL, clip_limit=CLAHE_CLIP)
    return np.asarray(np.round(equalised * 255.0), dtype=np.uint8)

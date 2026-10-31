"""
Barbs grown into predicted instances from the observation.
"""

import numpy as np

from src.inference.barbs import grow_barbs, local_contrast


def filament_with_barb() -> tuple[np.ndarray, np.ndarray]:
    """
    A bright disk with a dark filament body, a thin dark barb leaving it upwards and a
    dark blob touching it below; the instance labels cover the body only.
    """
    image = np.full((128, 128), 0.8, dtype=np.float32)
    image[60:68, 20:100] = 0.4
    image[44:60, 50:52] = 0.5
    image[68:80, 80:92] = 0.5
    labels = np.zeros((128, 128), dtype=np.int32)
    labels[60:68, 20:100] = 1
    return image, labels


def test_contrast_is_relative_to_the_background():
    """
    The bright disk has no contrast, the body is half as bright as it.
    """
    image, _ = filament_with_barb()
    contrast = local_contrast(image)
    assert abs(contrast[10, 10]) < 0.01
    assert 0.4 < contrast[64, 30] < 0.6


def test_thin_barb_joins_the_instance_and_the_blob_does_not():
    """
    The barb is long and narrow, so it is added to the body's instance; the blob is not.
    """
    image, labels = filament_with_barb()
    grown = grow_barbs(labels, 1, image)
    assert (grown[48:58, 50:52] == 1).mean() > 0.8
    assert (grown[70:80, 82:90] == 0).all()
    assert (grown[labels == 1] == 1).all()


def test_nothing_grows_on_a_plain_disk_or_without_instances():
    """
    Without dark structure next to it an instance stays as it was.
    """
    image = np.full((128, 128), 0.8, dtype=np.float32)
    labels = np.zeros((128, 128), dtype=np.int32)
    labels[60:68, 20:100] = 1
    assert (grow_barbs(labels, 1, image) == labels).all()
    assert (grow_barbs(np.zeros_like(labels), 0, image) == 0).all()

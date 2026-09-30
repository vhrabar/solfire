"""
Shared fixtures: small synthetic filaments, so the tests need neither the dataset nor a GPU.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from synthetic import filament_polygon


@pytest.fixture
def rng():
    """
    :return: A seeded numpy random generator
    """
    return np.random.default_rng(0)


@pytest.fixture
def annotation_record(rng):
    """
    A synthetic annotator record: COCO annotations of a handful of filaments.
    :return: List of annotation dicts
    """
    return [{"segmentation": [filament_polygon(rng)], "category_id": 1 + k % 3} for k in range(5)]

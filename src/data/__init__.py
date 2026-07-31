"""Data loading for the MAGFiLO solar filament segmentation challenge."""

from .dataset import (
    CATEGORY_NAMES,
    DATA_ROOT,
    TEST_IMAGES,
    TRAIN_ANNOTATIONS,
    TRAIN_IMAGES,
    FilamentDataset,
    TestImageDataset,
    load_annotations,
    load_image,
    polygon_to_mask,
    train_val_split,
)

__all__ = [
    "CATEGORY_NAMES",
    "DATA_ROOT",
    "TEST_IMAGES",
    "TRAIN_ANNOTATIONS",
    "TRAIN_IMAGES",
    "FilamentDataset",
    "TestImageDataset",
    "load_annotations",
    "load_image",
    "polygon_to_mask",
    "train_val_split",
]

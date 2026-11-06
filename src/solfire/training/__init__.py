"""Training loop, objectives and metrics for the filament segmentation challenge."""

from .augment import AugmentedDataset
from .calibrate import (
    CALIBRATION_THRESHOLDS,
    DEFAULT_MIN_AREAS,
    DEFAULT_MIN_SCORES,
    DEFAULT_OFFSETS,
    cache_directory,
    cache_probabilities,
    calibrate,
    evaluate_tiled,
    parse_variant,
    write_calibration,
)
from .engine import evaluate, fit, save_checkpoint, train_one_epoch
from .losses import BCEDiceLoss, estimate_pos_weight, lovasz_hinge_loss, soft_dice_loss
from .metrics import DEFAULT_THRESHOLDS, PanopticQuality, grid_edges, panoptic_quality, threshold_grid
from .optim import build_optimizer, build_scheduler
from .plots import plot_distribution, plot_evaluation, plot_matching_cases

__all__ = [
    "CALIBRATION_THRESHOLDS",
    "DEFAULT_MIN_AREAS",
    "DEFAULT_MIN_SCORES",
    "DEFAULT_OFFSETS",
    "DEFAULT_THRESHOLDS",
    "AugmentedDataset",
    "BCEDiceLoss",
    "PanopticQuality",
    "build_optimizer",
    "build_scheduler",
    "cache_directory",
    "cache_probabilities",
    "calibrate",
    "estimate_pos_weight",
    "evaluate",
    "evaluate_tiled",
    "fit",
    "grid_edges",
    "lovasz_hinge_loss",
    "panoptic_quality",
    "parse_variant",
    "plot_distribution",
    "plot_evaluation",
    "plot_matching_cases",
    "save_checkpoint",
    "soft_dice_loss",
    "threshold_grid",
    "train_one_epoch",
    "write_calibration",
]

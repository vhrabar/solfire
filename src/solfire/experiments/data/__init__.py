"""Training data of the experiments: extra targets, pseudo-labels, magnetograms, CLAHE."""

from .clahe import clahe
from .crops import CATEGORY_CHANNELS, ExperimentCropDataset
from .magnetogram import MagneticTargets
from .pseudo_labels import PseudoLabelledCropDataset

__all__ = ["CATEGORY_CHANNELS", "ExperimentCropDataset", "MagneticTargets", "PseudoLabelledCropDataset", "clahe"]

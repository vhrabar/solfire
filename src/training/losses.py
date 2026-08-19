"""
Objectives for the strongly imbalanced filament segmentation task.
"""

import numpy as np
import torch
from torch import Tensor, nn

from ..data import polygon_to_mask


def soft_dice_loss(logits: Tensor, targets: Tensor, eps: float = 1.0) -> Tensor:
    """
    Soft Dice loss, computed per sample and averaged over the batch.

    :param logits: ``(B, 1, H, W)`` raw predictions
    :param targets: ``(B, 1, H, W)`` binary targets
    :param eps: Smoothing added to numerator and denominator; also defines the
        loss of an empty prediction on an empty target
    :return: Scalar loss in ``[0, 1]``
    """
    probabilities = torch.sigmoid(logits).flatten(1)
    targets = targets.flatten(1)
    intersection = (probabilities * targets).sum(dim=1)
    cardinality = probabilities.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + eps) / (cardinality + eps)
    return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """
    Weighted sum of binary cross entropy and soft Dice.

    BCE -> pp grad
    DICE -> overlap optim
    """

    pos_weight: Tensor | None

    def __init__(self, bce_weight=0.5, dice_weight=0.5, pos_weight=None):
        """
        :param bce_weight: Weight of the cross entropy term
        :param dice_weight: Weight of the Dice term
        :param pos_weight: Weight of positive pixels inside the BCE term, or ``None``
        """
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        if pos_weight is not None:
            pos_weight = torch.as_tensor(float(pos_weight))
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, logits, targets):
        """
        :param logits: ``(B, 1, H, W)`` raw predictions
        :param targets: ``(B, 1, H, W)`` binary targets
        :return: Scalar loss
        """
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight)
        return self.bce_weight * bce + self.dice_weight * soft_dice_loss(logits, targets)


def estimate_pos_weight(dataset, max_samples: int = 128, cap: float = 20.0) -> float:
    """
    Estimate the ``negative / positive`` pixel ratio of the training samples.

    :param dataset: A :class:`data.FilamentDataset` or :class:`data.FilamentCropDataset`
    :param max_samples: Number of samples to measure
    :param cap: Upper bound on the returned weight, keeping early training stable
    :return: Suggested ``pos_weight`` for :class:`torch.nn.BCEWithLogitsLoss`
    """
    if hasattr(dataset, "crop"):
        step = max(1, len(dataset) // max_samples)
        positive = 0
        total = 0
        for index in range(0, len(dataset), step):
            _, mask = dataset[index]
            positive += int(mask.sum().item())
            total += mask.numel()
    else:
        size = dataset.size
        positive = 0
        total = 0

        for image_id in dataset.image_ids[:max_samples]:
            record = dataset.records[image_id]
            scale = size / record["width"]
            mask = np.zeros((size, size), dtype=bool)
            for ann in dataset.annotations[image_id]:
                if dataset.category_ids is not None and ann["category_id"] not in dataset.category_ids:
                    continue
                mask |= polygon_to_mask(ann["segmentation"][0], size, size, scale)
            positive += int(mask.sum())
            total += mask.size

    if positive == 0:
        return 1.0
    return min(cap, (total - positive) / positive)

"""
Objectives for the strongly imbalanced filament segmentation task.

The final models train on BCE + soft Dice, and fine-tune with the Lovász hinge added.
The objectives that were tried and dropped live in :mod:`solfire.experiments.losses`.
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


def _lovasz_gradient(sorted_labels: Tensor, sorted_weights: Tensor | None = None) -> Tensor:
    """
    Gradient of the Lovász extension of the Jaccard loss at the sorted errors.
    :param sorted_labels: ``(N,)`` binary labels, ordered by decreasing error
    :param sorted_weights: ``(N,)`` positive pixel weights in the same order, or ``None``
    :return: ``(N,)`` weights of the sorted errors
    """
    weights = torch.ones_like(sorted_labels) if sorted_weights is None else sorted_weights
    positive_mass = sorted_labels * weights
    positives = positive_mass.sum()
    intersection = positives - positive_mass.cumsum(0)
    union = positives + ((1.0 - sorted_labels) * weights).cumsum(0)
    jaccard = 1.0 - intersection / union
    jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def lovasz_hinge_loss(logits: Tensor, targets: Tensor, weights: Tensor | None = None) -> Tensor:
    """
    Lovász hinge: a convex surrogate of 1 - IoU, computed per sample and averaged.

    Panoptic Quality pays nothing below an IoU of a half, so an objective on IoU itself
    rather than on per-pixel likelihood is a closer fit (Berman et al., 2018). Soft
    targets are binarised at a half.
    :param logits: ``(B, 1, H, W)`` raw predictions
    :param targets: ``(B, 1, H, W)`` targets in ``[0, 1]``
    :param weights: ``(B, 1, H, W)`` positive pixel weights of a weighted Jaccard, or ``None``
    :return: Scalar loss
    """
    losses = []
    with torch.autocast(device_type=logits.device.type, enabled=False):
        flat_weights = [None] * len(logits) if weights is None else weights.float().flatten(1)
        for logit, target, weight in zip(logits.float().flatten(1), targets.float().flatten(1), flat_weights):
            labels = (target >= 0.5).float()
            errors = 1.0 - logit * (2.0 * labels - 1.0)
            sorted_errors, order = torch.sort(errors, descending=True)
            gradient = _lovasz_gradient(labels[order], None if weight is None else weight[order])
            losses.append(torch.dot(torch.relu(sorted_errors), gradient))
    return torch.stack(losses).mean()


class BCEDiceLoss(nn.Module):
    """
    Weighted sum of binary cross entropy, soft Dice and, optionally, the Lovász hinge.

    BCE -> pp grad
    DICE -> overlap optim
    Lovász -> IoU optim
    """

    pos_weight: Tensor | None

    def __init__(self, bce_weight=0.5, dice_weight=0.5, pos_weight=None, lovasz_weight=0.0):
        """
        :param bce_weight: Weight of the cross entropy term
        :param dice_weight: Weight of the Dice term
        :param pos_weight: Weight of positive pixels inside the BCE term, or ``None``
        :param lovasz_weight: Weight of the Lovász hinge term; ``0`` leaves it out
        """
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.lovasz_weight = float(lovasz_weight)
        if pos_weight is not None:
            pos_weight = torch.as_tensor(float(pos_weight))
        self.register_buffer("pos_weight", pos_weight)

    def split_targets(self, targets) -> tuple[Tensor, Tensor | None]:
        """
        Separate the target from per-pixel loss weights; the core targets carry none.
        :param targets: Target tensor as the dataset returns it
        :return: Tuple of (``(B, 1, H, W)`` target, ``(B, 1, H, W)`` pixel weights or ``None``)
        """
        return targets, None

    def overlap_loss(self, logits, targets) -> Tensor:
        """
        The overlap term: soft Dice.
        :param logits: ``(B, 1, H, W)`` raw predictions
        :param targets: ``(B, 1, H, W)`` targets
        :return: Scalar loss
        """
        return soft_dice_loss(logits, targets)

    def forward(self, logits, targets):
        """
        :param logits: ``(B, 1, H, W)`` raw predictions
        :param targets: ``(B, 1, H, W)`` binary targets
        :return: Scalar loss
        """
        targets, weights = self.split_targets(targets)
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, weight=weights, pos_weight=self.pos_weight)
        loss = self.bce_weight * bce + self.dice_weight * self.overlap_loss(logits, targets)
        if self.lovasz_weight:
            loss = loss + self.lovasz_weight * lovasz_hinge_loss(logits, targets, weights)
        return loss


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
            mask = mask[0] if mask.dim() == 3 else mask  # auxiliary channels follow the target
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

"""
Objectives for the strongly imbalanced filament segmentation task.
"""

from functools import cache
from typing import cast

import numpy as np
import torch
from torch import Tensor, nn

from ..data import polygon_to_mask

KERNEL_SHAPES = ("disk", "cross", "diamond", "square")


@cache
def _structuring_element(shape: str, size: int, device: torch.device) -> Tensor:
    """
    Build serron's flat additive structuring element (``0`` active, ``-inf`` inactive).

    :param shape: One of :data:`KERNEL_SHAPES`
    :param size: Odd edge length in pixels; ``disk`` and ``diamond`` take ``size // 2``as their radius
    :param device: Device the element is created on
    :return: ``float32`` ``(size, size)`` kernel serron's operators accept
    :raises ValueError: ``size`` is not a positive odd integer or ``shape`` is unknown
    """
    if size < 1 or size % 2 == 0:
        raise ValueError(f"structuring element size must be a positive odd integer, got {size}")
    if shape not in KERNEL_SHAPES:
        raise ValueError(f"unknown structuring element shape {shape!r}, expected one of {KERNEL_SHAPES}")
    from serron import structuring_element as sse

    if shape == "square":
        return sse.square(size, device=device)
    if shape == "cross":
        return sse.cross(size, device=device)
    if shape == "disk":
        return sse.disk(size // 2, device=device)
    return sse.diamond(size // 2, device=device)


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


def soft_tversky_loss(logits: Tensor, targets: Tensor, beta: float = 0.7, eps: float = 1.0) -> Tensor:
    """
    Soft Tversky loss: Dice with false negatives weighted ``beta`` and false positives
    ``1 - beta``, so ``beta`` above one half pushes the prediction outward.

    :param logits: ``(B, 1, H, W)`` raw predictions
    :param targets: ``(B, 1, H, W)`` binary targets
    :param beta: Weight of the false negatives
    :param eps: Smoothing added to numerator and denominator
    :return: Scalar loss in ``[0, 1]``
    """
    probabilities = torch.sigmoid(logits).flatten(1)
    targets = targets.flatten(1)
    true_positive = (probabilities * targets).sum(dim=1)
    false_positive = probabilities.sum(dim=1) - true_positive
    false_negative = targets.sum(dim=1) - true_positive
    index = (true_positive + eps) / (true_positive + (1.0 - beta) * false_positive + beta * false_negative + eps)
    return 1.0 - index.mean()


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

    def __init__(
        self, bce_weight=0.5, dice_weight=0.5, pos_weight=None, lovasz_weight=0.0, pixel_weights=False, tversky_beta=0.5
    ):
        """
        :param bce_weight: Weight of the cross entropy term
        :param dice_weight: Weight of the Dice term
        :param pos_weight: Weight of positive pixels inside the BCE term, or ``None``
        :param lovasz_weight: Weight of the Lovász hinge term; ``0`` leaves it out
        :param pixel_weights: The targets carry a second channel of per-pixel weights
            (:func:`~src.data.structure.thickness_weight`), applied to the BCE and Lovász terms
        :param tversky_beta: Weight of the false negatives in the overlap term; ``0.5`` is Dice
        """
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.lovasz_weight = float(lovasz_weight)
        self.pixel_weights = pixel_weights
        self.tversky_beta = float(tversky_beta)
        if pos_weight is not None:
            pos_weight = torch.as_tensor(float(pos_weight))
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, logits, targets):
        """
        :param logits: ``(B, 1, H, W)`` raw predictions
        :param targets: ``(B, 1, H, W)`` binary targets, or ``(B, 2, H, W)`` with the pixel
            weights second when ``pixel_weights`` is set
        :return: Scalar loss
        """
        weights = None
        if self.pixel_weights:
            targets, weights = targets[:, :1], targets[:, 1:2]
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, weight=weights, pos_weight=self.pos_weight)
        if self.tversky_beta == 0.5:
            overlap = soft_dice_loss(logits, targets)
        else:
            overlap = soft_tversky_loss(logits, targets, self.tversky_beta)
        loss = self.bce_weight * bce + self.dice_weight * overlap
        if self.lovasz_weight:
            loss = loss + self.lovasz_weight * lovasz_hinge_loss(logits, targets, weights)
        return loss


def _mean_over_batch(values: Tensor) -> Tensor:
    """
    Average a ``(B, ...)`` tensor per sample, then over the batch.
    :param values: Any tensor whose first dimension is the batch
    :return: Scalar
    """
    return values.flatten(1).mean(dim=1).mean()


def opening_residual(probabilities: Tensor, kernel: Tensor) -> Tensor:
    """
    Mean probability mass an opening of the prediction removes.

    :param probabilities: ``(B, 1, H, W)`` map in ``[0, 1]``
    :param kernel: serron flat structuring element from :func:`_structuring_element`
    :return: Scalar penalty in ``[0, 1]``
    """
    from serron import functional as sf

    return _mean_over_batch(torch.relu(probabilities - sf.opening(probabilities, kernel)))


def closing_residual(probabilities: Tensor, kernel: Tensor) -> Tensor:
    """
    Mean probability a closing of the prediction adds.

    :param probabilities: ``(B, 1, H, W)`` map in ``[0, 1]``
    :param kernel: serron flat structuring element from :func:`_structuring_element`
    :return: Scalar penalty in ``[0, 1]``
    """
    from serron import functional as sf

    return _mean_over_batch(torch.relu(sf.closing(probabilities, kernel) - probabilities))


def soft_skeleton(probabilities: Tensor, kernel: Tensor, iterations: int) -> Tensor:
    """
    Iterative morphological skeleton, differentiable in ``probabilities``.

    :param probabilities: ``(B, 1, H, W)`` map in ``[0, 1]``
    :param kernel: serron flat structuring element from :func:`_structuring_element`
    :param iterations: Erosion steps; about the half-width of the widest structure
        the skeleton should thin, measured in structuring elements
    :return: Soft skeleton in ``[0, 1]``, the shape of the input
    """
    from serron import functional as sf

    eroded = probabilities
    skeleton = torch.relu(eroded - sf.opening(eroded, kernel))
    for _ in range(iterations):
        eroded = sf.erosion(eroded, kernel)
        delta = torch.relu(eroded - sf.opening(eroded, kernel))
        skeleton = skeleton + delta - skeleton * delta
    return skeleton.clamp(0.0, 1.0)


def soft_cldice_loss(probabilities: Tensor, targets: Tensor, kernel: Tensor, iterations: int = 5, eps: float = 1.0) -> Tensor:
    """
    Soft centre-line Dice: how well the predicted and annotated skeletons cover each other.

    :param probabilities: ``(B, 1, H, W)`` prediction in ``[0, 1]``
    :param targets: ``(B, 1, H, W)`` binary target
    :param kernel: serron flat structuring element the skeletons are built with
    :param iterations: Erosion steps passed to :func:`soft_skeleton`
    :param eps: Smoothing on the ratios, also the score of an empty skeleton
    :return: Scalar loss in ``[0, 1]``
    """
    predicted_skeleton = soft_skeleton(probabilities, kernel, iterations)
    target_skeleton = soft_skeleton(targets, kernel, iterations)

    p = probabilities.flatten(1)
    t = targets.flatten(1)
    pred_skel = predicted_skeleton.flatten(1)
    true_skel = target_skeleton.flatten(1)

    precision = (pred_skel * t).sum(dim=1).add(eps) / pred_skel.sum(dim=1).add(eps)
    sensitivity = (true_skel * p).sum(dim=1).add(eps) / true_skel.sum(dim=1).add(eps)
    cldice = 1.0 - 2.0 * precision * sensitivity / (precision + sensitivity)
    return cldice.mean()


class MorphologyLoss(nn.Module):
    """
    A base objective with optional morphology-aware terms, all off by default.

    - ``open_weight`` penalises what an opening of the prediction removes: thin
      spurs and speckle, the source of false-positive instances.
    - ``close_weight`` penalises what a closing adds: the one- and two-pixel gaps
      that split one filament into two.
    - ``cldice_weight`` adds soft centre-line Dice, so a filament only scores well
      when it is recovered end to end.
    """

    def __init__(
        self,
        base: nn.Module,
        *,
        open_weight: float = 0.0,
        close_weight: float = 0.0,
        cldice_weight: float = 0.0,
        kernel_shape: str = "disk",
        kernel_size: int = 3,
        skeleton_iters: int = 5,
        warmup_epochs: int = 0,
    ) -> None:
        """
        :param base: The pixel objective the terms are added to
        :param open_weight: Weight of the opening residual term
        :param close_weight: Weight of the closing residual term
        :param cldice_weight: Weight of the soft centre-line Dice term
        :param kernel_shape: Structuring element shape for every term
        :param kernel_size: Odd structuring element edge length in pixels
        :param skeleton_iters: Erosion steps in the clDice skeleton
        :param warmup_epochs: Ramp the terms in linearly over this many epochs; ``0``
            applies them at full weight from the first step
        """
        super().__init__()
        self.base = base
        self.open_weight = float(open_weight)
        self.close_weight = float(close_weight)
        self.cldice_weight = float(cldice_weight)
        self.kernel_shape = kernel_shape
        self.kernel_size = kernel_size
        self.skeleton_iters = skeleton_iters
        self.warmup_epochs = warmup_epochs
        self._scale = 1.0

        self.terms_active = max(self.open_weight, self.close_weight, self.cldice_weight) > 0.0
        if self.terms_active:
            _structuring_element(self.kernel_shape, self.kernel_size, torch.device("cpu"))  # validate shape/size early

    def set_epoch(self, epoch: int) -> None:
        """
        Set the warm-up factor for the coming epoch.
        :param epoch: Zero-based epoch index
        :return: None
        """
        if self.warmup_epochs > 0:
            self._scale = min(1.0, (epoch + 1) / (self.warmup_epochs + 1))

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        :param logits: ``(B, 1, H, W)`` raw predictions
        :param targets: ``(B, 1, H, W)`` binary targets
        :return: Scalar loss
        """
        loss = cast(Tensor, self.base(logits, targets))
        if not self.terms_active:
            return loss

        if logits.device.type != "cuda":
            raise RuntimeError(
                "MorphologyLoss needs a CUDA device: serron's morphology operators only "
                f"implement their backward pass on CUDA, got a {logits.device.type!r} tensor"
            )
        kernel = _structuring_element(self.kernel_shape, self.kernel_size, logits.device)
        with torch.autocast(device_type=logits.device.type, enabled=False):
            probabilities = torch.sigmoid(logits.float())
            target = targets.float()
            extra = torch.zeros((), device=logits.device, dtype=torch.float32)
            if self.open_weight:
                extra = extra + self.open_weight * opening_residual(probabilities, kernel)
            if self.close_weight:
                extra = extra + self.close_weight * closing_residual(probabilities, kernel)
            if self.cldice_weight:
                extra = extra + self.cldice_weight * soft_cldice_loss(probabilities, target, kernel, self.skeleton_iters)

        return loss + self._scale * extra.to(loss.dtype)


class CategoryHeadLoss(nn.Module):
    """
    The segmentation objective on the first channel plus an auxiliary category head.

    The annotations tell Left, Right and Unidentifiable filaments apart. Predicting
    the category on annotated pixels is a second task that shares the features, and
    only its first channel, the filament mask, is ever used at inference.
    """

    def __init__(self, base: nn.Module, weight: float) -> None:
        """
        :param base: Objective for the filament channel
        :param weight: Weight of the category cross entropy
        """
        super().__init__()
        self.base = base
        self.weight = float(weight)

    def set_epoch(self, epoch: int) -> None:
        """
        Pass the epoch on to a base objective that ramps its terms in.
        :param epoch: Zero-based epoch index
        :return: None
        """
        set_epoch = getattr(self.base, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(epoch)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        :param logits: ``(B, 1 + C, H, W)``: the filament logit, then one per category
        :param targets: ``(B, 1 + C, H, W)``: the filament target, then one indicator per category
        :return: Scalar loss
        """
        loss = cast(Tensor, self.base(logits[:, :1], targets[:, :1]))
        indicators = targets[:, 1:]
        annotated = indicators.sum(dim=1) > 0
        if not annotated.any():
            return loss + 0.0 * logits[:, 1:].sum()
        with torch.autocast(device_type=logits.device.type, enabled=False):
            category_logits = logits[:, 1:].float().permute(0, 2, 3, 1)[annotated]
            classes = indicators.permute(0, 2, 3, 1)[annotated].argmax(dim=1)
            category = nn.functional.cross_entropy(category_logits, classes)
        return loss + self.weight * category


class MagneticHeadLoss(nn.Module):
    """
    The segmentation objective on the first channel plus an auxiliary magnetic head.

    Two extra channels regress targets derived from the co-temporal GONG magnetogram:
    proximity to a polarity-inversion line, where filaments sit, and strong single-polarity
    flux, where sunspots and plage sit. The magnetogram is a training-only signal; the head
    is dropped at inference and the model still sees only the H-alpha image.
    """

    def __init__(self, base: nn.Module, weight: float) -> None:
        """
        :param base: Objective for the filament channel
        :param weight: Weight of the magnetic cross entropy
        """
        super().__init__()
        self.base = base
        self.weight = float(weight)

    def set_epoch(self, epoch: int) -> None:
        """
        Pass the epoch on to a base objective that ramps its terms in.
        :param epoch: Zero-based epoch index
        :return: None
        """
        set_epoch = getattr(self.base, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(epoch)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        :param logits: ``(B, 3, H, W)``: the filament logit, then PIL proximity and strong flux
        :param targets: ``(B, 4, H, W)``: the filament target, the two magnetic targets in
            ``[0, 1]``, then the mask of pixels a magnetogram covers
        :return: Scalar loss
        """
        loss = cast(Tensor, self.base(logits[:, :1], targets[:, :1]))
        valid = targets[:, 3:4]
        if not valid.any():
            return loss + 0.0 * logits[:, 1:3].sum()
        with torch.autocast(device_type=logits.device.type, enabled=False):
            per_pixel = nn.functional.binary_cross_entropy_with_logits(
                logits[:, 1:3].float(), targets[:, 1:3].float(), reduction="none"
            )
            magnetic = (per_pixel * valid).sum() / (2 * valid.sum())
        return loss + self.weight * magnetic


class ThinHeadLoss(nn.Module):
    """
    The segmentation objective plus a training-only head for the fine structure.

    Two extra channels predict the thin part of the annotation (what an opening of radius 2
    removes: barbs and thin ends) and a heatmap of the annotated spine, the centre line every
    filament ships with. Both give the decoder an explicit objective on structures that are
    a two percent minority of the filament pixels; only the first channel is used at inference.
    """

    def __init__(self, base: nn.Module, weight: float, pos_weight: float = 10.0) -> None:
        """
        :param base: Objective for the filament channel; receives every target channel
            before the last two, so it can carry pixel weights
        :param weight: Weight of the head
        :param pos_weight: Weight of positive pixels in the head's cross entropy; both
            targets cover well under one percent of a crop
        """
        super().__init__()
        self.base = base
        self.weight = float(weight)
        self.register_buffer("pos_weight", torch.as_tensor(float(pos_weight)))

    def set_epoch(self, epoch: int) -> None:
        """
        Pass the epoch on to a base objective that ramps its terms in.
        :param epoch: Zero-based epoch index
        :return: None
        """
        set_epoch = getattr(self.base, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(epoch)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        :param logits: ``(B, 3, H, W)``: the filament logit, then thin and spine
        :param targets: ``(B, K + 2, H, W)``: the ``K`` channels of the base objective, then
            the thin mask and the spine heatmap, both in ``[0, 1]``
        :return: Scalar loss
        """
        loss = cast(Tensor, self.base(logits[:, :1], targets[:, :-2]))
        with torch.autocast(device_type=logits.device.type, enabled=False):
            head = nn.functional.binary_cross_entropy_with_logits(
                logits[:, 1:3].float(), targets[:, -2:].float(), pos_weight=self.pos_weight
            )
        return loss + self.weight * head


def build_criterion(base: nn.Module, args) -> nn.Module:
    """
    Wrap ``base`` in :class:`MorphologyLoss` when any morphology weight is set.
    :param base: The pixel objective, normally :class:`BCEDiceLoss`
    :param args: Parsed ``train`` arguments carrying the ``morph_*`` fields
    :return: ``base`` unchanged, or a :class:`MorphologyLoss` around it
    """
    weights = (args.morph_open_weight, args.morph_close_weight, args.morph_cldice_weight)
    if max(weights) <= 0.0:
        return base
    return MorphologyLoss(
        base,
        open_weight=args.morph_open_weight,
        close_weight=args.morph_close_weight,
        cldice_weight=args.morph_cldice_weight,
        kernel_shape=args.morph_kernel,
        kernel_size=args.morph_kernel_size,
        skeleton_iters=args.morph_skeleton_iters,
        warmup_epochs=args.morph_warmup_epochs,
    )


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

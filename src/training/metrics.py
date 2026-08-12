import numpy as np
import torch

from ..inference.postprocess import connected_components

DEFAULT_THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
IOU_THRESHOLD = 0.5


def instance_layers(mask) -> torch.Tensor:
    """
    Split a binary mask into one layer per connected filament.
    :param mask: ``(H, W)`` boolean array
    :return: ``(n, H, W)`` float tensor with values in ``{0, 1}``
    """
    labels, n_labels = connected_components(mask)
    layers = labels[None] == np.arange(1, n_labels + 1, dtype=np.int32)[:, None, None]
    return torch.from_numpy(layers).float()


def overlap_matrices(gt_layers, pred_layers) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pairwise IoU and Dice between the filaments of one observation.
    :param gt_layers: ``(n_gt, H, W)`` annotated instances
    :param pred_layers: ``(n_pred, H, W)`` predicted instances
    :return: Tuple of two ``(n_gt, n_pred)`` tensors, IoU and Dice of every pair
    """
    gt_flat = gt_layers.flatten(1)
    pred_flat = pred_layers.flatten(1)

    intersection = gt_flat @ pred_flat.t()
    gt_areas = gt_flat.sum(dim=1).view(-1, 1)
    pred_areas = pred_flat.sum(dim=1).view(1, -1)
    union = gt_areas + pred_areas - intersection

    empty = union == 0
    iou = torch.where(empty, 0.0, intersection / union.clamp(min=1.0))
    dice = torch.where(empty, 0.0, 2 * intersection / (gt_areas + pred_areas).clamp(min=1.0))
    return iou, dice


def panoptic_quality(overlaps) -> float:
    """
    Reduce per-observation overlaps to a single score.
    :param overlaps: List of ``(n_gt, n_pred)`` IoU tensors, one per observation
    :return: Panoptic Quality in ``[0, 1]``, ``0`` when there is nothing to score
    """
    hit_ious: list[float] = []
    false_positives = 0
    false_negatives = 0

    for iou in overlaps:
        hits = iou > IOU_THRESHOLD
        hit_ious.extend(iou[hits].tolist())
        false_positives += int((~hits.any(dim=0)).sum())
        false_negatives += int((~hits.any(dim=1)).sum())

    denominator = len(hit_ious) + 0.5 * false_positives + 0.5 * false_negatives
    if denominator == 0:
        return 0.0
    return sum(hit_ious) / denominator


class PanopticQuality:
    """
    Running panoptic quality at several decision thresholds.

    Filaments are the only foreground class, so the instances are simply the
    connected components of the annotation and of the thresholded probability
    map.
    """

    def __init__(self, thresholds=DEFAULT_THRESHOLDS):
        """
        :param thresholds: Probability thresholds to score the predictions at
        """
        self.thresholds = tuple(thresholds)
        self.overlaps: list[list[tuple[torch.Tensor, torch.Tensor]]] = []
        self.reset()

    @torch.no_grad()
    def update(self, logits, targets) -> None:
        """
        Accumulate one batch.
        :param logits: ``(B, 1, H, W)`` raw predictions
        :param targets: ``(B, 1, H, W)`` binary targets
        :return: None
        """
        probabilities = torch.sigmoid(logits.detach().float()).cpu().numpy()
        truth = targets.detach().cpu().numpy() > 0.5

        for probability, target in zip(probabilities[:, 0], truth[:, 0]):
            gt_layers = instance_layers(target)
            for index, threshold in enumerate(self.thresholds):
                pred_layers = instance_layers(probability >= threshold)
                self.overlaps[index].append(overlap_matrices(gt_layers, pred_layers))

    def reset(self) -> None:
        """
        Drop the accumulated overlaps.
        :return: None
        """
        self.overlaps = [[] for _ in self.thresholds]

    def compute(self) -> dict:
        """
        Reduce the accumulated overlaps.
        :return: Dict with the per-threshold ``pq`` list, the best threshold and its
            score, and the ``overlaps`` of that threshold as ``(iou, dice)`` pairs
            for the distribution plots
        """
        scores = [panoptic_quality([iou for iou, _ in overlaps]) for overlaps in self.overlaps]
        best = max(range(len(scores)), key=scores.__getitem__)
        return {
            "thresholds": list(self.thresholds),
            "pq": scores,
            "best_threshold": self.thresholds[best],
            "best_pq": scores[best],
            "overlaps": self.overlaps[best],
        }

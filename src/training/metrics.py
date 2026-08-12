import numpy as np
import torch

from ..inference.postprocess import MIN_AREA, SUBMISSION_SIZE, connected_components

DEFAULT_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
IOU_THRESHOLD = 0.5


def drop_small(labels, n_labels, min_area) -> tuple[np.ndarray, int]:
    """
    Remove the components a submission would discard, and close the gaps in the numbering.
    :param labels: ``(H, W)`` label array with 0 as background
    :param n_labels: Number of labels in ``labels``
    :param min_area: Components of at most this many pixels are dropped
    :return: Tuple of (relabelled array, number of components left)
    """
    areas = np.bincount(labels.ravel(), minlength=n_labels + 1)
    keep = areas > min_area
    keep[0] = False

    remap = np.zeros(n_labels + 1, dtype=np.int32)
    remap[keep] = np.arange(1, int(keep.sum()) + 1, dtype=np.int32)
    return remap[labels], int(keep.sum())


def overlap_matrices(gt_labels, n_gt, pred_labels, n_pred) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pairwise IoU and Dice between the filaments of one observation.

    The pair counts come out of a single histogram over ``gt * (n_pred + 1) + pred``,
    so the cost follows the image rather than the number of instances; an early
    epoch that finds a thousand specks is no more expensive than a clean one.
    :param gt_labels: ``(H, W)`` labelled annotation
    :param n_gt: Number of annotated filaments
    :param pred_labels: ``(H, W)`` labelled prediction
    :param n_pred: Number of predicted filaments
    :return: Tuple of two ``(n_gt, n_pred)`` tensors, IoU and Dice of every pair
    """
    pairs = gt_labels.astype(np.int64) * (n_pred + 1) + pred_labels
    counts = np.bincount(pairs.ravel(), minlength=(n_gt + 1) * (n_pred + 1)).reshape(n_gt + 1, n_pred + 1)

    intersection = counts[1:, 1:].astype(np.float64)
    gt_areas = counts[1:].sum(axis=1, keepdims=True)
    pred_areas = counts[:, 1:].sum(axis=0, keepdims=True)
    union = gt_areas + pred_areas - intersection

    iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
    dice = np.divide(2 * intersection, gt_areas + pred_areas, out=np.zeros_like(intersection), where=union > 0)
    return torch.from_numpy(iou).float(), torch.from_numpy(dice).float()


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
    map. Predictions too small to be submitted are dropped here as well, so the
    threshold picked on validation is the one the submission is written with.
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
            # the submission is written at 2048, so the area cut has to be scaled down with it
            min_area = MIN_AREA * probability.size / SUBMISSION_SIZE**2
            gt_labels, n_gt = connected_components(target)
            for index, threshold in enumerate(self.thresholds):
                pred_labels, n_pred = drop_small(*connected_components(probability >= threshold), min_area)
                self.overlaps[index].append(overlap_matrices(gt_labels, n_gt, pred_labels, n_pred))

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

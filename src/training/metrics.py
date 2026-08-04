"""
Overlap metrics accumulated over a whole split.
"""

import torch
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex

DEFAULT_THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)


class SegmentationMetrics:
    """
    Running overlap statistics at several decision thresholds.
    """

    def __init__(self, thresholds=DEFAULT_THRESHOLDS):
        """
        :param thresholds: Probability thresholds to score the predictions at
        """
        self.thresholds = tuple(thresholds)
        metrics: dict[str, Metric | MetricCollection] = {}

        for i, threshold in enumerate(self.thresholds):
            metrics[f"dice_{i}"] = BinaryF1Score(threshold=threshold)
            metrics[f"iou_{i}"] = BinaryJaccardIndex(threshold=threshold)

            metrics[f"image_dice_{i}"] = BinaryF1Score(threshold=threshold, multidim_average="samplewise", zero_division=1.0)

        self.metrics = MetricCollection(metrics, compute_groups=False)

    @torch.no_grad()
    def update(self, logits, targets):
        """
        Accumulate one batch.
        :param logits: ``(B, 1, H, W)`` raw predictions
        :param targets: ``(B, 1, H, W)`` binary targets
        :return: None
        """
        probabilities = torch.sigmoid(logits.detach().float())
        self.metrics.to(probabilities.device)
        self.metrics.update(probabilities, targets.detach().int())

    def reset(self) -> None:
        """
        Drop the accumulated statistics.
        :return: None
        """
        self.metrics.reset()

    def compute(self) -> dict:
        """
        Reduce the accumulated statistics.
        :return: Dict with per-threshold ``dice``, ``iou`` and ``image_dice`` lists,
            plus the best threshold and its dataset-level Dice
        """
        values = self.metrics.compute()
        indices = range(len(self.thresholds))
        dice = [float(values[f"dice_{i}"]) for i in indices]
        iou = [float(values[f"iou_{i}"]) for i in indices]
        image_dice = [float(values[f"image_dice_{i}"].mean()) for i in indices]

        best = max(range(len(dice)), key=dice.__getitem__)
        return {
            "thresholds": list(self.thresholds),
            "dice": dice,
            "iou": iou,
            "image_dice": image_dice,
            "best_threshold": self.thresholds[best],
            "best_dice": dice[best],
            "best_iou": iou[best],
        }

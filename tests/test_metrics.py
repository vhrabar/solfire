"""
PQ against the organizers' definition.
"""

import numpy as np
import pytest
import torch
from organizer_reference import get_overlap_matrices, get_pq_score
from pycocotools import mask as coco_mask
from synthetic import SIZE, filament_polygon

from src.inference.postprocess import connected_components
from src.training.metrics import annotation_pixels, instance_iou, pq_counts, pq_from_counts


def gt_layers(record, size=SIZE) -> torch.Tensor:
    """
    The organizers' view of a record: one decoded RLE layer per annotation.
    """
    if not record:
        return torch.zeros((0, size, size))
    rles = [coco_mask.merge(coco_mask.frPyObjects(a["segmentation"], size, size)) for a in record]
    return torch.from_numpy(coco_mask.decode(rles).transpose(2, 0, 1).astype(np.float32))


def pred_layers(labels, n_labels) -> torch.Tensor:
    """
    The organizers' view of a prediction: one layer per predicted instance.
    """
    return (
        torch.from_numpy(np.stack([labels == k for k in range(1, n_labels + 1)]).astype(np.float32))
        if n_labels
        else torch.zeros((0, *labels.shape))
    )


def ours(records, labels, n_labels) -> float:
    """
    PQ over several records scored against one prediction.
    """
    total = np.zeros(4)
    for record in records:
        total += pq_counts(instance_iou(annotation_pixels(record, SIZE, SIZE), labels, n_labels))
    return pq_from_counts(*total)


def theirs(records, labels, n_labels) -> float:
    """
    The organizers' PQ over the same records and prediction.
    """
    predicted = pred_layers(labels, n_labels)
    entries = []
    for record in records:
        layers = gt_layers(record)
        iou, _ = (
            get_overlap_matrices(layers, predicted)
            if len(layers) and len(predicted)
            else (torch.zeros((len(layers), len(predicted))), None)
        )
        entries.append((iou, len(layers), len(predicted)))
    return get_pq_score(entries)


@pytest.mark.parametrize("seed", range(20))
def test_pq_matches_organizers(seed):
    """
    Random annotations and a perturbed prediction: our PQ equals theirs.
    """
    rng = np.random.default_rng(seed)
    records = [[{"segmentation": [filament_polygon(rng)]} for _ in range(rng.integers(0, 6))] for _ in range(rng.integers(1, 4))]
    # prediction: the first record's filaments, dilated or eroded, some dropped, plus a spurious blob
    truth = gt_layers(records[0]).numpy().astype(bool)
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    for layer in truth:
        if rng.random() < 0.8:
            mask |= np.roll(layer, int(rng.integers(-2, 3)), axis=int(rng.integers(0, 2)))
    if rng.random() < 0.5:
        mask[5:12, 100:120] = True
    labels, n_labels = connected_components(mask)
    assert ours(records, labels, n_labels) == pytest.approx(theirs(records, labels, n_labels), abs=1e-6)


def test_pq_hand_cases():
    """
    A perfect match scores 1, a prediction elsewhere 0, and a filament split into
    halves that each miss IoU 0.5 scores 0 as well.
    """
    square = [{"segmentation": [[10, 10, 40, 10, 40, 40, 10, 40]]}]
    filled = gt_layers(square)[0].numpy().astype(bool)
    labels, n_labels = connected_components(filled)
    assert ours([square], labels, n_labels) == pytest.approx(1.0)

    elsewhere = np.zeros((SIZE, SIZE), dtype=bool)
    elsewhere[80:100, 80:100] = True
    labels, n_labels = connected_components(elsewhere)
    assert ours([square], labels, n_labels) == 0.0

    # a split into two halves: neither half reaches IoU 0.5, one miss and two false positives
    split = filled.copy()
    split[:, 25] = False
    labels, n_labels = connected_components(split)
    assert n_labels == 2
    assert ours([square], labels, n_labels) == 0.0


def test_every_record_is_scored():
    """
    Two annotators, the prediction matching only one of them: PQ counts the other's misses.
    """
    first = [{"segmentation": [[10, 10, 40, 10, 40, 40, 10, 40]]}]
    second = [{"segmentation": [[70, 70, 100, 70, 100, 100, 70, 100]]}]
    labels, n_labels = connected_components(gt_layers(first)[0].numpy().astype(bool))
    # record one: 1 TP with IoU 1; record two: 1 FP + 1 FN -> 1 / (1 + 0.5 + 0.5)
    assert ours([first, second], labels, n_labels) == pytest.approx(0.5)
    assert theirs([first, second], labels, n_labels) == pytest.approx(0.5)


def test_tversky_at_one_half_matches_dice_and_beta_favours_recall():
    """
    With ``beta`` one half the Tversky loss is Dice up to the smoothing; above it a
    prediction missing 28 filament pixels costs more than one adding 36.
    """
    import torch

    from src.training.losses import soft_dice_loss, soft_tversky_loss

    targets = torch.zeros(1, 1, 16, 16)
    targets[..., 4:12, 4:12] = 1
    logits = (targets * 2 - 1) * 3
    dice = soft_dice_loss(logits, targets, eps=1e-6)
    assert torch.isclose(soft_tversky_loss(logits, targets, beta=0.5, eps=1e-6), dice, atol=1e-4)
    shrunk = torch.full_like(targets, -6.0)
    shrunk[..., 5:11, 5:11] = 6.0
    grown = torch.full_like(targets, -6.0)
    grown[..., 3:13, 3:13] = 6.0
    assert soft_tversky_loss(shrunk, targets, beta=0.7) > soft_tversky_loss(grown, targets, beta=0.7)

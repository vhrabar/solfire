"""
Turning probability maps into scored instances and submission records.
"""

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools import mask as coco_mask
from scipy import ndimage

SUBMISSION_SIZE = 2048
# The smallest annotated filaments cover a few hundred pixels at that size, so anything
# below this is noise, and under Panoptic Quality every speck kept is half a point of penalty.
MIN_AREA = 256
EIGHT_CONNECTED = np.ones((3, 3), dtype=bool)


def connected_components(mask) -> tuple[np.ndarray, int]:
    """
    Label 8-connected foreground regions.
    :param mask: ``(H, W)`` boolean array
    :return: Tuple of (``(H, W)`` int32 label array with 0 as background, number of labels)
    """
    labels, n_labels = ndimage.label(mask, structure=EIGHT_CONNECTED)
    return labels.astype(np.int32, copy=False), n_labels


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """
    Grow a mask by ``radius`` steps of four connected neighbours.

    A filament is about eleven pixels thick at the submission resolution, so one
    pixel of systematic bias on each side already costs a quarter of the overlap
    with the annotation. That makes the offset worth tuning, and a diamond
    structuring element is the finest step available.
    :param mask: ``(H, W)`` boolean array
    :param radius: Number of dilation steps
    :return: ``(H, W)`` boolean array
    """
    for _ in range(max(0, radius)):
        grown = mask.copy()
        grown[1:] |= mask[:-1]
        grown[:-1] |= mask[1:]
        grown[:, 1:] |= mask[:, :-1]
        grown[:, :-1] |= mask[:, 1:]
        mask = grown
    return mask


def erode(mask: np.ndarray, radius: int) -> np.ndarray:
    """
    Shrink a mask by ``radius`` steps of four connected neighbours.
    :param mask: ``(H, W)`` boolean array
    :param radius: Number of erosion steps
    :return: ``(H, W)`` boolean array
    """
    return ~dilate(~mask, radius)


def apply_offset(mask: np.ndarray, offset: int) -> np.ndarray:
    """
    Dilate a positive offset, erode a negative one.
    :param mask: ``(H, W)`` boolean array
    :param offset: Signed number of morphological steps
    :return: ``(H, W)`` boolean array
    """
    if offset > 0:
        return dilate(mask, offset)
    if offset < 0:
        return erode(mask, -offset)
    return mask


def instance_scores(labels, n_labels, probabilities) -> np.ndarray:
    """
    Mean foreground probability of every labelled component.
    :param labels: ``(H, W)`` label array with 0 as background
    :param n_labels: Number of labels in ``labels``
    :param probabilities: ``(H, W)`` foreground probabilities
    :return: ``(n_labels,)`` scores, component ``k`` at index ``k - 1``
    """
    flat = labels.ravel()
    sums = np.bincount(flat, weights=probabilities.ravel(), minlength=n_labels + 1)[1:]
    areas = np.bincount(flat, minlength=n_labels + 1)[1:]
    return sums / np.maximum(areas, 1)


def keep_labels(labels, n_labels, keep) -> tuple[np.ndarray, int]:
    """
    Remove the components not kept, and close the gaps in the numbering.
    :param labels: ``(H, W)`` label array with 0 as background
    :param n_labels: Number of labels in ``labels``
    :param keep: ``(n_labels,)`` boolean, component ``k`` at index ``k - 1``
    :return: Tuple of (relabelled array, number of components left)
    """
    remap = np.zeros(n_labels + 1, dtype=np.int32)
    remap[1:][keep] = np.arange(1, int(np.count_nonzero(keep)) + 1, dtype=np.int32)
    return remap[labels], int(np.count_nonzero(keep))


def drop_small(labels, n_labels, min_area) -> tuple[np.ndarray, int]:
    """
    Remove the components a submission would discard, and close the gaps in the numbering.
    :param labels: ``(H, W)`` label array with 0 as background
    :param n_labels: Number of labels in ``labels``
    :param min_area: Components of at most this many pixels are dropped
    :return: Tuple of (relabelled array, number of components left)
    """
    if min_area <= 0:
        return labels, n_labels
    areas = np.bincount(labels.ravel(), minlength=n_labels + 1)[1:]
    return keep_labels(labels, n_labels, areas > min_area)


def label_instances(probabilities, threshold, offset=0, min_area=0, min_score=0.0) -> tuple[np.ndarray, int]:
    """
    Turn a probability map into labelled instances the way a submission does.

    The threshold is best set low, so a filament comes out whole rather than in
    pieces, and ``min_score`` then drops the components the model is unsure of as a
    whole: under Panoptic Quality every false positive costs as much as a miss.
    :param probabilities: ``(H, W)`` foreground probabilities
    :param threshold: Probability above which a pixel counts as filament
    :param offset: Morphological steps applied to the thresholded mask
    :param min_area: Drop components with at most this many pixels
    :param min_score: Drop components whose mean probability is below this
    :return: Tuple of (``(H, W)`` int32 label array with 0 as background, number of components)
    """
    mask = apply_offset(probabilities >= threshold, offset)
    labels, n_labels = drop_small(*connected_components(mask), min_area)
    if min_score > 0 and n_labels:
        labels, n_labels = keep_labels(labels, n_labels, instance_scores(labels, n_labels, probabilities) >= min_score)
    return labels, n_labels


def instances_from_labels(labels, n_labels, probabilities) -> list[dict]:
    """
    Describe every labelled component.

    The foreground is sorted by label once instead of being compared against every
    label in turn, so an early checkpoint that finds a thousand specks costs no
    more than a clean one.
    :param labels: ``(H, W)`` int32 label array with 0 as background
    :param n_labels: Number of labels in ``labels``
    :param probabilities: ``(H, W)`` float probabilities behind the mask
    :return: List of dicts with ``mask``, ``score``, ``area`` and ``bbox``
    """
    ys, xs = np.nonzero(labels)
    if ys.size == 0:
        return []
    values = labels[ys, xs]
    order = np.argsort(values, kind="stable")
    ys, xs, values = ys[order], xs[order], values[order]
    starts = np.searchsorted(values, np.arange(1, n_labels + 2))

    instances = []
    for label in range(n_labels):
        begin, end = starts[label], starts[label + 1]
        if begin == end:
            continue
        rows, columns = ys[begin:end], xs[begin:end]
        mask = np.zeros(labels.shape, dtype=bool)
        mask[rows, columns] = True
        x0, y0 = int(columns.min()), int(rows.min())
        instances.append(
            {
                "mask": mask,
                "score": float(probabilities[rows, columns].mean()),
                "area": int(end - begin),
                "bbox": [x0, y0, int(columns.max()) - x0 + 1, int(rows.max()) - y0 + 1],
            }
        )
    return instances


def instances_from_mask(mask, probabilities, min_area=0) -> list[dict]:
    """
    Split a binary mask into scored instances.
    :param mask: ``(H, W)`` boolean prediction
    :param probabilities: ``(H, W)`` float probabilities behind the mask
    :param min_area: Drop components with at most this many pixels
    :return: List of dicts with ``mask``, ``score``, ``area`` and ``bbox``
    """
    labels, n_labels = drop_small(*connected_components(mask), min_area)
    return instances_from_labels(labels, n_labels, probabilities)


def encode_rle(mask) -> dict[str, Any]:
    """
    Encode a boolean mask as a COCO compressed RLE.
    :param mask: ``(H, W)`` boolean array
    :return: Dict with ``size`` and a JSON-serialisable ``counts`` string
    """
    rle: dict[str, Any] = coco_mask.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def write_detections(detections, path) -> None:
    """
    Write COCO-style detections to disk, for local evaluation and debugging.
    :param detections: Records produced by :func:`inference.predict.predict_dataset`
    :param path: Destination JSON file
    :return: None
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(detections), encoding="utf-8")


def write_submission(detections, path) -> int:
    """
    Write the challenge submission: one row per predicted filament.

    :param detections: Records produced by :func:`inference.predict.predict_dataset`
    :param path: Destination CSV file
    :return: Number of rows written
    :raises ValueError: If a mask is not 2048 by 2048, or its counts would need escaping
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, quoting=csv.QUOTE_NONE, quotechar=None)
        writer.writerow(["filament_id", "segmentation_rle"])
        for detection in detections:
            height, width = detection["segmentation"]["size"]
            if (height, width) != (SUBMISSION_SIZE, SUBMISSION_SIZE):
                raise ValueError(
                    f"{detection['filament_id']}: masks must be {SUBMISSION_SIZE}x{SUBMISSION_SIZE}, got {height}x{width}"
                )
            counts = detection["segmentation"]["counts"]
            if any(character in counts for character in ",\"'\r\n"):
                raise ValueError(f"{detection['filament_id']}: counts need escaping")
            writer.writerow([detection["filament_id"], counts])

    return len(detections)

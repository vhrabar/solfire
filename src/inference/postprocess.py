"""
Turning probability maps into scored instances and submission records.
"""

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools import mask as coco_mask

SUBMISSION_SIZE = 2048


def connected_components(mask) -> tuple[np.ndarray, int]:
    """
    Label 8-connected foreground regions.

    Rows are reduced to runs of consecutive foreground pixels before the
    union-find, so cost scales with the number of runs rather than the number
    of pixels, which matters at the full 2048 by 2048 resolution.
    :param mask: ``(H, W)`` boolean array
    :return: Tuple of (``(H, W)`` int32 label array with 0 as background, number of labels)
    """
    height, width = mask.shape
    padded = np.zeros(width + 2, dtype=np.int8)
    runs: list[tuple[int, int, int]] = []
    row_slices: list[tuple[int, int]] = []

    for y in range(height):
        padded[1:-1] = mask[y]
        edges = np.flatnonzero(padded[1:] != padded[:-1])
        start = len(runs)
        runs.extend((y, int(a), int(b)) for a, b in zip(edges[::2], edges[1::2]))
        row_slices.append((start, len(runs)))

    parent = list(range(len(runs)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        i, j = find(i), find(j)
        if i != j:
            parent[max(i, j)] = min(i, j)

    for y in range(1, height):
        above_start, above_end = row_slices[y - 1]
        here_start, here_end = row_slices[y]
        i = above_start
        for j in range(here_start, here_end):
            _, start, end = runs[j]
            while i < above_end and runs[i][2] < start:
                i += 1
            k = i
            while k < above_end and runs[k][1] <= end:
                union(j, k)
                k += 1

    labels = np.zeros((height, width), dtype=np.int32)
    remap: dict[int, int] = {}
    for index, (y, start, end) in enumerate(runs):
        label = remap.setdefault(find(index), len(remap) + 1)
        labels[y, start:end] = label
    return labels, len(remap)


def instances_from_mask(mask, probabilities, min_area=0) -> list[dict]:
    """
    Split a binary mask into scored instances.
    :param mask: ``(H, W)`` boolean prediction
    :param probabilities: ``(H, W)`` float probabilities behind the mask
    :param min_area: Drop components with at most this many pixels
    :return: List of dicts with ``mask``, ``score``, ``area`` and ``bbox``
    """
    labels, n_labels = connected_components(mask)
    instances = []

    for label in range(1, n_labels + 1):
        component = labels == label
        area = int(component.sum())
        if area <= min_area:
            continue
        ys, xs = np.nonzero(component)
        x0, y0 = int(xs.min()), int(ys.min())
        instances.append(
            {
                "mask": component,
                "score": float(probabilities[component].mean()),
                "area": area,
                "bbox": [x0, y0, int(xs.max()) - x0 + 1, int(ys.max()) - y0 + 1],
            }
        )
    return instances


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

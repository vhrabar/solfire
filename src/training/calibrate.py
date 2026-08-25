"""
Scoring and post-processing calibration on the path the submission is written with.

The model is applied to whole observations at native resolution, exactly as at
prediction time, and the decision threshold, the morphological offset and the
minimum area are chosen on that output rather than on a downsampled proxy.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

from ..inference.postprocess import SUBMISSION_SIZE, apply_offset, connected_components
from ..inference.tiling import DEFAULT_OVERLAP, DEFAULT_TILE, predict_probabilities, predict_tiled
from .metrics import DEFAULT_THRESHOLDS, grid_edges, overlaps_from_counts, pairwise_counts, panoptic_quality

DEFAULT_OFFSETS = (-2, -1, 0, 1, 2)
DEFAULT_MIN_AREAS = (0, 128, 256, 512, 1024)


def resize(tensor: Tensor, size) -> Tensor:
    """
    Resample a single-channel map, filtering on the way down.
    :param tensor: ``(1, H, W)`` map
    :param size: Edge length to resample to
    :return: ``(1, size, size)`` map
    """
    if tensor.shape[-1] == size and tensor.shape[-2] == size:
        return tensor
    resampled = F.interpolate(tensor[None], size=(size, size), mode="bilinear", align_corners=False, antialias=True)
    return resampled[0]


@torch.no_grad()
def observation_probabilities(
    model,
    loader,
    device,
    tile=DEFAULT_TILE,
    overlap=DEFAULT_OVERLAP,
    tta=True,
    batch_size=8,
    limit=None,
    model_size=None,
    original_size=None,
):
    """
    Run inference over whole observations and hand back the submission's view of them.

    :param model: Model in eval mode
    :param loader: Loader yielding ``(image, mask)`` batches of one observation
    :param device: Device to run on
    :param tile: Edge length of the inference windows, or ``None`` to run the model
        on the whole observation at once
    :param overlap: Fraction of a window shared with its neighbour
    :param tta: Average predictions over flips
    :param batch_size: Windows evaluated per forward pass
    :param limit: Stop after this many observations, or ``None`` for all of them
    :param model_size: Edge length the observation is resized to before the model,
        or ``None`` to feed it as the loader yields it
    :param original_size: Edge length the probabilities are resampled to before they
        are scored, or ``None`` to score them as the model produced them
    :return: Iterator of (``(H, W)`` probabilities, ``(H, W)`` boolean annotation)
    """
    model.eval()
    seen = 0
    total = len(loader) if limit is None else min(limit, len(loader))
    for images, masks in tqdm(loader, desc="inference", total=total, leave=False):
        for image, mask in zip(images, masks):
            if model_size is not None:
                image = resize(image, model_size)
            if tile:
                tiled = predict_tiled(model, image, device, tile=tile, overlap=overlap, tta=tta, batch_size=batch_size)
                probabilities = tiled[None]
            else:
                probabilities = predict_probabilities(model, image[None].to(device), tta=tta)[0].cpu()
            if original_size is not None:
                probabilities = resize(probabilities, original_size)

            yield probabilities[0].numpy(), mask.numpy() > 0.5
            seen += 1
            if limit is not None and seen >= limit:
                return


def score_grid(
    probabilities,
    target,
    thresholds=DEFAULT_THRESHOLDS,
    offsets=DEFAULT_OFFSETS,
    min_areas=DEFAULT_MIN_AREAS,
) -> dict:
    """
    Overlap matrices of one observation for every post-processing setting.

    Connected components are labelled once per (threshold, offset) pair; dropping
    the components below a minimum area only removes columns from the overlap
    matrix, so the areas are swept for free.
    :param probabilities: ``(H, W)`` foreground probabilities
    :param target: ``(H, W)`` boolean annotation
    :param thresholds: Probability thresholds to score at
    :param offsets: Morphological steps applied to the thresholded mask
    :param min_areas: Minimum instance areas to score at
    :return: Dict keyed by ``(threshold, offset, min_area)`` holding ``(n_gt, n_pred)`` IoU matrices
    """
    gt_labels, n_gt = connected_components(target)
    scores = {}

    for threshold in thresholds:
        mask = probabilities >= threshold
        for offset in offsets:
            pred_labels, n_pred = connected_components(apply_offset(mask, offset))
            iou, _, areas = overlaps_from_counts(pairwise_counts(gt_labels, n_gt, pred_labels, n_pred))
            for min_area in min_areas:
                keep = areas > min_area
                scores[(threshold, offset, min_area)] = torch.from_numpy(iou[:, keep]).float()
    return scores


def calibrate(
    model,
    loader,
    device,
    thresholds=DEFAULT_THRESHOLDS,
    offsets=DEFAULT_OFFSETS,
    min_areas=DEFAULT_MIN_AREAS,
    tile=DEFAULT_TILE,
    overlap=DEFAULT_OVERLAP,
    tta=True,
    batch_size=8,
    limit=None,
    model_size=None,
    original_size=SUBMISSION_SIZE,
) -> dict:
    """
    Choose the post-processing that maximises Panoptic Quality on a validation split.

    :param model: Model in eval mode
    :param loader: Loader yielding ``(image, mask)`` batches at native resolution
    :param device: Device to run on
    :param thresholds: Probability thresholds to search
    :param offsets: Morphological steps to search
    :param min_areas: Minimum instance areas to search
    :param tile: Edge length of the inference windows
    :param overlap: Fraction of a window shared with its neighbour
    :param tta: Average predictions over flips
    :param batch_size: Windows evaluated per forward pass
    :param limit: Score only this many observations, or ``None`` for all of them
    :param model_size: Edge length the observation is resized to before the model,
        or ``None`` to feed it at the resolution the loader yields
    :param original_size: Edge length everything is scored at, which is the size the
        submission masks are written at
    :return: Dict with the winning ``threshold``, ``offset``, ``min_area``, its ``pq``,
        the ``at_edge`` settings whose winner ran out of grid, the number of scored
        ``observations`` and the full ``table`` of settings
    """
    accumulated: dict[tuple, list] = {}
    observations = 0

    for probabilities, target in observation_probabilities(
        model,
        loader,
        device,
        tile=tile,
        overlap=overlap,
        tta=tta,
        batch_size=batch_size,
        limit=limit,
        model_size=model_size,
        original_size=original_size,
    ):
        for key, iou in score_grid(probabilities, target, thresholds, offsets, min_areas).items():
            accumulated.setdefault(key, []).append(iou)
        observations += 1

    table = [
        {"threshold": float(threshold), "offset": int(offset), "min_area": int(min_area), "pq": panoptic_quality(overlaps)}
        for (threshold, offset, min_area), overlaps in accumulated.items()
    ]
    table.sort(key=lambda row: -row["pq"])
    best = table[0] if table else {"threshold": 0.5, "offset": 0, "min_area": 0, "pq": 0.0}
    edges = grid_edges(
        {"threshold": best["threshold"], "offset": best["offset"], "min_area": best["min_area"]},
        {"threshold": thresholds, "offset": offsets, "min_area": min_areas},
    )

    return {
        "threshold": best["threshold"],
        "offset": best["offset"],
        "min_area": best["min_area"],
        "pq": best["pq"],
        "at_edge": edges,
        "tile": tile,
        "overlap": overlap,
        "tta": tta,
        "scored_at": original_size,
        "observations": observations,
        "table": table,
    }


def evaluate_tiled(
    model,
    loader,
    device,
    thresholds=DEFAULT_THRESHOLDS,
    offset=0,
    min_area=0,
    tile=DEFAULT_TILE,
    overlap=DEFAULT_OVERLAP,
    tta=False,
    batch_size=8,
    limit=None,
) -> dict:
    """
    Validation on whole observations, in the shape :func:`training.engine.fit` expects.

    :param model: Model to score
    :param loader: Loader yielding ``(image, mask)`` batches at native resolution
    :param device: Device to run on
    :param thresholds: Probability thresholds to score at
    :param offset: Morphological steps applied to the thresholded mask
    :param min_area: Drop instances with at most this many pixels
    :param tile: Edge length of the inference windows
    :param overlap: Fraction of a window shared with its neighbour
    :param tta: Average predictions over flips
    :param batch_size: Windows evaluated per forward pass
    :param limit: Score only this many observations, or ``None`` for all of them
    :return: Dict with ``thresholds``, per-threshold ``pq``, ``best_threshold``,
        ``best_pq``, ``overlaps`` of the best threshold and the mean ``loss``
    """
    thresholds = tuple(thresholds)
    accumulated: dict[float, list] = {threshold: [] for threshold in thresholds}
    total_loss = 0.0
    observations = 0

    for probabilities, target in observation_probabilities(
        model, loader, device, tile=tile, overlap=overlap, tta=tta, batch_size=batch_size, limit=limit
    ):
        clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
        truth = target.astype(np.float32)
        total_loss += float(-np.mean(truth * np.log(clipped) + (1.0 - truth) * np.log(1.0 - clipped)))

        gt_labels, n_gt = connected_components(target)
        for threshold in thresholds:
            pred_mask = apply_offset(probabilities >= threshold, offset)
            pred_labels, n_pred = connected_components(pred_mask)
            try:
                iou, dice, areas = overlaps_from_counts(pairwise_counts(gt_labels, n_gt, pred_labels, n_pred))
            except ValueError as exc:
                import time as _time

                dump_path = f"/tmp/pairwise_counts_failure_{int(_time.time())}.npz"
                np.savez(
                    dump_path,
                    probabilities=probabilities,
                    target=target,
                    gt_labels=gt_labels,
                    pred_labels=pred_labels,
                    pred_mask=pred_mask,
                    n_gt=n_gt,
                    n_pred=n_pred,
                    threshold=threshold,
                    offset=offset,
                )
                print(
                    f"pairwise_counts failed at threshold={threshold} offset={offset} "
                    f"n_gt={n_gt} n_pred={n_pred} gt_labels[min,max]=({gt_labels.min()},{gt_labels.max()}) "
                    f"pred_labels[min,max]=({pred_labels.min()},{pred_labels.max()}) "
                    f"probabilities[min,max]=({np.nanmin(probabilities)},{np.nanmax(probabilities)}) "
                    f"nan_in_probabilities={bool(np.isnan(probabilities).any())} error={exc} "
                    f"dumped to {dump_path}"
                )
                iou = np.zeros((n_gt, 0), dtype=np.float64)
                dice = np.zeros((n_gt, 0), dtype=np.float64)
                areas = np.zeros((0,), dtype=np.float64)
            keep = areas > min_area
            accumulated[threshold].append((torch.from_numpy(iou[:, keep]).float(), torch.from_numpy(dice[:, keep]).float()))
        observations += 1

    scores = [panoptic_quality([iou for iou, _ in accumulated[threshold]]) for threshold in thresholds]
    best = max(range(len(scores)), key=scores.__getitem__) if scores else 0
    return {
        "thresholds": list(thresholds),
        "pq": scores,
        "best_threshold": thresholds[best],
        "best_pq": scores[best],
        "overlaps": accumulated[thresholds[best]],
        "loss": total_loss / max(1, observations),
    }


def write_calibration(path, calibration) -> None:
    """
    Store a calibration next to the checkpoint it belongs to.
    :param path: Destination JSON file
    :param calibration: Result of :func:`calibrate`
    :return: None
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(calibration, indent=2), encoding="utf-8")

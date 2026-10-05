"""
Scoring and post-processing calibration on the path the submission is written with.

The model is applied to whole observations at native resolution, exactly as at
prediction time, and the post-processing is chosen on that output, scored the way
the organizers score it: every annotated filament is its own instance, filled the
way pycocotools fills it, and every annotator record of an observation counts.
"""

import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

from ..data import TRAIN_IMAGES, load_image
from ..data.preprocess import observation_path
from ..inference.postprocess import apply_offset, connected_components, instance_scores
from ..inference.tiling import DEFAULT_OVERLAP, DEFAULT_TILE, predict_probabilities, predict_tiled
from .metrics import (
    DEFAULT_THRESHOLDS,
    annotation_pixels,
    grid_edges,
    instance_iou,
    overlaps_from_counts,
    pairwise_counts,
    panoptic_quality,
    pq_counts,
    pq_from_counts,
)

# fine around the optimum: chosen on 4 folds this grid scores +0.0016 PQ on the 5th over a coarse one
CALIBRATION_THRESHOLDS = (0.25, 0.3, 0.325, 0.35, 0.375, 0.4, 0.45, 0.5, 0.65)
DEFAULT_OFFSETS = (-2, -1, 0, 1)
DEFAULT_MIN_AREAS = (0, 128, 256, 512)
DEFAULT_MIN_SCORES = (0.0, 0.6, 0.625, 0.65, 0.675, 0.7, 0.725, 0.75, 0.8, 0.85, 0.88, 0.91, 0.94)
CACHE_SCALE = 255


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
    for images, masks in tqdm(loader, desc="inference", total=total, leave=False, mininterval=2.0):
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


def cache_directory(checkpoint) -> Path:
    """
    Where the validation probabilities of a checkpoint are kept.
    :param checkpoint: Path to the checkpoint
    :return: Directory next to the checkpoint
    """
    path = Path(checkpoint)
    return path.with_name(f"{path.stem}.val")


@torch.no_grad()
def cache_probabilities(
    model,
    file_names,
    cache_dir,
    device,
    images_dir=TRAIN_IMAGES,
    tile=DEFAULT_TILE,
    overlap=DEFAULT_OVERLAP,
    tta=True,
    batch_size=8,
    identity=None,
) -> None:
    """
    Run a model over whole validation observations once and keep what it said.

    Every annotator record of an observation is scored against the same prediction,
    so the model runs once per file rather than once per record, and ensembles and
    pooled out-of-fold calibrations are built from the cache without a GPU. The
    probabilities are stored as 8 bit, a resolution well below anything the
    threshold search can tell apart.
    :param model: Model in eval mode
    :param file_names: Observations to cover; the ones already cached are skipped
    :param cache_dir: Directory the ``.npy`` files are written to
    :param device: Device to run on
    :param images_dir: Directory holding the observations
    :param tile: Edge length of the inference windows
    :param overlap: Fraction of a window shared with its neighbour
    :param tta: Average predictions over flips
    :param batch_size: Windows evaluated per forward pass
    :param identity: What the cached maps depend on beyond the inference settings,
        such as the epoch of the checkpoint; a cache made under anything else is dropped
    :return: None
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    settings = {"tile": tile, "overlap": overlap, "tta": tta, **(identity or {})}
    stamp = cache_dir / "settings.json"
    if not stamp.is_file() or json.loads(stamp.read_text()) != settings:
        for stale in cache_dir.glob("*.npy"):
            stale.unlink()
        stamp.write_text(json.dumps(settings))
    missing = [name for name in file_names if not (cache_dir / f"{Path(name).stem}.npy").is_file()]
    model.eval()
    for name in tqdm(missing, desc="inference", leave=False, mininterval=2.0):
        image = load_image(observation_path(images_dir, name))
        probabilities = predict_tiled(model, image, device, tile=tile, overlap=overlap, tta=tta, batch_size=batch_size)
        np.save(cache_dir / f"{Path(name).stem}.npy", (probabilities.numpy() * CACHE_SCALE).round().astype(np.uint8))


def load_probabilities(cache_dirs, file_name) -> np.ndarray:
    """
    The mean of the cached probabilities several models gave one observation.
    :param cache_dirs: Directories written by :func:`cache_probabilities`
    :param file_name: Observation to load
    :return: ``(H, W)`` float32 probabilities
    """
    total = None
    for cache_dir in cache_dirs:
        values = np.load(Path(cache_dir) / f"{Path(file_name).stem}.npy").astype(np.float32)
        total = values if total is None else total + values
    assert total is not None
    return np.asarray(total / (CACHE_SCALE * len(cache_dirs)), dtype=np.float32)


def score_observation(
    probabilities,
    records,
    thresholds=CALIBRATION_THRESHOLDS,
    offsets=DEFAULT_OFFSETS,
    min_areas=DEFAULT_MIN_AREAS,
    min_scores=DEFAULT_MIN_SCORES,
) -> dict:
    """
    Panoptic Quality terms of one observation for every post-processing setting.

    Components are labelled once per (threshold, offset) pair; the minimum area and
    the minimum score only remove columns from the IoU matrices, so they are swept
    for free. Every annotator record of the observation is scored against the same
    prediction, the way the organizers score a test image annotated more than once.
    :param probabilities: ``(H, W)`` foreground probabilities
    :param records: Per annotator record, the pixels of its filaments from
        :func:`metrics.annotation_pixels`
    :param thresholds: Probability thresholds to score at
    :param offsets: Morphological steps applied to the thresholded mask
    :param min_areas: Minimum instance areas to score at
    :param min_scores: Minimum instance mean probabilities to score at
    :return: Dict keyed by ``(threshold, offset, min_area, min_score)`` holding
        ``[matched iou, tp, fp, fn]`` summed over the records
    """
    terms = {}
    for threshold in thresholds:
        mask = probabilities >= threshold
        for offset in offsets:
            labels, n_labels = connected_components(apply_offset(mask, offset))
            areas = np.bincount(labels.ravel(), minlength=n_labels + 1)[1:]
            scores = instance_scores(labels, n_labels, probabilities)
            overlaps = [instance_iou(pixels, labels, n_labels) for pixels in records]
            for min_area in min_areas:
                for min_score in min_scores:
                    keep = (areas > min_area) & (scores >= min_score)
                    total = np.zeros(4)
                    for iou in overlaps:
                        total += pq_counts(iou[:, keep])
                    terms[(threshold, offset, min_area, min_score)] = total
    return terms


def calibrate(
    sources,
    records,
    thresholds=CALIBRATION_THRESHOLDS,
    offsets=DEFAULT_OFFSETS,
    min_areas=DEFAULT_MIN_AREAS,
    min_scores=DEFAULT_MIN_SCORES,
    workers=8,
) -> dict:
    """
    Choose the post-processing that maximises Panoptic Quality on held-out observations.

    :param sources: Per observation file, the cache directories of the models that
        did not train on it; their probabilities are averaged
    :param records: Per observation file, the annotations of each of its records
    :param thresholds: Probability thresholds to search
    :param offsets: Morphological steps to search
    :param min_areas: Minimum instance areas to search
    :param min_scores: Minimum instance mean probabilities to search
    :param workers: Processes scoring observations in parallel
    :return: Dict with the winning ``threshold``, ``offset``, ``min_area`` and
        ``min_score``, its ``pq`` and ``tp``/``fp``/``fn``, the ``at_edge`` settings
        whose winner ran out of grid, the number of scored ``files`` and ``records``,
        and the ``table`` of settings
    """
    grid = (tuple(thresholds), tuple(offsets), tuple(min_areas), tuple(min_scores))
    jobs = [(name, sources[name], records[name], grid) for name in sorted(sources)]
    accumulated: dict[tuple, np.ndarray] = {}
    with ProcessPoolExecutor(max(1, workers), mp_context=multiprocessing.get_context("fork")) as pool:
        for terms in tqdm(pool.map(_score_job, jobs), total=len(jobs), desc="scoring", leave=False, mininterval=2.0):
            for key, value in terms.items():
                accumulated[key] = accumulated.get(key, 0) + value

    table = [
        {
            "threshold": float(threshold),
            "offset": int(offset),
            "min_area": int(min_area),
            "min_score": float(min_score),
            "pq": pq_from_counts(*total),
            "tp": int(total[1]),
            "fp": int(total[2]),
            "fn": int(total[3]),
        }
        for (threshold, offset, min_area, min_score), total in accumulated.items()
    ]
    table.sort(key=lambda row: -row["pq"])
    best = table[0]
    names = ("threshold", "offset", "min_area", "min_score")
    edges = grid_edges({name: best[name] for name in names}, dict(zip(names, grid)))
    return {
        **best,
        "at_edge": edges,
        "files": len(jobs),
        "records": sum(len(value) for value in records.values()),
        "table": table,
    }


def _score_job(job) -> dict:
    """
    Score one observation in a worker process.
    :param job: Tuple of (file name, cache directories, annotations per record, grid)
    :return: Result of :func:`score_observation`
    """
    file_name, cache_dirs, annotations, (thresholds, offsets, min_areas, min_scores) = job
    probabilities = load_probabilities(cache_dirs, file_name)
    records = [annotation_pixels(record) for record in annotations]
    return score_observation(probabilities, records, thresholds, offsets, min_areas, min_scores)


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

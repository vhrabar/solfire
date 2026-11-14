"""
The quantitative eval on held-out samples
"""

import multiprocessing
import pickle
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from ..data import TRAIN_IMAGES, load_image
from ..data.preprocess import observation_path
from ..inference.postprocess import SUBMISSION_SIZE, keep_labels, label_instances
from ..inference.refiner import assemble, offset_labels
from .calibrate import load_probabilities
from .metrics import IOU_THRESHOLD, annotation_pixels, instance_iou, pq_from_counts

_GROUP_FILTERS: dict[int, Any] = {}


def evaluate_observation(labels, n_labels, records) -> dict:
    """
    Every quantity the judging uses, for one observation and all its annotator records.
    :param labels: ``(H, W)`` predicted labels with 0 as background
    :param n_labels: Number of predictions
    :param records: Per annotator record, the pixels of its filaments from :func:`metrics.annotation_pixels`
    :return: Dict with per-record ``dice`` (whole-mask Dice), the ``pairs`` IoU of
        every overlapping pair, the IoU of the ``matched`` pairs, the ``gt_degree`` and
        ``pred_degree`` counts, the ``overlaps`` matrices, and PQ ``counts``
    """
    predicted = labels.ravel() > 0
    n_predicted = int(predicted.sum())
    result: dict = {"dice": [], "pairs": [], "matched": [], "gt_degree": [], "pred_degree": [], "overlaps": []}
    counts = np.zeros(4)
    for pixels in records:
        iou = instance_iou(pixels, labels, n_labels)
        union = np.unique(np.concatenate(pixels)) if pixels else np.zeros(0, dtype=np.int64)
        intersection = int(predicted[union].sum())
        total = n_predicted + len(union)
        result["dice"].append(2 * intersection / total if total else 1.0)

        hits = iou > IOU_THRESHOLD
        touching = iou > 0
        result["pairs"] += iou[touching].tolist()
        result["matched"] += iou[hits].tolist()
        result["gt_degree"] += touching.sum(axis=1).tolist()
        result["pred_degree"] += touching.sum(axis=0).tolist()
        result["overlaps"].append(iou)
        counts += (float(iou[hits].sum()), int(hits.sum()), int((~hits.any(axis=0)).sum()), int((~hits.any(axis=1)).sum()))
    result["counts"] = counts
    return result


def bootstrap_pq(counts, other=None, draws: int = 2000, seed: int = 0) -> tuple[float, float]:
    """
    Percentile bootstrap interval of PQ, resampling whole observations with replacement.

    PQ is a ratio of sums, so every draw only re-adds the per-observation counts. Observations are the
    resampling unit because the records of one observation are not independent. This covers the sampling
    noise of the evaluation set, not the training seed noise, which needs several runs.
    :param counts: ``(n_observations, 4)`` per-observation ``matched_iou, tp, fp, fn`` of :func:`evaluate_observation`
    :param other: Counts of a second method over the same observations; the interval is then of the paired
        difference ``counts - other``, drawing the same observations for both
    :param draws: Number of bootstrap draws
    :param seed: Seed of the draws, so the interval is reproducible
    :return: Tuple of the 2.5 % and 97.5 % percentiles
    """
    counts = np.asarray(counts, dtype=float)
    picks = np.random.default_rng(seed).integers(len(counts), size=(draws, len(counts)))

    def pq_of(table):
        total = np.asarray(table, dtype=float)[picks].sum(axis=1)
        denominator = total[:, 1] + 0.5 * total[:, 2] + 0.5 * total[:, 3]
        return np.divide(total[:, 0], denominator, out=np.zeros(draws), where=denominator > 0)

    values = pq_of(counts) - (pq_of(other) if other is not None else 0.0)
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def summarise(results) -> dict:
    """
    Reduce per-observation results to the numbers the judging looks at.
    :param results: Results of :func:`evaluate_observation`
    :return: Summary dict; see the keys for the quantities
    """
    counts = sum((r["counts"] for r in results), np.zeros(4))
    matched_iou, tp, fp, fn = counts
    dice = [value for r in results for value in r["dice"]]
    pairs = np.asarray([value for r in results for value in r["pairs"]])
    matched = np.asarray([value for r in results for value in r["matched"]])
    gt_degree = Counter(min(int(d), 3) for r in results for d in r["gt_degree"])
    pred_degree = Counter(min(int(d), 3) for r in results for d in r["pred_degree"])
    n_gt, n_pred = sum(gt_degree.values()), sum(pred_degree.values())

    def dice_of(iou):
        return 2 * iou / (1 + iou)

    return {
        "pq": pq_from_counts(matched_iou, tp, fp, fn),
        "pq_ci": bootstrap_pq([r["counts"] for r in results]),
        "sq": float(matched_iou / tp) if tp else 0.0,
        "rq": float(tp / (tp + 0.5 * fp + 0.5 * fn)) if tp + fp + fn else 0.0,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "mean_dice": float(np.mean(dice)) if dice else 0.0,
        "matched_dice": float(dice_of(matched).mean()) if len(matched) else 0.0,
        "pair_iou": float(pairs.mean()) if len(pairs) else 0.0,
        "pair_dice": float(dice_of(pairs).mean()) if len(pairs) else 0.0,
        "gt_degree": {("3+" if k == 3 else str(k)): gt_degree[k] for k in sorted(gt_degree)},
        "pred_degree": {("3+" if k == 3 else str(k)): pred_degree[k] for k in sorted(pred_degree)},
        "fragmented_share": float(sum(v for k, v in gt_degree.items() if k >= 2) / n_gt) if n_gt else 0.0,
        "merged_share": float(sum(v for k, v in pred_degree.items() if k >= 2) / n_pred) if n_pred else 0.0,
        "missed_share": float(gt_degree[0] / n_gt) if n_gt else 0.0,
        "spurious_share": float(pred_degree[0] / n_pred) if n_pred else 0.0,
        "records": len(dice),
        "predictions_per_record": float(n_pred / len(dice)) if dice else 0.0,
    }


def evaluate(sources, records, calibration, images_dir=TRAIN_IMAGES, filters=None, workers=8, refined_dir=None):
    """
    Score a calibration's post-processing on held-out maps with every judged quantity.

    :param sources: Per observation file, the cache directories of the models that held it out
    :param records: Per observation file, the annotations of each of its records
    :param calibration: Calibration dict: ``threshold``, ``offset``, ``min_area`` and ``min_score``
    :param images_dir: Directory holding the observations the instance filters read
    :param filters: ``(group_of, filters)``: per observation file its group, and per group an
        object whose ``keep(labels, n_labels, probabilities, image)`` says which instances to
        submit, fitted without that group; ``None`` for no filter. A calibration naming a learned
        ``filter`` needs them (:func:`solfire.experiments.filter_fit.cross_validated_filters`)
    :param workers: Processes working in parallel
    :param refined_dir: Refined out-of-fold instances (``scripts/refine.py refine``); the
        instances are then assembled with the calibration's ``refiner`` block instead of
        cut from the probabilities
    :return: Tuple of (summary from :func:`summarise`, per-observation ``(iou, dice)`` matrices)
    """
    if refined_dir is not None:
        if "refiner" not in calibration:
            raise ValueError("refined instances need a calibration with a refiner block")
        if calibration.get("filter"):
            raise ValueError("a learned filter and the refiner are alternatives; the calibration has both")
    if calibration.get("filter") and filters is None:
        raise ValueError("a calibration with a learned filter is scored with its cross-validated filters; use solfire-lab")
    names = sorted(sources)
    group_of, group_filters = filters or ({}, {})

    settings = {key: calibration.get(key, 0) for key in ("threshold", "offset", "min_area", "min_score")}
    if refined_dir is not None:
        refiner = calibration["refiner"]
        settings = {
            "refined_dir": Path(refined_dir),
            "threshold": refiner["threshold"],
            "min_score": refiner["min_score"],
            "final_offset": refiner.get("final_offset", 0),
        }
    filtered = bool(group_filters)
    jobs = [
        (name, sources[name], records[name], settings, group_of.get(name, 0) if filtered else None, images_dir) for name in names
    ]
    results = []

    context = multiprocessing.get_context("forkserver" if filtered else "fork")
    with ProcessPoolExecutor(max(1, workers), mp_context=context, initializer=_set_filters, initargs=(group_filters,)) as pool:
        for result in tqdm(pool.map(_evaluate_one, jobs), total=len(jobs), desc="evaluating", leave=False, mininterval=2.0):
            results.append(result)
    overlaps = [(iou, 2 * iou / np.maximum(1 + iou, 1e-9)) for r in results for iou in r["overlaps"]]
    return summarise(results), overlaps


def _set_filters(filters) -> None:
    """
    Install the group filters in a worker process.
    :param filters: Filters keyed by observation group
    :return: None
    """
    # one thread per worker: the pool already fills the cores, nested thread pools only thrash
    threadpool_limits(1)
    _GROUP_FILTERS.clear()
    _GROUP_FILTERS.update(filters)


def _evaluate_one(job) -> dict:
    """
    Evaluate one observation in a worker process.
    :param job: Tuple of (file name, cache directories, annotations per record, settings, filter group, images dir)
    :return: Result of :func:`evaluate_observation`
    """
    name, cache_dirs, annotations, settings, group, images_dir = job
    if "refined_dir" in settings:
        with open(settings["refined_dir"] / f"{Path(name).stem}.pkl", "rb") as handle:
            stored = pickle.load(handle)
        labels, n_labels, _ = assemble(
            stored["refined"], SUBMISSION_SIZE, SUBMISSION_SIZE, threshold=settings["threshold"], min_score=settings["min_score"]
        )
        labels, n_labels = offset_labels(labels, n_labels, settings["final_offset"])
        return evaluate_observation(labels, n_labels, [annotation_pixels(record) for record in annotations])
    probabilities = load_probabilities(cache_dirs, name)
    labels, n_labels = label_instances(
        probabilities, settings["threshold"], settings["offset"], settings["min_area"], settings["min_score"]
    )
    if group is not None:
        image = load_image(observation_path(images_dir, name)).numpy()[0]
        labels, n_labels = keep_labels(labels, n_labels, _GROUP_FILTERS[group].keep(labels, n_labels, probabilities, image))
    return evaluate_observation(labels, n_labels, [annotation_pixels(record) for record in annotations])

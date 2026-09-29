"""
The quantitative eval on hgeld-out samples
"""

import multiprocessing
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from ..data import TRAIN_IMAGES, load_image
from ..data.preprocess import observation_path
from ..inference.instance_filter import InstanceFilter, build_regressor
from ..inference.postprocess import keep_labels, label_instances
from .calibrate import load_probabilities
from .metrics import IOU_THRESHOLD, annotation_pixels, instance_iou, pq_from_counts

_GROUP_FILTERS: dict[int, InstanceFilter] = {}


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


def evaluate(sources, records, calibration, images_dir=TRAIN_IMAGES, instance_data=None, folds=5, seed=0, workers=8):
    """
    Score a calibration's post-processing on held-out maps with every judged quantity.

    :param sources: Per observation file, the cache directories of the models that held it out
    :param records: Per observation file, the annotations of each of its records
    :param calibration: Calibration dict: ``threshold``, ``offset``, ``min_area``,
        ``min_score`` and optionally a ``filter`` path
    :param images_dir: Directory holding the observations the filter reads
    :param instance_data: Result of :func:`filter_fit.collect_instances` at the
        calibration's threshold, required when it names a filter
    :param folds: Groups the filter is cross-validated over
    :param seed: Seed of the grouping and the regressor
    :param workers: Processes working in parallel
    :return: Tuple of (summary from :func:`summarise`, per-observation ``(iou, dice)`` matrices)
    """
    names = sorted(sources)
    group_of: dict[str, int] = {}
    filters: dict[int, InstanceFilter] = {}
    if calibration.get("filter"):
        if instance_data is None:
            raise ValueError("a calibration with a learned filter needs its instance data to be cross-validated")
        shuffled = sorted(set(instance_data["files"].tolist()))
        random.Random(seed).shuffle(shuffled)
        group_of = {name: index % folds for index, name in enumerate(shuffled)}
        groups = np.asarray([group_of[name] for name in instance_data["files"]])
        cut = float(calibration.get("filter_report", {}).get("cut", InstanceFilter.load(calibration["filter"]).cut))
        for group in range(folds):
            model = build_regressor(seed)
            model.fit(instance_data["features"][groups != group], instance_data["earned"][groups != group])
            filters[group] = InstanceFilter(model=model, cut=cut, threshold=calibration["threshold"])

    settings = {key: calibration.get(key, 0) for key in ("threshold", "offset", "min_area", "min_score")}
    filtered = bool(filters)
    jobs = [
        (name, sources[name], records[name], settings, group_of.get(name, 0) if filtered else None, images_dir) for name in names
    ]
    results = []

    context = multiprocessing.get_context("forkserver" if filtered else "fork")
    with ProcessPoolExecutor(max(1, workers), mp_context=context, initializer=_set_filters, initargs=(filters,)) as pool:
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
    probabilities = load_probabilities(cache_dirs, name)
    labels, n_labels = label_instances(
        probabilities, settings["threshold"], settings["offset"], settings["min_area"], settings["min_score"]
    )
    if group is not None:
        image = load_image(observation_path(images_dir, name)).numpy()[0]
        labels, n_labels = keep_labels(labels, n_labels, _GROUP_FILTERS[group].keep(labels, n_labels, probabilities, image))
    return evaluate_observation(labels, n_labels, [annotation_pixels(record) for record in annotations])

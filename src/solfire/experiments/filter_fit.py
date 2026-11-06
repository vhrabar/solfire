"""
Fitting the learned instance filter on out-of-fold probability maps.
"""

import multiprocessing
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from solfire.data import TRAIN_IMAGES, load_image
from solfire.data.preprocess import observation_path
from solfire.inference.postprocess import apply_offset, connected_components
from solfire.training.calibrate import load_probabilities
from solfire.training.metrics import annotation_pixels, instance_iou

from .instance_filter import InstanceFilter, build_regressor, instance_features, matched_iou_per_record

DEFAULT_CUTS = tuple(round(value, 2) for value in np.arange(0.10, 0.45, 0.01))


def collect_instances(sources, records, threshold, offset=0, images_dir=TRAIN_IMAGES, workers=8) -> dict:
    """
    Features and earned IoU of every candidate instance on the held-out observations.
    :param sources: Per observation file, the cache directories of the models that held it out
    :param records: Per observation file, the annotations of each of its records
    :param threshold: Probability threshold the components are cut at
    :param offset: Morphological steps applied to the thresholded mask
    :param images_dir: Directory holding the observations the features read
    :param workers: Processes working in parallel
    :return: Dict with ``features``, ``earned`` (mean matched IoU per record),
        ``records`` (annotator records scoring each instance), ``files`` and the
        ``gt`` count summed over all records
    """
    jobs = [(name, sources[name], records[name], threshold, offset, images_dir) for name in sorted(sources)]
    features, earned, weights, files = [], [], [], []
    gt = 0
    # forkserver: a forked child hangs in OpenMP code once this process has fitted a regressor
    with ProcessPoolExecutor(
        max(1, workers), mp_context=multiprocessing.get_context("forkserver"), initializer=_single_thread
    ) as pool:
        for name, rows, gains, n_records, n_gt in tqdm(
            pool.map(_collect_one, jobs), total=len(jobs), desc="instances", leave=False, mininterval=2.0
        ):
            features.append(rows)
            earned.append(gains)
            weights.append(np.full(len(rows), n_records, dtype=np.float32))
            files += [name] * len(rows)
            gt += n_gt
    return {
        "features": np.concatenate(features),
        "earned": np.concatenate(earned),
        "records": np.concatenate(weights),
        "files": np.asarray(files),
        "gt": gt,
    }


def _single_thread() -> None:
    """
    Keep a worker to one thread: the pool already fills the cores, nested thread pools only thrash.
    :return: None
    """
    threadpool_limits(1)


def _collect_one(job):
    """
    Collect the instances of one observation in a worker process.
    :param job: Tuple of (file name, cache directories, annotations per record, threshold, offset, images dir)
    :return: Tuple of (file name, features, earned IoU, number of records, number of GT filaments)
    """
    name, cache_dirs, annotations, threshold, offset, images_dir = job
    probabilities = load_probabilities(cache_dirs, name)
    image = load_image(observation_path(images_dir, name)).numpy()[0]
    labels, n_labels = connected_components(apply_offset(probabilities >= threshold, offset))
    rows, described = instance_features(labels, n_labels, probabilities, image)
    pixels = [annotation_pixels(record) for record in annotations]
    earned = matched_iou_per_record([instance_iou(record, labels, n_labels) for record in pixels], n_labels)
    return name, rows, earned[described - 1].astype(np.float32), len(annotations), sum(len(p) for p in pixels)


def pq_of(keep, earned, records, gt) -> float:
    """
    Panoptic Quality of a selection of instances.

    :param keep: Boolean selection
    :param earned: Mean matched IoU per record of each instance
    :param records: Number of records scoring each instance
    :param gt: Annotated filaments summed over every record
    :return: Panoptic Quality
    """
    denominator = 0.5 * records[keep].sum() + 0.5 * gt
    return float((earned[keep] * records[keep]).sum() / denominator) if denominator else 0.0


def fit_instance_filter(data, threshold, offset=0, folds=5, cuts=DEFAULT_CUTS, seed=0) -> tuple[InstanceFilter, dict]:
    """
    Cross-validate the regressor over observations, choose its cut, and refit on everything.
    :param data: Result of :func:`collect_instances`
    :param threshold: Probability threshold the instances were cut at
    :param offset: Morphological offset the instances were cut with
    :param folds: Groups of observations the regressor is cross-validated over
    :param cuts: Cuts on the predicted matched IoU to search
    :param seed: Seed of the grouping and the regressor
    :return: Tuple of (fitted filter, report with the out-of-fold ``pq`` per cut and the winner)
    """
    features, earned, records, files, gt = (data[key] for key in ("features", "earned", "records", "files", "gt"))
    names = sorted(set(files.tolist()))
    random.Random(seed).shuffle(names)
    group_of = {name: index % folds for index, name in enumerate(names)}
    groups = np.asarray([group_of[name] for name in files])

    predicted = np.zeros(len(earned))
    for group in range(folds):
        held_out = groups == group
        model = build_regressor(seed)
        model.fit(features[~held_out], earned[~held_out])
        predicted[held_out] = model.predict(features[held_out])

    scores = {float(cut): pq_of(predicted >= cut, earned, records, gt) for cut in cuts}
    cut = max(scores, key=scores.__getitem__)
    model = build_regressor(seed)
    model.fit(features, earned)
    report = {
        "pq": scores[cut],
        "cut": cut,
        "threshold": threshold,
        "offset": offset,
        "instances": len(earned),
        "kept": int((predicted >= cut).sum()),
        "files": len(names),
        "gt": int(gt),
        "by_cut": scores,
    }
    return InstanceFilter(model=model, cut=cut, threshold=threshold, offset=offset, metadata=report), report


def filter_path(output) -> Path:
    """
    Where the pickled filter of a calibration JSON is kept.
    :param output: Calibration JSON path
    :return: ``<output stem>.filter.pkl`` next to it
    """
    path = Path(output)
    return path.with_name(path.name.removesuffix(".json").removesuffix(".calibration") + ".filter.pkl")


def cross_validated_filters(instance_data, calibration, folds=5, seed=0) -> tuple[dict[str, int], dict[int, InstanceFilter]]:
    """
    Refit a calibration's learned filter once per group of observations, each without that group,
    so :func:`solfire.training.evaluate.evaluate` scores every observation with a filter that never saw it.
    :param instance_data: Result of :func:`collect_instances` at the calibration's threshold
    :param calibration: Calibration naming the ``filter``, and its ``filter_report`` cut when stored
    :param folds: Groups the filter is cross-validated over
    :param seed: Seed of the grouping and the regressor
    :return: Tuple of (per observation file its group, per group the filter fitted without it)
    """
    shuffled = sorted(set(instance_data["files"].tolist()))
    random.Random(seed).shuffle(shuffled)
    group_of = {name: index % folds for index, name in enumerate(shuffled)}
    groups = np.asarray([group_of[name] for name in instance_data["files"]])
    cut = float(calibration.get("filter_report", {}).get("cut", InstanceFilter.load(calibration["filter"]).cut))
    filters = {}
    for group in range(folds):
        model = build_regressor(seed)
        model.fit(instance_data["features"][groups != group], instance_data["earned"][groups != group])
        filters[group] = InstanceFilter(model=model, cut=cut, threshold=calibration["threshold"])
    return group_of, filters

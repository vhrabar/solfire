"""
The learned instance filter and the sums its cut is chosen on.
"""

import numpy as np
import pytest
from synthetic import SIZE, filament_polygon

from solfire.experiments.filter_fit import pq_of
from solfire.experiments.instance_filter import (
    FEATURES,
    MIN_PIXELS,
    InstanceFilter,
    instance_features,
    matched_iou_per_record,
)
from solfire.inference.postprocess import connected_components
from solfire.training.metrics import annotation_pixels, instance_iou, pq_counts, pq_from_counts


def scene(rng):
    """
    A bright synthetic disk, annotated filaments and a prediction of some of them plus noise.
    """
    image = np.full((SIZE, SIZE), 0.5, dtype=np.float32)
    record = [{"segmentation": [filament_polygon(rng)]} for _ in range(4)]
    pixels = annotation_pixels(record, SIZE, SIZE)
    probabilities = np.zeros((SIZE, SIZE), dtype=np.float32)
    for index, flat in enumerate(pixels):
        if index < 3:
            probabilities.ravel()[flat] = 0.9
            image.ravel()[flat] = 0.3
    probabilities[2:8, 2:20] = 0.7  # a spurious detection
    return image, probabilities, record


def test_features_describe_every_large_component(rng):
    """
    One row per component above the minimum size, in the order of FEATURES.
    """
    image, probabilities, _ = scene(rng)
    labels, n_labels = connected_components(probabilities >= 0.5)
    features, described = instance_features(labels, n_labels, probabilities, image)
    areas = np.bincount(labels.ravel(), minlength=n_labels + 1)[1:]
    assert features.shape == (int((areas >= MIN_PIXELS).sum()), len(FEATURES))
    assert np.isfinite(features).all()
    assert set(described) == set(np.flatnonzero(areas >= MIN_PIXELS) + 1)


def test_pq_from_sums_equals_pq_from_matching(rng):
    """
    With disjoint predictions, PQ of any selection follows from per-instance sums.
    """
    _, probabilities, record = scene(rng)
    labels, n_labels = connected_components(probabilities >= 0.5)
    iou = instance_iou(annotation_pixels(record, SIZE, SIZE), labels, n_labels)
    earned = matched_iou_per_record([iou], n_labels)
    for keep in ([True] * n_labels, [k % 2 == 0 for k in range(n_labels)]):
        keep = np.asarray(keep)
        direct = pq_from_counts(*pq_counts(iou[:, keep]))
        assert pq_of(keep, earned, np.ones(n_labels), len(record)) == pytest.approx(direct)


class Constant:
    """
    A stand-in regressor predicting one value.
    """

    def __init__(self, value):
        self.value = value

    def predict(self, features):
        return np.full(len(features), self.value)


def test_filter_keeps_by_cut_and_round_trips(rng, tmp_path):
    """
    A prediction above the cut keeps every described instance; the pickle loads back.
    """
    image, probabilities, _ = scene(rng)
    labels, n_labels = connected_components(probabilities >= 0.5)
    keep_all = InstanceFilter(model=Constant(0.5), cut=0.25, threshold=0.5).keep(labels, n_labels, probabilities, image)
    keep_none = InstanceFilter(model=Constant(0.1), cut=0.25, threshold=0.5).keep(labels, n_labels, probabilities, image)
    assert keep_all.sum() > 0 and keep_none.sum() == 0
    path = tmp_path / "filter.pkl"
    InstanceFilter(model=Constant(0.5), cut=0.25, threshold=0.5).save(path)
    assert InstanceFilter.load(path).cut == 0.25

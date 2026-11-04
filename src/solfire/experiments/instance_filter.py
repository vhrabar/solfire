"""
A learned decision on which predicted instances to submit.
"""

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

FEATURES = (
    "log_area",
    "mean_probability",
    "max_probability",
    "std_probability",
    "confident_share",
    "height",
    "width",
    "fill",
    "elongation",
    "major_axis",
    "disk_radius",
    "contrast",
    "intensity",
    "background",
)
MIN_PIXELS = 32
BACKGROUND_WINDOW = 61
DISK_LEVEL = 10 / 255
CONFIDENT = 0.8


def solar_disk(image) -> tuple[float, float, float, float]:
    """
    Centre, radius and typical brightness of the solar disk.

    Everything off the disk is black in these observations, so the disk is the set
    of lit pixels, and a circle of the same area and centroid describes it.
    :param image: ``(H, W)`` observation in ``[0, 1]``
    :return: Tuple of (centre y, centre x, radius, median disk brightness)
    """
    disk = image > DISK_LEVEL
    ys, xs = np.nonzero(disk)
    if len(ys) == 0:
        height, width = image.shape
        return height / 2, width / 2, min(height, width) / 2, float(np.median(image))
    return float(ys.mean()), float(xs.mean()), float(np.sqrt(len(ys) / np.pi)), float(np.median(image[disk]))


def instance_features(labels, n_labels, probabilities, image) -> tuple[np.ndarray, np.ndarray]:
    """
    Describe every labelled component for the filter, in the order of :data:`FEATURES`.

    Components under :data:`MIN_PIXELS` are not described: they are never submitted.
    :param labels: ``(H, W)`` label array with 0 as background
    :param n_labels: Number of labels in ``labels``
    :param probabilities: ``(H, W)`` foreground probabilities
    :param image: ``(H, W)`` observation in ``[0, 1]``, at the resolution of ``labels``
    :return: Tuple of (``(n, len(FEATURES))`` features, ``(n,)`` labels they describe)
    """
    if n_labels == 0:
        return np.zeros((0, len(FEATURES)), dtype=np.float32), np.zeros(0, dtype=np.int64)
    index = np.arange(1, n_labels + 1)
    area = np.bincount(labels.ravel(), minlength=n_labels + 1)[1:].astype(np.float64)
    described = np.flatnonzero(area >= MIN_PIXELS)
    if len(described) == 0:
        return np.zeros((0, len(FEATURES)), dtype=np.float32), np.zeros(0, dtype=np.int64)

    mean = np.asarray(ndimage.mean(probabilities, labels, index))
    maximum = np.asarray(ndimage.maximum(probabilities, labels, index))
    spread = np.asarray(ndimage.standard_deviation(probabilities, labels, index))
    confident = np.asarray(ndimage.sum(probabilities >= CONFIDENT, labels, index)) / np.maximum(area, 1)
    background = ndimage.uniform_filter(image, BACKGROUND_WINDOW)
    intensity = np.asarray(ndimage.mean(image, labels, index))
    local = np.asarray(ndimage.mean(background, labels, index))
    centre_y, centre_x, radius, brightness = solar_disk(image)
    brightness = max(brightness, 1e-3)
    slices = ndimage.find_objects(labels)

    rows = []
    for k in described:
        window = slices[k]
        ys, xs = np.nonzero(labels[window] == k + 1)
        height, width = labels[window].shape
        covariance = np.cov(np.stack([ys, xs])) if len(ys) > 2 else np.eye(2)
        minor, major = np.sort(np.linalg.eigvalsh(covariance))
        distance = np.hypot(ys.mean() + window[0].start - centre_y, xs.mean() + window[1].start - centre_x)
        rows.append(
            [
                np.log(area[k]),
                mean[k],
                maximum[k],
                spread[k],
                confident[k],
                height,
                width,
                area[k] / (height * width),
                np.sqrt(major / max(minor, 1e-3)),
                np.sqrt(major),
                distance / radius,
                intensity[k] - local[k],
                intensity[k] / brightness,
                local[k] / brightness,
            ]
        )
    return np.asarray(rows, dtype=np.float32), described + 1


def matched_iou_per_record(overlaps, n_labels) -> np.ndarray:
    """
    What each prediction earns, averaged over the annotator records it is scored against.
    :param overlaps: Per record, the ``(n_gt, n_labels)`` IoU matrix
    :param n_labels: Number of predictions
    :return: ``(n_labels,)`` mean IoU of the match, 0 where there is none
    """
    earned = np.zeros(n_labels)
    for iou in overlaps:
        if iou.shape[0]:
            best = iou.max(axis=0)
            earned += np.where(best > 0.5, best, 0.0)
    return earned / max(len(overlaps), 1)


@dataclass
class InstanceFilter:
    """
    The fitted regressor, the cut on its prediction, and the labelling it was fitted on.
    """

    model: Any
    cut: float
    threshold: float
    offset: int = 0
    metadata: dict = field(default_factory=dict)

    def keep(self, labels, n_labels, probabilities, image) -> np.ndarray:
        """
        Decide which components to submit.
        :param labels: ``(H, W)`` label array with 0 as background
        :param n_labels: Number of labels in ``labels``
        :param probabilities: ``(H, W)`` foreground probabilities
        :param image: ``(H, W)`` observation in ``[0, 1]``
        :return: ``(n_labels,)`` boolean, component ``k`` at index ``k - 1``
        """
        keep = np.zeros(n_labels, dtype=bool)
        features, described = instance_features(labels, n_labels, probabilities, image)
        if len(described):
            keep[described - 1] = self.model.predict(features) >= self.cut
        return keep

    def save(self, path) -> None:
        """
        Pickle the filter.
        :param path: Destination file
        :return: None
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump(self, handle)

    @staticmethod
    def load(path) -> InstanceFilter:
        """
        Read a filter written by :meth:`save`.
        :param path: File to read
        :return: The filter
        """
        with open(path, "rb") as handle:
            loaded = _RenamedPackageUnpickler(handle).load()
        if not isinstance(loaded, InstanceFilter):
            raise TypeError(f"{path} does not hold an InstanceFilter")
        return loaded


class _RenamedPackageUnpickler(pickle.Unpickler):
    """
    Reads filters pickled before this module moved: first the package was ``src``, and the
    filter lived in ``inference.instance_filter`` before it joined the experiments.
    """

    MOVED = {"src.inference.instance_filter": __name__, "solfire.inference.instance_filter": __name__}

    def find_class(self, module, name):
        module = self.MOVED.get(module, module)
        if module == "src" or module.startswith("src."):
            module = "solfire" + module.removeprefix("src")
        return super().find_class(module, name)


def build_regressor(seed=0) -> Any:
    """
    The regressor the filter is fitted with; small, and regularised against 16k samples.
    :param seed: Random state
    :return: An unfitted scikit-learn regressor
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0, random_state=seed
    )

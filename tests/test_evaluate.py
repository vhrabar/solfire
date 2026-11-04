"""
The judged quantities on refined out-of-fold instances.
"""

import pickle

import numpy as np
import pytest

from solfire.inference.refiner import RefinedInstance, Window
from solfire.training.evaluate import evaluate


def square(top, left, side) -> list:
    """:return: COCO polygon of a square"""
    return [[left, top, left + side, top, left + side, top + side, left, top + side]]


def test_refined_instances_are_assembled_with_the_refiner_settings(tmp_path):
    """
    The refined instance that matches the annotation counts, the one below ``min_score`` is
    dropped, and a calibration without a refiner block is refused.
    """
    match = np.zeros((64, 64), dtype=np.uint8)
    match[8:40, 8:40] = 255
    stray = np.zeros((64, 64), dtype=np.uint8)
    stray[10:20, 10:20] = 255
    instances = [RefinedInstance(Window(100, 100, 64), match, 0.9), RefinedInstance(Window(500, 500, 64), stray, 0.1)]
    with open(tmp_path / "obs.pkl", "wb") as handle:
        pickle.dump({"refined": instances, "seeds": instances}, handle)
    records = {"obs.jpeg": [[{"segmentation": square(108, 108, 32)}]]}
    calibration = {"threshold": 0.5, "refiner": {"threshold": 0.5, "min_score": 0.5}}

    summary, _ = evaluate({"obs.jpeg": []}, records, calibration, workers=1, refined_dir=tmp_path)
    assert summary["pq"] > 0.85
    assert summary["spurious_share"] == 0

    calibration["refiner"]["min_score"] = 0.0
    summary, _ = evaluate({"obs.jpeg": []}, records, calibration, workers=1, refined_dir=tmp_path)
    assert summary["spurious_share"] == pytest.approx(0.5)

    with pytest.raises(ValueError):
        evaluate({"obs.jpeg": []}, records, {"threshold": 0.5}, workers=1, refined_dir=tmp_path)

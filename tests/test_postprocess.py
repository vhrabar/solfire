"""
Turning probabilities into instances and instances into a valid submission.
"""

import csv

import numpy as np
from pycocotools import mask as coco_mask

from src.inference.postprocess import (
    SUBMISSION_SIZE,
    apply_offset,
    connected_components,
    encode_rle,
    instance_scores,
    label_instances,
    write_submission,
)


def two_blobs() -> np.ndarray:
    """
    A probability map with a confident large blob and a faint small one.
    """
    probabilities = np.zeros((64, 64), dtype=np.float32)
    probabilities[5:25, 5:25] = 0.95
    probabilities[40:44, 40:44] = 0.6
    return probabilities


def test_connected_components_are_eight_connected():
    """
    Diagonal neighbours belong to one component, as in the organizers' instances.
    """
    mask = np.zeros((5, 5), dtype=bool)
    mask[1, 1] = mask[2, 2] = mask[3, 3] = True
    _, n_labels = connected_components(mask)
    assert n_labels == 1


def test_min_area_and_min_score_filter_instances():
    """
    The faint 16-pixel blob goes with either filter, the confident one stays.
    """
    probabilities = two_blobs()
    assert label_instances(probabilities, 0.5)[1] == 2
    assert label_instances(probabilities, 0.5, min_area=16)[1] == 1
    labels, n_labels = label_instances(probabilities, 0.5, min_score=0.9)
    assert n_labels == 1
    assert labels[10, 10] == 1 and labels[41, 41] == 0


def test_instance_scores_are_mean_probabilities():
    """
    A component's score is the mean probability over its pixels.
    """
    probabilities = two_blobs()
    labels, n_labels = connected_components(probabilities >= 0.5)
    assert np.allclose(sorted(instance_scores(labels, n_labels, probabilities)), [0.6, 0.95])


def test_offset_grows_and_shrinks():
    """
    A positive offset dilates, a negative one erodes, by one diamond step each.
    """
    mask = np.zeros((9, 9), dtype=bool)
    mask[4, 4] = True
    assert apply_offset(mask, 1).sum() == 5
    assert apply_offset(apply_offset(mask, 1), -1).sum() == 1


def test_rle_round_trip():
    """
    The submission RLE decodes back to the same mask.
    """
    rng = np.random.default_rng(0)
    mask = rng.random((SUBMISSION_SIZE, SUBMISSION_SIZE)) > 0.999
    rle = encode_rle(mask)
    decoded = coco_mask.decode({"size": rle["size"], "counts": rle["counts"].encode()})
    assert np.array_equal(decoded.astype(bool), mask)


def test_submission_format(tmp_path):
    """
    One unquoted row per filament, ``filament_id,segmentation_rle``, masks at 2048.
    """
    mask = np.zeros((SUBMISSION_SIZE, SUBMISSION_SIZE), dtype=bool)
    mask[100:120, 200:260] = True
    detections = [{"filament_id": "20110109104734Ch_1", "segmentation": encode_rle(mask)}]
    path = tmp_path / "submission.csv"
    assert write_submission(detections, path) == 1
    text = path.read_text()
    assert text.splitlines()[0] == "filament_id,segmentation_rle"
    assert '"' not in text
    row = next(csv.DictReader(path.open()))
    decoded = coco_mask.decode({"size": [SUBMISSION_SIZE, SUBMISSION_SIZE], "counts": row["segmentation_rle"].encode()})
    assert np.array_equal(decoded.astype(bool), mask)

"""
Crop refiner: windows, inputs, the network's shapes, the loss and the pasting of instances.
"""

import numpy as np
import pytest
import torch
from scipy import ndimage

from solfire.inference.refiner import (
    MIN_WINDOW,
    RefinedInstance,
    Refiner,
    Window,
    assemble,
    component_window,
    offset_labels,
    refine_observation,
    refiner_inputs,
    window_of,
)
from solfire.training.refine import MAX_RECORDS, _flat_iou, record_weights, refiner_loss


def test_window_is_square_centred_and_padded_past_the_edge():
    """
    A component in the corner gets a window reaching off the image, filled with zeros there.
    """
    mask = np.zeros((300, 300), dtype=bool)
    mask[0:100, 0:40] = True
    (box,) = ndimage.find_objects(mask.astype(int))
    window = component_window(box)
    assert window.side == 150
    assert (window.top, window.left) == (-25, -55)
    cut = window_of(np.ones((300, 300), dtype=np.float32), window)
    assert cut.shape == (150, 150)
    assert cut[:25].sum() == 0 and cut[:, :55].sum() == 0
    assert cut[25:, 55:].min() == 1


def test_small_components_get_the_minimum_window():
    """
    A speck is refined on a window of at least :data:`MIN_WINDOW` pixels.
    """
    window = component_window((slice(50, 60), slice(70, 75)))
    assert window.side == MIN_WINDOW


def test_inputs_hold_image_probability_and_seed():
    """
    The three channels are resampled to the refiner size and stay in ``[0, 1]``.
    """
    image = np.full((200, 200), 0.5, dtype=np.float32)
    probabilities = np.zeros((200, 200), dtype=np.float32)
    seed = np.zeros((200, 200), dtype=bool)
    seed[90:110, 60:140] = True
    probabilities[seed] = 0.9
    inputs = refiner_inputs(image, probabilities, seed, Window(0, 0, 200), size=64)
    assert inputs.shape == (3, 64, 64)
    assert inputs.min() >= 0 and inputs.max() <= 1
    assert inputs[2, 32, 32] == pytest.approx(1.0)
    assert inputs[2, 5, 5] == 0


def test_refiner_returns_a_mask_and_a_score():
    """
    In eval mode the network maps ``(B, 3, S, S)`` to a mask and one score per sample.
    """
    model = Refiner(encoder_weights=None).eval()
    with torch.no_grad():
        masks, scores = model(torch.rand(2, 3, 64, 64))
    assert masks.shape == (2, 1, 64, 64)
    assert scores.shape == (2,)


def test_loss_rewards_the_matching_mask():
    """
    A mask logit that matches the target scores a lower loss than its inverse, and the
    score term learns the IoU the predicted mask earns.
    """
    targets = torch.zeros(1, MAX_RECORDS, 32, 32)
    targets[0, 0, 8:24, 8:24] = 1
    good = (targets[:, :1] * 2 - 1) * 8
    n_records = torch.tensor([1])
    loss_good, _, _ = refiner_loss(good, torch.tensor([4.0]), targets, n_records)
    loss_bad, _, _ = refiner_loss(-good, torch.tensor([4.0]), targets, n_records)
    assert loss_good < loss_bad
    _, _, confident = refiner_loss(good, torch.tensor([6.0]), targets, n_records)
    _, _, doubtful = refiner_loss(good, torch.tensor([-6.0]), targets, n_records)
    assert confident < doubtful


def test_strict_score_target_follows_the_record_with_fewest_filaments():
    """
    Under the strict target a filament only the liberal record drew earns nothing, so a
    confident score costs more than under the record average.
    """
    weights = record_weights({"a": [[1, 2, 3], [1]], "b": [[1], [2]]}, "strict")
    assert weights["a"].tolist() == [0.0, 1.0, 0.0]
    assert weights["b"].tolist() == [0.5, 0.5, 0.0]
    assert record_weights({"a": [[1, 2, 3], [1]]})["a"].tolist() == [0.5, 0.5, 0.0]
    with pytest.raises(ValueError):
        record_weights({}, "loose")

    targets = torch.zeros(1, MAX_RECORDS, 32, 32)
    targets[0, 0, 8:24, 8:24] = 1
    good = (targets[:, :1] * 2 - 1) * 8
    n_records = torch.tensor([2])
    confident = torch.tensor([4.0])
    _, _, mean = refiner_loss(good, confident, targets, n_records)
    _, _, liberal = refiner_loss(good, confident, targets, n_records, weights=torch.tensor([[1.0, 0.0, 0.0]]))
    _, _, strict = refiner_loss(good, confident, targets, n_records, weights=torch.tensor([[0.0, 1.0, 0.0]]))
    assert liberal < mean < strict


def test_assemble_drops_a_second_copy_of_the_same_filament():
    """
    Two fragments both redrawn as the whole filament are submitted once, under the higher score.
    """
    whole = np.zeros((40, 40), dtype=np.uint8)
    whole[10:30, 5:35] = 255
    first = RefinedInstance(Window(0, 0, 40), whole, 0.8)
    second = RefinedInstance(Window(0, 0, 40), whole, 0.6)
    labels, n_labels, scores = assemble([second, first], 40, 40)
    assert n_labels == 1
    assert scores.tolist() == pytest.approx([0.8])
    assert (labels > 0).sum() == 600


def test_assemble_cuts_masks_and_scores():
    """
    The mask threshold and the score cut both apply, and a window off the edge is clipped.
    """
    soft = np.zeros((40, 40), dtype=np.uint8)
    soft[10:30, 10:30] = 100
    soft[15:25, 15:25] = 220
    instance = RefinedInstance(Window(-10, -10, 40), soft, 0.5)
    labels, n_labels, _ = assemble([instance], 30, 30, threshold=0.5)
    assert n_labels == 1 and (labels > 0).sum() == 100
    labels, n_labels, _ = assemble([instance], 30, 30, threshold=0.3)
    assert (labels > 0).sum() == 400
    _, n_labels, _ = assemble([instance], 30, 30, min_score=0.6)
    assert n_labels == 0


def test_flat_iou():
    """
    IoU of sorted pixel index sets, with the disjoint shortcut.
    """
    assert _flat_iou(np.arange(0, 10), np.arange(5, 15)) == pytest.approx(5 / 15)
    assert _flat_iou(np.arange(0, 10), np.arange(20, 30)) == 0


def test_refine_observation_labels_every_kept_instance():
    """
    End to end on a small observation with two untrained refiners averaged: proposals are
    cut, redrawn, scored and painted, and nothing is kept past a score cut above one.
    """
    torch.manual_seed(0)
    models = [Refiner(encoder_weights=None).eval() for _ in range(2)]
    probabilities = np.zeros((160, 160), dtype=np.float32)
    probabilities[40:52, 20:140] = 0.9
    image = np.full((160, 160), 0.5, dtype=np.float32)
    settings = {"proposal_threshold": 0.5, "proposal_offset": 0, "threshold": 0.0, "min_score": 0.0}
    labels, n_labels = refine_observation(models, image, probabilities, settings, torch.device("cpu"), tta=False)
    assert n_labels == 1
    assert labels.shape == (160, 160) and labels.max() == 1
    _, n_labels = refine_observation(models, image, probabilities, {**settings, "min_score": 1.1}, torch.device("cpu"), tta=False)
    assert n_labels == 0


def test_offset_labels_grows_shrinks_and_drops_vanished_instances():
    """
    ``+1`` grows both instances, ``-1`` shrinks the large one and removes the 1-px line,
    renumbering densely, and ``0`` is the identity.
    """
    labels = np.zeros((20, 20), dtype=np.int32)
    labels[2, 2:10] = 1
    labels[8:14, 8:14] = 2
    same, n = offset_labels(labels, 2, 0)
    assert n == 2 and same is labels
    grown, n = offset_labels(labels, 2, 1)
    assert n == 2
    assert (grown == 1).sum() > (labels == 1).sum() and (grown == 2).sum() == 8 * 8
    assert np.all(grown[labels > 0] == labels[labels > 0])
    shrunk, n = offset_labels(labels, 2, -1)
    assert n == 1 and set(np.unique(shrunk)) == {0, 1} and (shrunk == 1).sum() == 4 * 4


def test_refine_observation_honours_final_offset():
    """
    The ``final_offset`` key of the settings grows the painted instance.
    """
    torch.manual_seed(0)
    models = [Refiner(encoder_weights=None).eval()]
    probabilities = np.zeros((160, 160), dtype=np.float32)
    probabilities[40:52, 20:140] = 0.9
    image = np.full((160, 160), 0.5, dtype=np.float32)
    settings = {"proposal_threshold": 0.5, "proposal_offset": 0, "threshold": 0.0, "min_score": 0.0}
    base, _ = refine_observation(models, image, probabilities, settings, torch.device("cpu"), tta=False)
    grown, n = refine_observation(models, image, probabilities, {**settings, "final_offset": 1}, torch.device("cpu"), tta=False)
    assert n == 1 and (grown > 0).sum() >= (base > 0).sum()


def test_per_record_lovasz_is_free_for_a_confident_match_and_costs_otherwise():
    """
    The Lovász term adds nothing when the mask matches every record with a margin, and adds
    loss when the records disagree, since no single mask then matches each of them.
    """
    targets = torch.zeros(1, MAX_RECORDS, 32, 32)
    targets[0, 0, 8:24, 8:24] = 1
    good = (targets[:, :1] * 2 - 1) * 8
    score = torch.tensor([4.0])
    plain, _, _ = refiner_loss(good, score, targets, torch.tensor([1]))
    with_term, _, _ = refiner_loss(good, score, targets, torch.tensor([1]), lovasz_weight=0.5)
    assert torch.isclose(plain, with_term)
    targets[0, 1, 8:24, 8:16] = 1
    plain, _, _ = refiner_loss(good, score, targets, torch.tensor([2]))
    with_term, _, _ = refiner_loss(good, score, targets, torch.tensor([2]), lovasz_weight=0.5)
    assert with_term > plain


def test_load_encoder_fills_the_observation_channel_and_keeps_the_others(tmp_path):
    refiner = Refiner(encoder="resnet18", encoder_weights=None)
    state = {key: torch.full_like(value, 0.5) for key, value in refiner.state_dict().items() if key.startswith("net.encoder.")}
    state["net.encoder.conv1.weight"] = torch.ones(64, 1, 7, 7)
    state["net.segmentation_head.0.weight"] = torch.zeros(1)  # not the encoder's: ignored
    torch.save({"model_state": state}, tmp_path / "encoder.pt")
    before = refiner.state_dict()["net.encoder.conv1.weight"].clone()
    assert refiner.load_encoder(tmp_path / "encoder.pt") == len(state) - 1
    conv = refiner.state_dict()["net.encoder.conv1.weight"]
    assert torch.equal(conv[:, :1], torch.ones(64, 1, 7, 7))
    assert torch.equal(conv[:, 1:], before[:, 1:])
    assert (refiner.state_dict()["net.encoder.layer1.0.conv1.weight"] == 0.5).all()

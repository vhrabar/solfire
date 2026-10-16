"""
Crop refiner: windows, inputs, the network's shapes, the loss and the pasting of instances.
"""

import numpy as np
import pytest
import torch
from scipy import ndimage

from src.inference.refiner import (
    MIN_WINDOW,
    RefinedInstance,
    Refiner,
    Window,
    assemble,
    component_window,
    refiner_inputs,
    window_of,
)
from src.training.refine import MAX_RECORDS, _flat_iou, refiner_loss


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

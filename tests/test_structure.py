"""
Fine-structure targets: thickness weights, the thin mask, the spine heatmap and their losses.
"""

import numpy as np
import pytest
import torch

from src.data.structure import REFERENCE_HALF_WIDTH, spine_heatmap, thickness_weight, thin_mask
from src.training.losses import BarbHeadLoss, BCEDiceLoss, ThinHeadLoss, lovasz_hinge_loss


def body_with_barb():
    """
    A 13 px thick horizontal body with a 3 px wide barb hanging off it.
    """
    mask = np.zeros((64, 64), dtype=bool)
    mask[20:33, 5:60] = True
    mask[33:50, 30:33] = True
    return mask


def test_thin_structures_weigh_more_than_the_body():
    """
    Body pixels keep weight 1, barb pixels get about reference / half-width, background 1.
    """
    weight = thickness_weight(body_with_barb(), cap=8)
    assert weight[26, 15] == pytest.approx(1.0)
    assert weight[0, 0] == 1.0
    assert 2.0 < weight[45, 31] <= REFERENCE_HALF_WIDTH / 2 + 1e-6
    assert weight.max() <= 8


def test_crop_edge_does_not_make_a_body_thin():
    """
    A wide filament cut by the crop edge keeps weight 1 and is not marked thin there.
    """
    mask = np.zeros((32, 32), dtype=bool)
    mask[10:23, :] = True
    assert np.allclose(thickness_weight(mask, cap=8)[10:23, 0], 1.0)
    assert not thin_mask(mask)[10:23, 0].any()


def test_thin_mask_is_the_barb():
    """
    The opening residual keeps the barb and drops the body.
    """
    thin = thin_mask(body_with_barb())
    assert thin[33:50, 30:33].mean() > 0.8
    assert not thin[22:31, 10:55].any()


def test_spine_heatmap_peaks_on_the_line_inside_the_window():
    """
    1 on the drawn centre line, shifted by the window origin, near 0 away from it.
    """
    heat = spine_heatmap([[0, 40, 200, 40]], 32, 32, origin=(10, 24))
    assert heat[16, 16] == pytest.approx(1.0, abs=0.02)
    assert heat[2, 16] < 0.01
    assert heat.max() <= 1.0


def test_uniform_pixel_weights_leave_lovasz_unchanged():
    """
    The weighted Jaccard with weight 1 everywhere is the plain one.
    """
    torch.manual_seed(0)
    logits, target = torch.randn(2, 1, 32, 32), (torch.rand(2, 1, 32, 32) > 0.7).float()
    assert lovasz_hinge_loss(logits, target, torch.ones_like(target)) == pytest.approx(lovasz_hinge_loss(logits, target).item())


def test_pixel_weights_raise_the_cost_of_a_missed_barb():
    """
    Missing heavily weighted pixels costs more than missing the same pixels at weight 1.
    """
    target = torch.zeros(1, 1, 32, 32)
    target[:, :, 10:20, 5:25] = 1
    logits = (2 * target - 1) * 5
    logits[:, :, 10:20, 20:25] = -5
    weights = torch.ones_like(target)
    plain = BCEDiceLoss(0.5, 0.5, 5.0, lovasz_weight=0.5, pixel_weights=True)(logits, torch.cat([target, weights], 1))
    weights[:, :, 10:20, 20:25] = 6
    heavy = BCEDiceLoss(0.5, 0.5, 5.0, lovasz_weight=0.5, pixel_weights=True)(logits, torch.cat([target, weights], 1))
    assert heavy > plain


def test_thin_head_loss_trains_the_head_and_passes_weights_to_the_base():
    """
    The head's channels get gradients; the base sees the filament target and the weights.
    """
    logits = torch.randn(1, 3, 32, 32, requires_grad=True)
    target = torch.zeros(1, 4, 32, 32)
    target[:, 0, 10:20, 5:25] = 1
    target[:, 1] = 1
    target[:, 2, 10:12, 5:25] = 1
    target[:, 3, 15, 5:25] = 1
    criterion = ThinHeadLoss(BCEDiceLoss(0.5, 0.5, 5.0, lovasz_weight=0.5, pixel_weights=True), 0.3)
    criterion(logits, target).backward()
    assert logits.grad[:, 1:].abs().sum() > 0 and torch.isfinite(logits.grad).all()


def test_barb_head_loss_ignores_samples_without_a_barb_label():
    """
    The head gets gradients from valid samples only; the filament channel always does.
    """
    logits = torch.randn(2, 2, 32, 32, requires_grad=True)
    target = torch.zeros(2, 3, 32, 32)
    target[:, 0, 10:20, 5:25] = 1
    target[0, 1, 8:10, 12:14] = 1
    target[0, 2] = 1
    criterion = BarbHeadLoss(BCEDiceLoss(0.5, 0.5, 5.0, lovasz_weight=0.5), 0.3)
    criterion(logits, target).backward()
    assert logits.grad[0, 1].abs().sum() > 0 and logits.grad[1, 1].abs().sum() == 0
    assert logits.grad[1, 0].abs().sum() > 0

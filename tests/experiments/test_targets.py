"""
Targets and objectives of the experiments: grown barbs, the barb and category heads, Tversky.
"""

import json

import numpy as np
import pytest
import torch
from PIL import Image
from test_models import small_model

from solfire.experiments.data import ExperimentCropDataset
from solfire.experiments.losses import CategoryHeadLoss, soft_tversky_loss
from solfire.training.losses import BCEDiceLoss, soft_dice_loss


def test_barb_targets_raise_the_target_outside_the_annotation(tmp_path):
    """
    Barb pixels grown from the image join the target at the barb level; annotated pixels keep theirs.
    """
    Image.fromarray(np.full((64, 64), 200, dtype=np.uint8)).save(tmp_path / "a.png")
    square = [8.0, 8.0, 56.0, 8.0, 56.0, 56.0, 8.0, 56.0]
    annotations = {
        "images": [{"id": 1, "file_name": "a.png", "width": 64, "height": 64}],
        "annotations": [{"image_id": 1, "segmentation": [square], "category_id": 1}],
    }
    (tmp_path / "annotations.json").write_text(json.dumps(annotations))
    barbs = np.zeros((64, 64), dtype=np.uint8)
    barbs[2:8, 30] = 255
    (tmp_path / "barbs").mkdir()
    Image.fromarray(barbs).save(tmp_path / "barbs" / "a.png")
    common = {"images_dir": tmp_path, "annotations_path": tmp_path / "annotations.json", "raster": "coco"}

    _, target = ExperimentCropDataset(crop=64, barb_targets=tmp_path / "barbs", barb_level=0.5, **common)[0]

    assert target.sum() == 48 * 48 + 6 * 0.5
    assert float(target[4, 30]) == 0.5 and float(target[20, 20]) == 1.0
    with pytest.raises(ValueError):
        ExperimentCropDataset(crop=128, input_scale=2.0, barb_targets=tmp_path / "barbs", **common)


def test_barb_head_learns_only_from_the_records_that_draw_barbs(tmp_path):
    """
    The barb channel is the thin part of the barb-drawing record's mask, averaged over those records
    alone; a sample without such a record is marked invalid instead of taught that there are no barbs.
    """
    Image.fromarray(np.full((64, 64), 200, dtype=np.uint8)).save(tmp_path / "a.png")
    Image.fromarray(np.full((64, 64), 200, dtype=np.uint8)).save(tmp_path / "b.png")
    body = [16.0, 16.0, 48.0, 16.0, 48.0, 48.0, 16.0, 48.0]
    barb = [30.0, 4.0, 32.0, 4.0, 32.0, 16.0, 30.0, 16.0]
    annotations = {
        "images": [
            {"id": 1, "file_name": "a.png", "width": 64, "height": 64},
            {"id": 2, "file_name": "a.png", "width": 64, "height": 64},
            {"id": 3, "file_name": "b.png", "width": 64, "height": 64},
        ],
        "annotations": [
            {"image_id": 1, "segmentation": [body], "category_id": 1},
            {"image_id": 1, "segmentation": [barb], "category_id": 1},
            {"image_id": 2, "segmentation": [body], "category_id": 1},
            {"image_id": 3, "segmentation": [body], "category_id": 1},
        ],
    }
    (tmp_path / "annotations.json").write_text(json.dumps(annotations))
    common = {"images_dir": tmp_path, "annotations_path": tmp_path / "annotations.json", "raster": "coco", "targets": "mean"}
    dataset = ExperimentCropDataset(crop=64, barb_records=[1], **common)

    _, drawn = dataset[0]
    _, undrawn = dataset[1]

    assert drawn.shape == (3, 64, 64)
    assert float(drawn[1, 10, 31]) == 1.0 and float(drawn[1, 32, 32]) == 0.0
    assert float(drawn[2].min()) == 1.0
    assert float(undrawn[1].max()) == 0.0 and float(undrawn[2].max()) == 0.0


def test_category_head_loss_trains_the_head():
    """
    The category cross entropy reaches the extra channels.
    """
    model = small_model(aux_channels=3).train()
    target = torch.zeros(1, 4, 64, 64)
    target[:, 0, 20:30, 10:50] = 1
    target[:, 2, 20:30, 10:50] = 1
    loss = CategoryHeadLoss(BCEDiceLoss(0.5, 0.5, 5.0), 0.2)(model(torch.rand(1, 1, 64, 64)), target)
    loss.backward()
    assert model.net.segmentation_head[0].weight.grad[1:].abs().sum() > 0


def test_tversky_at_one_half_matches_dice_and_beta_favours_recall():
    """
    With ``beta`` one half the Tversky loss is Dice up to the smoothing; above it a
    prediction missing 28 filament pixels costs more than one adding 36.
    """
    targets = torch.zeros(1, 1, 16, 16)
    targets[..., 4:12, 4:12] = 1
    logits = (targets * 2 - 1) * 3
    dice = soft_dice_loss(logits, targets, eps=1e-6)
    assert torch.isclose(soft_tversky_loss(logits, targets, beta=0.5, eps=1e-6), dice, atol=1e-4)
    shrunk = torch.full_like(targets, -6.0)
    shrunk[..., 5:11, 5:11] = 6.0
    grown = torch.full_like(targets, -6.0)
    grown[..., 3:13, 3:13] = 6.0
    assert soft_tversky_loss(shrunk, targets, beta=0.7) > soft_tversky_loss(grown, targets, beta=0.7)

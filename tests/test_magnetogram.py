"""
Magnetogram registration, the auxiliary magnetic targets and their loss.
"""

import numpy as np
import torch
from astropy.io import fits
from test_models import small_model

from src.data.magnetogram import (
    HA_CENTRE,
    HA_RADIUS,
    MagneticTargets,
    magnetic_targets,
    magnetogram_on_grid,
    pil_map,
)
from src.training.losses import BCEDiceLoss, MagneticHeadLoss

BZI_SIZE = 200


def bzi_header(centre=(96.0, 104.0), radius=80.0) -> fits.Header:
    """
    :return: Header of a fake bzi frame with its limb fit
    """
    header = fits.Header()
    header["FNDLMBXC"], header["FNDLMBYC"] = centre
    header["FNDLMBMI"] = header["FNDLMBMA"] = radius
    return header


def test_quarter_turn_maps_bzi_down_to_halpha_right():
    """
    The registered convention: a feature below disk centre in the bzi frame lands right of
    centre in the H-alpha frame.
    """
    header = bzi_header()
    field = np.zeros((BZI_SIZE, BZI_SIZE), np.float32)
    row = int(header["FNDLMBYC"] + 0.5 * header["FNDLMBMI"])
    column = int(header["FNDLMBXC"])
    field[row - 3 : row + 4, column - 3 : column + 4] = 500.0
    resampled, valid = magnetogram_on_grid(field, header, 256)
    y, x = np.argwhere(resampled > 250).mean(axis=0)
    expected_x = (HA_CENTRE + 0.5 * HA_RADIUS) * 256 / 2048
    assert abs(x - expected_x) <= 2 and abs(y - 128) <= 2
    assert not valid[0, 0] and valid[128, 128]


def bipole(size=BZI_SIZE) -> np.ndarray:
    """
    :return: A field positive left of the bzi disk centre and negative right of it
    """
    columns = np.arange(size)[None, :].repeat(size, 0)
    return np.where(columns < bzi_header()["FNDLMBXC"], 50.0, -50.0).astype(np.float32)


def test_pil_follows_the_polarity_boundary():
    """
    The inversion line of a bipole is the straight boundary between its halves, and
    its proximity target peaks there and vanishes off the disk.
    """
    resampled, valid = magnetogram_on_grid(bipole(), bzi_header(), 128)
    pil = pil_map(resampled, valid, sigma=1.0, gradient=1.0)
    rows = np.nonzero(pil.any(axis=1))[0]
    # the bzi columns turn into H-alpha rows, so the boundary is a horizontal line at the centre
    assert pil.sum() > 20 and np.all(np.abs(rows - 64) <= 2)
    proximity, _, inside = magnetic_targets(bipole(), bzi_header(), size=512, sigma_mm=12.0, gradient=0.5)
    assert proximity[254:258, 256].max() > 0.8 and proximity[150, 256] < 0.1
    assert inside[0, 0] == 0 and proximity[0, 0] == 0


def test_window_crops_and_upsamples_the_grid(tmp_path):
    """
    A window of the stored grid matches the region it covers; an unmatched observation
    gets all-zero targets, so its valid channel switches the loss off.
    """
    grid = np.zeros((3, 64, 64), np.uint8)
    grid[0, 16:32, 16:32] = 255
    grid[2] = 255
    np.save(tmp_path / "20110101000000Bh.npy", grid)
    store = MagneticTargets(tmp_path)
    window = store.window("20110101000000Bh.jpeg", x=512, y=512, crop=128, size=2048)
    # the crop spans grid pixels 15.5..19.5; past the first half pixel it is inside the filled square
    assert window.shape == (3, 128, 128)
    assert window[0, 16:, 16:].min() > 0.99 and window[2].all()
    assert not store.window("20990101000000Bh.jpeg", 0, 0, 128, 2048).any()


def test_magnetic_head_loss_trains_the_head_and_respects_valid():
    """
    The magnetic term reaches the two extra channels, and without a magnetogram the loss
    is just the filament objective.
    """
    torch.manual_seed(0)
    model = small_model(aux_channels=2).train()
    image = torch.rand(1, 1, 64, 64)
    target = torch.zeros(1, 4, 64, 64)
    target[:, 0, 20:30, 10:50] = 1
    target[:, 1, 18:32, :] = 0.8
    target[:, 3] = 1
    base = BCEDiceLoss(0.5, 0.5, 5.0)
    logits = model(image)
    MagneticHeadLoss(base, 0.5)(logits, target).backward()
    assert model.net.segmentation_head[0].weight.grad[1:3].abs().sum() > 0

    target[:, 3] = 0
    with torch.no_grad():
        logits = model(image)
        assert torch.allclose(MagneticHeadLoss(base, 0.5)(logits, target), base(logits[:, :1], target[:, :1]))

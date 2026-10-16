"""
Losses, the auxiliary-head model and tiled inference with test-time augmentation.
"""

import pytest
import torch

from src.inference.tiling import predict_probabilities, predict_tiled
from src.main import load_matching
from src.networks import ARCHITECTURES
from src.training.losses import BCEDiceLoss, CategoryHeadLoss, lovasz_hinge_loss


def stripe_target():
    """
    A ``(2, 1, 32, 32)`` target with one horizontal band.
    """
    target = torch.zeros(2, 1, 32, 32)
    target[:, :, 10:20, 5:25] = 1
    return target


def test_lovasz_is_zero_when_perfect_and_grows_with_errors():
    """
    Confidently right costs nothing; more wrong pixels cost more.
    """
    target = stripe_target()
    perfect = (2 * target - 1) * 5
    half = perfect.clone()
    half[:, :, 10:20, 15:25] = -5
    assert lovasz_hinge_loss(perfect, target) == pytest.approx(0.0)
    assert 0 < lovasz_hinge_loss(half, target) < lovasz_hinge_loss(-perfect, target)


def test_lovasz_has_gradients():
    """
    The term trains: finite, non-zero gradients.
    """
    logits = torch.randn(2, 1, 32, 32, requires_grad=True)
    lovasz_hinge_loss(logits, stripe_target()).backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0


def test_lovasz_weight_zero_changes_nothing():
    """
    The default loss is the one every earlier run was trained with.
    """
    logits, target = torch.randn(2, 1, 32, 32), stripe_target()
    assert BCEDiceLoss(0.5, 0.5, 5.0)(logits, target) == BCEDiceLoss(0.5, 0.5, 5.0, lovasz_weight=0.0)(logits, target)


def small_model(aux_channels=0):
    """
    A light smp model built offline, no pretrained download.
    """
    return ARCHITECTURES["pretrained"](encoder="resnet18", encoder_weights=None, aux_channels=aux_channels)


def test_auxiliary_head_only_in_training_and_starts_from_one_channel_checkpoint():
    """
    A one-channel checkpoint starts the auxiliary model; in eval mode it predicts exactly like the original.
    """
    torch.manual_seed(0)
    original = small_model().eval()
    extended = small_model(aux_channels=3)
    load_matching(extended, original.state_dict())
    image = torch.rand(1, 1, 64, 64)
    extended.eval()
    assert torch.allclose(extended(image), original(image), atol=1e-5)
    extended.train()
    assert extended(image).shape == (1, 4, 64, 64)


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


def symmetric_model():
    """
    A convolution with a symmetric kernel: equivariant to flips and quarter turns.
    """
    conv = torch.nn.Conv2d(1, 1, 3, padding=1, bias=False)
    with torch.no_grad():
        conv.weight[:] = torch.tensor([[1.0, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
    return conv.eval()


@pytest.mark.parametrize("tta", [True, "d4"])
def test_tta_is_exact_for_an_equivariant_model(tta):
    """
    Undoing each view's transform correctly means an equivariant model is unchanged by TTA.
    """
    model, images = symmetric_model(), torch.rand(2, 1, 48, 48)
    assert torch.allclose(predict_probabilities(model, images, tta), predict_probabilities(model, images, False), atol=1e-6)


def test_tiled_inference_matches_whole_image():
    """
    For a pointwise model, overlapping blended tiles reproduce the whole-image prediction.
    """
    model = torch.nn.Conv2d(1, 1, 1).eval()
    image = torch.rand(1, 100, 130)
    whole = predict_probabilities(model, image[None], tta=False)[0, 0]
    tiled = predict_tiled(model, image, torch.device("cpu"), tile=32, overlap=0.25, tta=False)
    assert torch.allclose(tiled, whole, atol=1e-5)


def test_flips_and_turned_views_average_to_d4():
    """
    The ``turned`` half and the ``flips`` half of the square's symmetries average to ``d4``.
    """
    torch.manual_seed(0)
    model = small_model().eval()
    image = torch.rand(1, 1, 64, 64)
    halves = (predict_probabilities(model, image, tta=True) + predict_probabilities(model, image, tta="turned")) / 2
    assert torch.allclose(halves, predict_probabilities(model, image, tta="d4"), atol=1e-6)


def test_scaled_tiling_returns_the_native_size():
    """
    A scaled pass comes back at the observation's own size, in ``[0, 1]``.
    """
    model = small_model().eval()
    image = torch.rand(1, 64, 64)
    for scale in (0.8, 1.25):
        tiled = predict_tiled(model, image, torch.device("cpu"), tile=32, overlap=0.25, tta=False, scale=scale)
        assert tiled.shape == (64, 64) and tiled.min() >= 0 and tiled.max() <= 1

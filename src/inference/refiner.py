"""
A second look at every predicted filament: the crop refiner.
"""

from dataclasses import dataclass

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch import Tensor, nn

from ..networks.pretrained import IMAGENET_MEAN, IMAGENET_STD
from .postprocess import connected_components

CROP_SIZE = 256
# window edge as a multiple of the longer side of the component's box
CONTEXT = 1.5
MIN_WINDOW = 128
MIN_PIXELS = 32
# share of a refined mask that must survive the higher scoring instances before it
MIN_SURVIVING = 0.5


@dataclass(frozen=True)
class Window:
    """
    A square window of an observation, possibly reaching past its edges.
    """

    top: int
    left: int
    side: int

    def clipped(self, height, width) -> tuple[slice, slice, slice, slice]:
        """
        :param height: Height of the observation
        :param width: Width of the observation
        :return: Tuple of (rows, columns) in the observation and (rows, columns) in the window
        """
        y0, x0 = max(self.top, 0), max(self.left, 0)
        y1, x1 = min(self.top + self.side, height), min(self.left + self.side, width)
        return (
            slice(y0, y1),
            slice(x0, x1),
            slice(y0 - self.top, y1 - self.top),
            slice(x0 - self.left, x1 - self.left),
        )


def component_window(box, context=CONTEXT, min_side=MIN_WINDOW) -> Window:
    """
    The window a component is refined on.
    :param box: ``(rows, columns)`` slices of the component, as from ``ndimage.find_objects``
    :param context: Window edge as a multiple of the longer side of the box
    :param min_side: Smallest window edge
    :return: Window centred on the box
    """
    rows, columns = box
    height, width = rows.stop - rows.start, columns.stop - columns.start
    side = max(min_side, round(context * max(height, width)))
    centre_y, centre_x = (rows.start + rows.stop) / 2, (columns.start + columns.stop) / 2
    return Window(round(centre_y - side / 2), round(centre_x - side / 2), side)


def window_of(array, window) -> np.ndarray:
    """
    Cut a window out of an observation, with zeros past its edges.
    :param array: ``(H, W)`` array
    :param window: Window to cut
    :return: ``(side, side)`` array of the same dtype
    """
    out = np.zeros((window.side, window.side), dtype=array.dtype)
    rows, columns, window_rows, window_columns = window.clipped(*array.shape)
    out[window_rows, window_columns] = array[rows, columns]
    return out


def resample(windows, size) -> Tensor:
    """
    Bring windows of one edge length to another, filtering on the way down.
    :param windows: ``(B, C, side, side)`` float tensor
    :param size: Edge length to resample to
    :return: ``(B, C, size, size)`` tensor
    """
    if windows.shape[-1] == size:
        return windows
    return F.interpolate(windows, size=(size, size), mode="bilinear", align_corners=False, antialias=True)


def refiner_inputs(image, probabilities, seed, window, size=CROP_SIZE) -> Tensor:
    """
    The three channels the refiner reads for one component.
    :param image: ``(H, W)`` observation in ``[0, 1]``
    :param probabilities: ``(H, W)`` ensemble probabilities
    :param seed: ``(H, W)`` boolean mask of the component
    :param window: Window of the component
    :param size: Edge length the refiner works at
    :return: ``(3, size, size)`` float tensor in ``[0, 1]``
    """
    stacked = np.stack(
        [
            window_of(image.astype(np.float32, copy=False), window),
            window_of(probabilities.astype(np.float32, copy=False), window),
            window_of(seed, window).astype(np.float32),
        ]
    )
    # antialiasing can overshoot the range by a rounding error
    return resample(torch.from_numpy(stacked)[None], size)[0].clamp_(0, 1)


class Refiner(nn.Module):
    """
    U-Net that redraws one filament, with a head on the bottleneck that scores it.
    """

    def __init__(self, encoder="resnet34", encoder_weights="imagenet", in_channels=3):
        """
        :param encoder: Encoder name as smp spells it
        :param encoder_weights: Pretrained weights, or ``None`` when a state dict follows
        :param in_channels: Input channels: observation, ensemble probability, seed mask
        """
        super().__init__()
        self.net = smp.Unet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1,
            aux_params={"classes": 1, "dropout": 0.2},
        )

    @property
    def encoder(self) -> nn.Module:
        """
        :return: The pretrained part of the network, for a separate learning rate
        """
        encoder: nn.Module = self.net.encoder
        return encoder

    def forward(self, x) -> tuple[Tensor, Tensor]:
        """
        :param x: ``(B, 3, S, S)`` inputs from :func:`refiner_inputs`, ``S`` divisible by 32
        :return: Tuple of (``(B, 1, S, S)`` mask logits, ``(B,)`` logits of the earned IoU)
        """
        masks, scores = self.net((x - IMAGENET_MEAN) / IMAGENET_STD)
        return masks, scores[:, 0]


@dataclass
class RefinedInstance:
    """
    What the refiner made of one component, at the native resolution of its window.
    """

    window: Window
    # uint8, probability times 255
    probabilities: np.ndarray
    score: float


@torch.no_grad()
def refine_components(model, image, probabilities, labels, n_labels, device, size=CROP_SIZE, batch_size=32, tta=True):
    """
    Run the refiner over every component of an observation.
    :param model: :class:`Refiner` in eval mode
    :param image: ``(H, W)`` observation in ``[0, 1]``
    :param probabilities: ``(H, W)`` ensemble probabilities
    :param labels: ``(H, W)`` candidate components, 0 as background
    :param n_labels: Number of labels in ``labels``
    :param device: Device to run on
    :param size: Edge length the refiner works at
    :param batch_size: Components per forward pass
    :param tta: Average over horizontal and vertical flips
    :return: List of :class:`RefinedInstance`, one per component of at least
        :data:`MIN_PIXELS` pixels
    """
    areas = np.bincount(labels.ravel(), minlength=n_labels + 1)
    boxes = ndimage.find_objects(labels)
    jobs = [(index + 1, component_window(box)) for index, box in enumerate(boxes) if box and areas[index + 1] >= MIN_PIXELS]
    refined = []
    for start in range(0, len(jobs), batch_size):
        chunk = jobs[start : start + batch_size]
        inputs = torch.stack([refiner_inputs(image, probabilities, labels == k, window, size) for k, window in chunk])
        masks, scores = predict_refiner(model, inputs.to(device), tta=tta)
        for (_, window), mask, score in zip(chunk, masks, scores):
            native = resample(mask[None].float(), window.side)[0, 0]
            refined.append(RefinedInstance(window, (native * 255).round().byte().cpu().numpy(), float(score)))
    return refined


def predict_refiner(model, inputs, tta=True) -> tuple[Tensor, Tensor]:
    """
    Refiner probabilities, averaged over flips.
    :param model: :class:`Refiner` in eval mode
    :param inputs: ``(B, 3, S, S)`` inputs on the model's device
    :param tta: Average over horizontal and vertical flips
    :return: Tuple of (``(B, 1, S, S)`` mask probabilities, ``(B,)`` predicted earned IoU)
    """
    flips = [(), (-1,), (-2,)] if tta else [()]
    masks, scores = 0, 0
    for dims in flips:
        logits, score = model(torch.flip(inputs, dims) if dims else inputs)
        masks = masks + torch.sigmoid(torch.flip(logits, dims) if dims else logits)
        scores = scores + torch.sigmoid(score)
    return masks / len(flips), scores / len(flips)


def assemble(refined, height, width, threshold=0.5, min_score=0.0) -> tuple[np.ndarray, int, np.ndarray]:
    """
    Paint refined instances into one label array, the most confident first.

    Two fragments of one filament are both redrawn as the whole filament, so a mask
    that the higher scoring instances already cover is dropped rather than submitted
    twice.
    :param refined: Output of :func:`refine_components`
    :param height: Height of the observation
    :param width: Width of the observation
    :param threshold: Probability the refined masks are cut at
    :param min_score: Predicted earned IoU an instance needs to be kept
    :return: Tuple of (``(H, W)`` labels, number of labels, ``(n,)`` scores of the labels)
    """
    labels = np.zeros((height, width), dtype=np.int32)
    scores = []
    for instance in sorted(refined, key=lambda item: -item.score):
        if instance.score < min_score:
            break
        rows, columns, window_rows, window_columns = instance.window.clipped(height, width)
        mask = instance.probabilities[window_rows, window_columns] >= round(threshold * 255)
        mask = largest_component(mask)
        area = int(mask.sum())
        if area < MIN_PIXELS:
            continue
        free = mask & (labels[rows, columns] == 0)
        if free.sum() < MIN_SURVIVING * area:
            continue
        scores.append(instance.score)
        labels[rows, columns][free] = len(scores)
    return labels, len(scores), np.asarray(scores, dtype=np.float32)


def largest_component(mask) -> np.ndarray:
    """
    :param mask: Boolean mask
    :return: Its largest connected component
    """
    labels, n_labels = connected_components(mask)
    if n_labels <= 1:
        return mask
    areas = np.bincount(labels.ravel())[1:]
    return labels == int(areas.argmax()) + 1

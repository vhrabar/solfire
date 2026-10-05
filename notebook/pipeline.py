# %% [markdown]
# # Solar filament segmentation: full pipeline
#
# Segments solar filaments in GONG H-alpha observations (MAGFiLO 1.0) and writes the competition's
# `submission.csv`, one compressed-RLE mask per filament.
#
# **Model.** An ensemble of 15 networks: 3 encoder-decoder families x 5 folds, all with ImageNet encoders.
#
# | Family | Decoder | Encoder | Pooled out-of-fold PQ |
# |---|---|---|---|
# | `unetpp-r34` | U-Net++ | ResNet-34 | 0.4503 |
# | `unet-mitb2` | U-Net | MiT-B2 (SegFormer) | 0.4514 |
# | `unet-hrnet32` | U-Net | HRNet-W32 | 0.4521 |
# | **ensemble** | | | **0.4594** |
#
# The ensemble's post-processing is tuned on the same predictions it is scored on. Tuned on four folds and scored
# on the fifth instead, it reaches 0.4564; that is the estimate to expect on unseen observations. Public LB: 0.40.
#
# **Recipe**, per family and fold:
# 1. 40 epochs from ImageNet on random 512 px windows at native resolution (filaments are ~11 px wide, so
#    nothing is downsampled). Most windows are centred on an annotated filament. Targets are averaged over the
#    annotators of an observation. Loss: BCE (positive weight 5) + soft Dice.
# 2. An 8-epoch fine-tune from the best epoch of stage 1 at a lower learning rate, with a Lovász hinge term added.
#    Lovász is a surrogate for IoU, the quantity Panoptic Quality is built on.
#
# **Inference.** Tiled 512 px windows (overlap 0.25, cosine-blended) with 4-flip TTA. Probabilities are averaged
# over the 15 models, then thresholded, eroded by one pixel, and split into 8-connected instances. Instances with a
# mean probability below the cut are dropped. Threshold, offset, minimum area and score cut are chosen by grid
# search, scored with the organizers' Panoptic Quality on the pooled out-of-fold predictions of all 707 training
# observations.
#
# **Running it.**
# - `TRAIN = False` (default): downloads the published checkpoints and their calibration, then predicts the test set.
# - `TRAIN = True`: trains all 30 runs, calibrates, then predicts. That is about 20 GPU-hours on an RTX 5070 Ti,
#   more than one Kaggle session. Finished runs are skipped, so the notebook can be rerun until it completes.
#
# Source: https://github.com/vhrabar/SolarFilamentSegmentationChallenge
# (this notebook is generated from `notebook/pipeline.py`).

# %%
import csv
import json
import math
import multiprocessing
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

TRAIN = False  # True: train every model from the competition data; False: fetch the published checkpoints
SEED = 1
FOLDS = 5  # grouped by observation, the assignment is fixed (seed 0) whatever SEED is

MODEL_HANDLE = "vhrabar/filament-segmentation-2026/pyTorch/cv15-hrnet/2"  # Kaggle Models: 15 checkpoints + calibration
CALIBRATION_NAME = "cv15-hrnet-fine-oof.calibration.json"
WEIGHTS_DIR = None  # a local folder holding the checkpoints and the calibration; None downloads MODEL_HANDLE

FAMILIES = {
    "unetpp-r34": {"decoder": "unetplusplus", "encoder": "resnet34"},
    "unet-mitb2": {"decoder": "unet", "encoder": "mit_b2"},
    "unet-hrnet32": {"decoder": "unet", "encoder": "tu-hrnet_w32"},
}
TRAINING = {  # shared by both stages
    "crop": 512,  # window edge in native pixels
    "crops_per_image": 7,  # windows per observation and epoch
    "foreground_prob": 0.7,  # share of windows placed on an annotated filament
    "batch_size": 8,
    "weight_decay": 1e-2,
    "encoder_lr_scale": 0.25,  # the pretrained encoder learns at a quarter of the decoder's rate
    "pos_weight": 5.0,
    "dice_weight": 0.5,  # BCE gets the rest
    "grad_clip": 1.0,
    "val_limit": 64,  # held-out records scored after every epoch to pick the best one
}
STAGES = {
    "stage1": {"epochs": 40, "lr": 4e-4, "warmup_epochs": 3, "lovasz_weight": 0.0},
    "stage2": {"epochs": 8, "lr": 1e-4, "warmup_epochs": 0, "lovasz_weight": 0.5},
}
TILE, OVERLAP, TILE_BATCH = 512, 0.25, 8
CALIBRATION_GRID = {
    "threshold": (0.25, 0.3, 0.325, 0.35, 0.375, 0.4, 0.45, 0.5, 0.65),
    "offset": (-2, -1, 0, 1),  # signed one-pixel dilation/erosion steps
    "min_area": (0, 256),
    "min_score": (0.0, 0.6, 0.625, 0.65, 0.675, 0.7, 0.725, 0.75, 0.8, 0.85, 0.88, 0.91, 0.94),
}

SUBMISSION_SIZE = 2048
WORKERS = min(8, os.cpu_count() or 1)
WORK = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd() / "build" / "notebook"
WORK.mkdir(parents=True, exist_ok=True)

# %%
try:
    import segmentation_models_pytorch as smp
except ImportError:  # not in Kaggle's image
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "segmentation-models-pytorch==0.5.0", "timm==1.0.30"], check=True
    )
    import segmentation_models_pytorch as smp

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools import mask as coco_mask
from scipy import ndimage
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seeds(seed: int) -> None:
    """Seed every random source; called at the start of each training run."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seeds(SEED)
print(f"torch {torch.__version__}, smp {smp.__version__}, device {DEVICE}, workers {WORKERS}")

# %% [markdown]
# ## Data
#
# 707 training observations (2048 x 2048 grayscale JPEG) carry 1,154 annotator records: an observation can be
# annotated by up to three people. Every record is a COCO image entry whose polygons are the filaments.
# Folds group all records of one observation together.

# %%
ANNOTATIONS_NAME = "MAGFiLO_1.0_Annotations_kaggle2026_train.json"


def find_dataset() -> tuple[Path, Path, Path]:
    """
    Locate the competition data: a Kaggle input mount, $FILAMENT_DATA_ROOT, or ./data.
    :return: Tuple of (annotations file, train image directory, test image directory)
    """
    roots = [Path(os.environ["FILAMENT_DATA_ROOT"])] if "FILAMENT_DATA_ROOT" in os.environ else []
    roots += [Path("/kaggle/input"), Path.cwd() / "data"]
    for root in roots:
        annotations = next(root.rglob(ANNOTATIONS_NAME), None) if root.is_dir() else None
        if annotations is None:
            continue
        base = annotations.parent.parent if annotations.parent.name == "train" else annotations.parent
        train_images = next(path for path in base.rglob("train_images") if path.is_dir())
        test_images = next(path for path in base.rglob("test_images") if path.is_dir())
        return annotations, train_images, test_images
    raise FileNotFoundError(f"{ANNOTATIONS_NAME} not found under {', '.join(map(str, roots))}")


ANNOTATIONS, TRAIN_IMAGES, TEST_IMAGES = find_dataset()
raw = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))
RECORDS = {image["id"]: image for image in raw["images"]}
BY_RECORD: dict = {image_id: [] for image_id in RECORDS}
for annotation in raw["annotations"]:
    BY_RECORD[annotation["image_id"]].append(annotation)
TEST_FILES = sorted(path for path in TEST_IMAGES.iterdir() if path.suffix.lower() in {".jpeg", ".jpg", ".png"})

print(
    f"train: {len({r['file_name'] for r in RECORDS.values()})} observations, {len(RECORDS)} records, "
    f"{len(raw['annotations'])} filaments  ({TRAIN_IMAGES})"
)
print(f"test:  {len(TEST_FILES)} observations  ({TEST_IMAGES})")


def kfold_split(fold: int) -> tuple[list, list]:
    """
    Record ids for training and for validation in one fold; all records of an observation share a fold.
    :param fold: Fold held out, in [0, FOLDS)
    :return: Tuple of (train record ids, held-out record ids)
    """
    groups: dict[str, list] = {}
    for image in raw["images"]:
        groups.setdefault(image["file_name"], []).append(image["id"])
    names = sorted(groups)
    random.Random(0).shuffle(names)
    held_out = [i for index, name in enumerate(names) if index % FOLDS == fold for i in groups[name]]
    train = [i for index, name in enumerate(names) if index % FOLDS != fold for i in groups[name]]
    return train, held_out


def polygon_mask(polygon, height: int, width: int, origin=(0, 0)) -> np.ndarray:
    """
    Fill a flat [x0, y0, x1, y1, ...] polygon the way pycocotools fills the scored masks.

    The polygon is filled on its own bounding box and pasted, so a window can be rasterised without
    building the full 2048 x 2048 mask, and clipping matches cropping the full mask.
    :param polygon: Polygon in observation coordinates
    :param height: Height of the window
    :param width: Width of the window
    :param origin: (x, y) of the window inside the observation
    :return: (height, width) boolean mask
    """
    points = np.asarray(polygon, dtype=np.float64).reshape(-1, 2) - np.asarray(origin, dtype=np.float64)
    mask = np.zeros((height, width), dtype=bool)
    if len(points) < 3:
        return mask
    x0, y0 = np.floor(points.min(axis=0)).astype(int) - 1
    x1, y1 = np.ceil(points.max(axis=0)).astype(int) + 2
    if x1 <= 0 or y1 <= 0 or x0 >= width or y0 >= height:
        return mask
    local = (points - [x0, y0]).ravel().tolist()
    filled = coco_mask.decode(coco_mask.frPyObjects([local], int(y1 - y0), int(x1 - x0)))[:, :, 0].astype(bool)
    cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, width), min(y1, height)
    mask[cy0:cy1, cx0:cx1] = filled[cy0 - y0 : cy1 - y0, cx0 - x0 : cx1 - x0]
    return mask


def load_image(path, window=None) -> Tensor:
    """
    A grayscale observation, or one square window of it, as a (1, H, W) float tensor in [0, 1].
    :param path: Observation file
    :param window: (x, y, edge) of the window, or None for the whole observation
    :return: (1, H, W) tensor
    """
    with Image.open(path) as handle:
        image = handle.convert("L")
        if window is not None:
            x, y, edge = window
            image = image.crop((x, y, x + edge, y + edge))
        array = np.asarray(image, dtype=np.uint8)
    return torch.from_numpy(array.copy()).unsqueeze(0).float().div_(255.0)


def augment(image: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    """
    Random flips and quarter turns (the disk has no preferred orientation), gamma and brightness/contrast jitter.
    :param image: (1, H, W) window
    :param target: (H, W) target
    :return: Tuple of the augmented (image, target)
    """
    target = target.unsqueeze(0)
    if random.random() < 0.5:
        image, target = torch.flip(image, [-1]), torch.flip(target, [-1])
    if random.random() < 0.5:
        image, target = torch.flip(image, [-2]), torch.flip(target, [-2])
    turns = random.randint(0, 3)
    if turns:
        image, target = torch.rot90(image, turns, [-2, -1]), torch.rot90(target, turns, [-2, -1])
    image = image.clamp_min(0.0).pow(math.exp(random.gauss(0.0, 0.2)))
    scale, shift = 1.0 + random.uniform(-0.15, 0.15), random.uniform(-0.15, 0.15)
    return (image * scale + shift).clamp_(0.0, 1.0), target.squeeze(0)


class FilamentWindows(Dataset):
    """
    Random native-resolution windows of the training observations, with soft targets.

    One sample per observation: the masks of its annotator records are averaged, so a filament only one of three
    annotators drew gets a target of 1/3. Filaments cover about 0.2% of the disk, so most windows are placed on a
    random vertex of a random filament (anywhere inside the window), the rest uniformly.
    """

    def __init__(self, record_ids, crop: int, crops_per_image: int, foreground_prob: float):
        """
        :param record_ids: Training record ids
        :param crop: Window edge in pixels
        :param crops_per_image: Windows drawn per observation and epoch
        :param foreground_prob: Probability of placing a window on a filament
        """
        groups: dict[str, list] = {}
        for image_id in record_ids:
            groups.setdefault(RECORDS[image_id]["file_name"], []).append(image_id)
        self.samples = [groups[name] for name in sorted(groups)]
        self.crop = crop
        self.repeat = crops_per_image
        self.foreground_prob = foreground_prob

    def __len__(self) -> int:
        return len(self.samples) * self.repeat

    def corner(self, annotations, size: int) -> tuple[int, int]:
        """
        :param annotations: Every filament of the observation, over all its records
        :param size: Edge length of the observation
        :return: (x, y) top-left corner of the window, inside the observation
        """
        limit = max(0, size - self.crop)
        if annotations and random.random() < self.foreground_prob:
            points = np.asarray(random.choice(annotations)["segmentation"][0], dtype=np.float64).reshape(-1, 2)
            x, y = points[random.randrange(len(points))]
            x, y = int(x) - random.randrange(self.crop), int(y) - random.randrange(self.crop)
        else:
            x, y = random.randint(0, limit), random.randint(0, limit)
        return min(max(x, 0), limit), min(max(y, 0), limit)

    def __getitem__(self, index):
        group = self.samples[index % len(self.samples)]
        record = RECORDS[group[0]]
        x, y = self.corner([ann for image_id in group for ann in BY_RECORD[image_id]], record["width"])
        image = load_image(TRAIN_IMAGES / record["file_name"], (x, y, self.crop))

        target = np.zeros((self.crop, self.crop), dtype=np.float32)
        for image_id in group:
            mask = np.zeros((self.crop, self.crop), dtype=bool)
            for ann in BY_RECORD[image_id]:
                points = np.asarray(ann["segmentation"][0], dtype=np.float64).reshape(-1, 2)
                (x0, y0), (x1, y1) = points.min(axis=0), points.max(axis=0)
                if x1 < x or x0 > x + self.crop or y1 < y or y0 > y + self.crop:
                    continue
                mask |= polygon_mask(ann["segmentation"][0], self.crop, self.crop, origin=(x, y))
            target += mask
        target /= len(group)
        return augment(image, torch.from_numpy(target))


# %% [markdown]
# ## Model
#
# A `segmentation_models_pytorch` decoder on an ImageNet encoder. smp collapses the first convolution to one input
# channel by summing its RGB kernels, so the grayscale observation only needs ImageNet's mean and std.

# %%
IMAGENET_MEAN, IMAGENET_STD = 0.449, 0.226  # averaged over the colour channels
DECODERS = {"unet": smp.Unet, "unetplusplus": smp.UnetPlusPlus}


class Pretrained(nn.Module):
    """
    smp decoder + ImageNet encoder for one-channel input and one-channel logits, at any input size.
    """

    def __init__(self, decoder: str, encoder: str, encoder_weights=None):
        """
        :param decoder: Key of DECODERS
        :param encoder: smp encoder name, tu- prefixed for timm encoders
        :param encoder_weights: "imagenet" to start training, None when a checkpoint supplies the weights
        """
        super().__init__()
        self.net = DECODERS[decoder](encoder_name=encoder, encoder_weights=encoder_weights, in_channels=1, classes=1)
        self.divisor = 32  # five halvings in the encoder

    @property
    def encoder(self) -> nn.Module:
        return self.net.encoder

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: (B, 1, H, W) observations in [0, 1]
        :return: (B, 1, H, W) logits
        """
        height, width = x.shape[-2:]
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        x = F.pad(x, (0, -width % self.divisor, 0, -height % self.divisor), mode="replicate")
        return self.net(x)[..., :height, :width]


def load_model(path) -> tuple[nn.Module, dict]:
    """
    Rebuild a trained model from its checkpoint.
    :param path: Checkpoint file
    :return: Tuple of (model in eval mode on DEVICE, checkpoint without the weights)
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["model_config"]
    model = Pretrained(config["decoder"], config["encoder"])
    model.load_state_dict(checkpoint.pop("model_state"))
    return model.to(DEVICE).eval(), checkpoint


# %% [markdown]
# ## Inference
#
# The model sees 512 px windows in training, so a whole observation is predicted in overlapping windows, each
# averaged over the four flips. Overlaps are blended with a raised-cosine weight, so window borders, where the model
# lacks context, count least.


# %%
@torch.no_grad()
def predict_flips(model, images: Tensor) -> Tensor:
    """
    :param model: Model in eval mode
    :param images: (B, 1, H, W) windows on DEVICE
    :return: (B, 1, H, W) probabilities averaged over the identity and the three flips
    """
    total = torch.zeros_like(images, dtype=torch.float32)
    for flip_x in (False, True):
        for flip_y in (False, True):
            dims = [d for d, on in ((-1, flip_x), (-2, flip_y)) if on]
            view = torch.flip(images, dims) if dims else images
            probabilities = torch.sigmoid(model(view).float())
            total += torch.flip(probabilities, dims) if dims else probabilities
    return total / 4


def tile_offsets(size: int, tile: int, stride: int) -> list[int]:
    """
    :return: Window starts along one axis; the last window is pulled back to the edge rather than padded
    """
    if size <= tile:
        return [0]
    offsets = list(range(0, size - tile + 1, stride))
    if offsets[-1] != size - tile:
        offsets.append(size - tile)
    return offsets


@torch.no_grad()
def predict_tiled(model, image: Tensor, tile=TILE, overlap=OVERLAP, tta=True, batch_size=TILE_BATCH) -> Tensor:
    """
    Probabilities for one whole observation, assembled from overlapping windows.
    :param model: Model in eval mode
    :param image: (1, H, W) observation at native resolution
    :param tile: Window edge
    :param overlap: Fraction of a window shared with its neighbour
    :param tta: Average every window over the four flips
    :param batch_size: Windows per forward pass
    :return: (H, W) probabilities on the cpu
    """
    image = image.unsqueeze(0)
    height, width = image.shape[-2:]
    stride = max(1, round(tile * (1.0 - overlap)))
    corners = [(y, x) for y in tile_offsets(height, tile, stride) for x in tile_offsets(width, tile, stride)]

    ramp = torch.hann_window(tile + 2, periodic=False, device=DEVICE)[1:-1].clamp_min(0.05)
    weights = (ramp[:, None] * ramp[None, :])[None, None]
    total = torch.zeros((1, 1, height, width), dtype=torch.float32, device=DEVICE)
    counts = torch.zeros_like(total)
    for start in range(0, len(corners), batch_size):
        chunk = corners[start : start + batch_size]
        batch = torch.cat([image[..., y : y + tile, x : x + tile] for y, x in chunk]).to(DEVICE, non_blocking=True)
        probabilities = predict_flips(model, batch) if tta else torch.sigmoid(model(batch).float())
        for (y, x), probability in zip(chunk, probabilities):
            total[..., y : y + tile, x : x + tile] += probability[None] * weights
            counts[..., y : y + tile, x : x + tile] += weights
    return (total / counts)[0, 0].cpu()


# %% [markdown]
# ## Post-processing and scoring
#
# A probability map becomes instances by thresholding, a signed morphological offset (filaments are ~11 px thick,
# so one pixel per side matters), and 8-connected components. Components that are too small, or whose mean
# probability is too low, are dropped: under Panoptic Quality every false positive costs half a point, the same as
# a miss.
#
# Panoptic Quality as the organizers compute it: a prediction matches an annotated filament at IoU > 0.5, and
# `PQ = sum(matched IoU) / (TP + FP/2 + FN/2)`. Every annotator record of an observation is scored against the same
# prediction.

# %%
EIGHT_CONNECTED = np.ones((3, 3), dtype=bool)


def connected_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    labels, n_labels = ndimage.label(mask, structure=EIGHT_CONNECTED)
    return labels.astype(np.int32, copy=False), n_labels


def apply_offset(mask: np.ndarray, offset: int) -> np.ndarray:
    """
    Dilate (offset > 0) or erode (offset < 0) by |offset| steps of the four-neighbour diamond.
    """
    grow = offset > 0
    mask = mask if grow else ~mask
    for _ in range(abs(offset)):
        grown = mask.copy()
        grown[1:] |= mask[:-1]
        grown[:-1] |= mask[1:]
        grown[:, 1:] |= mask[:, :-1]
        grown[:, :-1] |= mask[:, 1:]
        mask = grown
    return mask if grow else ~mask


def component_stats(labels: np.ndarray, n_labels: int, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    :return: Tuple of (area, mean probability) per component, component k at index k - 1
    """
    flat = labels.ravel()
    areas = np.bincount(flat, minlength=n_labels + 1)[1:]
    sums = np.bincount(flat, weights=probabilities.ravel(), minlength=n_labels + 1)[1:]
    return areas, sums / np.maximum(areas, 1)


def label_instances(probabilities: np.ndarray, threshold, offset=0, min_area=0, min_score=0.0) -> tuple[np.ndarray, int]:
    """
    Turn a probability map into numbered instances, the way the submission is written.
    :return: Tuple of ((H, W) labels with 0 as background, number of instances)
    """
    labels, n_labels = connected_components(apply_offset(probabilities >= threshold, offset))
    areas, scores = component_stats(labels, n_labels, probabilities)
    keep = (areas > min_area) & (scores >= min_score)
    remap = np.zeros(n_labels + 1, dtype=np.int32)
    remap[1:][keep] = np.arange(1, int(keep.sum()) + 1, dtype=np.int32)
    return remap[labels], int(keep.sum())


def annotation_pixels(annotations) -> list[np.ndarray]:
    """
    :param annotations: The filaments of one annotator record
    :return: Flat pixel indices of each filament, filled by pycocotools like the scored masks
    """
    pixels = []
    for annotation in annotations:
        rle = coco_mask.merge(coco_mask.frPyObjects(annotation["segmentation"], SUBMISSION_SIZE, SUBMISSION_SIZE))
        pixels.append(np.flatnonzero(coco_mask.decode(rle)))
    return pixels


def instance_iou(gt_pixels, labels: np.ndarray, n_labels: int) -> np.ndarray:
    """
    :return: (n_gt, n_labels) IoU of every annotated filament with every predicted component
    """
    flat = labels.ravel()
    pred_areas = np.bincount(flat, minlength=n_labels + 1)[1:]
    iou = np.zeros((len(gt_pixels), n_labels))
    for row, pixels in enumerate(gt_pixels):
        intersection = np.bincount(flat[pixels], minlength=n_labels + 1)[1:]
        iou[row] = intersection / np.maximum(len(pixels) + pred_areas - intersection, 1)
    return iou


def pq_terms(iou: np.ndarray) -> np.ndarray:
    """
    :param iou: (n_gt, n_pred) IoU matrix of one record
    :return: [sum of matched IoU, TP, FP, FN]
    """
    hits = iou > 0.5
    return np.array([iou[hits].sum(), hits.sum(), (~hits.any(axis=0)).sum(), (~hits.any(axis=1)).sum()], dtype=np.float64)


def panoptic_quality(terms) -> float:
    matched, tp, fp, fn = terms
    denominator = tp + 0.5 * fp + 0.5 * fn
    return float(matched / denominator) if denominator else 0.0


# %% [markdown]
# ## Training
#
# AdamW without decay on biases and norms, linear warm-up then cosine decay, float16 autocast. After every epoch,
# 64 held-out records are predicted whole (tiled, no TTA) and scored over a threshold grid. The best epoch is
# kept as `best.pt`.


# %%
def soft_dice_loss(logits: Tensor, targets: Tensor, eps=1.0) -> Tensor:
    probabilities = torch.sigmoid(logits).flatten(1)
    targets = targets.flatten(1)
    dice = (2.0 * (probabilities * targets).sum(1) + eps) / (probabilities.sum(1) + targets.sum(1) + eps)
    return 1.0 - dice.mean()


def lovasz_hinge_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """
    Lovász hinge (Berman et al., 2018), a convex surrogate of 1 - IoU, per sample; soft targets are cut at a half.
    """
    losses = []
    with torch.autocast(device_type=logits.device.type, enabled=False):
        for logit, target in zip(logits.float().flatten(1), targets.float().flatten(1)):
            labels = (target >= 0.5).float()
            errors, order = torch.sort(1.0 - logit * (2.0 * labels - 1.0), descending=True)
            sorted_labels = labels[order]
            positives = sorted_labels.sum()
            jaccard = 1.0 - (positives - sorted_labels.cumsum(0)) / (positives + (1.0 - sorted_labels).cumsum(0))
            jaccard[1:] = jaccard[1:] - jaccard[:-1]
            losses.append(torch.dot(torch.relu(errors), jaccard))
    return torch.stack(losses).mean()


def segmentation_loss(logits: Tensor, targets: Tensor, lovasz_weight: float) -> Tensor:
    pos_weight = torch.tensor(TRAINING["pos_weight"], device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
    loss = (1.0 - TRAINING["dice_weight"]) * bce + TRAINING["dice_weight"] * soft_dice_loss(logits, targets)
    if lovasz_weight:
        loss = loss + lovasz_weight * lovasz_hinge_loss(logits, targets)
    return loss


def build_optimizer(model: Pretrained, lr: float) -> torch.optim.Optimizer:
    encoder_ids = {id(param) for param in model.encoder.parameters()}
    norms = nn.modules.batchnorm._BatchNorm | nn.LayerNorm | nn.GroupNorm
    groups: dict[tuple[bool, bool], list] = {}
    for module in model.modules():
        for name, param in module.named_parameters(recurse=False):
            if param.requires_grad:
                decays = not (name == "bias" or isinstance(module, norms))
                groups.setdefault((decays, id(param) in encoder_ids), []).append(param)
    return torch.optim.AdamW(
        [
            {
                "params": params,
                "weight_decay": TRAINING["weight_decay"] if decays else 0.0,
                "lr": lr * TRAINING["encoder_lr_scale"] if in_encoder else lr,
            }
            for (decays, in_encoder), params in groups.items()
        ],
        lr=lr,
    )


def build_scheduler(optimizer, epochs: int, steps_per_epoch: int, warmup_epochs: int):
    """Linear warm-up from zero, then cosine decay to 1% of the peak, stepped every batch."""
    total = max(2, epochs * steps_per_epoch)
    warmup = min(total - 1, max(1, warmup_epochs * steps_per_epoch))
    peak = max(group["lr"] for group in optimizer.param_groups)
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0 / warmup, end_factor=1.0, total_iters=warmup - 1),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total - warmup, eta_min=0.01 * peak),
        ],
        milestones=[warmup],
    )


def train_epoch(model, loader, optimizer, scheduler, scaler, lovasz_weight: float, desc: str) -> float:
    model.train()
    total, batches = 0.0, 0
    for images, targets in tqdm(loader, desc=desc, leave=False, mininterval=2.0):
        images = images.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True).unsqueeze(1)
        norm_stats = {k: v.detach().clone() for k, v in model.named_buffers() if k.endswith(("running_mean", "running_var"))}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16 if DEVICE.type == "cuda" else torch.bfloat16):
            loss = segmentation_loss(model(images), targets, lovasz_weight)
        if not torch.isfinite(loss):  # skip the batch and undo its effect on the batch-norm statistics
            buffers = dict(model.named_buffers())
            for name, value in norm_stats.items():
                buffers[name].copy_(value)
            continue
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total, batches = total + loss.item(), batches + 1
    return total / max(1, batches)


def threshold_grid(pos_weight: float, points=9, span=4.0) -> tuple[float, ...]:
    """
    Thresholds evenly spaced in logit space around the decision point the weighted BCE implies, w / (w + 1).
    """
    centre = math.log(pos_weight)
    return tuple(round(1.0 / (1.0 + math.exp(-(centre + offset))), 4) for offset in np.linspace(-span, span, points))


VALIDATION_THRESHOLDS = threshold_grid(TRAINING["pos_weight"])


@torch.no_grad()
def validate(model, record_ids) -> tuple[float, float]:
    """
    Predict held-out records whole and score them over VALIDATION_THRESHOLDS.

    Instances here are the connected components of each record's mask, and predictions under 256 px are dropped.
    This is a fast proxy for picking the epoch; the post-processing is calibrated properly later.
    :return: Tuple of (best PQ, its threshold)
    """
    model.eval()
    terms = {threshold: np.zeros(4) for threshold in VALIDATION_THRESHOLDS}
    for image_id in record_ids:
        record = RECORDS[image_id]
        probabilities = predict_tiled(model, load_image(TRAIN_IMAGES / record["file_name"]), overlap=0.0, tta=False).numpy()
        truth = np.zeros(probabilities.shape, dtype=bool)
        for ann in BY_RECORD[image_id]:
            truth |= polygon_mask(ann["segmentation"][0], *probabilities.shape)
        gt_labels, n_gt = connected_components(truth)
        for threshold in VALIDATION_THRESHOLDS:
            labels, n_pred = label_instances(probabilities, threshold, min_area=256)
            counts = np.bincount(
                (gt_labels.astype(np.int64) * (n_pred + 1) + labels).ravel(), minlength=(n_gt + 1) * (n_pred + 1)
            )
            counts = counts.reshape(n_gt + 1, n_pred + 1)
            intersection = counts[1:, 1:].astype(np.float64)
            union = counts[1:].sum(1, keepdims=True) + counts[:, 1:].sum(0, keepdims=True) - intersection
            terms[threshold] += pq_terms(np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0))
    best = max(VALIDATION_THRESHOLDS, key=lambda threshold: panoptic_quality(terms[threshold]))
    return panoptic_quality(terms[best]), best


def train_run(family: str, stage: str, fold: int, init=None) -> Path:
    """
    Train one family on one fold for one stage; a finished run is reused.
    :param family: Key of FAMILIES
    :param stage: Key of STAGES
    :param fold: Fold held out
    :param init: Checkpoint to start from, or None to start from ImageNet
    :return: Path to the best checkpoint
    """
    output = WORK / "runs" / f"{family}-{stage}-fold{fold}"
    best_path = output / "best.pt"
    if (output / "history.json").is_file():
        return best_path
    output.mkdir(parents=True, exist_ok=True)
    settings = STAGES[stage]
    set_seeds(SEED)

    train_ids, held_out = kfold_split(fold)
    windows = FilamentWindows(train_ids, TRAINING["crop"], TRAINING["crops_per_image"], TRAINING["foreground_prob"])
    loader = DataLoader(
        windows,
        batch_size=TRAINING["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=WORKERS,
        pin_memory=DEVICE.type == "cuda",
        persistent_workers=WORKERS > 0,
        # classes defined in a notebook cannot be pickled into a spawned worker, so fork it
        multiprocessing_context="fork" if WORKERS > 0 else None,
    )
    model = Pretrained(**FAMILIES[family], encoder_weights="imagenet").to(DEVICE)
    if init is not None:
        model.load_state_dict(torch.load(init, map_location=DEVICE, weights_only=False)["model_state"])
    optimizer = build_optimizer(model, settings["lr"])
    scheduler = build_scheduler(optimizer, settings["epochs"], len(loader), settings["warmup_epochs"])
    scaler = torch.amp.GradScaler(DEVICE.type, enabled=DEVICE.type == "cuda")

    history, best_pq = [], -1.0
    for epoch in range(1, settings["epochs"] + 1):
        started = time.time()
        desc = f"{output.name} epoch {epoch}/{settings['epochs']}"
        loss = train_epoch(model, loader, optimizer, scheduler, scaler, settings["lovasz_weight"], desc)
        pq, threshold = validate(model, held_out[: TRAINING["val_limit"]])
        history.append({"epoch": epoch, "loss": loss, "pq": pq, "threshold": threshold, "seconds": time.time() - started})
        print(f"{desc}  loss {loss:.4f}  pq {pq:.4f} @ {threshold:.2f}  ({history[-1]['seconds']:.0f}s)")
        if pq > best_pq:
            best_pq = pq
            model_config = {**FAMILIES[family], "encoder_weights": None}
            torch.save({"model_state": model.state_dict(), "model_config": model_config, "fold": fold, "epoch": epoch}, best_path)
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"{output.name}: best pq {best_pq:.4f} -> {best_path}")
    return best_path


# %% [markdown]
# ## Calibration
#
# Every observation is predicted by the three models (one per family) that did not train on it. The averaged map
# is scored against all its annotator records for every combination in `CALIBRATION_GRID`. Area and score cuts only
# remove columns from the IoU matrices, so components are labelled once per (threshold, offset). Maps are rounded
# to 8 bits before averaging.


# %%
def score_observation(probabilities: np.ndarray, records) -> dict:
    """
    :param probabilities: (H, W) averaged out-of-fold probabilities
    :param records: The annotations of each annotator record of the observation
    :return: PQ terms per (threshold, offset, min_area, min_score), summed over the records
    """
    gt = [annotation_pixels(annotations) for annotations in records]
    terms = {}
    for threshold in CALIBRATION_GRID["threshold"]:
        mask = probabilities >= threshold
        for offset in CALIBRATION_GRID["offset"]:
            labels, n_labels = connected_components(apply_offset(mask, offset))
            areas, scores = component_stats(labels, n_labels, probabilities)
            overlaps = [instance_iou(pixels, labels, n_labels) for pixels in gt]
            for min_area in CALIBRATION_GRID["min_area"]:
                for min_score in CALIBRATION_GRID["min_score"]:
                    keep = (areas > min_area) & (scores >= min_score)
                    terms[(threshold, offset, min_area, min_score)] = sum(pq_terms(iou[:, keep]) for iou in overlaps)
    return terms


def calibrate(checkpoints) -> dict:
    """
    Choose the post-processing that maximises pooled out-of-fold Panoptic Quality.
    :param checkpoints: The ensemble; each checkpoint knows its fold
    :return: Calibration with the winning settings, its PQ and the checkpoints it was made for
    """
    by_fold: dict[int, list] = {}
    for path in checkpoints:
        by_fold.setdefault(torch.load(path, map_location="cpu", weights_only=False)["fold"], []).append(path)

    pending = {}
    with ProcessPoolExecutor(WORKERS, mp_context=multiprocessing.get_context("fork")) as pool:
        for fold, paths in sorted(by_fold.items()):
            models = [load_model(path)[0] for path in paths]
            held_out = kfold_split(fold)[1]
            files: dict[str, list] = {}
            for image_id in held_out:
                files.setdefault(RECORDS[image_id]["file_name"], []).append(BY_RECORD[image_id])
            for name, records in tqdm(sorted(files.items()), desc=f"out-of-fold maps, fold {fold}", mininterval=2.0):
                image = load_image(TRAIN_IMAGES / name)
                total = sum(
                    (predict_tiled(model, image).numpy() * 255).round().astype(np.uint8).astype(np.float32) for model in models
                )
                pending[name] = pool.submit(score_observation, np.asarray(total / (255 * len(models)), dtype=np.float32), records)
            del models
            torch.cuda.empty_cache()
        accumulated: dict = {}
        for name in tqdm(sorted(pending), desc="scoring", mininterval=2.0):
            for key, value in pending[name].result().items():
                accumulated[key] = accumulated.get(key, 0) + value

    best_key, best_terms = max(accumulated.items(), key=lambda item: panoptic_quality(item[1]))
    calibration = dict(zip(CALIBRATION_GRID, best_key))
    return {
        **calibration,
        "pq": panoptic_quality(best_terms),
        "tp": int(best_terms[1]),
        "fp": int(best_terms[2]),
        "fn": int(best_terms[3]),
        "files": len(pending),
        "checkpoints": [str(path) for path in checkpoints],
    }


# %% [markdown]
# ## Models: train or fetch


# %%
def family_and_fold(path) -> tuple[int, int]:
    """Order checkpoints like the calibration: family as in FAMILIES, then fold."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    encoders = [family["encoder"] for family in FAMILIES.values()]
    return encoders.index(checkpoint["model_config"]["encoder"]), checkpoint["fold"]


if TRAIN:
    checkpoints = []
    for family in FAMILIES:
        for fold in range(FOLDS):
            first = train_run(family, "stage1", fold)
            checkpoints.append(train_run(family, "stage2", fold, init=first))
    calibration = calibrate(checkpoints)
    (WORK / CALIBRATION_NAME).write_text(json.dumps(calibration, indent=2), encoding="utf-8")
else:
    if WEIGHTS_DIR is None:
        import kagglehub

        weights = Path(kagglehub.model_download(MODEL_HANDLE))
    else:
        weights = Path(WEIGHTS_DIR)
    checkpoints = sorted(weights.glob("*.pt"), key=family_and_fold)
    calibration = json.loads((weights / CALIBRATION_NAME).read_text(encoding="utf-8"))

assert len(checkpoints) == len(calibration["checkpoints"]), (
    f"{len(checkpoints)} checkpoints, calibration made for {len(calibration['checkpoints'])}"
)
print(
    f"{len(checkpoints)} checkpoints; out-of-fold PQ {calibration['pq']:.4f} at threshold {calibration['threshold']}, "
    f"offset {calibration['offset']:+d}, min_area {calibration['min_area']}, min_score {calibration['min_score']}"
)

# %% [markdown]
# ## Predict the test set


# %%
def encode_rle(mask: np.ndarray) -> str:
    """Compressed COCO RLE counts of a 2048 x 2048 mask, as the submission expects them."""
    return coco_mask.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"].decode("ascii")


models = [load_model(path)[0] for path in checkpoints]
started = time.time()
rows = []
for path in tqdm(TEST_FILES, desc="predict", mininterval=2.0):
    image = load_image(path)
    probabilities = (sum(predict_tiled(model, image) for model in models) / len(models)).numpy()
    labels, n_labels = label_instances(
        probabilities, calibration["threshold"], calibration["offset"], calibration["min_area"], calibration["min_score"]
    )
    rows += [(f"{path.stem}_{label}", encode_rle(labels == label)) for label in range(1, n_labels + 1)]
seconds = time.time() - started

submission = WORK / "submission.csv"
with open(submission, "w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle, quoting=csv.QUOTE_NONE, quotechar=None)
    writer.writerow(["filament_id", "segmentation_rle"])
    writer.writerows(rows)

observations = {row[0].rsplit("_", 1)[0] for row in rows}
assert observations <= {path.stem for path in TEST_FILES}
assert all(coco_mask.area({"size": [SUBMISSION_SIZE] * 2, "counts": counts.encode()}) for _, counts in rows)
print(f"{len(rows)} filaments over {len(observations)}/{len(TEST_FILES)} observations -> {submission}")
print(
    f"{seconds:.0f} s, {seconds / len(TEST_FILES):.1f} s per observation on {torch.cuda.get_device_name() if DEVICE.type == 'cuda' else 'cpu'}"
)

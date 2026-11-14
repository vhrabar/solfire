# %% [markdown]
# # Solar filament segmentation: full pipeline
#
# This notebook finds solar filaments in GONG H-alpha observations (MAGFiLO 1.0) and writes the competition's
# `submission.csv`, one compressed-RLE mask per filament. The model is SOLFIRE (SOLar FIlament Refined Ensemble). Its
# weights are on Kaggle Models as `vhrabar/filament-segmentation-2026`, one variation per preset, and in `weights/final`
# of the repository. Pick a preset with `PRESET`:
#
# | `PRESET` | Models | TTA | Refiner | Out-of-fold PQ | Public LB | s / image |
# |---|---|---|---|---|---|---|
# | **`final`** (the evaluated submission) | SSL ResNet-34, MiT-B2, HRNet-W32, zoom HRNet-W32 | D4, overlap 0.5 | yes | 0.4668 | 0.40 | 10.8 |
# | `fast` | the same 4 | 4 flips | yes | 0.4642 | 0.40 | 4.1 |
# | `lite` | ImageNet ResNet-34, MiT-B2, HRNet-W32 | 4 flips | yes | 0.4614 | 0.39 | 2.0 |
# | `tiny` | HRNet-W32 | 4 flips | no | 0.4508 | 0.39 | 1.2 |
#
# Seconds per image are for the 180 test images including image loading, on an RTX 5070 Ti. A Kaggle T4 takes about
# 3 to 5 times as long (50.6, 17.7, 7.8 and 4.0 s). Out-of-fold PQ, with barbs, comes from the five-fold models of the same recipe,
# pooled over the 707 training observations; the full-data models reuse the cut-offs found there. The public
# leaderboard only shows two decimals.
#
# Quick start on Kaggle: add the competition, the model variation you want, `kx-plot` and the GONG dataset as
# inputs, keep `TRAIN = False` and run all. A T4 needs about 2.5 h for `final`, 53 min for `fast`, 24 min for `lite`
# and 12 min for `tiny`. You get `submission.csv` in the working folder.
#
# ![SOLFIRE pipeline](https://github.com/vhrabar/solfire/raw/main/paper/out/pipeline.png)
#
# *The pipeline of `final`. Dashed arrows carry what training produces into inference.
#
# ## How it works
#
# Models: U-Net++ on ResNet-34, U-Net on MiT-B2 (SegFormer) and U-Net on HRNet-W32, plus a second HRNet-W32 that
# sees the observation upsampled 2x (`zoom`), so thin filaments cover more pixels. The ResNet-34 of `final` and `fast`
# starts from masked-image pretraining on 1,300 unlabelled GONG frames from 2023-2025 (`scripts/pretrain_mim.py`); the
# other encoders start from ImageNet. Every model is trained on all 707 training observations and keeps its last epoch.
# The weights are float16 copies, which give the same submission as the float32 checkpoints under fp16 autocast.
#
# Training: each family and fold gets 40 epochs on random 512 px windows at native resolution (filaments are only
# about 11 px wide, so nothing is downsampled), most windows centred on an annotated filament. The target is the
# average of the annotators' masks, and the loss is BCE (positive weight 5) plus soft Dice. Then 8 epochs of
# fine-tuning at a lower learning rate with a Lovasz hinge added, a surrogate for the IoU that Panoptic Quality is
# built on.
#
# Inference: the observation is cut into overlapping 512 px windows (overlap 0.25, cosine-blended) and averaged
# over four flips; `final` adds the four flips of the quarter-turned image at overlap 0.5. The models' probabilities
# are averaged, thresholded, eroded by one pixel and split into 8-connected pieces. A crop refiner (a ResNet-34
# U-Net) then redraws each piece on a window around it and predicts the IoU it will earn, and that prediction decides
# whether the piece is kept (`tiny` has no refiner and keeps pieces by their mean probability). Last, barbs are grown from the image into every instance, since the organisers want
# them in the masks. The cut-offs come from a grid search that scores the organisers' PQ on the pooled out-of-fold
# predictions.
#
# ## Running it
#
# The notebook works out where it is. With `/kaggle/input` and `/kaggle/working` present it runs in Kaggle mode: it
# reads the competition data, the weights, `kx-plot` and the GONG data from the attached inputs and writes to
# `/kaggle/working`. Otherwise it runs in local mode, from a checkout of the repository: `data/` (or
# `$FILAMENT_DATA_ROOT`), `weights/final`, kx-plot from the `uv` environment and the GONG downloads in `runs/external`,
# writing to `build/notebook`. Without the GONG folder the GONG figures and the pretraining are skipped.
#
# - `TRAIN = False` (default) loads the `PRESET` weights and predicts. The weights are looked up in `WEIGHTS_DIR`,
#   then in `weights/final`, then among the attached inputs, and finally downloaded with `kagglehub` from
#   `MODEL_HANDLE`.
# - `TRAIN = True` trains the five-fold recipe for the three families (30 runs) and the five refiners, calibrates,
#   and predicts with those 15 models (0.4637 out of fold). With the GONG frames and `SSL = True`, the ResNet-34 first
#   learns masked-image reconstruction on them. The zoom family and the full-data retrain are not repeated. That is
#   about 20 GPU-hours on an RTX 5070 Ti, more than one Kaggle session, but finished runs are skipped, so you can
#   rerun until it completes.
# - A finished prediction is not repeated. If `submission.csv` and `inference.json` for this `PRESET` are already in
#   the working folder, the notebook reads them and only reruns the few observations the figures show. Set
#   `RERUN = True` in the prediction cell to predict everything again.
#
# Source: https://github.com/vhrabar/solfire. This notebook is generated from `notebook/pipeline.py`.

# %%
import csv
import hashlib
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

TRAIN = False  # True: train the models and the crop refiners from the competition data; False: fetch the published ones
SEED = 1
FOLDS = 5  # grouped by observation, the assignment is fixed (seed 0) whatever SEED is

# Kaggle mode when the Kaggle mounts exist, local mode otherwise
KAGGLE = Path("/kaggle/input").is_dir() and Path("/kaggle/working").is_dir()
SSL = True  # with TRAIN: start the ResNet-34 family from masked-image pretraining on the GONG H-alpha frames
SSL_EPOCHS = 30
SSL_PATCH, SSL_MASK_RATIO = 32, 0.6  # side of a hidden square (px) and the share of squares hidden

PRESET = "final"  # final, fast, lite or tiny (see the table above)
MODEL_HANDLE = f"vhrabar/filament-segmentation-2026/pyTorch/solfire-{PRESET}/1"  # Kaggle Models variation
CALIBRATION_NAME = f"{PRESET}.calibration.json"  # cut-offs and inference settings, next to the checkpoints
WEIGHTS_DIR = None  # an override folder holding the checkpoints and the calibration
# the repository checkout this runs in (the notebook sits in notebook/), else the working directory
REPO = next((path for path in (Path.cwd(), *Path.cwd().parents) if (path / "notebook" / "pipeline.py").is_file()), Path.cwd())
REPO_WEIGHTS = REPO / "weights" / "final"

# %% [markdown]
# The training recipe (used with `TRAIN = True`), then inference, calibration and refiner settings.


# %%
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
    "min_area": (0, 128, 256, 512),
    "min_score": (0.0, 0.6, 0.625, 0.65, 0.675, 0.7, 0.725, 0.75, 0.8, 0.85, 0.88, 0.91, 0.94),
}

# %%
REFINER = {
    "crop": 256,  # edge the refiner works at
    "context": 1.5,  # window edge as a multiple of the instance's longer box side
    "min_window": 128,
    "min_pixels": 32,  # smaller proposals are never refined nor submitted
    "proposal_thresholds": (0.3, 0.375, 0.45),  # training proposals; the calibrated cut sits in the middle
    "duplicate_iou": 0.85,  # a training proposal this close to one already taken is skipped
    "epochs": 12,
    "batch_size": 32,
    "lr": 3e-4,
    "encoder_lr_scale": 0.5,
    "score_weight": 0.5,
}
BARBS = None  # None follows the calibration's "barbs" block; True / False forces the barbs on (DEFAULT_BARBS) or off
REFINE_GRID = {
    "threshold": (0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6),
    "min_score": tuple(round(0.02 * step, 2) for step in range(31)),  # cut on the predicted IoU
}

SUBMISSION_SIZE = 2048
WORKERS = min(8, os.cpu_count() or 1)
WORK = Path("/kaggle/working") if KAGGLE else REPO / "build" / "notebook"
WORK.mkdir(parents=True, exist_ok=True)

# %%
try:
    import segmentation_models_pytorch as smp
except ImportError:  # not in Kaggle's image
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "segmentation-models-pytorch==0.6.0", "timm==1.0.30"], check=True
    )
    import segmentation_models_pytorch as smp

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools import mask as coco_mask
from scipy import ndimage
from skimage.filters import sato
from skimage.morphology import disk
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


# %%
FIGURES = WORK / "figures"


def import_kx():
    """
    :return: The kx module: installed in the environment, else from the attached Kaggle Dataset vhrabar/kx-plot
    """
    try:
        import kx

        return kx
    except ImportError:
        package = next(Path("/kaggle/input").rglob("kx/__init__.py"), None) if KAGGLE else None
        sys.path.append(str(package.parent.parent))
        import kx

        return kx


kx = import_kx()
import matplotlib.pyplot as plt
import pandas as pd

kx.use("light")
print(f"{'Kaggle' if KAGGLE else 'local'} mode, kx-plot {kx.__version__}, figures -> {FIGURES}")


def figure(name: str) -> None:
    """Save the current figure as FIGURES/<name>.png and show it."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    kx.save(str(FIGURES / f"{name}.png"))
    plt.show()
    plt.close("all")


# %% [markdown]
# ## Data
#
# 707 training observations (2048 x 2048 greyscale JPEG) carry 1,154 annotator records: an observation can be
# annotated by up to three people. Every record is a COCO image entry whose polygons are the filaments.
# Folds group all records of one observation together.

# %%
ANNOTATIONS_NAME = "MAGFiLO_1.0_Annotations_kaggle2026_train.json"


def find_dataset() -> tuple[Path, Path, Path]:
    """
    Locate the competition data: $FILAMENT_DATA_ROOT, a Kaggle input mount, or data/ of the repository.
    :return: Tuple of (annotations file, train image directory, test image directory)
    """
    roots = [Path(os.environ["FILAMENT_DATA_ROOT"])] if "FILAMENT_DATA_ROOT" in os.environ else []
    roots += [Path("/kaggle/input"), REPO / "data"]
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


# %% [markdown]
# Masks are filled with pycocotools, exactly as the scored masks are, and observations are read as greyscale in
# [0, 1].


# %%
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
    A greyscale observation, or one square window of it, as a (1, H, W) float tensor in [0, 1].
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


# %% [markdown]
# Training windows: native-resolution crops with soft targets, flipped, turned and jittered.


# %%
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


# %% [markdown]
# One sample per observation and window, its target the annotators' average.


# %%
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
# The annotations: how many people annotated each observation, how many filaments a record holds and how big they
# are; then four training windows as the model sees them (augmented), with every annotator's filaments.


# %%
def polygon_area(polygon) -> float:
    """:return: Area of a flat [x0, y0, x1, y1, ...] polygon (shoelace formula)"""
    x, y = np.asarray(polygon, dtype=np.float64).reshape(-1, 2).T
    return 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


annotators = pd.Series([record["file_name"] for record in RECORDS.values()]).value_counts()
per_record = np.array([len(BY_RECORD[image_id]) for image_id in RECORDS])
areas = np.array([polygon_area(annotation["segmentation"][0]) for annotation in raw["annotations"]])
axs = kx.grid(3)
kx.plot(annotators.to_numpy(), spec="count-ann", ax=axs[0], title="annotators per observation")
kx.plot(per_record, spec="hist", ax=axs[1], bins=np.arange(per_record.max() + 2) - 0.5, title="filaments per record")
kx.plot(
    areas,
    spec="hist-logx-median-ann",
    ax=axs[2],
    bins=np.geomspace(max(areas.min(), 1), areas.max(), 40),
    title="filament area (px)",
)
figure("annotations")

random.seed(SEED)
windows = FilamentWindows(list(RECORDS), TRAINING["crop"], 1, foreground_prob=1.0)
samples = [windows[index] for index in random.sample(range(len(windows)), 4)]
kx.plot([image[0].numpy() for image, _ in samples], [(target > 0).numpy() for _, target in samples], "segmask")
figure("training-windows")

# %% [markdown]
# Where the observations come from: GONG has six sites and the file name carries the year and the site letter
# (B Big Bear, C Cerro Tololo, L Learmonth, M Mauna Loa, T El Teide, U Udaipur; A and Z appear from 2023). The
# test set follows the training set's years and sites, shown as the share of each set.

# %%
SITES = {"B": "Big Bear", "C": "Cerro Tololo", "L": "Learmonth", "M": "Mauna Loa", "T": "El Teide", "U": "Udaipur"}


def year_site(name: str) -> tuple[int, str]:
    """:return: (year, site letter) of a file name such as 20110109104734Ch.jpeg"""
    return int(name[:4]), name[14]


train_names = sorted({record["file_name"] for record in RECORDS.values()})
origin = pd.DataFrame(
    [
        (*year_site(name), part)
        for part, names in (("train", train_names), ("test", [p.name for p in TEST_FILES]))
        for name in names
    ],
    columns=["year", "site", "set"],
)
axs = kx.grid(2)
by_year = origin.groupby(["year", "set"]).size().unstack(fill_value=0)
kx.plot(
    (100 * by_year / by_year.sum()).reset_index(),
    "year",
    "train",
    "test",
    spec="bar-leg",
    ax=axs[0],
    title="observations per year (%)",
)
by_site = origin.groupby(["site", "set"]).size().unstack(fill_value=0)
kx.plot(
    (100 * by_site / by_site.sum()).reset_index(),
    "site",
    "train",
    "test",
    spec="bar-leg",
    ax=axs[1],
    title="observations per GONG site (%)",
)
figure("observations")

# %% [markdown]
# ## GONG data
#
# Organisers allowed GONG data beyond MAGFiLO as long as the submitted model sees only H-alpha. The Kaggle Dataset
# `vhrabar/gong-solar-filaments` (and `runs/external` locally) holds 1,300 unlabelled 2023-2025 H-alpha frames, used
# for the masked-image pretraining below, and `bzi` line-of-sight magnetograms matched to the training
# observations, used only to look at the data here (a magnetogram auxiliary head did not improve the ensemble).


# %%
def find_gong() -> dict | None:
    """
    :return: Paths of the GONG data (frames, bursts, bzi, matches), or None locally when it was not downloaded.
        In Kaggle mode a missing Dataset is an error.
    """
    if KAGGLE:
        bursts = next(Path("/kaggle/input").rglob("halpha_bursts.json"), None)
        if bursts is None:
            raise FileNotFoundError("attach the Dataset vhrabar/gong-solar-filaments (Add Input)")
        root = bursts.parent
        return {
            "frames": root / "halpha_2023_2025",
            "bursts": bursts,
            "bzi": root / "bzi_fits",
            "matches": root / "bzi_matches.json",
        }
    root = REPO / "runs" / "external"
    if not (root / "gong" / "manifest.json").is_file():
        return None
    return {
        "frames": root / "gong" / "jpeg",
        "bursts": root / "gong" / "manifest.json",
        "bzi": root / "gong_bzi" / "fits",
        "matches": root / "gong_bzi" / "manifest.json",
    }


def burst_frames(gong: dict) -> list[list[Path]]:
    """:return: The H-alpha frames of every burst (five frames, 2 min apart) that exist on disk"""
    bursts = json.loads(gong["bursts"].read_text(encoding="utf-8"))["bursts"]
    found = [[gong["frames"] / f"{stem}.jpeg" for stem in burst] for burst in bursts]
    return [burst for burst in found if all(path.is_file() for path in burst)]


# %%
def read_bzi(gong: dict, stem: str):
    """
    :param stem: Training observation name without extension
    :return: (field in gauss, FITS header) of its matched bzi magnetogram, files are .fits.gz or gunzipped by Kaggle
    """
    from astropy.io import fits

    path = next(iter(sorted(gong["bzi"].glob(f"{stem}.fits*"))))
    if path.is_dir():
        path = next(iter(sorted(path.iterdir())))
    with fits.open(path) as hdus:
        return np.nan_to_num(hdus[0].data.astype(np.float32)), hdus[0].header


def bzi_on_halpha(field: np.ndarray, header, size: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """
    The magnetogram on the H-alpha grid (disk centre 1024, radius 900, north up) at size x size: the site image is
    a constant quarter-turn away, and the FNDLMB* keys give its disk centre and radius.
    :return: (field in gauss, on-disk mask)
    """
    steps = (np.arange(size) + 0.5) * 2048 / size
    v, u = np.meshgrid((steps - 1024) / 900, (steps - 1024) / 900, indexing="ij")
    turned_u, turned_v = -v, u  # rotation by 90 degrees
    radius = (header["FNDLMBMI"] + header["FNDLMBMA"]) / 2
    rows = header["FNDLMBYC"] + radius * turned_v
    columns = header["FNDLMBXC"] + radius * turned_u
    resampled = ndimage.map_coordinates(field, [rows, columns], order=1, cval=0.0)
    valid = u**2 + v**2 < 0.97**2
    return np.where(valid, resampled, 0.0), valid


# %%
def pil_proximity(field: np.ndarray, valid: np.ndarray, size: int = 512, smooth_mm=12.0, decay_mm=10.0) -> np.ndarray:
    """
    :return: exp(-d / decay) to the nearest polarity inversion line of the smoothed field (d in Mm)
    """
    mm = 696.0 / 900.0 * 2048 / size  # Mm per pixel at disk centre
    smooth = ndimage.gaussian_filter(field, smooth_mm / mm)
    positive = smooth > 0
    line = np.zeros_like(positive)
    line[:, :-1] |= positive[:, :-1] != positive[:, 1:]
    line[:-1, :] |= positive[:-1, :] != positive[1:, :]
    line &= np.hypot(*np.gradient(smooth)) >= 1.0
    line &= ndimage.binary_erosion(valid, iterations=int(2 * smooth_mm / mm) + 2)
    return np.exp(-ndimage.distance_transform_edt(~line) * mm / decay_mm) if line.any() else np.zeros_like(smooth)


# %%
GONG = find_gong()
if GONG is None:
    print("no GONG data under runs/external: GONG figures and pretraining are skipped")
else:
    bursts = burst_frames(GONG)
    matches = pd.DataFrame(json.loads(GONG["matches"].read_text(encoding="utf-8"))["images"])
    print(
        f"GONG: {len(bursts)} H-alpha bursts ({sum(map(len, bursts))} frames), {(matches.status == 'ok').sum()} magnetograms ({GONG['frames']})"
    )


# %% [markdown]
# How well the magnetograms match the training observations: the time between the two (minutes) and the plage
# correlation, i.e. how well |B| lines up with the H-alpha activity after registration (a registration and timing
# check; the median is about 0.67).

# %%
if GONG is not None:
    ok = matches[matches.status == "ok"]
    axs = kx.grid(2)
    kx.plot(
        ok.gap_minutes.to_numpy(float),
        spec="hist-median-ann",
        ax=axs[0],
        bins=np.linspace(0, 60, 31),
        title="time to the H-alpha observation (min)",
    )
    kx.plot(
        ok.plage_correlation.to_numpy(float),
        spec="hist-median-ann",
        ax=axs[1],
        bins=np.linspace(-0.2, 1, 37),
        title="plage correlation |B| vs H-alpha",
    )
    figure("gong-matches")


# %% [markdown]
# The best-registered training observations with annotations: H-alpha with every annotator's filaments, then the
# unsigned magnetogram |B| and the proximity to its polarity inversion lines laid over the same frame (each map
# scaled to 0-1). Filaments lie along the inversion lines.

# %%
if GONG is not None:
    names = {record["file_name"]: image_id for image_id, record in RECORDS.items() if BY_RECORD[image_id]}
    best = ok[ok.image.isin(names)].sort_values("plage_correlation", ascending=False).image.head(2)
    for name in best:
        stem = Path(name).stem
        field, header = bzi_on_halpha(*read_bzi(GONG, stem))
        proximity = pil_proximity(field, field != 0)
        image = load_image(TRAIN_IMAGES / name)[0].numpy()
        truth = np.zeros((2048, 2048), bool)
        for image_id, record in RECORDS.items():
            if record["file_name"] == name:
                for annotation in BY_RECORD[image_id]:
                    truth |= polygon_mask(annotation["segmentation"][0], 2048, 2048)
        kx.plot([image], [truth], "segmask", title=stem)
        figure(f"gong-filaments-{stem}")
        limit = float(np.percentile(np.abs(field), 99.5)) or 1.0
        kx.plot(
            [image, image],
            [np.minimum(np.abs(field), limit), proximity],
            ["|B| of the magnetogram", "proximity to inversion lines"],
            "saliency",
            title=stem,
        )
        figure(f"gong-registration-{stem}")


# %% [markdown]
# One unlabelled burst: five frames two minutes apart, the material of the pretraining.

# %%
if GONG is not None:
    burst = bursts[len(bursts) // 2]
    kx.plot([load_image(path)[0].numpy()[::4, ::4] for path in burst], "imgrid")
    figure("gong-burst")


# %% [markdown]
# ## Model
#
# A `segmentation_models_pytorch` decoder on an ImageNet encoder. smp collapses the first convolution to one input
# channel by summing its RGB kernels, so the greyscale observation only needs ImageNet's mean and std.

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
def predict_flips(model, images: Tensor, turned=False) -> Tensor:
    """
    :param model: Model in eval mode
    :param images: (B, 1, H, W) windows on DEVICE
    :param turned: Use the four flips of the quarter-turned windows instead (the other half of the square's symmetries)
    :return: (B, 1, H, W) probabilities averaged over the identity and the three flips
    """
    images = torch.rot90(images, 1, (-2, -1)) if turned else images
    total = torch.zeros_like(images, dtype=torch.float32)
    for flip_x in (False, True):
        for flip_y in (False, True):
            dims = [d for d, on in ((-1, flip_x), (-2, flip_y)) if on]
            view = torch.flip(images, dims) if dims else images
            probabilities = torch.sigmoid(model(view).float())
            total += torch.flip(probabilities, dims) if dims else probabilities
    return torch.rot90(total / 4, -1, (-2, -1)) if turned else total / 4


# %% [markdown]
# Tiling: windows step by `tile * (1 - overlap)`, the last one pulled back to the edge. `skip_dark` gives windows
# entirely off the disk probability 0 without running the model.


# %%
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
def predict_tiled(
    model, image: Tensor, tile=TILE, overlap=OVERLAP, tta=True, batch_size=TILE_BATCH, scale=1.0, turned=False, skip_dark=False
) -> Tensor:
    """
    Probabilities for one whole observation, assembled from overlapping windows.
    :param model: Model in eval mode
    :param image: (1, H, W) observation at native resolution
    :param tile: Window edge
    :param overlap: Fraction of a window shared with its neighbour
    :param tta: Average every window over the four flips
    :param batch_size: Windows per forward pass
    :param scale: Resize the observation by this factor first and the probabilities back afterwards
        (2 for the zoom family)
    :param turned: Average over the flips of the quarter-turned windows instead (see predict_flips)
    :param skip_dark: Give windows with nothing brighter than the black off the disk probability 0 without running
        the model on them
    :return: (H, W) probabilities on the cpu
    """
    if scale != 1.0:
        height, width = image.shape[-2:]
        size = (round(height * scale), round(width * scale))
        resized = F.interpolate(image[None].float(), size=size, mode="bilinear", align_corners=False, antialias=scale < 1.0)
        probabilities = predict_tiled(model, resized[0], tile, overlap, tta, batch_size, 1.0, turned, skip_dark)
        return F.interpolate(probabilities[None, None], size=(height, width), mode="bilinear", align_corners=False)[0, 0]
    image = image.unsqueeze(0)
    height, width = image.shape[-2:]
    stride = max(1, round(tile * (1.0 - overlap)))
    corners = [(y, x) for y in tile_offsets(height, tile, stride) for x in tile_offsets(width, tile, stride)]

    ramp = torch.hann_window(tile + 2, periodic=False, device=DEVICE)[1:-1].clamp_min(0.05)
    weights = (ramp[:, None] * ramp[None, :])[None, None]
    total = torch.zeros((1, 1, height, width), dtype=torch.float32, device=DEVICE)
    counts = torch.zeros_like(total)
    if skip_dark:
        lit = [(y, x) for y, x in corners if image[..., y : y + tile, x : x + tile].max() > 10 / 255]
        for y, x in set(corners) - set(lit):
            counts[..., y : y + tile, x : x + tile] += weights
        corners = lit
    for start in range(0, len(corners), batch_size):
        chunk = corners[start : start + batch_size]
        batch = torch.cat([image[..., y : y + tile, x : x + tile] for y, x in chunk]).to(DEVICE, non_blocking=True)
        probabilities = predict_flips(model, batch, turned) if tta else torch.sigmoid(model(batch).float())
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
# Panoptic Quality as the organisers compute it: a prediction matches an annotated filament at IoU > 0.5, and
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


# %% [markdown]
# Scoring: the organisers' Panoptic Quality, per annotator record.


# %%
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
# ## Masked-image pretraining
#
# With `TRAIN = True`, `SSL = True` and the GONG frames available, the ResNet-34 family does not start from ImageNet
# alone. SimMIM-style: random 32 px squares (60 %) of a native-resolution crop of an unlabelled H-alpha frame are
# replaced by the crop's mean, and the whole encoder-decoder reconstructs them. The target is standardised per square, so the
# loss rewards the dark filament and plage structure rather than limb darkening. The reconstruction head is dropped
# and the rest initialises the stage 1 runs of that family. The released `final` and `fast` presets were pretrained
# this way (on a dedicated GPU run of 30 epochs).


# %%
# %%
class MaskedCrops(Dataset):
    """Native-resolution crops centred on a random on-disk point of an H-alpha frame, flipped and turned."""

    def __init__(self, paths, crop=512, repeat=8, disk_level=0.1):
        self.paths, self.crop, self.repeat = list(paths), crop, repeat
        self.disk_level = round(disk_level * 255)

    def __len__(self) -> int:
        return len(self.paths) * self.repeat

    def __getitem__(self, index: int) -> Tensor:
        with Image.open(self.paths[index % len(self.paths)]) as handle:
            image = np.asarray(handle.convert("L"), dtype=np.uint8)
        limit = image.shape[0] - self.crop
        ys, xs = np.nonzero(image[::8, ::8] >= self.disk_level)
        pick = random.randrange(len(ys))
        x = min(max(int(xs[pick]) * 8 - self.crop // 2, 0), limit)
        y = min(max(int(ys[pick]) * 8 - self.crop // 2, 0), limit)
        window = torch.from_numpy(image[y : y + self.crop, x : x + self.crop].copy()).float().div_(255.0)
        if random.random() < 0.5:
            window = window.flip(-1)
        return torch.rot90(window, random.randint(0, 3), (-2, -1)).unsqueeze(0)


# %%
def hide_patches(batch: int, size: int, patch: int, ratio: float, device) -> Tensor:
    """:return: (B, 1, size, size) float mask, 1 where a square is hidden"""
    cells = size // patch
    hidden = torch.rand(batch, cells * cells, device=device).argsort(dim=1) < round(ratio * cells * cells)
    return F.interpolate(hidden.view(batch, 1, cells, cells).float(), scale_factor=patch, mode="nearest")


def patch_standardised(image: Tensor, patch: int) -> Tensor:
    """Each patch x patch square standardised on its own (the reconstruction target)."""
    mean = F.avg_pool2d(image, patch)
    std = (F.avg_pool2d(image**2, patch) - mean**2).clamp_min(0).add(1e-6).sqrt()
    return (image - F.interpolate(mean, scale_factor=patch, mode="nearest")) / F.interpolate(
        std, scale_factor=patch, mode="nearest"
    )


# %%
def pretrain_ssl(family: str) -> Path | None:
    """
    Masked-image pretraining of a family on the GONG frames; a finished run is reused.
    :param family: Key of FAMILIES
    :return: Checkpoint of the encoder and decoder without the reconstruction head, or None without GONG frames
    """
    if GONG is None:
        return None
    output = WORK / "runs" / f"{family}-ssl-pretrain"
    result = output / "encoder.pt"
    if result.is_file():
        return result
    output.mkdir(parents=True, exist_ok=True)
    set_seeds(SEED)
    used = {stem for burst in json.loads(GONG["bursts"].read_text(encoding="utf-8"))["bursts"] for stem in burst}
    frames = sorted(path for path in GONG["frames"].glob("*.jpeg") if path.stem in used)
    dataset = MaskedCrops(frames, TRAINING["crop"])
    loader = DataLoader(
        dataset,
        batch_size=TRAINING["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=WORKERS,
        pin_memory=DEVICE.type == "cuda",
        persistent_workers=WORKERS > 0,
        multiprocessing_context="fork" if WORKERS > 0 else None,
    )
    model = Pretrained(**FAMILIES[family], encoder_weights="imagenet").to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    steps, warmup = SSL_EPOCHS * len(loader), len(loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(step, steps) / steps))
    )
    history = []
    for epoch in range(1, SSL_EPOCHS + 1):
        model.train()
        total = 0.0
        for image in tqdm(loader, desc=f"mim epoch {epoch}/{SSL_EPOCHS}", mininterval=2.0):
            image = image.to(DEVICE, non_blocking=True)
            hidden = hide_patches(image.shape[0], TRAINING["crop"], SSL_PATCH, SSL_MASK_RATIO, DEVICE)
            masked = image * (1 - hidden) + image.mean(dim=(-2, -1), keepdim=True) * hidden
            with torch.autocast(DEVICE.type, dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
                prediction = model(masked)
            loss = ((prediction.float() - patch_standardised(image, SSL_PATCH)).abs() * hidden).sum() / hidden.sum().clamp_min(1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), TRAINING["grad_clip"])
            optimizer.step()
            scheduler.step()
            total += loss.item()
        history.append({"epoch": epoch, "loss": total / len(loader)})
        print(f"mim epoch {epoch}/{SSL_EPOCHS}  loss {history[-1]['loss']:.4f}")
    state = {k: v for k, v in model.state_dict().items() if "segmentation_head" not in k}
    torch.save({"model_state": state, "model_config": {**FAMILIES[family], "encoder_weights": None}}, result)
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return result


# %%
SSL_FAMILY = "unetpp-r34"  # the family that starts from the pretraining
SSL_START = None  # its checkpoint, set before the training runs


def run_dir(family: str, stage: str, fold: int) -> Path:
    """:return: Folder of one training run; the pretrained family has its own, so the two starts never mix"""
    suffix = "-ssl" if SSL_START is not None and family == SSL_FAMILY else ""
    return WORK / "runs" / f"{family}{suffix}-{stage}-fold{fold}"


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


# %% [markdown]
# Optimizer, schedule and one training epoch. A batch whose loss is not finite is skipped, and its effect on the
# batch-norm statistics undone.


# %%
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


# %% [markdown]
# Epoch selection: a fast PQ proxy on 64 held-out records.


# %%
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


# %% [markdown]
# One run: a family, a stage and a fold. A run with a `history.json` is finished and reused.


# %%
def train_run(family: str, stage: str, fold: int, init=None) -> Path:
    """
    Train one family on one fold for one stage; a finished run is reused.
    :param family: Key of FAMILIES
    :param stage: Key of STAGES
    :param fold: Fold held out
    :param init: Checkpoint to start from, or None to start from ImageNet (or SSL_START for SSL_FAMILY)
    :return: Path to the best checkpoint
    """
    output = run_dir(family, stage, fold)
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
    if init is None and SSL_START is not None and family == SSL_FAMILY:
        model.load_state_dict(torch.load(SSL_START, map_location=DEVICE, weights_only=False)["model_state"], strict=False)
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
# Every observation is predicted by the three models (one per family) that did not train on it. The averaged map is
# scored against all its annotator records for every combination in `CALIBRATION_GRID`.
#
# Area and score cuts only remove columns from the IoU matrices, so components are labelled once per
# (threshold, offset). Maps are rounded to 8 bits before averaging. The averaged maps are kept in `OOF_DIR`, because
# the crop refiner learns from them.


# %%
OOF_DIR = WORK / "oof"


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


# %% [markdown]
# The grid search over the pooled out-of-fold maps; scoring runs in worker processes while the GPU predicts.


# %%
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
                probabilities = np.asarray(total / (255 * len(models)), dtype=np.float32)
                OOF_DIR.mkdir(parents=True, exist_ok=True)
                np.save(OOF_DIR / f"{Path(name).stem}.npy", (probabilities * 255).round().astype(np.uint8))
                pending[name] = pool.submit(score_observation, probabilities, records)
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
# ## Crop refiner
#
# The ensemble finds most filaments but draws many of them a little wrong. A partial mask (IoU under 0.5) counts as
# both a miss and a false positive, so the refiner takes a second look at every predicted instance.
#
# It cuts a square window around the instance (1.5x its longer box side, at least 128 px) and resamples it to 256 px.
# It reads three channels: the observation, the ensemble's probabilities and the instance's own mask. A ResNet-34
# U-Net redraws the instance, and a head on its bottleneck predicts the IoU the redrawn mask will earn. That
# prediction is the keep/drop score. Two fragments of one filament are both redrawn as the whole filament, so a mask
# mostly covered by a higher-scoring one is dropped.
#
# It learns from the instances of the pooled out-of-fold maps, cut at three thresholds, so it trains on the proposals
# it meets at test time. The refiner of fold k trains on the other folds only and is scored on fold k. The released
# presets keep the fold-0 refiner. With `TRAIN = True`, the five fold refiners are averaged.


# %%
def component_window(box, context=REFINER["context"]) -> tuple[int, int, int]:
    """
    :param box: (rows, columns) slices of a component, from ndimage.find_objects
    :param context: Window edge as a multiple of the longer box side
    :return: (top, left, side) of the square window it is refined on; it may reach past the image
    """
    rows, columns = box
    side = max(REFINER["min_window"], round(context * max(rows.stop - rows.start, columns.stop - columns.start)))
    return round((rows.start + rows.stop - side) / 2), round((columns.start + columns.stop - side) / 2), side


def clipped(window, height: int, width: int) -> tuple[slice, slice, slice, slice]:
    """:return: (rows, columns) of the window inside the observation, then the same region inside the window"""
    top, left, side = window
    y0, x0, y1, x1 = max(top, 0), max(left, 0), min(top + side, height), min(left + side, width)
    return slice(y0, y1), slice(x0, x1), slice(y0 - top, y1 - top), slice(x0 - left, x1 - left)


def window_of(array: np.ndarray, window) -> np.ndarray:
    """:return: (side, side) cut of the array, zeros past its edges"""
    out = np.zeros((window[2], window[2]), dtype=array.dtype)
    rows, columns, window_rows, window_columns = clipped(window, *array.shape)
    out[window_rows, window_columns] = array[rows, columns]
    return out


def resample(windows: Tensor, size: int) -> Tensor:
    """(B, C, s, s) -> (B, C, size, size), antialiased on the way down"""
    if windows.shape[-1] == size:
        return windows
    return F.interpolate(windows, size=(size, size), mode="bilinear", align_corners=False, antialias=True)


def refiner_inputs(image: np.ndarray, probabilities: np.ndarray, seed: np.ndarray, window) -> Tensor:
    """:return: (3, crop, crop) observation, ensemble probability and seed mask, in [0, 1]"""
    stacked = np.stack([window_of(image, window), window_of(probabilities, window), window_of(seed, window).astype(np.float32)])
    return resample(torch.from_numpy(stacked)[None], REFINER["crop"])[0].clamp_(0, 1)


# %% [markdown]
# The network: a ResNet-34 U-Net on three channels, with an IoU head on its bottleneck.


# %%
class Refiner(nn.Module):
    """ResNet-34 U-Net that redraws one instance, with a head on the bottleneck predicting the IoU it will earn."""

    def __init__(self, encoder_weights=None):
        super().__init__()
        self.net = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
            aux_params={"classes": 1, "dropout": 0.2},
        )

    @property
    def encoder(self) -> nn.Module:
        return self.net.encoder

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """:return: Tuple of ((B, 1, S, S) mask logits, (B,) logits of the earned IoU)"""
        masks, scores = self.net((x - IMAGENET_MEAN) / IMAGENET_STD)
        return masks, scores[:, 0]


def load_refiner(path) -> Refiner:
    model = Refiner()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["state_dict"])
    return model.to(DEVICE).eval()


@torch.no_grad()
def predict_refiner(refiners, inputs: Tensor) -> tuple[Tensor, Tensor]:
    """
    :return: Mask probabilities and predicted IoU, averaged over the refiners and three flips; stacked and averaged
        at once (under autocast a running float16 sum rounds differently from the solfire package)
    """
    masks, scores = [], []
    for model in refiners:
        for dims in ((), (-1,), (-2,)):
            logits, score = model(torch.flip(inputs, dims) if dims else inputs)
            masks.append(torch.sigmoid(torch.flip(logits, dims) if dims else logits))
            scores.append(torch.sigmoid(score))
    return torch.stack(masks).mean(0).float(), torch.stack(scores).mean(0).float()


# %% [markdown]
# Refining an observation: every proposal of at least `min_pixels` is redrawn on its window, then painted back most
# confident first.


# %%
def refine_components(
    refiners, image: np.ndarray, probabilities: np.ndarray, labels: np.ndarray, n_labels: int, context=REFINER["context"]
) -> list:
    """
    :param context: Window edge as a multiple of the longer box side, as the refiners were trained
    :return: Per proposal of at least min_pixels pixels, (window, uint8 refined probabilities at native size, score)
    """
    areas = np.bincount(labels.ravel(), minlength=n_labels + 1)
    jobs = [
        (index + 1, component_window(box, context))
        for index, box in enumerate(ndimage.find_objects(labels))
        if box and areas[index + 1] >= REFINER["min_pixels"]
    ]
    refined = []
    for start in range(0, len(jobs), REFINER["batch_size"]):
        chunk = jobs[start : start + REFINER["batch_size"]]
        inputs = torch.stack([refiner_inputs(image, probabilities, labels == k, window) for k, window in chunk])
        masks, scores = predict_refiner(refiners, inputs.to(DEVICE))
        for (_, window), mask, score in zip(chunk, masks, scores):
            native = resample(mask[None].float(), window[2])[0, 0]
            refined.append((window, (native * 255).round().byte().cpu().numpy(), float(score)))
    return refined


def assemble(refined, threshold: float, min_score=0.0) -> tuple[np.ndarray, int, np.ndarray]:
    """
    Paint refined instances into one label array, most confident first; a mask that more than half falls on
    instances already painted is a duplicate and is dropped.
    :return: Tuple of ((H, W) labels, number of instances, their scores)
    """
    labels = np.zeros((SUBMISSION_SIZE, SUBMISSION_SIZE), dtype=np.int32)
    scores = []
    for window, probabilities, score in sorted(refined, key=lambda item: -item[2]):
        if score < min_score:
            break
        rows, columns, window_rows, window_columns = clipped(window, SUBMISSION_SIZE, SUBMISSION_SIZE)
        components, count = connected_components(probabilities[window_rows, window_columns] >= round(threshold * 255))
        mask = components == int(np.bincount(components.ravel())[1:].argmax()) + 1 if count > 1 else components > 0
        area = int(mask.sum())
        free = mask & (labels[rows, columns] == 0)
        if area < REFINER["min_pixels"] or free.sum() < 0.5 * area:
            continue
        scores.append(score)
        labels[rows, columns][free] = len(scores)
    return labels, len(scores), np.asarray(scores)


def refine_observation(refiners, image: np.ndarray, probabilities: np.ndarray, settings: dict) -> tuple[np.ndarray, int, dict]:
    """
    The instances of one observation the way the refined submission is written.
    :return: Tuple of (labels, number of instances, details): the proposals the refiner started from and the
        predicted IoU of the kept instances and of every candidate (the ones under the cut are dropped)
    """
    proposals = apply_offset(probabilities >= settings["proposal_threshold"], settings["proposal_offset"])
    proposed, n_proposed = connected_components(proposals)
    context = settings.get("context", REFINER["context"])
    refined = refine_components(refiners, image, probabilities, proposed, n_proposed, context)
    labels, n_labels, scores = assemble(refined, settings["threshold"], settings["min_score"])
    details = {"proposals": proposed, "kept": scores, "candidates": np.array([item[2] for item in refined])}
    return labels, n_labels, details


# %% [markdown]
# Training: every proposal of the out-of-fold maps becomes a sample. Its target, per annotator record, is the
# annotated filament it overlaps most (empty for a false positive); the mask loss is BCE + soft Dice against the
# record average, and the score head regresses the IoU the refined mask, cut at a half, earns against each record.


# %%
REFINER_DIR = WORK / "refiner"
MAX_RECORDS = 3


def fold_of_file() -> dict:
    """:return: The fold that holds each training observation out"""
    folds = {}
    for fold in range(FOLDS):
        for image_id in kfold_split(fold)[1]:
            folds[RECORDS[image_id]["file_name"]] = fold
    return folds


def records_of_file() -> dict:
    """:return: Per training observation, the annotations of each of its records"""
    files: dict = {}
    for image_id, record in RECORDS.items():
        files.setdefault(record["file_name"], []).append(BY_RECORD[image_id])
    return files


# %% [markdown]
# Every proposal of one observation, cut at the three thresholds, with its target per annotator record.


# %%
def collect_one(job):
    """:return: (inputs, targets, n_records, file) of every training proposal of one observation"""
    name, records, offset = job
    probabilities = np.load(OOF_DIR / f"{Path(name).stem}.npy").astype(np.float32) / 255
    image = load_image(TRAIN_IMAGES / name).numpy()[0]
    gt = [annotation_pixels(record) for record in records]
    taken, inputs, targets = [], [], []
    for threshold in REFINER["proposal_thresholds"]:
        labels, n_labels = connected_components(apply_offset(probabilities >= threshold, offset))
        areas = np.bincount(labels.ravel(), minlength=n_labels + 1)
        overlaps = [instance_iou(pixels, labels, n_labels) for pixels in gt]
        for index, box in enumerate(ndimage.find_objects(labels)):
            if box is None or areas[index + 1] < REFINER["min_pixels"]:
                continue
            seed = labels == index + 1
            flat = np.flatnonzero(seed)
            if any(
                len(np.intersect1d(flat, other)) / len(np.union1d(flat, other)) >= REFINER["duplicate_iou"] for other in taken
            ):
                continue
            taken.append(flat)
            window = component_window(box)
            target = np.zeros((MAX_RECORDS, REFINER["crop"], REFINER["crop"]), dtype=np.uint8)
            for record, (pixels, iou) in enumerate(zip(gt, overlaps)):
                if iou.shape[0] and iou[:, index].max() > 0:
                    mask = np.zeros(SUBMISSION_SIZE * SUBMISSION_SIZE, dtype=np.float32)
                    mask[pixels[int(iou[:, index].argmax())]] = 1
                    cut = window_of(mask.reshape(SUBMISSION_SIZE, SUBMISSION_SIZE), window)
                    target[record] = (resample(torch.from_numpy(cut)[None, None], REFINER["crop"])[0, 0].numpy() * 255).round()
            inputs.append((refiner_inputs(image, probabilities, seed, window).numpy() * 255).round().astype(np.uint8))
            targets.append(target)
    return inputs, targets, len(records), name


# %% [markdown]
# All proposals, stored once as two uint8 memory maps.


# %%
def collect_refiner_samples(offset: int) -> dict:
    """
    Write every training proposal to REFINER_DIR as two uint8 arrays; reused when present.
    :param offset: Morphological offset of the calibration, which the proposals are cut with
    """
    meta_path = REFINER_DIR / "samples.json"
    if meta_path.is_file():
        return json.loads(meta_path.read_text())
    REFINER_DIR.mkdir(parents=True, exist_ok=True)
    chunks, records, files = [], [], []
    jobs = [(name, records, offset) for name, records in sorted(records_of_file().items())]
    with ProcessPoolExecutor(WORKERS, mp_context=multiprocessing.get_context("fork")) as pool:
        for inputs, targets, n_records, name in tqdm(pool.map(collect_one, jobs), total=len(jobs), desc="proposals"):
            if inputs:
                path = REFINER_DIR / f"chunk-{len(chunks)}.npz"
                np.savez(path, inputs=np.stack(inputs), targets=np.stack(targets))
                chunks.append(path)
                records += [n_records] * len(inputs)
                files += [name] * len(inputs)
    shape = (len(records), 3, REFINER["crop"], REFINER["crop"])
    stored_inputs = np.lib.format.open_memmap(REFINER_DIR / "inputs.npy", "w+", np.uint8, shape)
    stored_targets = np.lib.format.open_memmap(REFINER_DIR / "targets.npy", "w+", np.uint8, (shape[0], MAX_RECORDS, *shape[2:]))
    start = 0
    for path in chunks:
        with np.load(path) as chunk:
            stored_inputs[start : start + len(chunk["inputs"])] = chunk["inputs"]
            stored_targets[start : start + len(chunk["inputs"])] = chunk["targets"]
            start += len(chunk["inputs"])
        path.unlink()
    stored_inputs.flush()
    stored_targets.flush()
    folds = fold_of_file()
    meta = {"records": records, "folds": [folds[name] for name in files]}
    meta_path.write_text(json.dumps(meta))
    print(f"{len(records)} refiner samples")
    return meta


# %% [markdown]
# Refiner training: flips, transposes and a gamma/gain jitter of the observation channel; BCE + soft Dice on the
# redrawn mask and BCE on the IoU it earns.


# %%
class RefinerSamples(Dataset):
    """The collected samples, with random flips, transposes and a gamma/gain jitter of the observation channel."""

    def __init__(self, indices, records):
        self.indices, self.records = indices, records
        self.inputs = self.targets = None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item):
        if self.inputs is None:  # opened in the worker
            self.inputs = np.load(REFINER_DIR / "inputs.npy", mmap_mode="r")
            self.targets = np.load(REFINER_DIR / "targets.npy", mmap_mode="r")
        index = self.indices[item]
        inputs = torch.from_numpy(np.array(self.inputs[index], dtype=np.float32) / 255)
        targets = torch.from_numpy(np.array(self.targets[index], dtype=np.float32) / 255)
        flip_x, flip_y, transpose, gain, gamma = torch.rand(5).tolist()
        if flip_x < 0.5:
            inputs, targets = inputs.flip(-1), targets.flip(-1)
        if flip_y < 0.5:
            inputs, targets = inputs.flip(-2), targets.flip(-2)
        if transpose < 0.5:
            inputs, targets = inputs.transpose(-1, -2), targets.transpose(-1, -2)
        inputs[0] = (inputs[0] ** (0.8 + 0.45 * gamma) * (0.85 + 0.3 * gain)).clamp(0, 1)
        return inputs, targets, self.records[index]


def refiner_loss(masks: Tensor, scores: Tensor, targets: Tensor, n_records: Tensor) -> Tensor:
    valid = (torch.arange(MAX_RECORDS, device=targets.device)[None] < n_records[:, None]).float()
    soft = (targets * valid[:, :, None, None]).sum(1, keepdim=True) / n_records[:, None, None, None].float()
    logits = masks.float()
    probabilities = torch.sigmoid(logits)
    dice = 1 - (2 * (probabilities * soft).sum((1, 2, 3)) + 1) / (probabilities.sum((1, 2, 3)) + soft.sum((1, 2, 3)) + 1)
    with torch.no_grad():
        predicted, truth = probabilities >= 0.5, targets >= 0.5
        union = (predicted | truth).sum((2, 3)).float()
        iou = (predicted & truth).sum((2, 3)).float() / union.clamp(min=1)
        earned = (torch.where(iou > 0.5, iou, torch.zeros_like(iou)) * valid).sum(1) / n_records.float()
    score = F.binary_cross_entropy_with_logits(scores.float(), earned)
    return F.binary_cross_entropy_with_logits(logits, soft) + dice.mean() + REFINER["score_weight"] * score


# %% [markdown]
# One refiner per fold, trained on the proposals of the other folds.


# %%
def train_refiner(fold: int, meta: dict) -> Path:
    """Train the refiner of one fold on the samples of the other folds; a finished one is reused."""
    path = REFINER_DIR / f"refiner-fold{fold}.pt"
    if path.is_file():
        return path
    set_seeds(SEED)
    indices = [index for index, sample_fold in enumerate(meta["folds"]) if sample_fold != fold]
    loader = DataLoader(
        RefinerSamples(indices, meta["records"]),
        batch_size=REFINER["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=WORKERS,
        persistent_workers=WORKERS > 0,
        multiprocessing_context="fork" if WORKERS > 0 else None,
    )
    model = Refiner(encoder_weights="imagenet").to(DEVICE)
    encoder_ids = {id(param) for param in model.encoder.parameters()}
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [p for p in model.parameters() if id(p) in encoder_ids],
                "lr": REFINER["lr"] * REFINER["encoder_lr_scale"],
            },
            {"params": [p for p in model.parameters() if id(p) not in encoder_ids], "lr": REFINER["lr"]},
        ],
        weight_decay=TRAINING["weight_decay"],
    )
    scheduler = build_scheduler(optimizer, REFINER["epochs"], len(loader), warmup_epochs=1)
    for epoch in range(1, REFINER["epochs"] + 1):
        model.train()
        total = 0.0
        for inputs, targets, n_records in tqdm(loader, desc=f"refiner fold {fold} epoch {epoch}", leave=False, mininterval=2.0):
            inputs, targets, n_records = inputs.to(DEVICE), targets.to(DEVICE), n_records.to(DEVICE)
            with torch.autocast(DEVICE.type, dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
                masks, scores = model(inputs)
            loss = refiner_loss(masks, scores, targets, n_records)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += loss.item()
        print(f"refiner fold {fold} epoch {epoch}/{REFINER['epochs']}  loss {total / len(loader):.4f}")
    torch.save({"state_dict": model.state_dict(), "fold": fold}, path)
    return path


# %% [markdown]
# Refiner calibration: the refined mask threshold and the predicted-IoU cut, chosen on pooled out-of-fold PQ.


# %%
def score_refined_one(job) -> dict:
    """:return: PQ terms of one refined observation per (mask threshold, min_score) of REFINE_GRID"""
    refined, records = job
    gt = [annotation_pixels(record) for record in records]
    terms = {}
    for threshold in REFINE_GRID["threshold"]:
        labels, n_labels, scores = assemble(refined, threshold)
        overlaps = [instance_iou(pixels, labels, n_labels) for pixels in gt]
        for min_score in REFINE_GRID["min_score"]:
            keep = scores >= min_score
            terms[(threshold, min_score)] = sum(pq_terms(iou[:, keep]) for iou in overlaps)
    return terms


def calibrate_refiner(refiner_paths, calibration: dict) -> dict:
    """
    Refine every out-of-fold map with the refiner of its fold and choose the two cuts on pooled PQ.
    :param refiner_paths: Refiner checkpoint per fold
    :param calibration: The ensemble's calibration, whose threshold and offset cut the proposals
    :return: The ``refiner`` block of the calibration
    """
    folds, files = fold_of_file(), records_of_file()
    accumulated: dict = {}
    with ProcessPoolExecutor(WORKERS, mp_context=multiprocessing.get_context("fork")) as pool:
        pending = []
        for fold, path in enumerate(refiner_paths):
            refiners = [load_refiner(path)]
            for name in tqdm([name for name in sorted(files) if folds[name] == fold], desc=f"refining fold {fold}"):
                probabilities = np.load(OOF_DIR / f"{Path(name).stem}.npy").astype(np.float32) / 255
                image = load_image(TRAIN_IMAGES / name).numpy()[0]
                proposals = apply_offset(probabilities >= calibration["threshold"], calibration["offset"])
                labels, n_labels = connected_components(proposals)
                refined = refine_components(refiners, image, probabilities, labels, n_labels)
                pending.append(pool.submit(score_refined_one, (refined, files[name])))
        for future in tqdm(pending, desc="scoring"):
            for key, value in future.result().items():
                accumulated[key] = accumulated.get(key, 0) + value
    (threshold, min_score), terms = max(accumulated.items(), key=lambda item: panoptic_quality(item[1]))
    return {
        "checkpoints": [str(path) for path in refiner_paths],
        "proposal_threshold": calibration["threshold"],
        "proposal_offset": calibration["offset"],
        "threshold": threshold,
        "min_score": min_score,
        "pq": panoptic_quality(terms),
    }


# %% [markdown]
# ## Barbs
#
# The models draw the body of a filament well but rarely its barbs, the thin dark offshoots the annotators draw only
# sometimes. Around every predicted instance, pixels darker than the local background (a median over ~60 px) that
# also lie on a thin dark ridge (Sato filter) are candidates; the long, narrow pieces touching the instance are added
# to that same instance. No labels are involved. Out of fold this adds 49k pixels and costs 0.0024 PQ (0.4692 to
# 0.4668), since the annotators leave most barbs out.


# %%
# contrast and ridge cut the candidates; a piece is kept when length / mean width >= min_ratio and width <= max_width
DEFAULT_BARBS = {"contrast": 0.05, "ridge": 0.025, "band": 20, "min_ratio": 4.0, "max_width": 4.0}


def local_contrast(image: np.ndarray) -> np.ndarray:
    """(background - image) / background, smoothed; the background is a median taken on a 4x smaller copy."""
    height, width = image.shape
    rows, columns = -(-height // 4) * 4, -(-width // 4) * 4
    padded = np.pad(image.astype(np.float32), ((0, rows - height), (0, columns - width)), mode="edge")
    small = padded.reshape(rows // 4, 4, columns // 4, 4).mean((1, 3))
    background = np.kron(ndimage.median_filter(small, 15), np.ones((4, 4), np.float32))[:height, :width]
    contrast = np.where(background > 20 / 255, (background - image) / np.maximum(background, 1e-6), 0)
    return ndimage.gaussian_filter(contrast.astype(np.float32), 1)


def thin_pieces(pieces: np.ndarray, mask: np.ndarray, min_ratio: float, max_width: float) -> np.ndarray:
    """The long, narrow pieces, measured from where they leave the mask."""
    labels, n_labels = ndimage.label(pieces, structure=EIGHT_CONNECTED)
    distance = ndimage.distance_transform_edt(~mask)
    keep = np.zeros(n_labels + 1, dtype=bool)
    for number, window in enumerate(ndimage.find_objects(labels), start=1):
        if window is None:
            continue
        points = np.argwhere(labels[window] == number) + np.array([window[0].start, window[1].start])
        base = points[np.argmin(distance[points[:, 0], points[:, 1]])]
        length = np.sqrt(((points - base) ** 2).sum(1).max()) + 1
        width = len(points) / length
        keep[number] = length / width >= min_ratio and width <= max_width
    return keep[labels]


def grow_barbs(labels: np.ndarray, n_labels: int, image: np.ndarray, settings: dict) -> np.ndarray:
    """Add the visible barbs of every instance to it; grown pixels only fill background."""
    settings = {**DEFAULT_BARBS, **settings}
    if n_labels == 0:
        return labels
    contrast = local_contrast(image)
    grown = labels.copy()
    margin = settings["band"] + 2
    height, width = labels.shape
    for number, window in enumerate(ndimage.find_objects(labels), start=1):
        if window is None:
            continue
        window = (
            slice(max(window[0].start - margin, 0), min(window[0].stop + margin, height)),
            slice(max(window[1].start - margin, 0), min(window[1].stop + margin, width)),
        )
        mask = labels[window] == number
        local = contrast[window]
        ridge = sato(local, sigmas=[1, 2], black_ridges=False)
        band = ndimage.binary_dilation(mask, disk(int(settings["band"]))) & ~mask
        candidates = band & (local > settings["contrast"]) & (ridge > settings["ridge"])
        pieces, _ = ndimage.label(candidates, structure=EIGHT_CONNECTED)
        touching = np.unique(pieces[ndimage.binary_dilation(mask, EIGHT_CONNECTED) & candidates])
        barbs = thin_pieces(np.isin(pieces, touching[touching > 0]), mask, settings["min_ratio"], settings["max_width"])
        target = grown[window]
        target[barbs & (target == 0)] = number
    return grown


# %% [markdown]
# ## Models: train or fetch
#
# With `TRAIN = False` the `PRESET` folder is found (see *Running it*), every checkpoint its calibration lists is
# resolved inside it, and when it ships a `manifest.json` each file is checked against its SHA-256.


# %%
def find_weights() -> Path:
    """
    :return: The folder holding CALIBRATION_NAME and its checkpoints: WEIGHTS_DIR, the repository's weights/final,
        an attached Kaggle input, or a kagglehub download of MODEL_HANDLE
    """
    if WEIGHTS_DIR is not None:
        return Path(WEIGHTS_DIR)
    if (REPO_WEIGHTS / CALIBRATION_NAME).is_file():
        return REPO_WEIGHTS
    attached = Path("/kaggle/input")
    found = next(attached.rglob(CALIBRATION_NAME), None) if attached.is_dir() else None
    if found is not None:
        return found.parent
    import kagglehub

    return Path(kagglehub.model_download(MODEL_HANDLE))


def resolve(weights: Path, path: str) -> Path:
    """
    A checkpoint the calibration lists, inside the weights folder.
    :param weights: Folder of the preset
    :param path: The calibration's entry: weights/final/<file> for a release, runs/<sweep>/<run>/<stem>.pt or
        runs/refiner/<set>/fold<k>.pt for an older one (published as <run>-<stem>.pt, refiner-fold<k>.pt)
    :return: The file
    """
    source = Path(path)
    names = (source.name, f"{source.parent.name}-{source.stem}.pt", f"refiner-{source.stem}.pt")
    found = next((weights / name for name in names if (weights / name).is_file()), None)
    assert found is not None, f"{weights} holds none of {names} for {path}"
    return found


def verify(weights: Path, paths) -> None:
    """Check the files against the release's manifest.json, when there is one: a truncated or stale download fails."""
    manifest = weights / "manifest.json"
    if not manifest.is_file():
        return
    expected = {entry["file"]: entry["sha256"] for entry in json.loads(manifest.read_text(encoding="utf-8"))["files"]}
    for path in paths:
        if path.name not in expected:
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        assert digest.hexdigest() == expected[path.name], f"{path} does not match {manifest.name}"


# %%
if TRAIN:
    SSL_START = pretrain_ssl(SSL_FAMILY) if SSL else None
    print(f"{SSL_FAMILY} starts from {'masked-image pretraining' if SSL_START else 'ImageNet'}")
    checkpoints = []
    for family in FAMILIES:
        for fold in range(FOLDS):
            first = train_run(family, "stage1", fold)
            checkpoints.append(train_run(family, "stage2", fold, init=first))
    calibration = calibrate(checkpoints)
    meta = collect_refiner_samples(calibration["offset"])  # the refiners are part of TRAIN
    refiner_paths = [train_refiner(fold, meta) for fold in range(FOLDS)]
    calibration["refiner"] = calibrate_refiner(refiner_paths, calibration)
    (WORK / CALIBRATION_NAME).write_text(json.dumps(calibration, indent=2), encoding="utf-8")
else:
    weights = find_weights()
    print(f"weights from {weights}")
    calibration = json.loads((weights / CALIBRATION_NAME).read_text(encoding="utf-8"))
    checkpoints = [resolve(weights, path) for path in calibration["checkpoints"]]  # in the calibration's order
    if calibration.get("refiner"):
        calibration["refiner"]["checkpoints"] = [str(resolve(weights, path)) for path in calibration["refiner"]["checkpoints"]]
    verify(weights, [*checkpoints, *map(Path, (calibration.get("refiner") or {}).get("checkpoints", []))])
    preset = calibration.get("preset") or {}
    if preset:
        print(f"preset {preset['name']}: {preset['summary']}; out-of-fold PQ {preset['oof_pq_without_barbs']:.4f} without barbs")


# %% [markdown]
# The released presets side by side, from the release's `manifest.json`: how much each step up in inference cost buys
# in out-of-fold PQ, and what the barbs cost.


# %%
manifest_path = weights / "manifest.json" if not TRAIN else None
if manifest_path is not None and manifest_path.is_file():
    released = pd.DataFrame(json.loads(manifest_path.read_text(encoding="utf-8"))["presets"]).T
    released = released.dropna(subset=["oof_pq", "seconds_per_image"])
    released = released.astype({"oof_pq": float, "oof_pq_without_barbs": float, "seconds_per_image": float})
    print(released[["oof_pq", "oof_pq_without_barbs", "public_lb", "seconds_per_image"]].round(4).to_string())
    axs = kx.grid(2)
    kx.plot(
        released.rename(columns={"seconds_per_image": "seconds per image", "oof_pq": "out-of-fold PQ"}),
        "seconds per image",
        "out-of-fold PQ",
        "scatter",
        ax=axs[0],
        title="presets: out-of-fold PQ (y) against seconds per image (x)",
    )
    for name, row in released.iterrows():
        label = f"{name}\nLB {float(row['public_lb']):.2f}" if row["public_lb"] == row["public_lb"] else name
        axs[0].annotate(label, (row["seconds_per_image"], row["oof_pq"]), textcoords="offset points", xytext=(6, -14), fontsize=8)
    axs[0].set_xlim(0, released["seconds_per_image"].max() * 1.2)
    kx.plot(
        released.reset_index(names="preset").rename(columns={"oof_pq": "with barbs", "oof_pq_without_barbs": "without barbs"}),
        "preset",
        "with barbs",
        "without barbs",
        spec="bar-leg",
        ax=axs[1],
        title="out-of-fold PQ with and without barbs (axis from 0.44)",
    )
    axs[1].set_ylim(0.44, 0.475)
    figure("presets")


# %% [markdown]
# The training curves, when the models were trained here: the held-out PQ proxy of every fold, stage 2 (from epoch 41)
# after stage 1.


# %%
if TRAIN:
    axs = kx.grid(len(FAMILIES))
    for ax, family in zip(axs, FAMILIES):
        for fold in range(FOLDS):
            epochs, pq = [], []
            for stage, shift in (("stage1", 0), ("stage2", STAGES["stage1"]["epochs"])):
                history_path = run_dir(family, stage, fold) / "history.json"
                for row in json.loads(history_path.read_text()) if history_path.is_file() else []:
                    epochs.append(row["epoch"] + shift)
                    pq.append(row["pq"])
            if epochs:
                curve = pd.DataFrame({"epoch": epochs, f"fold {fold}": pq})
                kx.plot(curve, "epoch", f"fold {fold}", spec="line-leg", ax=ax, title=f"{family}: held-out PQ (proxy)")
    figure("training")

if TRAIN and SSL_START is not None:
    history = pd.DataFrame(json.loads((SSL_START.parent / "history.json").read_text(encoding="utf-8")))
    kx.plot(history, "epoch", "loss", "line", title="masked-image pretraining: L1 on the hidden squares")
    figure("pretraining")


# %% [markdown]
# What is about to run.


# %%
assert len(checkpoints) == len(calibration["checkpoints"]), (
    f"{len(checkpoints)} checkpoints, calibration made for {len(calibration['checkpoints'])}"
)
print(
    f"{len(checkpoints)} checkpoints; out-of-fold PQ {calibration['pq']:.4f} at threshold {calibration['threshold']}, "
    f"offset {calibration['offset']:+d}, min_area {calibration['min_area']}, min_score {calibration['min_score']}"
)
REFINE = bool(calibration.get("refiner", {}).get("checkpoints"))  # the preset has a crop refiner (TRAIN always trains one)
if REFINE:
    refiner_settings = calibration["refiner"]
    print(
        f"{len(refiner_settings['checkpoints'])} crop refiners; out-of-fold PQ {refiner_settings['pq']:.4f} with refined "
        f"masks at {refiner_settings['threshold']} and predicted IoU >= {refiner_settings['min_score']}"
    )
else:
    print("no crop refiner: instances are cut and filtered by the calibration above")

# %% [markdown]
# ## Predict the test set


# %%
def encode_rle(mask: np.ndarray) -> str:
    """Compressed COCO RLE counts of a 2048 x 2048 mask, as the submission expects them."""
    return coco_mask.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"].decode("ascii")


def parse_variant(name: str) -> dict:
    """Inference variant of a calibration: base, or '-'-joined o<overlap> and turned (e.g. o0.5, turned)."""
    settings = {"overlap": OVERLAP, "turned": False}
    for part in [] if name == "base" else name.split("-"):
        if part == "turned":
            settings["turned"] = True
        elif part.startswith("o"):
            settings["overlap"] = float(part[1:])
        else:
            raise ValueError(f"unknown part {part!r} of inference variant {name!r}")
    return settings


# %% [markdown]
# The inference settings come from the calibration: its variants are averaged per model, float16 autocast when it
# says so, and its barbs block (or `BARBS`).


# %%
variants = [parse_variant(name) for name in calibration.get("variants") or ["base"]]
# float16 autocast when the calibration asks for it; the cached out-of-fold maps were float32, the masks match
half = calibration.get("precision") == "fp16" and DEVICE.type == "cuda"
skip_dark = bool(calibration.get("skip_dark"))
barbs = calibration.get("barbs") if BARBS is None else ((calibration.get("barbs") or {}) if BARBS else None)
print("barbs grown into the instances" if barbs is not None else "no barbs grown")
models, refiners = [], []


def load_inference() -> None:
    """Load the ensemble and the refiners on first use, so a finished prediction run does not need them."""
    if models:
        return
    for path in checkpoints:
        model, meta = load_model(path)
        models.append((model, meta.get("input_scale") or 1.0))  # the zoom family predicts at 2x
    if REFINE:
        refiners.extend(load_refiner(path) for path in refiner_settings["checkpoints"])
    print(f"{len(models)} models x {len(variants)} variants, {len(refiners)} refiners, {'fp16' if half else 'fp32'}")


# %% [markdown]
# One observation through the pipeline, keeping every stage the figures below show: the ensemble's probabilities
# and how much the models disagree, the proposals the refiner starts from, the refined instances, and the final
# instances with their barbs. The refiner's predicted IoU is each instance's confidence (the mean probability
# without a refiner).


# %%
def infer(image: Tensor) -> dict:
    """:return: The stages of one observation: probabilities, spread, proposals, refined, labels, n_labels, scores, candidates"""
    load_inference()
    with torch.autocast(DEVICE.type, dtype=torch.float16, enabled=half):
        maps = [
            predict_tiled(model, image, scale=scale, skip_dark=skip_dark, **variant)
            for model, scale in models
            for variant in variants
        ]
        probabilities = (sum(maps) / len(maps)).numpy()
        refined = refine_observation(refiners, image.numpy()[0], probabilities, refiner_settings) if REFINE else None
    if refined is not None:
        labels, n_labels, details = refined
    else:
        labels, n_labels = label_instances(
            probabilities, calibration["threshold"], calibration["offset"], calibration["min_area"], calibration["min_score"]
        )
        mean = ndimage.mean(probabilities, labels, range(1, n_labels + 1)) if n_labels else []
        details = {"proposals": labels, "kept": np.asarray(mean, float), "candidates": np.asarray(mean, float)}
    per_model = [sum(maps[i : i + len(variants)]) / len(variants) for i in range(0, len(maps), len(variants))]
    result = {
        "probabilities": probabilities,
        "spread": torch.stack(per_model).std(dim=0).numpy(),  # standard deviation between the models
        "refined": labels,
        "n_labels": n_labels,
        **details,
    }
    if barbs is not None:
        labels = grow_barbs(labels, n_labels, image.numpy()[0], barbs)
    result["labels"] = labels
    return result


# %% [markdown]
# Every test observation: averaged probabilities, refined instances, barbs, one row per filament. The first `SHOW`
# observations are kept for the figures, and so are the extremes: the most and least confident observation (mean
# confidence of its filaments), the one with the most filaments, and one without any.


# %%
SHOW = 2
RERUN = False  # True: predict again even when a finished prediction of this PRESET is in WORK
SUBMISSION = WORK / "submission.csv"
SUMMARY = WORK / "inference.json"  # what the figures need from a finished prediction, next to the submission
RUN_TAG = "trained" if TRAIN else PRESET
examples, extremes, summary = [], {}, []


def roles_of(n: int, confidence) -> list:
    """:return: (role, value) pairs: an observation holds a role when it has the highest value so far"""
    return [
        ("most confident", confidence),
        ("least confident", None if confidence is None else -confidence),
        ("most filaments", n),
        ("no filaments", 0 if n == 0 else None),
    ]


def remember(name: str, image: np.ndarray, result: dict, count: bool = True) -> None:
    """Keep the stages of an observation when it is one of the first SHOW or sets an extreme."""
    n = result["n_labels"]
    confidence = float(result["kept"].mean()) if n else None
    if count:
        summary.append({"name": name, "filaments": n, "confidence": confidence, "candidates": result["candidates"]})
    entry = {"name": name, "image": image, **result}
    if len(examples) < SHOW and name in names_shown:
        examples.append(entry)
    for role, value in roles_of(n, confidence):
        if value is not None and (role not in extremes or value > extremes[role][0]):
            extremes[role] = (value, entry)


def finished_run() -> dict | None:
    """:return: The summary of a finished prediction of this PRESET over these test files, or None"""
    if RERUN or not (SUBMISSION.is_file() and SUMMARY.is_file()):
        return None
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    return data if data["tag"] == RUN_TAG and data["files"] == [path.stem for path in TEST_FILES] else None


names_shown = {path.stem for path in TEST_FILES[:SHOW]}
finished = finished_run()
started = time.time()
if finished is None:
    rows = []
    for path in tqdm(TEST_FILES, desc="predict", mininterval=2.0):
        image = load_image(path)
        result = infer(image)
        labels = result["labels"]
        rows += [(f"{path.stem}_{label}", encode_rle(labels == label)) for label in range(1, result["n_labels"] + 1)]
        remember(path.stem, image.numpy()[0], result)
    seconds = time.time() - started
else:
    # no new prediction: the rows come from the submission, and only the few observations the figures show are run again
    print(f"{SUBMISSION} is a finished {RUN_TAG} prediction; not predicted again (RERUN = True to override)")
    with open(SUBMISSION, encoding="utf-8", newline="") as handle:
        rows = [tuple(line.rstrip("\r\n").split(",", 1)) for line in handle][1:]
    seconds = finished["seconds"]
    summary = [{**row, "candidates": np.asarray(row["candidates"], float)} for row in finished["summary"]]
    best: dict = {}
    for row in summary:
        for role, value in roles_of(row["filaments"], row["confidence"]):
            if value is not None and (role not in best or value > best[role][0]):
                best[role] = (value, row["name"])
    figure_names = names_shown | {name for _, name in best.values()}
    for path in TEST_FILES:
        if path.stem in figure_names:
            image = load_image(path)
            remember(path.stem, image.numpy()[0], infer(image), count=False)


# %% [markdown]
# The submission: one row per filament, its compressed RLE counts unquoted.


# %%
submission = SUBMISSION
if finished is None:
    with open(submission, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, quoting=csv.QUOTE_NONE, quotechar=None)
        writer.writerow(["filament_id", "segmentation_rle"])
        writer.writerows(rows)
    state = {"tag": RUN_TAG, "files": [path.stem for path in TEST_FILES], "seconds": seconds}
    state["summary"] = [{**row, "candidates": [float(x) for x in row["candidates"]]} for row in summary]
    SUMMARY.write_text(json.dumps(state), encoding="utf-8")

observations = {row[0].rsplit("_", 1)[0] for row in rows}
assert observations <= {path.stem for path in TEST_FILES}
assert all(coco_mask.area({"size": [SUBMISSION_SIZE] * 2, "counts": counts.encode()}) for _, counts in rows)
print(f"{len(rows)} filaments over {len(observations)}/{len(TEST_FILES)} observations -> {submission}")
print(
    f"{seconds:.0f} s{' (first run)' if finished else ''}, {seconds / len(TEST_FILES):.1f} s per observation on "
    f"{torch.cuda.get_device_name() if DEVICE.type == 'cuda' else 'cpu'}"
)


# %% [markdown]
# ## Figures of the test predictions
#
# How many filaments each test observation got and how big they are; then the first observations whole, with the
# submitted outlines.


# %%
if rows:
    found = pd.Series([name.rsplit("_", 1)[0] for name, _ in rows]).value_counts()
    per_observation = found.reindex([path.stem for path in TEST_FILES], fill_value=0).to_numpy()
    sizes = np.array([coco_mask.area({"size": [SUBMISSION_SIZE] * 2, "counts": counts.encode()}) for _, counts in rows], float)
    axs = kx.grid(2)
    kx.plot(
        per_observation,
        spec="hist-median-ann",
        ax=axs[0],
        bins=np.arange(per_observation.max() + 2) - 0.5,
        title="filaments per test observation",
    )
    kx.plot(
        sizes,
        spec="hist-logx-median-ann",
        ax=axs[1],
        bins=np.geomspace(sizes.min(), sizes.max(), 40),
        title="submitted filament area (px)",
    )
    figure("submission")

    boxes = np.array([coco_mask.toBbox({"size": [SUBMISSION_SIZE] * 2, "counts": counts.encode()}) for _, counts in rows])
    submitted = np.c_[boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3] / 2]
    corners = [np.asarray(a["segmentation"][0]).reshape(-1, 2) for a in raw["annotations"]]
    drawn = np.array([(points.min(axis=0) + points.max(axis=0)) / 2 for points in corners])
    axs = kx.grid(2)
    for ax, points, title in (
        (axs[0], drawn, "annotated filaments (training)"),
        (axs[1], submitted, "submitted filaments (test)"),
    ):
        positions = pd.DataFrame({"x (px)": points[:, 0], "y (px, north up)": SUBMISSION_SIZE - points[:, 1]})
        kx.plot(
            positions,
            "x (px)",
            "y (px, north up)",
            "hexbin-eq",
            ax=ax,
            title=title,
            gridsize=24,
            extent=(0, 2048, 0, 2048),
            mincnt=1,
        )
    figure("filament-positions")

if examples:
    kx.plot([example["image"] for example in examples], [example["labels"] > 0 for example in examples], "segmask")
    figure("test-observations")


# %% [markdown]
# The ranked observations whole, with the submitted outlines. Without ground truth, "best" and "worst" mean the
# model's own confidence, the refiner's predicted IoU averaged over the observation's filaments.


# %%
roles = [role for role in ("most confident", "least confident", "most filaments", "no filaments") if role in extremes]
for role in roles:
    entry = extremes[role][1]
    kx.plot([entry["image"]], [entry["labels"] > 0], "segmask", title=f"{role}: {entry['name']}, {entry['n_labels']} filaments")
    figure(f"ranked-{role.replace(' ', '-')}")


# %% [markdown]
# Confidence of every candidate the refiner scored, against the cut that decides which are kept, and each
# observation's filament count against its mean confidence.


# %%
cut = refiner_settings["min_score"] if REFINE else calibration["min_score"]
candidates = np.concatenate([row["candidates"] for row in summary] or [np.zeros(0)])
axs = kx.grid(2)
split = pd.DataFrame(
    {f"kept (>= {cut:g})": pd.Series(candidates[candidates >= cut]), "dropped": pd.Series(candidates[candidates < cut])}
)
kx.plot(split, split.columns[0], "dropped", "hist-leg", ax=axs[0], bins=np.linspace(0, 1, 41), title="candidate confidence")
scored = pd.DataFrame(summary).dropna(subset=["confidence"])[["filaments", "confidence"]].astype(float)
kx.plot(scored, "filaments", "confidence", "scatter", ax=axs[1], title="observations: filaments and mean confidence")
figure("confidence")


# %% [markdown]
# Close-ups of the (up to) three largest filaments of an observation. kx-plot's segmask draws the first mask solid
# (legend `true`) and the second dashed (`predicted`); here those are the earlier and the later stage, and the IoU
# between them heads each row (title: observation and instance numbers). Rows: the ensemble's cut and the refined
# instance.


# %%
def largest(labels: np.ndarray, count: int = 3, key=None) -> tuple[list, list]:
    """:return: The ids of the `count` instances with the biggest key (area by default) and their windows"""
    size = np.bincount(labels.ravel() if key is None else key.ravel(), minlength=labels.max() + 1)[1:]
    boxes = ndimage.find_objects(labels)
    ids = [int(k) + 1 for k in np.argsort(size)[::-1][:count] if size[k] and boxes[k]]
    return ids, [component_window(boxes[k - 1]) for k in ids]


def compare(entry: dict, ids: list, windows: list, before: np.ndarray, after: np.ndarray, title: str, figure_name: str) -> None:
    """One row per filament: the image, the `before` mask (solid outline) and the `after` mask (dashed); the IoU between them heads the row."""
    kx.plot(
        [window_of(entry["image"], window) for window in windows],
        [window_of(before, window) for window in windows],
        [window_of(after, window) for window in windows],
        "segmask",
        title=f"{entry['name']} {ids}",
    )
    figure(figure_name)


for role in ("most filaments", "least confident"):
    if role in extremes:
        entry = extremes[role][1]
        ids, windows = largest(entry["labels"])
        if ids:
            compare(
                entry,
                ids,
                windows,
                entry["proposals"] > 0,
                entry["refined"] > 0,
                "ensemble cut (solid) vs refined (dashed)",
                f"refinement-{role.replace(' ', '-')}",
            )


# %% [markdown]
# The barbs: refined (`true`, solid) against with barbs (`predicted`, dashed), on the filaments that gained the most
# pixels.


# %%
if barbs is not None:
    entry = max(extremes.values(), key=lambda item: int(((item[1]["labels"] > 0) & (item[1]["refined"] == 0)).sum()))[1]
    added = (entry["labels"] > 0) & (entry["refined"] == 0)
    ids, windows = largest(entry["labels"], key=np.where(added, entry["labels"], 0))
    if ids:
        compare(entry, ids, windows, entry["refined"] > 0, entry["labels"] > 0, "refined (solid) vs with barbs (dashed)", "barbs")
        print(f"{entry['name']}: {int(added.sum())} barb pixels added over {entry['n_labels']} filaments")


# %% [markdown]
# One observation through every stage: the image, the ensemble's probability, how much its models disagree (the
# standard deviation between the models, large where thin filaments are found by some and missed by others), the
# proposals, the refined instances and the final ones with barbs. In the second figure the first row compares the
# proposals (`true`, solid) with the refined instances (`predicted`, dashed), the second row the refined instances with
# the final ones.


# %%
entry = extremes["most filaments"][1] if "most filaments" in extremes else examples[0]
kx.plot(
    [entry["image"], entry["image"]],
    [entry["probabilities"], entry["spread"]],
    ["ensemble probability", "model disagreement (std)"],
    "saliency",
    title=f"{entry['name']}",
)
figure("pipeline-probabilities")
kx.plot(
    [entry["image"], entry["image"]],
    [entry["proposals"] > 0, entry["refined"] > 0],
    [entry["refined"] > 0, entry["labels"] > 0],
    "segmask",
    title=f"{entry['name']}: proposals to refined to final",
)
figure("pipeline-instances")


# %% [markdown]
# ## Self-evaluation on training observations
#
# The organisers' self-evaluation notebook scores a prediction file against the ground truth. It draws the IoU and
# Dice distributions of every (annotated, predicted) pair, and the many-to-one and one-to-many matches. Here the same
# is done for `SELF_EVAL` training observations run through the pipeline above.
#
# The Panoptic Quality is the organisers' formula. It was checked against their notebook on 40 observations with
# perturbed annotations, and the two agree to 1e-9.
#
# **The released models were trained on these observations**, so the scores are optimistic. They show that the
# pipeline and the scoring work, not how well the model generalises. For that, see the out-of-fold PQ above.


# %%
SELF_EVAL = 10
rng = random.Random(0)
chosen = rng.sample(sorted({record["file_name"] for record in RECORDS.values()}), min(SELF_EVAL, len(RECORDS)))
SELF_CACHE = WORK / f"self_evaluation-{RUN_TAG}.json"  # labels and overlaps per observation, reused unless RERUN
done = json.loads(SELF_CACHE.read_text(encoding="utf-8")) if SELF_CACHE.is_file() and not RERUN else {}
scored_runs = []  # per observation: name, image, final labels, union of the annotated filaments, IoU matrix per record
for name in tqdm(chosen, desc="self-evaluation", mininterval=2.0):
    image = load_image(TRAIN_IMAGES / name)
    records = [image_id for image_id, record in RECORDS.items() if record["file_name"] == name]
    if name not in done:
        result = infer(image)
        done[name] = {
            "n_labels": result["n_labels"],
            "rle": [encode_rle(result["labels"] == k) for k in range(1, result["n_labels"] + 1)],
        }
    labels = np.zeros((SUBMISSION_SIZE, SUBMISSION_SIZE), np.int16)
    for k, counts in enumerate(done[name]["rle"], start=1):
        labels[coco_mask.decode({"size": [SUBMISSION_SIZE] * 2, "counts": counts.encode()}).astype(bool)] = k
    matrices = [instance_iou(annotation_pixels(BY_RECORD[i]), labels, done[name]["n_labels"]) for i in records]
    truth = np.zeros((SUBMISSION_SIZE, SUBMISSION_SIZE), bool)
    for i in records:
        for annotation in BY_RECORD[i]:
            truth |= polygon_mask(annotation["segmentation"][0], SUBMISSION_SIZE, SUBMISSION_SIZE)
    scored_runs.append({"name": name, "image": image.numpy()[0], "labels": labels, "truth": truth, "matrices": matrices})
SELF_CACHE.write_text(json.dumps(done), encoding="utf-8")

per_observation_terms = [sum(pq_terms(matrix) for matrix in run["matrices"]) for run in scored_runs]
total = sum(per_observation_terms)


def strict_terms(iou: np.ndarray) -> np.ndarray:
    """PQ terms with one-to-one matching: each annotated and each predicted filament takes part in at most one match."""
    from scipy.optimize import linear_sum_assignment

    rows, columns = linear_sum_assignment(-(iou * (iou > 0.5)))
    hit = iou[rows, columns] > 0.5
    return np.array([iou[rows, columns][hit].sum(), hit.sum(), iou.shape[1] - hit.sum(), iou.shape[0] - hit.sum()], float)


strict = sum(strict_terms(matrix) for run in scored_runs for matrix in run["matrices"])
print(
    f"strict one-to-one PQ {panoptic_quality(strict):.4f}: equal to the PQ below unless a prediction matches two annotated filaments"
)
print(
    f"PQ {panoptic_quality(total):.4f} over {len(scored_runs)} training observations (in-sample); TP {total[1]:.0f}, FP {total[2]:.0f}, FN {total[3]:.0f}"
)


# %% [markdown]
# Distributions of the overlap of every annotated filament with every predicted one that touches it (pairs with no
# overlap are left out): IoU, then Dice (= 2 IoU / (1 + IoU)). A pair counts as a match above IoU 0.5.


# %%
iou_pairs = np.concatenate([matrix.ravel() for run in scored_runs for matrix in run["matrices"]] or [np.zeros(0)])
iou_pairs = iou_pairs[iou_pairs > 0]
axs = kx.grid(2)
for ax, scores, title in ((axs[0], iou_pairs, "IoU"), (axs[1], 2 * iou_pairs / (1 + iou_pairs), "Dice")):
    kx.plot(scores, spec="hist-mean-ann", ax=ax, bins=np.linspace(0, 1, 51), title=f"{title} distribution")
figure("self-evaluation-overlap")


# %% [markdown]
# Many-to-one and one-to-many: how many predictions touch each annotated filament (`annotated` bars) and how many
# annotated filaments each prediction touches (`predicted` bars). 0 touched are missed filaments and predictions
# with no filament under them; 1 is the clean one-to-one case.


# %%
gt_degree = np.concatenate([(matrix > 0).sum(axis=1) for run in scored_runs for matrix in run["matrices"]] or [np.zeros(0, int)])
pred_degree = np.concatenate(
    [(matrix > 0).sum(axis=0) for run in scored_runs for matrix in run["matrices"]] or [np.zeros(0, int)]
)
cases = ["0", "1", "2", "3+"]


def tally(degrees: np.ndarray) -> list:
    return [int((degrees == k).sum()) for k in range(3)] + [int((degrees >= 3).sum())]


matches = pd.DataFrame({"touching": cases, "annotated": tally(gt_degree), "predicted": tally(pred_degree)})
kx.plot(
    matches,
    "touching",
    "annotated",
    "predicted",
    "bar-ann-leg",
    title="how many predictions touch an annotated filament, and how many annotated ones a prediction (0 = missed, spurious)",
)
figure("self-evaluation-matches")


# %% [markdown]
# The best and the worst of these observations by their own Panoptic Quality, with the annotated filaments (any
# annotator, `true`, solid) against the submitted ones (`predicted`, dashed); the IoU of the two unions heads the row.


# %%
quality = [panoptic_quality(terms) for terms in per_observation_terms]
order = np.argsort(quality)
for label, index in (("worst", int(order[0])), ("best", int(order[-1]))):
    run = scored_runs[index]
    print(f"{label}: {run['name']}")
    kx.plot(
        [run["image"]],
        [run["truth"]],
        [run["labels"] > 0],
        "segmask",
        title=f"{label}: PQ {quality[index]:.3f}",
    )

    figure(f"self-evaluation-{label}")

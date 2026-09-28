"""
Loading of the MAGFiLO 1.0 solar filament dataset.
"""

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from pycocotools import mask as coco_mask
from torch.utils.data import Dataset

ANNOTATIONS_NAME = "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
REPO_DATA = Path(__file__).resolve().parents[2] / "data"
KAGGLE_INPUT = Path("/kaggle/input")

CATEGORY_NAMES = {1: "Left", 2: "Right", 3: "Unidentifiable", 4: "Ambiguous"}

RASTERS = ("coco", "pil")
TARGETS = ("each", "mean")


def candidate_roots() -> list[Path]:
    """
    Directories that might hold the dataset, most specific first.

    :return: Roots to search, in the order they are tried
    """
    roots = []
    override = os.environ.get("FILAMENT_DATA_ROOT")
    if override:
        roots.append(Path(override))
    roots.append(REPO_DATA)
    if KAGGLE_INPUT.is_dir():
        # Kaggle mounts every attached dataset under its own slug, and the slug is
        # whatever the competition chose, so enumerate rather than guess.
        roots.extend(sorted(path for path in KAGGLE_INPUT.iterdir() if path.is_dir()))
    return roots


def find_annotations() -> Path | None:
    """
    Locate the training annotations under the first candidate root that holds them.

    The annotations file is the landmark for the whole dataset: wherever it
    lands, the images are nearby. Searching for it beats hardcoding a root,
    because the same checkout runs against a local ``data/`` directory and
    against a read-only Kaggle mount whose path is not known until runtime.

    :return: Path to the annotations, or ``None`` if no root contains them
    """
    for root in candidate_roots():
        if not root.is_dir():
            continue
        direct = next((path for path in (root / "train" / ANNOTATIONS_NAME, root / ANNOTATIONS_NAME) if path.is_file()), None)
        found = direct or next(root.rglob(ANNOTATIONS_NAME), None)
        if found is not None:
            return found
    return None


def _find_directory(root: Path, name: str, preferred: Path, fallback: Path) -> Path:
    """
    Resolve one image directory, checking the expected spot before walking the tree.

    :param root: Dataset root to search under
    :param name: Directory name to look for
    :param preferred: Location to check first, so the common layout costs no walk
    :param fallback: Returned when nothing matches
    :return: The directory to read observations from
    """
    if preferred.is_dir():
        return preferred
    for candidate in root.rglob(name):
        if candidate.is_dir():
            return candidate
    return fallback


def resolve_dataset_paths() -> tuple[Path, Path, Path, Path]:
    """
    Work out where the dataset lives on this machine.

    :return: Tuple of (data root, train images, test images, annotations)
    """
    annotations = find_annotations()
    if annotations is None:
        return (
            REPO_DATA,
            REPO_DATA / "train" / "train_images",
            REPO_DATA / "test" / "test_images",
            REPO_DATA / "train" / ANNOTATIONS_NAME,
        )

    root = annotations.parent.parent if annotations.parent.name == "train" else annotations.parent
    train_images = _find_directory(root, "train_images", annotations.parent / "train_images", annotations.parent)
    test_images = _find_directory(root, "test_images", root / "test" / "test_images", root / "test")
    return root, train_images, test_images, annotations


DATA_ROOT, TRAIN_IMAGES, TEST_IMAGES, TRAIN_ANNOTATIONS = resolve_dataset_paths()


def load_annotations(path=TRAIN_ANNOTATIONS) -> tuple[Any, dict[Any, list[Any]]]:
    """
    Return ``(images, annotations_by_image_id)`` from the COCO-style JSON.
    :param path: path to COCO-style annotations
    :return: Tuple of (image records, annotations keyed by image id)
    """
    try:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
    except FileNotFoundError:
        searched = ", ".join(str(root) for root in candidate_roots())
        raise FileNotFoundError(
            f"could not find {ANNOTATIONS_NAME}; searched {searched}. "
            f"Set FILAMENT_DATA_ROOT to the directory holding the dataset."
        ) from None
    by_image: dict[Any, list[Any]] = {img["id"]: [] for img in raw["images"]}
    for ann in raw["annotations"]:
        by_image[ann["image_id"]].append(ann)
    return raw["images"], by_image


def polygon_to_mask(polygon, height, width, scale=1.0, origin=(0, 0), raster="pil") -> np.ndarray:
    """Fill a flat ``[x0, y0, x1, y1, ...]`` polygon into a boolean mask.
    :param polygon: Input polygon
    :param height: Height of the output mask
    :param width: Width of the output mask
    :param scale: Factor applied to the polygon coordinates before filling
    :param origin: ``(x, y)`` of the mask inside the scaled observation, so a crop
        can be rasterised without ever building the full-resolution mask
    :param raster: One of :data:`RASTERS`
    :return: ``(height, width)`` boolean mask
    """
    points = np.asarray(polygon, dtype=np.float64).reshape(-1, 2) * scale
    points = points - np.asarray(origin, dtype=np.float64)
    if raster == "coco":
        return _coco_polygon_mask(points, height, width)
    if raster != "pil":
        raise ValueError(f"unknown raster {raster!r}, expected one of {RASTERS}")
    canvas = Image.new("1", (width, height), 0)
    if len(points) >= 3:
        ImageDraw.Draw(canvas).polygon([(float(x), float(y)) for x, y in points], fill=1)
    return np.asarray(canvas, dtype=bool)


def _coco_polygon_mask(points, height, width) -> np.ndarray:
    """
    Fill a polygon with pycocotools, on a canvas just large enough to hold it.

    pycocotools clips a polygon that leaves its canvas differently from cropping the
    full mask, so the polygon is filled on its own bounding box, shifted by whole
    pixels, and pasted into the window.
    :param points: ``(N, 2)`` polygon vertices in window coordinates
    :param height: Height of the window
    :param width: Width of the window
    :return: ``(height, width)`` boolean mask
    """
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


def polygon_bounds(polygon, scale=1.0) -> tuple[float, float, float, float]:
    """
    The extent of a polygon in the coordinates of a scaled observation.

    Taken from the polygon rather than from the ``bbox`` field of the annotation:
    a few dozen annotations carry a box that does not cover their own outline, and
    a filament dropped from a crop would be trained on as background.
    :param polygon: Flat ``[x0, y0, x1, y1, ...]`` polygon
    :param scale: Factor applied to the coordinates
    :return: Tuple of (x0, y0, x1, y1)
    """
    points = np.asarray(polygon, dtype=np.float64).reshape(-1, 2) * scale
    return points[:, 0].min(), points[:, 1].min(), points[:, 0].max(), points[:, 1].max()


def train_val_split(images, val_fraction=0.2, seed=0) -> tuple[list[Any], list[Any]]:
    """
    Split image ids into ``(train_ids, val_ids)``, grouped by observation.
    :param images: Images to split
    :param val_fraction: Fraction of images to use for validation
    :param seed: Random seed for reproducibility
    :return: Tuple of (train_ids, val_ids)
    """
    groups: dict[str, list[Any]] = {}
    for img in images:
        groups.setdefault(img["file_name"], []).append(img["id"])

    file_names = sorted(groups)
    random.Random(seed).shuffle(file_names)
    n_val = round(len(file_names) * val_fraction)

    val_ids = [i for f in file_names[:n_val] for i in groups[f]]
    train_ids = [i for f in file_names[n_val:] for i in groups[f]]
    return train_ids, val_ids


def kfold_split(images, fold, folds=5, seed=0) -> tuple[list[Any], list[Any]]:
    """
    Split image ids into ``(train_ids, val_ids)`` for one fold of a grouped k-fold.

    Every record of an observation lands in the same fold, whoever annotated it, and
    the assignment depends only on ``seed``, so models trained with different seeds
    still share their folds and their out-of-fold predictions can be pooled.
    :param images: Images to split
    :param fold: Index of the fold held out, in ``[0, folds)``
    :param folds: Number of folds
    :param seed: Seed of the fold assignment
    :return: Tuple of (train_ids, val_ids)
    """
    if not 0 <= fold < folds:
        raise ValueError(f"fold must be in [0, {folds}), got {fold}")
    groups: dict[str, list[Any]] = {}
    for img in images:
        groups.setdefault(img["file_name"], []).append(img["id"])

    file_names = sorted(groups)
    random.Random(seed).shuffle(file_names)
    val_ids = [i for index, f in enumerate(file_names) if index % folds == fold for i in groups[f]]
    train_ids = [i for index, f in enumerate(file_names) if index % folds != fold for i in groups[f]]
    return train_ids, val_ids


def validation_split(images, folds=0, fold=0, val_fraction=0.2, seed=0) -> tuple[list[Any], list[Any]]:
    """
    Hold out the validation records a run is selected and calibrated on.
    :param images: Image records
    :param folds: Number of grouped folds, or ``0`` for a random ``val_fraction`` split
    :param fold: Fold held out when ``folds`` is set
    :param val_fraction: Fraction held out otherwise
    :param seed: Seed of the random split; the folds are fixed across seeds
    :return: Tuple of (train_ids, val_ids)
    """
    if folds:
        return kfold_split(images, fold, folds)
    return train_val_split(images, val_fraction=val_fraction, seed=seed)


def load_image(path, size=None) -> torch.Tensor:
    """
    Load a grayscale observation as a ``(1, H, W)`` float tensor in [0, 1].
    :param path: Path to img
    :param size: Edge length to resize to, or ``None`` to keep the original size
    :return: ``(1, H, W)`` float tensor
    """
    with Image.open(path) as handle:
        if size is not None:
            handle.draft("L", (size, size))  # fast approximate JPEG downscale
        img = handle.convert("L")
        if size is not None and img.size != (size, size):
            img = img.resize((size, size), Image.Resampling.BILINEAR)
        array = np.asarray(img, dtype=np.uint8)
    return torch.from_numpy(array.copy()).unsqueeze(0).float().div_(255.0)


class FilamentDataset(Dataset):
    """
    Observations with a binary filament mask.

    Yields ``(image, mask)`` pairs, where the mask is the union of every
    annotated filament polygon of the selected categories.
    """

    def __init__(
        self,
        image_ids=None,
        size=512,
        category_ids=None,
        images_dir=TRAIN_IMAGES,
        annotations_path=TRAIN_ANNOTATIONS,
        raster="pil",
    ):
        """
        Build the dataset from a COCO-style annotation file.
        :param image_ids: Image ids to include, or ``None`` for all of them
        :param size: Edge length the observations are resized to
        :param category_ids: Category ids to draw into the mask, or ``None`` for all
        :param images_dir: Directory holding the observation files
        :param annotations_path: Path to the COCO-style annotations
        :param raster: How polygons are filled, one of :data:`RASTERS`
        """
        images, by_image = load_annotations(annotations_path)
        self.images_dir = Path(images_dir)
        self.size = size
        self.raster = raster
        self.category_ids = None if category_ids is None else set(category_ids)

        records = {img["id"]: img for img in images}
        self.image_ids = list(records) if image_ids is None else list(image_ids)
        self.records = records
        self.annotations = by_image

    def __len__(self):
        """
        :return: Number of observations in the split
        """
        return len(self.image_ids)

    def __getitem__(self, index):
        """
        Load one observation and rasterise its filament annotations.
        :param index: Position in :attr:`image_ids`
        :return: Tuple of (``(1, H, W)`` image tensor, ``(H, W)`` float mask tensor)
        """
        image_id = self.image_ids[index]
        record = self.records[image_id]
        image = load_image(self.images_dir / record["file_name"], self.size)

        height, width = image.shape[-2:]
        scale = width / record["width"]
        mask = np.zeros((height, width), dtype=bool)
        for ann in self.annotations[image_id]:
            if self.category_ids is not None and ann["category_id"] not in self.category_ids:
                continue
            mask |= polygon_to_mask(ann["segmentation"][0], height, width, scale, raster=self.raster)

        return image, torch.from_numpy(mask).float()


def load_image_window(path, size, window, crop) -> torch.Tensor:
    """
    Load one square window of an observation without resizing it away.

    The observation is decoded at ``size`` and only ``crop`` by ``crop`` pixels are
    kept, so training can happen at the resolution the submission is written at
    while a batch still fits in memory.
    :param path: Path to the observation
    :param size: Edge length the observation is decoded at, or ``None`` for its own
    :param window: ``(x, y)`` top left corner of the window in decoded pixels
    :param crop: Edge length of the window
    :return: ``(1, crop, crop)`` float tensor in [0, 1], zero padded past the edges
    """
    with Image.open(path) as handle:
        if size is not None:
            handle.draft("L", (size, size))  # fast approximate JPEG downscale
        img = handle.convert("L")
        if size is not None and img.size != (size, size):
            img = img.resize((size, size), Image.Resampling.BILINEAR)
        x, y = window
        array = np.asarray(img.crop((x, y, x + crop, y + crop)), dtype=np.uint8)
    return torch.from_numpy(array.copy()).unsqueeze(0).float().div_(255.0)


class FilamentCropDataset(Dataset):
    """
    Random crops taken at the resolution the observations were annotated at.

    Filaments are roughly eleven pixels thick at 2048, so resizing a whole
    observation down to what fits in a batch costs more than half of the overlap
    with the annotation before the model has made a single mistake. Cropping keeps
    the native scale instead, and because the filaments cover about two parts in a
    thousand of the disk, most crops are drawn around an annotation rather than
    uniformly, which would return empty sky most of the time.
    """

    def __init__(
        self,
        image_ids=None,
        crop=512,
        native_size=None,
        repeat=1,
        foreground_prob=0.7,
        category_ids=None,
        images_dir=TRAIN_IMAGES,
        annotations_path=TRAIN_ANNOTATIONS,
        seed=None,
        raster="pil",
        targets="each",
    ):
        """
        :param image_ids: Image ids to sample from, or ``None`` for all of them
        :param crop: Edge length of the returned window
        :param native_size: Edge length the observation is decoded at, or ``None``
            to keep the resolution it was annotated at
        :param repeat: Crops drawn per observation and epoch
        :param foreground_prob: Probability of centring the crop on an annotation
        :param category_ids: Category ids to draw into the mask, or ``None`` for all
        :param images_dir: Directory holding the observation files
        :param annotations_path: Path to the COCO-style annotations
        :param seed: Seed for a private RNG; ``None`` uses the global :mod:`random`,
            which the data loader reseeds per worker
        :param raster: How polygons are filled, one of :data:`RASTERS`
        :param targets: How the records of one observation become samples, one of :data:`TARGETS`
        """
        if targets not in TARGETS:
            raise ValueError(f"unknown targets {targets!r}, expected one of {TARGETS}")
        images, by_image = load_annotations(annotations_path)
        self.images_dir = Path(images_dir)
        self.crop = crop
        self.raster = raster
        self.native_size = native_size
        self.repeat = max(1, repeat)
        self.foreground_prob = foreground_prob
        self.category_ids = None if category_ids is None else set(category_ids)

        records = {img["id"]: img for img in images}
        self.image_ids = list(records) if image_ids is None else list(image_ids)
        self.records = records
        self.annotations = by_image
        self._rng = None if seed is None else random.Random(seed)

        # each sample is the list of records whose masks are averaged into its target
        if targets == "mean":
            groups: dict[str, list[Any]] = {}
            for image_id in self.image_ids:
                groups.setdefault(records[image_id]["file_name"], []).append(image_id)
            self.samples = [groups[name] for name in sorted(groups)]
        else:
            self.samples = [[image_id] for image_id in self.image_ids]

    @property
    def rng(self):
        """
        The random source to draw windows from.
        :return: A :class:`random.Random` or the :mod:`random` module
        """
        return random if self._rng is None else self._rng

    def __len__(self):
        """
        :return: Number of crops in one epoch
        """
        return len(self.samples) * self.repeat

    def selected(self, image_id) -> list:
        """
        The annotations of one observation that are drawn into the mask.
        :param image_id: Image id to look up
        :return: List of annotations
        """
        annotations = self.annotations[image_id]
        if self.category_ids is None:
            return annotations
        return [ann for ann in annotations if ann["category_id"] in self.category_ids]

    def sample_window(self, annotations, size, scale) -> tuple[int, int]:
        """
        Pick the top left corner of a crop.

        :param annotations: Annotations of the observation
        :param size: Edge length the observation is decoded at
        :param scale: Factor the annotation coordinates are scaled by
        :return: ``(x, y)`` corner, clamped to keep the crop inside the observation
        """
        limit = max(0, size - self.crop)
        if annotations and self.rng.random() < self.foreground_prob:
            annotation = self.rng.choice(annotations)
            points = np.asarray(annotation["segmentation"][0], dtype=np.float64).reshape(-1, 2)
            x, y = points[self.rng.randrange(len(points))] * scale
            # place the point anywhere inside the crop, so filaments are not always centred
            x = int(x) - self.rng.randrange(self.crop)
            y = int(y) - self.rng.randrange(self.crop)
        else:
            x = self.rng.randint(0, limit)
            y = self.rng.randint(0, limit)
        return min(max(x, 0), limit), min(max(y, 0), limit)

    def __getitem__(self, index):
        """
        Draw one crop and rasterise the annotations that reach into it.
        :param index: Position in the epoch; samples are cycled ``repeat`` times
        :return: Tuple of (``(1, crop, crop)`` image tensor, ``(crop, crop)`` float mask,
            the fraction of the sample's records that mark each pixel)
        """
        group = self.samples[index % len(self.samples)]
        record = self.records[group[0]]
        size = self.native_size or record["width"]
        scale = size / record["width"]

        per_record = [self.selected(image_id) for image_id in group]
        x, y = self.sample_window([ann for annotations in per_record for ann in annotations], size, scale)
        image = load_image_window(self.images_dir / record["file_name"], self.native_size, (x, y), self.crop)

        target = np.zeros((self.crop, self.crop), dtype=np.float32)
        for annotations in per_record:
            mask = np.zeros((self.crop, self.crop), dtype=bool)
            for ann in annotations:
                x0, y0, x1, y1 = polygon_bounds(ann["segmentation"][0], scale)
                if x1 < x or x0 > x + self.crop or y1 < y or y0 > y + self.crop:
                    continue
                mask |= polygon_to_mask(ann["segmentation"][0], self.crop, self.crop, scale, origin=(x, y), raster=self.raster)
            target += mask
        target /= len(per_record)

        return image, torch.from_numpy(target)


class TestImageDataset(Dataset):
    """
    The unlabelled test observations; yields ``(image, file_name)``.
    """

    def __init__(self, size=512, images_dir=TEST_IMAGES):
        """
        Collect the test observations found on disk.
        :param size: Edge length the observations are resized to
        :param images_dir: Directory holding the ``.jpeg`` test observations
        """
        self.images_dir = Path(images_dir)
        self.size = size
        # Kaggle repackaging sometimes renames the extension, so accept both spellings.
        self.file_names = sorted(p.name for p in self.images_dir.iterdir() if p.suffix.lower() in {".jpeg", ".jpg"})

    def __len__(self):
        """
        :return: Number of test observations
        """
        return len(self.file_names)

    def __getitem__(self, index):
        """
        Load one test observation.
        :param index: Position in :attr:`file_names`
        :return: Tuple of (``(1, H, W)`` image tensor, file name)
        """
        file_name = self.file_names[index]
        return load_image(self.images_dir / file_name, self.size), file_name

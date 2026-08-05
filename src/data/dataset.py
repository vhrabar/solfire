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
from torch.utils.data import Dataset

ANNOTATIONS_NAME = "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
REPO_DATA = Path(__file__).resolve().parents[2] / "data"
KAGGLE_INPUT = Path("/kaggle/input")

CATEGORY_NAMES = {1: "Left", 2: "Right", 3: "Unidentifiable", 4: "Ambiguous"}


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
        return REPO_DATA, REPO_DATA / "train" / "train_images", REPO_DATA / "test" / "test_images", REPO_DATA / "train" / ANNOTATIONS_NAME

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


def polygon_to_mask(polygon, height, width, scale=1.0) -> np.ndarray:
    """Fill a flat ``[x0, y0, x1, y1, ...]`` polygon into a boolean mask.
    :param polygon: Input polygon
    :param height: Height of the output mask
    :param width: Width of the output mask
    :param scale: Factor applied to the polygon coordinates before filling
    :return: ``(height, width)`` boolean mask
    """
    points = np.asarray(polygon, dtype=np.float64).reshape(-1, 2) * scale
    canvas = Image.new("1", (width, height), 0)
    if len(points) >= 3:
        ImageDraw.Draw(canvas).polygon([(float(x), float(y)) for x, y in points], fill=1)
    return np.asarray(canvas, dtype=bool)


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

    def __init__(self, image_ids=None, size=512, category_ids=None, images_dir=TRAIN_IMAGES, annotations_path=TRAIN_ANNOTATIONS):
        """
        Build the dataset from a COCO-style annotation file.
        :param image_ids: Image ids to include, or ``None`` for all of them
        :param size: Edge length the observations are resized to
        :param category_ids: Category ids to draw into the mask, or ``None`` for all
        :param images_dir: Directory holding the observation files
        :param annotations_path: Path to the COCO-style annotations
        """
        images, by_image = load_annotations(annotations_path)
        self.images_dir = Path(images_dir)
        self.size = size
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
            mask |= polygon_to_mask(ann["segmentation"][0], height, width, scale)

        return image, torch.from_numpy(mask).float()


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

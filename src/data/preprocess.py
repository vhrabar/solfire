"""
Image preprocessing that is applied and cached fr lalter experiments
"""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import exposure

PREPROCESSING = ("none", "clahe")
CACHE_ROOT = Path(__file__).resolve().parents[2] / "runs" / "preprocessed"
CLAHE_KERNEL = 256
CLAHE_CLIP = 0.01
IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png"}


def clahe(image: np.ndarray) -> np.ndarray:
    """
    Contrast limited adaptive histogram equalisation of one observation.
    :param image: ``(H, W)`` uint8 observation
    :return: ``(H, W)`` uint8 equalised observation
    """
    equalised = exposure.equalize_adapthist(image, kernel_size=CLAHE_KERNEL, clip_limit=CLAHE_CLIP)
    return np.asarray(np.round(equalised * 255.0), dtype=np.uint8)


def preprocessed_dir(images_dir, method) -> Path:
    """
    Where the preprocessed copies of one image directory live.
    :param images_dir: Directory holding the original observations
    :param method: One of :data:`PREPROCESSING`
    :return: ``images_dir`` itself for ``none``, else its cache directory
    """
    if method in (None, "none"):
        return Path(images_dir)
    if method not in PREPROCESSING:
        raise ValueError(f"unknown preprocessing {method!r}, expected one of {PREPROCESSING}")
    source = Path(images_dir)
    return CACHE_ROOT / str(method) / source.name


def observation_path(images_dir, file_name) -> Path:
    """
    The file holding one observation, original or preprocessed.
    :param images_dir: Directory returned by :func:`preprocessed_dir`
    :param file_name: File name of the original observation
    :return: Path to read
    """
    path = Path(images_dir) / str(file_name)
    return path if path.is_file() else path.with_suffix(".png")


def build_cache(images_dir, method, workers=8) -> Path:
    """
    Preprocess every observation of a directory that is not cached yet.
    :param images_dir: Directory holding the original observations
    :param method: One of :data:`PREPROCESSING` other than ``none``
    :param workers: Processes working in parallel
    :return: The cache directory
    """
    output = preprocessed_dir(images_dir, method)
    output.mkdir(parents=True, exist_ok=True)
    sources = sorted(p for p in Path(images_dir).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    jobs = [(source, output / f"{source.stem}.png") for source in sources if not (output / f"{source.stem}.png").is_file()]
    with ProcessPoolExecutor(max(1, workers), mp_context=multiprocessing.get_context("fork")) as pool:
        list(pool.map(_process_one, jobs))
    return output


def _process_one(job) -> None:
    """
    Preprocess one observation in a worker process.
    :param job: Tuple of (source path, destination path)
    :return: None
    """
    source, destination = job
    with Image.open(source) as handle:
        image = np.asarray(handle.convert("L"), dtype=np.uint8)
    partial = destination.with_suffix(".tmp.png")
    Image.fromarray(clahe(image)).save(partial)
    partial.replace(destination)

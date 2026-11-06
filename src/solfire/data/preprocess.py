"""
Image preprocessing that is applied once and cached for later runs.

The final models read the observations as they are (``none``). Other methods register
themselves in :data:`PREPROCESSORS` (the experiments add ``clahe``); a checkpoint records
the method it was trained on and inference reads the same cache.
"""

import multiprocessing
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from .. import PROJECT_ROOT

CACHE_ROOT = PROJECT_ROOT / "runs" / "preprocessed"
IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png"}

# method name -> function from a ``(H, W)`` uint8 observation to its preprocessed uint8 copy
PREPROCESSORS: dict[str, Callable[[np.ndarray], np.ndarray]] = {}


def register_preprocessing(name: str, function: Callable[[np.ndarray], np.ndarray]) -> None:
    """
    Make a preprocessing method available to ``train --preprocess`` and to the inference of its checkpoints.
    :param name: Name the checkpoints record
    :param function: Maps a ``(H, W)`` uint8 observation to a ``(H, W)`` uint8 image
    :return: None
    """
    PREPROCESSORS[name] = function


def preprocessing_methods() -> tuple[str, ...]:
    """
    :return: ``none`` and every registered method
    """
    return ("none", *PREPROCESSORS)


def preprocessed_dir(images_dir, method) -> Path:
    """
    Where the preprocessed copies of one image directory live.
    :param images_dir: Directory holding the original observations
    :param method: One of :func:`preprocessing_methods`
    :return: ``images_dir`` itself for ``none``, else its cache directory
    """
    if method in (None, "none"):
        return Path(images_dir)
    if method not in PREPROCESSORS:
        raise ValueError(
            f"unknown preprocessing {method!r}, expected one of {preprocessing_methods()}; "
            "checkpoints of the experiments need the solfire-lab command"
        )
    source = Path(images_dir)
    return CACHE_ROOT / method / source.name


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
    :param method: A registered method, not ``none``
    :param workers: Processes working in parallel
    :return: The cache directory
    """
    output = preprocessed_dir(images_dir, method)
    output.mkdir(parents=True, exist_ok=True)
    sources = sorted(p for p in Path(images_dir).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    jobs = [(source, output / f"{source.stem}.png") for source in sources if not (output / f"{source.stem}.png").is_file()]
    with ProcessPoolExecutor(max(1, workers), mp_context=multiprocessing.get_context("fork")) as pool:
        list(pool.map(_process_one, [(source, destination, method) for source, destination in jobs]))
    return output


def _process_one(job) -> None:
    """
    Preprocess one observation in a worker process.
    :param job: Tuple of (source path, destination path, method)
    :return: None
    """
    source, destination, method = job
    with Image.open(source) as handle:
        image = np.asarray(handle.convert("L"), dtype=np.uint8)
    partial = destination.with_suffix(".tmp.png")
    Image.fromarray(PREPROCESSORS[method](image)).save(partial)
    partial.replace(destination)

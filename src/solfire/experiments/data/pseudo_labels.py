"""
Training on GONG observations without annotations, with soft targets from teacher models
(scripts/experiments/pseudo_label.py): pseudo-labels from the ensemble, and distillation.
"""

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from solfire.data.dataset import load_image_window


class PseudoLabelledCropDataset(Dataset):
    """
    Native-resolution crops of unlabeled observations with soft pseudo-label targets.
    """

    def __init__(self, images_dir, target_dir, crop=512, repeat=1, foreground_prob=0.7, foreground_level=0.5, seed=None):
        """
        :param images_dir: Directory holding the observations
        :param target_dir: Directory holding one target PNG per observation
        :param crop: Edge length of the returned window
        :param repeat: Crops drawn per observation and epoch
        :param foreground_prob: Probability of centring the crop on a likely filament pixel
        :param foreground_level: Target value a pixel needs to count as a likely filament
        :param seed: Seed for a private RNG; ``None`` uses the global :mod:`random`
        """
        self.images_dir = Path(images_dir)
        self.targets = sorted(Path(target_dir).glob("*.png"))
        if not self.targets:
            raise FileNotFoundError(f"no pseudo-label targets in {target_dir}")
        self.crop = crop
        self.repeat = max(1, repeat)
        self.foreground_prob = foreground_prob
        self.foreground_level = foreground_level
        self._rng = None if seed is None else random.Random(seed)

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
        return len(self.targets) * self.repeat

    def __getitem__(self, index):
        """
        Draw one crop of an observation and its target.
        :param index: Position in the epoch; observations are cycled ``repeat`` times
        :return: Tuple of (``(1, crop, crop)`` image tensor, ``(crop, crop)`` float target)
        """
        target_path = self.targets[index % len(self.targets)]
        with Image.open(target_path) as handle:
            target = np.asarray(handle, dtype=np.uint8)
        size = target.shape[0]
        limit = max(0, size - self.crop)
        ys, xs = np.nonzero(target >= round(self.foreground_level * 255))
        if len(ys) and self.rng.random() < self.foreground_prob:
            pick = self.rng.randrange(len(ys))
            x = int(xs[pick]) - self.rng.randrange(self.crop)
            y = int(ys[pick]) - self.rng.randrange(self.crop)
        else:
            x, y = self.rng.randint(0, limit), self.rng.randint(0, limit)
        x, y = min(max(x, 0), limit), min(max(y, 0), limit)
        image = load_image_window(self.images_dir / f"{target_path.stem}.jpeg", None, (x, y), self.crop)
        window = target[y : y + self.crop, x : x + self.crop].astype(np.float32) / 255.0
        return image, torch.from_numpy(window)

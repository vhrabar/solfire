"""
Label-preserving augmentation for the filament observations.
"""

import math
import random

import torch
from torch.utils.data import Dataset


class AugmentedDataset(Dataset):
    """
    Wrap a segmentation dataset with random flips, rotations and intensity jitter.
    """

    def __init__(self, dataset, flip=True, rotate=True, intensity=0.15, gamma=0.2, seed=None):
        """
        :param dataset: Dataset yielding ``(image, mask)`` tensors
        :param flip: Apply random horizontal and vertical flips
        :param rotate: Apply a random multiple of 90 degrees
        :param intensity: Half-width of the uniform brightness/contrast jitter, ``0`` to disable
        :param gamma: Standard deviation of the log-normal gamma jitter, ``0`` to disable
        :param seed: Seed for a private RNG; ``None`` uses the global :mod:`random`,
            which the data loader reseeds per worker
        """
        self.dataset = dataset
        self.flip = flip
        self.rotate = rotate
        self.intensity = intensity
        self.gamma = gamma
        self.seed = seed
        self._rng = None if seed is None else random.Random(seed)

    @property
    def rng(self):
        """
        The random source to draw from.
        :return: A :class:`random.Random` or the :mod:`random` module
        """
        return random if self._rng is None else self._rng

    def __len__(self):
        """
        :return: Number of observations in the wrapped dataset
        """
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Load and augment one observation.
        :param index: Position in the wrapped dataset
        :return: Tuple of (``(1, H, W)`` image tensor, ``(H, W)`` float mask tensor)
        """
        image, mask = self.dataset[index]
        mask = mask.unsqueeze(0)

        if self.flip:
            if self.rng.random() < 0.5:
                image, mask = torch.flip(image, [-1]), torch.flip(mask, [-1])
            if self.rng.random() < 0.5:
                image, mask = torch.flip(image, [-2]), torch.flip(mask, [-2])
        if self.rotate:
            k = self.rng.randint(0, 3)
            if k:
                image, mask = torch.rot90(image, k, [-2, -1]), torch.rot90(mask, k, [-2, -1])

        if self.gamma:
            image = image.clamp_min(0.0).pow(math.exp(self.rng.gauss(0.0, self.gamma)))
        if self.intensity:
            scale = 1.0 + self.rng.uniform(-self.intensity, self.intensity)
            shift = self.rng.uniform(-self.intensity, self.intensity)
            image = (image * scale + shift).clamp_(0.0, 1.0)

        return image, mask.squeeze(0)

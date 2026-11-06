"""
Training crops with the extra targets and sampling the experiments tried.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from solfire.data.dataset import Crop, FilamentCropDataset

from .structure import spine_heatmap, thickness_weight, thin_mask

# category ids the category head predicts, one channel each
CATEGORY_CHANNELS = (1, 2, 3)


class ExperimentCropDataset(FilamentCropDataset):
    """
    :class:`~solfire.data.dataset.FilamentCropDataset` plus hard-negative sampling and the targets
    of the auxiliary heads, the thin-structure pixel weights and the grown barbs.

    With every option at its default it draws exactly the crops and targets of the core dataset.
    """

    def __init__(
        self,
        image_ids=None,
        *,
        hard_negatives=None,
        hard_negative_prob=0.0,
        categories=False,
        magnetic=None,
        thickness_cap=0.0,
        thin_head=False,
        barb_targets=None,
        barb_level=1.0,
        barb_records=None,
        **kwargs,
    ):
        """
        :param image_ids: Image ids to sample from, or ``None`` for all of them
        :param hard_negatives: Per observation file, ``(x, y)`` points of dark features that
            models mistake for filaments, or ``None``
        :param hard_negative_prob: Probability of centring a crop on one of them, taken
            from the uniformly placed crops
        :param categories: Also return one indicator channel per category of
            :data:`CATEGORY_CHANNELS` after the filament target, for an auxiliary head
        :param magnetic: :class:`~solfire.experiments.data.magnetogram.MagneticTargets` whose PIL
            proximity, strong-flux and valid channels follow the filament target, or ``None``
        :param thickness_cap: Above ``0``, a per-pixel loss weight channel follows the filament
            target, growing as the annotated structure gets thinner, up to this value
            (:func:`~solfire.experiments.data.structure.thickness_weight`)
        :param thin_head: Append the thin mask and the spine heatmap of the annotations as
            the last two channels, for an auxiliary head
        :param barb_targets: Directory of 0/255 PNGs, one per observation, marking barbs grown
            from the image outside the annotations (scripts/experiments/barb_targets.py), or ``None``
        :param barb_level: Target value at those barb pixels
        :param barb_records: Image ids of the records that draw barbs (scripts/experiments/barb_drawers.py),
            or ``None``; when given, two channels follow the filament target for a barb head: the
            thin mask of those records alone, and 1 where it is valid, i.e. over the whole crop
            when the sample holds at least one of them and 0 otherwise
        :param kwargs: Passed on to :class:`~solfire.data.dataset.FilamentCropDataset`
        """
        heads = (bool(categories), magnetic is not None, bool(thickness_cap) or thin_head, barb_records is not None)
        if sum(heads) > 1:
            raise ValueError("the category, magnetic, thin-structure and barb targets cannot be combined")
        if barb_targets is not None and (kwargs.get("native_size") or kwargs.get("input_scale", 1.0) != 1):
            raise ValueError("barb targets are drawn at the annotated resolution only")
        super().__init__(image_ids, **kwargs)
        self.hard_negatives = hard_negatives or {}
        self.hard_negative_prob = hard_negative_prob
        self.categories = categories
        self.magnetic = magnetic
        self.thickness_cap = float(thickness_cap)
        self.thin_head = thin_head
        self.barb_targets = None if barb_targets is None else Path(barb_targets)
        self.barb_level = float(barb_level)
        self.barb_records = None if barb_records is None else set(barb_records)

    def sample_window(self, annotations, size, scale, record) -> tuple[int, int]:
        """
        Pick the top left corner of a crop, centred on a hard negative with ``hard_negative_prob``.

        :param annotations: Annotations of the observation
        :param size: Edge length the observation is decoded at
        :param scale: Factor the annotation coordinates are scaled by
        :param record: Image record of the observation
        :return: ``(x, y)`` corner, clamped to keep the crop inside the observation
        """
        hard_negatives = self.hard_negatives.get(record["file_name"], ())
        limit = max(0, size - self.crop)
        draw = self.rng.random()
        if hard_negatives and self.foreground_prob <= draw < self.foreground_prob + self.hard_negative_prob:
            x, y = self.rng.choice(hard_negatives)
            x = int(x * scale) - self.rng.randrange(self.crop)
            y = int(y * scale) - self.rng.randrange(self.crop)
        elif annotations and draw < self.foreground_prob:
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

    def make_target(self, target, crop: Crop) -> torch.Tensor:
        """
        Add the channels of whichever experiment is switched on after the filament target.
        :param target: ``(crop, crop)`` fraction of the sample's records that mark each pixel
        :param crop: The window and its rasterised annotations
        :return: ``(crop, crop)`` target, or ``(C, crop, crop)`` with the extra channels after it
        """
        size, records = self.crop, len(crop.masks)
        if self.barb_targets is not None:
            with Image.open(self.barb_targets / f"{Path(crop.record['file_name']).stem}.png") as handle:
                barbs = np.asarray(handle.crop((crop.x, crop.y, crop.x + size, crop.y + size))) > 0
            target[barbs] = np.maximum(target[barbs], self.barb_level)

        if self.thickness_cap or self.thin_head:
            channels = [target]
            if self.thickness_cap:
                channels.append(thickness_weight(np.logical_or.reduce(crop.masks), self.thickness_cap))
            if self.thin_head:
                thin = np.zeros((size, size), dtype=np.float32)
                spine = np.zeros((size, size), dtype=np.float32)
                for mask, drawn in zip(crop.masks, crop.drawn):
                    thin += thin_mask(mask)
                    spines = [ann["spine"] for ann, _ in drawn if ann.get("spine")]
                    spine += spine_heatmap(spines, size, size, crop.scale, origin=(crop.x, crop.y))
                channels += [thin / records, spine / records]
            return torch.from_numpy(np.stack(channels))

        if self.barb_records is not None:
            barb = np.zeros((size, size), dtype=np.float32)
            drawers = 0
            for image_id, mask in zip(crop.group, crop.masks):
                if image_id in self.barb_records:
                    barb += thin_mask(mask)
                    drawers += 1
            valid = np.full((size, size), float(drawers > 0), dtype=np.float32)
            return torch.from_numpy(np.stack([target, barb / max(drawers, 1), valid]))

        if self.magnetic is not None:
            extra = self.magnetic.window(crop.record["file_name"], crop.x, crop.y, size, crop.size)
            return torch.from_numpy(np.concatenate([target[None], extra]))
        if not self.categories:
            return torch.from_numpy(target)
        categories = np.zeros((len(CATEGORY_CHANNELS), size, size), dtype=bool)
        for drawn in crop.drawn:
            for ann, filled in drawn:
                if ann["category_id"] in CATEGORY_CHANNELS:
                    categories[CATEGORY_CHANNELS.index(ann["category_id"])] |= filled
        return torch.from_numpy(np.concatenate([target[None], categories.astype(np.float32)]))

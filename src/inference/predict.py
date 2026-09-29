"""
Prediction on unlabelled observations.
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from ..networks import ARCHITECTURES
from .postprocess import encode_rle, instances_from_labels, keep_labels, label_instances
from .tiling import DEFAULT_OVERLAP, predict_probabilities, predict_tiled


def load_model(checkpoint_path, device) -> tuple[torch.nn.Module, dict]:
    """
    Rebuild a trained model from a checkpoint written by :func:`training.save_checkpoint`.
    :param checkpoint_path: Path to the checkpoint file
    :param device: Device to place the model on
    :return: Tuple of (model in eval mode, checkpoint metadata)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    arch = checkpoint.get("arch", "unet")
    if arch not in ARCHITECTURES:
        raise ValueError(f"unknown architecture {arch!r} in {checkpoint_path}, expected one of {sorted(ARCHITECTURES)}")
    model = ARCHITECTURES[arch](**(checkpoint.get("model_config") or {}))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, checkpoint


@torch.no_grad()
def observation_probabilities(models, images, device, tta=True, tile=None, overlap=DEFAULT_OVERLAP, tile_batch=8) -> torch.Tensor:
    """
    Foreground probabilities for a batch of whole observations.

    :param models: Model in eval mode, or a list of them whose probabilities are averaged
    :param images: ``(B, 1, H, W)`` batch of observations
    :param device: Device to run on
    :param tta: Average predictions over flips
    :param tile: Edge length of the sliding window, or ``None`` to run the model on
        the whole observation at once
    :param overlap: Fraction of a window shared with its neighbour
    :param tile_batch: Windows evaluated per forward pass
    :return: ``(B, 1, H, W)`` probabilities on the cpu
    """
    models = models if isinstance(models, (list, tuple)) else [models]
    total = None
    for model in models:
        if tile is None:
            probabilities = predict_probabilities(model, images.to(device), tta=tta).cpu()
        else:
            maps = [
                predict_tiled(model, image, device, tile=tile, overlap=overlap, tta=tta, batch_size=tile_batch)
                for image in images
            ]
            probabilities = torch.stack(maps).unsqueeze(1)
        total = probabilities if total is None else total + probabilities
    assert total is not None
    return total / len(models)


@torch.no_grad()
def predict_dataset(
    models,
    loader,
    device,
    threshold=0.5,
    tta=True,
    original_size=None,
    min_area=0,
    min_score=0.0,
    instance_filter=None,
    offset=0,
    category_id=1,
    save_masks_dir=None,
    tile=None,
    overlap=DEFAULT_OVERLAP,
    tile_batch=8,
) -> list[dict]:
    """
    Run the model over a loader of unlabelled observations.

    :param models: Model in eval mode, or a list of them whose probabilities are averaged
    :param loader: Loader yielding ``(image, file_name)`` batches
    :param device: Device to run on
    :param threshold: Probability above which a pixel counts as filament
    :param tta: Average predictions over flips
    :param original_size: Edge length to upsample to, or ``None`` to stay at model resolution
    :param min_area: Drop instances with at most this many pixels
    :param min_score: Drop instances whose mean probability is below this
    :param instance_filter: A fitted :class:`InstanceFilter` deciding which instances to keep, or ``None``
    :param offset: Morphological steps applied to the thresholded mask
    :param category_id: Category assigned to every predicted instance
    :param save_masks_dir: Directory to write binary mask PNGs to, or ``None``
    :param tile: Edge length of the sliding window, or ``None`` to run the model on
        the whole observation at once
    :param overlap: Fraction of a window shared with its neighbour
    :param tile_batch: Windows evaluated per forward pass
    :return: List of COCO-style detection records
    """
    if save_masks_dir is not None:
        save_masks_dir = Path(save_masks_dir)
        save_masks_dir.mkdir(parents=True, exist_ok=True)

    detections = []
    for images, file_names in tqdm(loader, desc="predict", mininterval=2.0):
        probabilities = observation_probabilities(
            models, images, device, tta=tta, tile=tile, overlap=overlap, tile_batch=tile_batch
        )
        if original_size is not None and probabilities.shape[-1] != original_size:
            probabilities = F.interpolate(
                probabilities,
                size=(original_size, original_size),
                mode="bilinear",
                align_corners=False,
            )

        for probability, image, file_name in zip(probabilities[:, 0].numpy(), images[:, 0].numpy(), file_names):
            image_id = Path(file_name).stem
            labels, n_labels = label_instances(probability, threshold, offset=offset, min_area=min_area, min_score=min_score)
            if instance_filter is not None:
                if image.shape != probability.shape:
                    raise ValueError("the instance filter needs the observation at the resolution of the probabilities")
                labels, n_labels = keep_labels(labels, n_labels, instance_filter.keep(labels, n_labels, probability, image))
            if save_masks_dir is not None:
                Image.fromarray(((labels > 0) * 255).astype("uint8")).save(save_masks_dir / f"{image_id}.png")
            for number, instance in enumerate(instances_from_labels(labels, n_labels, probability), start=1):
                detections.append(
                    {
                        "filament_id": f"{image_id}_{number}",
                        "image_id": image_id,
                        "file_name": file_name,
                        "category_id": category_id,
                        "segmentation": encode_rle(instance["mask"]),
                        "bbox": instance["bbox"],
                        "area": instance["area"],
                        "score": instance["score"],
                    }
                )
    return detections

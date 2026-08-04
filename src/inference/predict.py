"""
Prediction on unlabelled observations.
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from networks import UNet

from .postprocess import encode_rle, instances_from_mask


def load_model(checkpoint_path, device) -> tuple[UNet, dict]:
    """
    Rebuild a trained model from a checkpoint written by :func:`training.save_checkpoint`.
    :param checkpoint_path: Path to the checkpoint file
    :param device: Device to place the model on
    :return: Tuple of (model in eval mode, checkpoint metadata)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = UNet(**(checkpoint.get("model_config") or {}))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, checkpoint


@torch.no_grad()
def predict_probabilities(model, images, tta=True) -> torch.Tensor:
    """
    Foreground probabilities for a batch, optionally averaged over flips.

    :param model: Model in eval mode
    :param images: ``(B, 1, H, W)`` batch of observations
    :param tta: Average over horizontal and vertical flips
    :return: ``(B, 1, H, W)`` probabilities in ``[0, 1]``
    """
    if not tta:
        return torch.sigmoid(model(images).float())

    total = torch.zeros_like(images, dtype=torch.float32)
    for flip_x in (False, True):
        for flip_y in (False, True):
            dims = [d for d, on in ((-1, flip_x), (-2, flip_y)) if on]
            transformed = torch.flip(images, dims) if dims else images
            probabilities = torch.sigmoid(model(transformed).float())
            total += torch.flip(probabilities, dims) if dims else probabilities
    return total / 4.0


@torch.no_grad()
def predict_dataset(
    model, loader, device, threshold=0.5, tta=True, original_size=None, min_area=0, category_id=1, save_masks_dir=None
) -> list[dict]:
    """
    Run the model over a loader of unlabelled observations.

    :param model: Model in eval mode
    :param loader: Loader yielding ``(image, file_name)`` batches
    :param device: Device to run on
    :param threshold: Probability above which a pixel counts as filament
    :param tta: Average predictions over flips
    :param original_size: Edge length to upsample to, or ``None`` to stay at model resolution
    :param min_area: Drop instances with at most this many pixels
    :param category_id: Category assigned to every predicted instance
    :param save_masks_dir: Directory to write binary mask PNGs to, or ``None``
    :return: List of COCO-style detection records
    """
    if save_masks_dir is not None:
        save_masks_dir = Path(save_masks_dir)
        save_masks_dir.mkdir(parents=True, exist_ok=True)

    detections = []
    for images, file_names in tqdm(loader, desc="predict"):
        probabilities = predict_probabilities(model, images.to(device), tta=tta).cpu()
        if original_size is not None and probabilities.shape[-1] != original_size:
            probabilities = F.interpolate(
                probabilities,
                size=(original_size, original_size),
                mode="bilinear",
                align_corners=False,
            )

        for probability, file_name in zip(probabilities[:, 0].numpy(), file_names):
            image_id = Path(file_name).stem
            mask = probability >= threshold
            if save_masks_dir is not None:
                Image.fromarray((mask * 255).astype("uint8")).save(save_masks_dir / f"{image_id}.png")
            instances = instances_from_mask(mask, probability, min_area)
            for number, instance in enumerate(instances, start=1):
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

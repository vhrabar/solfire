"""
Training and validation loops.
"""

import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from .metrics import DEFAULT_THRESHOLDS, SegmentationMetrics


def autocast_context(device, enabled):
    """
    Mixed precision context appropriate for the device.

    :param device: Device the model runs on
    :param enabled: Whether mixed precision is requested
    :return: Tuple of (autocast context manager, dtype or ``None``)
    """
    if not enabled:
        return torch.autocast(device_type=device.type, enabled=False), None
    dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype), dtype


def train_one_epoch(
    model, loader, criterion, optimizer, device, scheduler=None, scaler=None, grad_clip=1.0, amp=False, epoch=0, epochs=0
) -> float:
    """
    Run one pass over the training split.
    :param model: Model to update
    :param loader: Loader yielding ``(image, mask)`` batches
    :param criterion: Loss taking ``(logits, targets)``
    :param optimizer: Optimiser to step
    :param device: Device to run on
    :param scheduler: Per-step learning rate scheduler, or ``None``
    :param scaler: Gradient scaler for float16 training, or ``None``
    :param grad_clip: Global gradient norm clip, ``0`` to disable
    :param amp: Whether to run the forward pass in mixed precision
    :param epoch: Index of the current epoch, for the progress bar
    :param epochs: Total number of epochs, for the progress bar
    :return: Mean training loss over the epoch
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    progress = tqdm(loader, desc=f"epoch {epoch + 1}/{epochs} train", leave=False)

    for images, masks in progress:
        images = images.to(device, non_blocking=True)
        targets = masks.to(device, non_blocking=True).unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)
        context, _ = autocast_context(device, amp)
        with context:
            loss = criterion(model(images), targets)

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        n_batches += 1
        progress.set_postfix(loss=f"{total_loss / n_batches:.4f}")

    return total_loss / max(1, n_batches)


@torch.no_grad()
def evaluate(model, loader, criterion, device, thresholds=DEFAULT_THRESHOLDS, amp=False) -> dict:
    """
    Score the model on a validation split across several thresholds.
    :param model: Model to evaluate
    :param loader: Loader yielding ``(image, mask)`` batches
    :param criterion: Loss taking ``(logits, targets)``
    :param device: Device to run on
    :param thresholds: Probability thresholds to score at
    :param amp: Whether to run the forward pass in mixed precision
    :return: Metrics dict from :meth:`SegmentationMetrics.compute` plus ``loss``
    """
    model.eval()
    metrics = SegmentationMetrics(thresholds)
    total_loss = 0.0
    n_batches = 0

    for images, masks in tqdm(loader, desc="validate", leave=False):
        images = images.to(device, non_blocking=True)
        targets = masks.to(device, non_blocking=True).unsqueeze(1)

        context, _ = autocast_context(device, amp)
        with context:
            logits = model(images)
            loss = criterion(logits, targets)

        metrics.update(logits.float(), targets)
        total_loss += loss.item()
        n_batches += 1

    results = metrics.compute()
    results["loss"] = total_loss / max(1, n_batches)
    return results


def save_checkpoint(path, model, model_config, epoch, metrics, image_size) -> None:
    """
    Write the weights together with everything inference needs to rebuild them.
    :param path: Destination file
    :param model: Trained model
    :param model_config: Keyword arguments the model was constructed with
    :param epoch: Epoch the weights come from
    :param metrics: Validation metrics of this epoch
    :param image_size: Edge length the model was trained at
    :return: None
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "arch": type(model).__name__.lower(),
            "model_config": model_config,
            "image_size": image_size,
            "epoch": epoch,
            "metrics": metrics,
            "threshold": metrics.get("best_threshold", 0.5),
        },
        path,
    )


def fit(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs,
    scheduler=None,
    model_config=None,
    image_size=None,
    output_dir="runs/unet",
    amp=False,
    grad_clip=1.0,
    patience=0,
    thresholds=DEFAULT_THRESHOLDS,
) -> dict:
    """
    Train for a fixed budget, keeping the checkpoint with the best validation Dice.

    Selection uses dataset-level Dice at the best threshold rather than the
    validation loss, since that is what the segmentation is judged on.
    :param model: Model to train
    :param train_loader: Loader over the training split
    :param val_loader: Loader over the validation split
    :param criterion: Loss taking ``(logits, targets)``
    :param optimizer: Optimiser to step
    :param device: Device to run on
    :param epochs: Number of epochs to run
    :param scheduler: Per-step learning rate scheduler, or ``None``
    :param model_config: Keyword arguments the model was constructed with
    :param image_size: Edge length the model is trained at
    :param output_dir: Directory for checkpoints and the history file
    :param amp: Whether to run forward passes in mixed precision
    :param grad_clip: Global gradient norm clip, ``0`` to disable
    :param patience: Stop after this many epochs without improvement, ``0`` to disable
    :param thresholds: Probability thresholds to validate at
    :return: Dict with ``best_dice``, ``best_threshold``, ``best_epoch`` and ``history``
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scaler = torch.amp.GradScaler("cuda") if (amp and device.type == "cuda") else None

    history = []
    best_dice = -1.0
    best_epoch = -1
    best_threshold = 0.5
    epochs_without_improvement = 0

    for epoch in range(epochs):
        started = time.time()
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scheduler=scheduler,
            scaler=scaler,
            grad_clip=grad_clip,
            amp=amp,
            epoch=epoch,
            epochs=epochs,
        )
        metrics = evaluate(model, val_loader, criterion, device, thresholds, amp=amp)

        entry = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": metrics["loss"],
            "val_dice": metrics["best_dice"],
            "val_iou": metrics["best_iou"],
            "threshold": metrics["best_threshold"],
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - started,
        }
        history.append(entry)
        print(
            f"epoch {epoch + 1}/{epochs}  train {train_loss:.4f}  val {metrics['loss']:.4f}  "
            f"dice {metrics['best_dice']:.4f} @ t={metrics['best_threshold']:.2f}  "
            f"iou {metrics['best_iou']:.4f}  ({entry['seconds']:.0f}s)"
        )

        save_checkpoint(output_dir / "last.pt", model, model_config, epoch + 1, metrics, image_size)
        if metrics["best_dice"] > best_dice:
            best_dice = metrics["best_dice"]
            best_epoch = epoch + 1
            best_threshold = metrics["best_threshold"]
            epochs_without_improvement = 0
            save_checkpoint(output_dir / "best.pt", model, model_config, epoch + 1, metrics, image_size)
            print(f"  saved best.pt (dice {best_dice:.4f})")
        else:
            epochs_without_improvement += 1
            if patience and epochs_without_improvement >= patience:
                print(f"early stopping after {patience} epochs without improvement")
                break

        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    return {
        "best_dice": best_dice,
        "best_threshold": best_threshold,
        "best_epoch": best_epoch,
        "history": history,
    }

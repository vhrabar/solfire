"""
Training and validation loops.
"""

import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from .metrics import DEFAULT_THRESHOLDS, PanopticQuality, grid_edges
from .plots import plot_evaluation


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
    :return: Metrics dict from :meth:`PanopticQuality.compute` plus ``loss``
    """
    model.eval()
    metrics = PanopticQuality(thresholds)
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


def save_checkpoint(path, model, model_config, epoch, metrics, image_size, extra=None) -> None:
    """
    Write the weights together with everything inference needs to rebuild them.
    :param path: Destination file
    :param model: Trained model
    :param model_config: Keyword arguments the model was constructed with
    :param epoch: Epoch the weights come from
    :param metrics: Validation metrics of this epoch
    :param image_size: Edge length the model was trained at, or ``None`` when it was
        trained on crops of observations at their native resolution
    :param extra: Further fields to store, such as the post-processing calibration
    :return: None
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model_state": model.state_dict(),
        "arch": type(model).__name__.lower(),
        "model_config": model_config,
        "image_size": image_size,
        "epoch": epoch,
        "metrics": metrics,
        "threshold": metrics.get("best_threshold", 0.5),
    }
    state.update(extra or {})
    torch.save(state, path)


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
    evaluate_fn=None,
    extra=None,
    swa_epochs=0,
) -> dict:
    """
    Train for a fixed budget, keeping the checkpoint with the best validation PQ.

    Selection uses the Panoptic Quality at the best threshold rather than the
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
    :param evaluate_fn: Callable taking the model and returning the metrics dict, used
        to validate the way the submission is written; ``None`` scores ``val_loader``
        at the resolution it yields
    :param extra: Further fields to store in every checkpoint
    :param swa_epochs: Average the weights over this many final epochs into ``swa.pt``;
        ``0`` disables it. Worth using once the validation score has flattened, where the
        run is moving around a minimum rather than still descending into one
    :return: Dict with ``best_pq``, ``best_threshold``, ``best_epoch``, ``history`` and,
        when the weights were averaged, ``swa_pq``
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scaler = torch.amp.GradScaler("cuda") if (amp and device.type == "cuda") else None

    averaged = torch.optim.swa_utils.AveragedModel(model) if swa_epochs else None
    swa_start = epochs - swa_epochs

    history = []
    best_pq = -1.0
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
        if evaluate_fn is not None:
            metrics = evaluate_fn(model)
        else:
            metrics = evaluate(model, val_loader, criterion, device, thresholds, amp=amp)
        overlaps = metrics.pop("overlaps")

        entry = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": metrics["loss"],
            "val_pq": metrics["best_pq"],
            "threshold": metrics["best_threshold"],
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - started,
        }
        history.append(entry)
        print(
            f"epoch {epoch + 1}/{epochs}  train {train_loss:.4f}  val {metrics['loss']:.4f}  "
            f"pq {metrics['best_pq']:.4f} @ t={metrics['best_threshold']:.2f}  ({entry['seconds']:.0f}s)"
        )

        if averaged is not None and epoch >= swa_start:
            averaged.update_parameters(model)

        save_checkpoint(output_dir / "last.pt", model, model_config, epoch + 1, metrics, image_size, extra=extra)
        if metrics["best_pq"] > best_pq:
            best_pq = metrics["best_pq"]
            best_epoch = epoch + 1
            best_threshold = metrics["best_threshold"]
            epochs_without_improvement = 0
            save_checkpoint(output_dir / "best.pt", model, model_config, epoch + 1, metrics, image_size, extra=extra)
            plot_evaluation(overlaps, output_dir / "evaluation.png")
            print(f"  saved best.pt (pq {best_pq:.4f})")
        else:
            epochs_without_improvement += 1
            if patience and epochs_without_improvement >= patience:
                print(f"early stopping after {patience} epochs without improvement")
                break

        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    summary = {
        "best_pq": best_pq,
        "best_threshold": best_threshold,
        "best_epoch": best_epoch,
        "history": history,
    }
    if grid_edges({"threshold": best_threshold}, {"threshold": thresholds}):
        print(
            f"warning: the best threshold {best_threshold:.2f} sits at the edge of the grid "
            f"{min(thresholds):.2f}..{max(thresholds):.2f}, so the optimum is outside it"
        )

    if averaged is not None and int(averaged.n_averaged) > 0:
        summary |= average_weights(
            averaged,
            train_loader,
            val_loader,
            criterion,
            device,
            model_config,
            image_size,
            output_dir,
            thresholds,
            amp,
            evaluate_fn,
            extra,
        )
    return summary


def average_weights(
    averaged,
    train_loader,
    val_loader,
    criterion,
    device,
    model_config,
    image_size,
    output_dir,
    thresholds,
    amp,
    evaluate_fn,
    extra,
) -> dict:
    """
    Score and store the averaged weights of the final epochs.

    The batch normalisation statistics belong to no single set of weights, so they are
    recomputed over the training split before the average is worth scoring.
    :param averaged: The :class:`torch.optim.swa_utils.AveragedModel` collected while training
    :param train_loader: Loader the batch normalisation statistics are recomputed over
    :param val_loader: Loader over the validation split
    :param criterion: Loss taking ``(logits, targets)``
    :param device: Device to run on
    :param model_config: Keyword arguments the model was constructed with
    :param image_size: Edge length the model was trained at
    :param output_dir: Directory to write ``swa.pt`` to
    :param thresholds: Probability thresholds to validate at
    :param amp: Whether to run forward passes in mixed precision
    :param evaluate_fn: Callable taking the model and returning the metrics dict, or ``None``
    :param extra: Further fields to store in the checkpoint
    :return: Dict with ``swa_pq``, ``swa_threshold`` and ``swa_epochs``
    """
    epochs = int(averaged.n_averaged)
    print(f"averaging the weights of the last {epochs} epochs, recomputing batch statistics")
    torch.optim.swa_utils.update_bn(train_loader, averaged, device=device)

    model = averaged.module
    metrics = (
        evaluate_fn(model) if evaluate_fn is not None else evaluate(model, val_loader, criterion, device, thresholds, amp=amp)
    )
    metrics.pop("overlaps", None)
    save_checkpoint(Path(output_dir) / "swa.pt", model, model_config, epochs, metrics, image_size, extra=extra)
    print(f"swa pq {metrics['best_pq']:.4f} @ t={metrics['best_threshold']:.2f} -> {output_dir}/swa.pt")

    return {"swa_pq": metrics["best_pq"], "swa_threshold": metrics["best_threshold"], "swa_epochs": epochs}

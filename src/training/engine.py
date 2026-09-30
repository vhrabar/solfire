"""
Training and validation loops.
"""

import json
import random
import time
from pathlib import Path

import numpy as np
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


def batchnorm_running_stats(model) -> dict[str, torch.Tensor]:
    """
    Snapshot every BatchNorm running mean/variance, so a bad step can be rolled back.

    :param model: Model to snapshot
    :return: Mapping of buffer name to a detached clone
    """
    return {name: buf.detach().clone() for name, buf in model.named_buffers() if name.endswith(("running_mean", "running_var"))}


def restore_batchnorm_running_stats(model, snapshot: dict[str, torch.Tensor]) -> None:
    """
    Undo a bad step's damage to the running statistics.

    :param model: Model to restore
    :param snapshot: Mapping produced by :func:`batchnorm_running_stats`
    :return: None
    """
    buffers = dict(model.named_buffers())
    for name, value in snapshot.items():
        buffers[name].copy_(value)


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
    if hasattr(criterion, "set_epoch"):
        criterion.set_epoch(epoch)
    total_loss = 0.0
    n_batches = 0
    n_skipped = 0
    progress = tqdm(loader, desc=f"epoch {epoch + 1}/{epochs} train", leave=False, mininterval=2.0)

    for images, masks in progress:
        images = images.to(device, non_blocking=True)
        targets = masks.to(device, non_blocking=True)
        targets = targets.unsqueeze(1) if targets.dim() == 3 else targets

        bn_stats = batchnorm_running_stats(model)

        optimizer.zero_grad(set_to_none=True)
        context, _ = autocast_context(device, amp)
        with context:
            loss = criterion(model(images), targets)

        if not torch.isfinite(loss):
            restore_batchnorm_running_stats(model, bn_stats)
            n_skipped += 1
            continue

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

    if n_skipped:
        print(f"  epoch {epoch + 1}: skipped {n_skipped} batch(es) with non-finite loss, rolled back batch-norm stats")

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

    for images, masks in tqdm(loader, desc="validate", leave=False, mininterval=2.0):
        images = images.to(device, non_blocking=True)
        targets = masks.to(device, non_blocking=True)
        targets = targets.unsqueeze(1) if targets.dim() == 3 else targets

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


def _rng_state() -> dict:
    """Snapshot every RNG source :func:`fit` draws from, for a byte-for-byte resume."""
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "python": random.getstate(),
        "numpy": np.random.get_state(),  # noqa: NPY002 - matches the legacy global RNG main.py seeds
    }


def _restore_rng_state(state: dict) -> None:
    """Best-effort restore of :func:`_rng_state`; a mismatch is logged, not fatal."""
    try:
        torch.set_rng_state(state["torch"])
        if state.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])  # noqa: NPY002
    except Exception as error:
        print(f"  could not restore RNG state ({error}); continuing with fresh randomness")


def _resume_fields(
    optimizer, scheduler, scaler, averaged, total_epochs, best_pq, best_epoch, best_threshold,
    epochs_without_improvement, history,
) -> dict:
    """
    Extra fields ``last.pt`` carries on top of the model weights so :func:`fit` can pick a
    killed or interrupted run back up where it left off.

    :return: Merged into ``last.pt``'s ``extra``; ``None`` entries are skipped by ``save_checkpoint``
    """
    return {
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "averaged_state": averaged.state_dict() if averaged is not None else None,
        "rng_state": _rng_state(),
        "total_epochs": total_epochs,
        "best_pq": best_pq,
        "best_epoch": best_epoch,
        "best_threshold": best_threshold,
        "epochs_without_improvement": epochs_without_improvement,
        "history": history,
    }


def _load_resume_state(last_path, model, optimizer, scheduler, scaler, averaged, epochs, device) -> dict:
    """
    Restore model/optimizer/scheduler/scaler/SWA-average/RNG from ``last_path`` in place.

    :param last_path: ``last.pt`` written by a previous, incomplete call to :func:`fit`
    :param model: Model to load the saved weights into
    :param optimizer: Optimiser to restore state into
    :param scheduler: Per-step scheduler to restore state into, or ``None``
    :param scaler: AMP gradient scaler to restore state into, or ``None``
    :param averaged: SWA :class:`~torch.optim.swa_utils.AveragedModel` to restore, or ``None``
    :param epochs: Total epochs requested for this call, to size the remaining range
    :param device: Device the checkpoint is loaded onto
    :return: ``{"start_epoch", "best_pq", "best_epoch", "best_threshold",
        "epochs_without_improvement", "history"}``
    """
    checkpoint = torch.load(last_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    if checkpoint.get("optimizer_state") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    else:
        print("  last.pt has no optimizer state (checkpoint predates --resume); resuming with a fresh optimizer")
    if scheduler is not None and checkpoint.get("scheduler_state") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    if scaler is not None and checkpoint.get("scaler_state") is not None:
        scaler.load_state_dict(checkpoint["scaler_state"])
    if averaged is not None and checkpoint.get("averaged_state") is not None:
        averaged.load_state_dict(checkpoint["averaged_state"])
    if checkpoint.get("rng_state") is not None:
        _restore_rng_state(checkpoint["rng_state"])

    total_epochs = checkpoint.get("total_epochs")
    if total_epochs not in (None, epochs):
        print(
            f"  warning: last.pt was saved with --epochs {total_epochs}, resuming under --epochs {epochs}; "
            "the LR schedule and SWA window were built for the old total"
        )

    start_epoch = min(int(checkpoint.get("epoch", 0)), epochs)
    state = {
        "start_epoch": start_epoch,
        "best_pq": float(checkpoint.get("best_pq", -1.0)),
        "best_epoch": int(checkpoint.get("best_epoch", -1)),
        "best_threshold": float(checkpoint.get("best_threshold", 0.5)),
        "epochs_without_improvement": int(checkpoint.get("epochs_without_improvement", 0)),
        "history": list(checkpoint.get("history", [])),
    }
    if start_epoch >= epochs:
        print(f"  last.pt already reached epoch {start_epoch}/{epochs}; nothing left to train")
    else:
        print(
            f"  resuming from epoch {start_epoch}/{epochs} "
            f"(best pq so far {state['best_pq']:.4f} @ epoch {state['best_epoch']})"
        )
    return state


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
    resume=False,
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
    :param resume: Continue from ``output_dir/last.pt`` when it exists, restoring the
        optimiser, scheduler, AMP scaler, SWA average and RNG state where the last
        completed epoch left them; ``False`` (or no ``last.pt`` yet) starts from scratch
    :return: Dict with ``best_pq``, ``best_threshold``, ``best_epoch``, ``history`` and,
        when the weights were averaged, ``swa_pq``
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scaler = torch.amp.GradScaler("cuda") if (amp and device.type == "cuda") else None

    averaged = torch.optim.swa_utils.AveragedModel(model) if swa_epochs else None
    swa_start = epochs - swa_epochs

    last_path = output_dir / "last.pt"
    if resume and last_path.is_file():
        state = _load_resume_state(last_path, model, optimizer, scheduler, scaler, averaged, epochs, device)
    else:
        if resume:
            print(f"  --resume set but {last_path} does not exist; starting from scratch")
        state = {"start_epoch": 0, "best_pq": -1.0, "best_epoch": -1, "best_threshold": 0.5, "epochs_without_improvement": 0, "history": []}

    history = state["history"]
    best_pq = state["best_pq"]
    best_epoch = state["best_epoch"]
    best_threshold = state["best_threshold"]
    epochs_without_improvement = state["epochs_without_improvement"]

    for epoch in range(state["start_epoch"], epochs):
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

        improved = metrics["best_pq"] > best_pq
        if improved:
            best_pq = metrics["best_pq"]
            best_epoch = epoch + 1
            best_threshold = metrics["best_threshold"]
            epochs_without_improvement = 0
            save_checkpoint(output_dir / "best.pt", model, model_config, epoch + 1, metrics, image_size, extra=extra)
            plot_evaluation(overlaps, output_dir / "evaluation.png")
            print(f"  saved best.pt (pq {best_pq:.4f})")
        else:
            epochs_without_improvement += 1

        # last.pt carries everything a resumed run needs, on top of what best.pt/swa.pt store
        resume_extra = {**(extra or {}), **_resume_fields(
            optimizer, scheduler, scaler, averaged, epochs, best_pq, best_epoch, best_threshold,
            epochs_without_improvement, history,
        )}
        save_checkpoint(output_dir / "last.pt", model, model_config, epoch + 1, metrics, image_size, extra=resume_extra)
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        if not improved and patience and epochs_without_improvement >= patience:
            print(f"early stopping after {patience} epochs without improvement")
            break

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

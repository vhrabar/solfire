"""
Training and out-of-fold scoring of the crop refiner (:mod:`inference.refiner`).

The refiner learns from the components of the pooled out-of-fold ensemble maps, so
it is trained on the proposals it will see at test time rather than on boxes drawn
around the annotations. The refiner of fold ``k`` never sees an observation of fold
``k``: it is trained on the held-out maps of the other folds and scored on fold ``k``.
"""

import multiprocessing
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from threadpoolctl import threadpool_limits
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..data import TRAIN_IMAGES, load_image
from ..data.preprocess import observation_path
from ..inference.postprocess import SUBMISSION_SIZE, apply_offset, connected_components
from ..inference.refiner import (
    CONTEXT,
    CROP_SIZE,
    MIN_PIXELS,
    RefinedInstance,
    Refiner,
    assemble,
    component_window,
    refine_components,
    refiner_inputs,
    resample,
    window_of,
)
from .calibrate import load_probabilities
from .engine import atomic_save
from .metrics import IOU_THRESHOLD, annotation_pixels, instance_iou, pq_counts, pq_from_counts
from .optim import build_optimizer, build_scheduler

MAX_RECORDS = 3
# the cuts the proposals are taken at in training; the calibrated one sits in the middle
TRAIN_THRESHOLDS = (0.3, 0.375, 0.45)
# a proposal this close to one already taken from the observation adds nothing new
DUPLICATE_IOU = 0.85
REFINE_THRESHOLDS = (0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6)
REFINE_MIN_SCORES = tuple(round(value, 2) for value in np.arange(0.0, 0.62, 0.02))


def collect_crops(
    sources, records, folds, output_dir, offsets=(-1,), thresholds=TRAIN_THRESHOLDS, workers=8, size=CROP_SIZE, context=CONTEXT
):
    """
    Cut every proposal of the held-out maps into a refiner sample, on disk.

    :param sources: Per observation file, the cache directories of the models that held it out
    :param records: Per observation file, the annotations of each of its records
    :param folds: Per observation file, the fold that held it out
    :param output_dir: Directory the samples are written to
    :param offsets: Morphological steps the proposals are cut with
    :param thresholds: Probability thresholds the proposals are cut at
    :param workers: Processes working in parallel
    :param size: Edge length the refiner works at
    :param context: Window edge as a multiple of the longer side of a proposal's box
    :return: Number of samples written
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(name, sources[name], records[name], offsets, thresholds, size, context) for name in sorted(sources)]
    chunks, meta = [], []
    chunk_dir = output_dir / "chunks"
    chunk_dir.mkdir(exist_ok=True)
    with ProcessPoolExecutor(
        max(1, workers), mp_context=multiprocessing.get_context("forkserver"), initializer=_single_thread
    ) as pool:
        for name, inputs, targets, rows in tqdm(
            pool.map(_collect_one, jobs), total=len(jobs), desc="proposals", leave=False, mininterval=2.0
        ):
            if not len(rows):
                continue
            path = chunk_dir / f"{Path(name).stem}.npz"
            np.savez(path, inputs=inputs, targets=targets)
            chunks.append(path)
            meta += [{**row, "file": name, "fold": folds[name]} for row in rows]

    total = len(meta)
    stored_inputs = np.lib.format.open_memmap(output_dir / "inputs.npy", "w+", np.uint8, (total, 3, size, size))
    stored_targets = np.lib.format.open_memmap(output_dir / "targets.npy", "w+", np.uint8, (total, MAX_RECORDS, size, size))
    start = 0
    for path in chunks:
        with np.load(path) as chunk:
            count = len(chunk["inputs"])
            stored_inputs[start : start + count] = chunk["inputs"]
            stored_targets[start : start + count] = chunk["targets"]
        start += count
        path.unlink()
    chunk_dir.rmdir()
    stored_inputs.flush()
    stored_targets.flush()
    with open(output_dir / "meta.pkl", "wb") as handle:
        pickle.dump(meta, handle)
    return total


def _single_thread() -> None:
    """
    Keep a worker to one thread: the pool already fills the cores.
    :return: None
    """
    threadpool_limits(1)
    torch.set_num_threads(1)


def _collect_one(job):
    """
    The refiner samples of one observation.
    :param job: Tuple of (file name, cache directories, annotations per record, offsets, thresholds, size, context)
    :return: Tuple of (file name, ``(n, 3, S, S)`` uint8 inputs, ``(n, MAX_RECORDS, S, S)``
        uint8 targets, per sample a dict of descriptive fields)
    """
    name, cache_dirs, annotations, offsets, thresholds, size, context = job
    probabilities = load_probabilities(cache_dirs, name)
    image = load_image(observation_path(TRAIN_IMAGES, name)).numpy()[0]
    height, width = probabilities.shape
    gt = [annotation_pixels(record, height, width) for record in annotations]
    if len(gt) > MAX_RECORDS:
        raise ValueError(f"{name} has {len(gt)} records, more than the {MAX_RECORDS} the samples hold")

    taken: list[np.ndarray] = []
    inputs, targets, rows = [], [], []
    for offset, threshold in ((offset, threshold) for offset in offsets for threshold in thresholds):
        labels, n_labels = connected_components(apply_offset(probabilities >= threshold, offset))
        if not n_labels:
            continue
        areas = np.bincount(labels.ravel(), minlength=n_labels + 1)
        boxes = ndimage.find_objects(labels)
        overlaps = [instance_iou(pixels, labels, n_labels) for pixels in gt]
        for index, box in enumerate(boxes):
            if box is None or areas[index + 1] < MIN_PIXELS:
                continue
            seed = labels == index + 1
            flat = np.flatnonzero(seed)
            if any(_flat_iou(flat, other) >= DUPLICATE_IOU for other in taken):
                continue
            taken.append(flat)
            window = component_window(box, context)
            sample = refiner_inputs(image, probabilities, seed, window, size)
            target = np.zeros((MAX_RECORDS, size, size), dtype=np.uint8)
            earned = []
            for record, (pixels, iou) in enumerate(zip(gt, overlaps)):
                column = iou[:, index] if iou.shape[0] else np.zeros(0)
                if len(column) and column.max() > 0:
                    best = int(column.argmax())
                    target[record] = _target_window(pixels[best], window, height, width, size)
                    earned.append(column[best] if column[best] > IOU_THRESHOLD else 0.0)
                else:
                    earned.append(0.0)
            inputs.append((sample.numpy() * 255).round().astype(np.uint8))
            targets.append(target)
            rows.append(
                {
                    "threshold": threshold,
                    "offset": offset,
                    "area": int(areas[index + 1]),
                    "score": float(probabilities[seed].mean()),
                    "records": len(gt),
                    "earned": float(np.mean(earned)),
                }
            )
    if not rows:
        return name, None, None, []
    return name, np.stack(inputs), np.stack(targets), rows


def _flat_iou(first, second) -> float:
    """
    :param first: Sorted flat pixel indices
    :param second: Sorted flat pixel indices
    :return: IoU of the two pixel sets
    """
    if first[0] > second[-1] or second[0] > first[-1]:
        return 0.0
    shared = len(np.intersect1d(first, second, assume_unique=True))
    return shared / (len(first) + len(second) - shared)


def _target_window(pixels, window, height, width, size) -> np.ndarray:
    """
    One annotated filament on a refiner window.
    :param pixels: Flat pixel indices of the filament
    :param window: Window of the proposal
    :param height: Height of the observation
    :param width: Width of the observation
    :param size: Edge length the refiner works at
    :return: ``(size, size)`` uint8 soft mask
    """
    mask = np.zeros(height * width, dtype=np.float32)
    mask[pixels] = 1.0
    cut = window_of(mask.reshape(height, width), window)
    return (resample(torch.from_numpy(cut)[None, None], size)[0, 0].numpy() * 255).round().astype(np.uint8)


class RefinerCrops(Dataset):
    """
    The samples written by :func:`collect_crops`, with flips, transposes and a brightness jitter.
    """

    def __init__(self, directory, indices, augment=True):
        """
        :param directory: Directory written by :func:`collect_crops`
        :param indices: Samples to serve
        :param augment: Apply the random symmetries and jitter
        """
        self.directory = Path(directory)
        self.indices = np.asarray(indices)
        self.augment = augment
        with open(self.directory / "meta.pkl", "rb") as handle:
            self.meta = pickle.load(handle)
        self.inputs: np.ndarray | None = None
        self.targets: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item):
        """
        :param item: Position in ``indices``
        :return: Tuple of (``(3, S, S)`` inputs, ``(MAX_RECORDS, S, S)`` targets, number of records)
        """
        if self.inputs is None or self.targets is None:
            self.inputs = np.load(self.directory / "inputs.npy", mmap_mode="r")
            self.targets = np.load(self.directory / "targets.npy", mmap_mode="r")
        index = int(self.indices[item])
        inputs = torch.from_numpy(np.array(self.inputs[index], dtype=np.float32) / 255.0)
        targets = torch.from_numpy(np.array(self.targets[index], dtype=np.float32) / 255.0)
        if self.augment:
            flip_x, flip_y, transpose, gain, gamma = torch.rand(5).tolist()
            if flip_x < 0.5:
                inputs, targets = inputs.flip(-1), targets.flip(-1)
            if flip_y < 0.5:
                inputs, targets = inputs.flip(-2), targets.flip(-2)
            if transpose < 0.5:
                inputs, targets = inputs.transpose(-1, -2), targets.transpose(-1, -2)
            gain, gamma = 0.85 + 0.3 * gain, 0.8 + 0.45 * gamma
            inputs[0] = (inputs[0].clamp(0, 1) ** gamma * gain).clamp(0, 1)
        return inputs, targets, self.meta[index]["records"]


def refiner_loss(masks, scores, targets, n_records, score_weight=0.5):
    """
    Mask loss against the record average, plus the error of the predicted earned IoU.

    The earned IoU is measured on the refined mask itself, cut at a half, against each
    record's filament, so the score learns what the submitted mask will be worth.
    :param masks: ``(B, 1, S, S)`` mask logits
    :param scores: ``(B,)`` logits of the earned IoU
    :param targets: ``(B, MAX_RECORDS, S, S)`` per record soft targets
    :param n_records: ``(B,)`` records of each sample
    :param score_weight: Weight of the score term
    :return: Tuple of (total loss, mask loss, score loss)
    """
    valid = (torch.arange(MAX_RECORDS, device=targets.device)[None] < n_records[:, None]).float()
    soft = (targets * valid[:, :, None, None]).sum(1, keepdim=True) / n_records[:, None, None, None].float()
    logits = masks.float()
    bce = F.binary_cross_entropy_with_logits(logits, soft)
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * soft).sum((1, 2, 3))
    dice = 1 - (2 * intersection + 1) / (probabilities.sum((1, 2, 3)) + soft.sum((1, 2, 3)) + 1)

    with torch.no_grad():
        predicted = probabilities >= 0.5
        truth = targets >= 0.5
        shared = (predicted & truth).sum((2, 3)).float()
        union = (predicted | truth).sum((2, 3)).float()
        iou = torch.where(union > 0, shared / union.clamp(min=1), torch.zeros_like(union))
        earned = (torch.where(iou > IOU_THRESHOLD, iou, torch.zeros_like(iou)) * valid).sum(1) / n_records.float()
    score = F.binary_cross_entropy_with_logits(scores.float(), earned)
    mask_loss = bce + dice.mean()
    return mask_loss + score_weight * score, mask_loss, score


def train_refiner(
    directory, fold, output, device, epochs=12, batch_size=32, lr=3e-4, encoder_lr_scale=0.5, workers=8, seed=1
) -> dict:
    """
    Train the refiner of one fold on the samples of every other fold.
    :param directory: Directory written by :func:`collect_crops`
    :param fold: Fold left out
    :param output: Checkpoint path
    :param device: Device to train on
    :param epochs: Passes over the samples
    :param batch_size: Samples per step
    :param lr: Peak learning rate of the decoder and heads
    :param encoder_lr_scale: Learning rate of the encoder as a fraction of ``lr``
    :param workers: Loader processes
    :param seed: Seed of the initialisation and the sampling
    :return: Per epoch mean losses
    """
    torch.manual_seed(seed)
    with open(Path(directory) / "meta.pkl", "rb") as handle:
        meta = pickle.load(handle)
    indices = [index for index, row in enumerate(meta) if row["fold"] != fold]
    loader = DataLoader(
        RefinerCrops(directory, indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        drop_last=True,
        persistent_workers=workers > 0,
    )
    model = Refiner().to(device)
    optimizer = build_optimizer(model, lr=lr, encoder_lr_scale=encoder_lr_scale)
    scheduler = build_scheduler(optimizer, epochs, len(loader), warmup_epochs=1)
    history = []
    # every finished epoch is kept here, so a killed run picks up where it stopped
    last = Path(output).with_suffix(".last.pt")
    if last.is_file():
        state = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["rng"])
        history = state["history"]
        print(f"fold {fold}: resuming after epoch {len(history)}/{epochs}")
    for epoch in range(len(history), epochs):
        model.train()
        totals = np.zeros(3)
        progress = tqdm(loader, desc=f"fold {fold} epoch {epoch + 1}/{epochs}", leave=False, mininterval=2.0)
        for inputs, targets, n_records in progress:
            inputs, targets, n_records = inputs.to(device), targets.to(device), n_records.to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                masks, scores = model(inputs)
            loss, mask_loss, score_loss = refiner_loss(masks, scores, targets, n_records)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            totals += [loss.item(), mask_loss.item(), score_loss.item()]
        history.append(dict(zip(("loss", "mask", "score"), (totals / len(loader)).tolist())))
        print(f"fold {fold} epoch {epoch + 1}/{epochs} " + " ".join(f"{k} {v:.4f}" for k, v in history[-1].items()))
        last.parent.mkdir(parents=True, exist_ok=True)
        atomic_save(
            {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "rng": torch.get_rng_state(),
                "history": history,
            },
            last,
        )
    atomic_save({"state_dict": model.state_dict(), "fold": fold, "epochs": epochs, "size": CROP_SIZE, "history": history}, output)
    last.unlink(missing_ok=True)
    return {"history": history}


def refine_held_out(models, sources, folds, output_dir, device, threshold=0.375, offset=-1, tta=True, context=CONTEXT) -> None:
    """
    Refine every held-out observation with the refiner that did not train on its fold.
    :param models: Per fold, a :class:`Refiner` in eval mode
    :param sources: Per observation file, the cache directories of the models that held it out
    :param folds: Per observation file, its fold
    :param output_dir: Directory a pickle of the refined instances per observation goes to
    :param device: Device to run on
    :param threshold: Probability the proposals are cut at
    :param offset: Morphological step the proposals are cut with
    :param tta: Average the refiner over flips
    :param context: Window edge as a multiple of the longer side of a proposal's box
    :return: None
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in tqdm(sorted(sources), desc="refining", leave=False, mininterval=2.0):
        path = output_dir / f"{Path(name).stem}.pkl"
        if path.is_file():
            continue
        probabilities = load_probabilities(sources[name], name)
        image = load_image(observation_path(TRAIN_IMAGES, name)).numpy()[0]
        labels, n_labels = connected_components(apply_offset(probabilities >= threshold, offset))
        refined = refine_components(models[folds[name]], image, probabilities, labels, n_labels, device, tta=tta, context=context)
        seeds = seed_instances(labels, n_labels, probabilities)
        with open(path, "wb") as handle:
            pickle.dump({"refined": refined, "seeds": seeds}, handle)


def seed_instances(labels, n_labels, probabilities) -> list[RefinedInstance]:
    """
    The unrefined proposals in the shape of refined ones, scored by their mean probability.
    :param labels: ``(H, W)`` proposals
    :param n_labels: Number of proposals
    :param probabilities: ``(H, W)`` ensemble probabilities
    :return: One :class:`RefinedInstance` per proposal of at least :data:`MIN_PIXELS` pixels
    """
    areas = np.bincount(labels.ravel(), minlength=n_labels + 1)
    seeds = []
    for index, box in enumerate(ndimage.find_objects(labels)):
        if box is None or areas[index + 1] < MIN_PIXELS:
            continue
        window = component_window(box)
        mask = window_of(labels == index + 1, window).astype(np.uint8) * 255
        seeds.append(RefinedInstance(window, mask, float(probabilities[labels == index + 1].mean())))
    return seeds


def score_refined(
    refined_dir, records, folds=None, thresholds=REFINE_THRESHOLDS, min_scores=REFINE_MIN_SCORES, workers=8
) -> dict:
    """
    Panoptic Quality of the refined instances over a grid of mask cuts and score cuts.

    Three readings are scored: ``refined`` (redrawn masks, refiner scores), ``rescored``
    (proposal masks, refiner scores) and ``seeds`` (proposal masks and mean probability,
    the shipped rule), so the gain of the masks and of the scores can be told apart.
    :param refined_dir: Directory written by :func:`refine_held_out`
    :param records: Per observation file, the annotations of each of its records
    :param folds: Per observation file, its fold; given, each reading also gets a
        ``nested`` score, its cuts chosen on the other folds and scored on the held-out one
    :param thresholds: Cuts on the refined probabilities
    :param min_scores: Cuts on the instance score
    :param workers: Processes working in parallel
    :return: Per reading, the winning setting with its ``pq`` and counts, its table, and
        with ``folds`` the ``nested`` pooled score and the cuts each fold chose
    """
    names = sorted(records)
    jobs = [(name, Path(refined_dir), records[name], tuple(thresholds), tuple(min_scores)) for name in names]
    by_fold: dict[int, dict[tuple, np.ndarray]] = {}
    with ProcessPoolExecutor(
        max(1, workers), mp_context=multiprocessing.get_context("forkserver"), initializer=_single_thread
    ) as pool:
        for name, terms in zip(
            names, tqdm(pool.map(_score_one, jobs), total=len(jobs), desc="scoring", leave=False, mininterval=2.0)
        ):
            fold_terms = by_fold.setdefault(folds[name] if folds else 0, {})
            for key, value in terms.items():
                fold_terms[key] = fold_terms.get(key, 0) + value
    accumulated: dict[tuple, np.ndarray] = {}
    for fold_terms in by_fold.values():
        for key, value in fold_terms.items():
            accumulated[key] = accumulated.get(key, 0) + value
    report = {}
    for reading in ("refined", "rescored", "seeds"):
        table = [
            {"threshold": t, "min_score": s, "pq": pq_from_counts(*v), "tp": int(v[1]), "fp": int(v[2]), "fn": int(v[3])}
            for (r, t, s), v in accumulated.items()
            if r == reading
        ]
        table.sort(key=lambda row: -row["pq"])
        report[reading] = {**table[0], "table": table}
        if folds:
            report[reading]["nested"] = _nested(by_fold, accumulated, reading)
    return report


def _nested(by_fold, accumulated, reading) -> dict:
    """
    Choose the cuts of a reading on all folds but one, score them on that one, and pool.
    :param by_fold: Per fold, the summed terms keyed by ``(reading, threshold, min_score)``
    :param accumulated: The terms summed over every fold
    :param reading: Reading to score
    :return: Dict with the pooled ``pq`` and counts and the ``chosen`` cuts per fold
    """
    pooled = np.zeros(4)
    chosen = {}
    keys = [key for key in accumulated if key[0] == reading]
    for fold, fold_terms in sorted(by_fold.items()):
        best = max(keys, key=lambda key: pq_from_counts(*(accumulated[key] - fold_terms[key])))
        pooled += fold_terms[best]
        chosen[fold] = {"threshold": best[1], "min_score": float(best[2])}
    return {"pq": pq_from_counts(*pooled), "tp": int(pooled[1]), "fp": int(pooled[2]), "fn": int(pooled[3]), "chosen": chosen}


def _score_one(job) -> dict:
    """
    :param job: Tuple of (file name, refined directory, annotations per record, thresholds, min scores)
    :return: Dict keyed by ``(reading, threshold, min_score)`` of summed ``[iou, tp, fp, fn]``
    """
    name, refined_dir, annotations, thresholds, min_scores = job
    with open(refined_dir / f"{Path(name).stem}.pkl", "rb") as handle:
        stored = pickle.load(handle)
    gt = [annotation_pixels(record) for record in annotations]
    readings = {
        "refined": (stored["refined"], thresholds),
        "rescored": ([_with_score(seed, done.score) for seed, done in zip(stored["seeds"], stored["refined"])], (0.5,)),
        "seeds": (stored["seeds"], (0.5,)),
    }
    terms = {}
    for reading, (instances, cuts) in readings.items():
        scores_for_grid = min_scores if reading != "seeds" else SEED_MIN_SCORES
        for threshold in cuts:
            labels, n_labels, scores = assemble(instances, SUBMISSION_SIZE, SUBMISSION_SIZE, threshold=threshold)
            overlaps = [instance_iou(pixels, labels, n_labels) for pixels in gt]
            for min_score in scores_for_grid:
                keep = scores >= min_score
                total = np.zeros(4)
                for iou in overlaps:
                    total += pq_counts(iou[:, keep])
                terms[(reading, threshold, min_score)] = total
    return terms


# the shipped min_score grid, for the unrefined reading
SEED_MIN_SCORES = (0.0, 0.6, 0.625, 0.65, 0.675, 0.7, 0.725, 0.75, 0.8)


def _with_score(instance, score) -> RefinedInstance:
    """
    :param instance: An instance
    :param score: Score to give it
    :return: A copy of the instance with that score
    """
    return RefinedInstance(instance.window, instance.probabilities, score)

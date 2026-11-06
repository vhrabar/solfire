"""
Barbs from the barb head (train --barb-head-weight), scored on the refined out-of-fold instances.
"""

import argparse
import json
import pickle
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from skimage.morphology import disk

from solfire.data import load_annotations
from solfire.data.dataset import validation_split
from solfire.experiments.data.structure import thin_mask
from solfire.inference.barbs import thin_pieces
from solfire.inference.postprocess import EIGHT_CONNECTED
from solfire.inference.predict import load_model
from solfire.inference.refiner import assemble
from solfire.training.calibrate import cache_probabilities, load_probabilities
from solfire.training.metrics import annotation_pixels, instance_iou, pq_counts, pq_from_counts

SIZE = 2048
THRESHOLDS = (0.1, 0.12, 0.13, 0.15, 0.17, 0.2, 0.3)
BAND = 20


def cache_dir(checkpoint) -> Path:
    """
    :param checkpoint: Fold checkpoint with a barb head
    :return: Directory its held-out barb maps are kept in
    """
    path = Path(checkpoint)
    return path.with_name(f"{path.stem}.barbhead.val")


def held_out_files(checkpoint, images) -> list[str]:
    """
    :param checkpoint: Fold checkpoint
    :param images: Image records
    :return: The observation files the checkpoint's fold held out
    """
    found = torch.load(checkpoint, map_location="cpu", weights_only=False)
    _, val_ids = validation_split(
        images, found.get("folds") or 0, found.get("fold") or 0, found.get("val_fraction") or 0.2, found.get("seed", 1)
    )
    names = {image["id"]: image["file_name"] for image in images}
    return sorted({names[image_id] for image_id in val_ids})


def cache(args) -> None:
    images, _ = load_annotations()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for checkpoint in args.checkpoints:
        model, found = load_model(checkpoint, device)
        if not found.get("barb_head_weight") or getattr(model, "aux_channels", 0) != 1:
            raise SystemExit(f"{checkpoint} has no barb head")
        model.inference_channels = slice(1, 2)
        files = held_out_files(checkpoint, images)
        print(f"{checkpoint}: barb maps for {len(files)} held-out observations", flush=True)
        cache_probabilities(
            model,
            files,
            cache_dir(checkpoint),
            device,
            tile=512,
            overlap=0.25,
            tta=True,
            batch_size=8,
            identity={"head": "barb", "epoch": found.get("epoch")},
        )


def grow(labels, n_labels, head, threshold, thin) -> np.ndarray:
    """
    :param labels: ``(H, W)`` refined instance labels
    :param n_labels: Number of instances
    :param head: ``(H, W)`` barb probabilities
    :param threshold: Barb probability a pixel needs
    :param thin: Keep only long, narrow pieces (:func:`~solfire.inference.barbs.thin_pieces`)
    :return: Labels with the barb pieces that touch an instance added to it
    """
    grown = labels.copy()
    height, width = labels.shape
    margin = BAND + 2
    for number, window in enumerate(ndimage.find_objects(labels), start=1):
        if window is None:
            continue
        window = (
            slice(max(window[0].start - margin, 0), min(window[0].stop + margin, height)),
            slice(max(window[1].start - margin, 0), min(window[1].stop + margin, width)),
        )
        mask = labels[window] == number
        band = ndimage.binary_dilation(mask, disk(BAND)) & (labels[window] == 0)
        candidates = band & (head[window] > threshold)
        pieces, _ = ndimage.label(candidates, structure=EIGHT_CONNECTED)
        touching = np.unique(pieces[ndimage.binary_dilation(mask, EIGHT_CONNECTED) & candidates])
        barbs = np.isin(pieces, touching[touching > 0])
        if thin:
            barbs = thin_pieces(barbs, mask, 4.0, 4.0)
        target = grown[window]
        target[barbs & (target == 0)] = number
    return grown


def score_job(job) -> dict:
    name, records, dirs, refined, cut, min_score = job
    with open(Path(refined) / f"{Path(name).stem}.pkl", "rb") as handle:
        stored = pickle.load(handle)
    labels, n_labels, _ = assemble(stored["refined"], SIZE, SIZE, threshold=cut, min_score=min_score)
    head = load_probabilities(dirs, name)
    gts = [annotation_pixels(record) for record in records]
    union = np.zeros(SIZE * SIZE, dtype=bool)
    thin = np.zeros((SIZE, SIZE), dtype=bool)
    for gt in gts:
        record_mask = np.zeros(SIZE * SIZE, dtype=bool)
        for pixels in gt:
            record_mask[pixels] = True
        union |= record_mask
        thin |= thin_mask(record_mask.reshape(SIZE, SIZE))
    union = union.reshape(SIZE, SIZE)
    out = {}
    variants = [("none", None, False)] + [
        (f"{'thin ' if t else ''}{threshold}", threshold, t) for t in (False, True) for threshold in THRESHOLDS
    ]
    for key, threshold, thin_only in variants:
        grown = labels if threshold is None else grow(labels, n_labels, head, threshold, thin_only)
        added = (grown > 0) & (labels == 0)
        counts = np.zeros(4)
        for gt in gts:
            counts += pq_counts(instance_iou(gt, grown, n_labels))
        out[key] = [*counts, int(added.sum()), int((added & union).sum()), int(((grown > 0) & thin).sum()), int(thin.sum())]
    return out


def score(args) -> None:
    images, by_image = load_annotations()
    files = defaultdict(list)
    for image in sorted(images, key=lambda item: item["id"]):
        files[image["file_name"]].append(by_image[image["id"]])
    sources: dict[str, list[Path]] = defaultdict(list)
    for checkpoint in args.checkpoints:
        for name in held_out_files(checkpoint, images):
            sources[name].append(cache_dir(checkpoint))
    jobs = [(name, records, sources[name], args.refined, args.cut, args.min_score) for name, records in sorted(files.items())]
    missing = [job[0] for job in jobs if not job[2]]
    if missing:
        raise SystemExit(f"{len(missing)} observations have no barb map, e.g. {missing[0]}")
    totals: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(8))
    with ProcessPoolExecutor(args.workers) as pool:
        for result in pool.map(score_job, jobs, chunksize=4):
            for key, values in result.items():
                totals[key] += values
    base = pq_from_counts(*totals["none"][:4])
    print(
        f"{len(jobs)} observations, refined {args.refined} at {args.cut} / {args.min_score}, {len(args.checkpoints)} barb heads"
    )
    rows = {}
    for key, values in totals.items():
        pq = pq_from_counts(*values[:4])
        added, inside, thin_hit, thin_all = values[4:]
        rows[key] = {"pq": pq, "added": int(added), "inside_gt": inside / max(added, 1), "thin_recall": thin_hit / thin_all}
        per_million = (pq - base) / added * 1e6 if added else 0.0
        print(
            f"  {key:9s} PQ {pq:.4f} ({pq - base:+.4f})  added {int(added):8d} px  inside GT {inside / max(added, 1):.3f}  "
            f"thin recall {thin_hit / thin_all:.3f}  PQ per 1M px {per_million:+.3f}"
        )
    if args.output:
        Path(args.output).write_text(json.dumps(rows, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    cached = commands.add_parser("cache", help="write the held-out barb maps of fold checkpoints")
    cached.add_argument("checkpoints", nargs="+")
    scored = commands.add_parser("score", help="grow the barb maps into the refined OOF instances and score them")
    scored.add_argument("checkpoints", nargs="+")
    scored.add_argument("--refined", default="runs/refiner/zoom-ssl-tta-rt/refined")
    scored.add_argument("--cut", type=float, default=0.45)
    scored.add_argument("--min-score", type=float, default=0.32)
    scored.add_argument("--workers", type=int, default=6)
    scored.add_argument("--output", default=None)
    args = parser.parse_args()
    cache(args) if args.command == "cache" else score(args)


if __name__ == "__main__":
    main()

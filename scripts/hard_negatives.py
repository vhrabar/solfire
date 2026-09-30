"""
Find the dark features models mistake for filaments, for hard-negative crop sampling.`.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_annotations, validation_split
from src.inference.postprocess import connected_components, instance_scores
from src.training.calibrate import cache_directory, load_probabilities
from src.training.metrics import annotation_pixels


def main() -> int:
    """
    Write the hard-negative points of every held-out observation of the checkpoints.
    :return: Process exit code
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoints", nargs="+", type=Path, help="fold checkpoints whose cached held-out maps are used")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-score", type=float, default=0.5, help="mean probability a component needs")
    parser.add_argument("--min-area", type=int, default=128)
    parser.add_argument("--output", type=Path, default=Path("runs/ensembles/hard_negatives.json"))
    args = parser.parse_args()

    images, by_image = load_annotations()
    file_names = {image["id"]: image["file_name"] for image in images}
    sources: dict[str, list[Path]] = {}
    records: dict[str, set] = {}
    for checkpoint in args.checkpoints:
        found = torch.load(checkpoint, map_location="cpu", weights_only=False)
        _, val_ids = validation_split(
            images, found.get("folds") or 0, found.get("fold") or 0, found.get("val_fraction") or 0.2, found.get("seed", 1)
        )
        for image_id in val_ids:
            name = file_names[image_id]
            if cache_directory(checkpoint) not in sources.setdefault(name, []):
                sources[name].append(cache_directory(checkpoint))
            records.setdefault(name, set()).add(image_id)

    points: dict[str, list[list[int]]] = {}
    for name in sorted(sources):
        probabilities = load_probabilities(sources[name], name)
        labels, n_labels = connected_components(probabilities >= args.threshold)
        if not n_labels:
            continue
        annotated = np.zeros(probabilities.size, dtype=bool)
        for image_id in records[name]:
            for pixels in annotation_pixels(by_image[image_id]):
                annotated[pixels] = True
        index = np.arange(1, n_labels + 1)
        areas = np.bincount(labels.ravel(), minlength=n_labels + 1)[1:]
        touched = np.bincount(labels.ravel()[annotated], minlength=n_labels + 1)[1:] > 0
        scores = instance_scores(labels, n_labels, probabilities)
        chosen = index[(areas > args.min_area) & (scores >= args.min_score) & ~touched]
        if len(chosen):
            centres = ndimage.center_of_mass(np.ones_like(probabilities), labels, chosen)
            points[name] = [[round(x), round(y)] for y, x in centres]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(points), encoding="utf-8")
    print(
        f"{sum(len(v) for v in points.values())} hard negatives over {len(points)}/{len(sources)} observations -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

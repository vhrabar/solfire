"""
Barb training targets without barb labels.
"""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from solfire.data import load_annotations
from solfire.data.dataset import TRAIN_IMAGES, load_image
from solfire.data.preprocess import observation_path
from solfire.inference.barbs import grow_barbs
from solfire.inference.postprocess import connected_components
from solfire.training.metrics import annotation_pixels


def job(args) -> tuple[str, int, int]:
    file_name, records, output = args
    image = load_image(observation_path(TRAIN_IMAGES, file_name)).numpy()[0]
    height, width = image.shape
    union = np.zeros(height * width, dtype=bool)
    for record in records:
        for pixels in annotation_pixels(record, height, width):
            union[pixels] = True
    union = union.reshape(height, width)
    labels, n_labels = connected_components(union)
    barbs = (grow_barbs(labels, n_labels, image) > 0) & ~union
    Image.fromarray(barbs.astype(np.uint8) * 255).save(output / f"{Path(file_name).stem}.png", optimize=True)
    return file_name, int(barbs.sum()), int(union.sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--output", type=Path, default=Path("runs/barb_targets"))
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    images, by_image = load_annotations()
    files = defaultdict(list)
    for image in sorted(images, key=lambda item: item["id"]):
        files[image["file_name"]].append(by_image[image["id"]])
    jobs = [(name, records, args.output) for name, records in sorted(files.items())]
    barbs = annotated = 0
    with ProcessPoolExecutor(args.workers) as pool:
        for _, grown, union in pool.map(job, jobs, chunksize=4):
            barbs += grown
            annotated += union
    print(f"{len(jobs)} observations -> {args.output}: {barbs} barb pixels, {barbs / annotated:.1%} of the annotated union")


if __name__ == "__main__":
    main()

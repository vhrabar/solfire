"""
The annotation records that draw barbs.
"""

import argparse
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from solfire.data import load_annotations
from solfire.data.structure import thin_mask
from solfire.training.metrics import annotation_pixels


def spine_length(annotation) -> float:
    """
    :param annotation: One annotation with an optional ``spine`` polyline
    :return: Length of the polyline in pixels, 0 without one
    """
    points = np.asarray(annotation.get("spine") or [], dtype=np.float64).reshape(-1, 2)
    return float(sum(np.hypot(*(b - a)) for a, b in itertools.pairwise(points)))


def job(args) -> tuple[int, int, int, float]:
    image, annotations = args
    height, width = image["height"], image["width"]
    union = np.zeros(height * width, dtype=bool)
    for pixels in annotation_pixels(annotations, height, width):
        union[pixels] = True
    union = union.reshape(height, width)
    return image["id"], int(thin_mask(union).sum()), int(union.sum()), sum(map(spine_length, annotations))


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--output", type=Path, default=Path("runs/barb_drawers.json"))
    parser.add_argument("--share", type=float, default=1 / 3, help="fraction of the records kept")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    images, by_image = load_annotations()
    jobs = [(image, by_image[image["id"]]) for image in images if by_image[image["id"]]]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(job, jobs, chunksize=8))
    density = {image_id: thin / max(length, 1.0) for image_id, thin, _, length in rows}
    cut = float(np.quantile(list(density.values()), 1 - args.share))
    kept = sorted(image_id for image_id, value in density.items() if value >= cut)
    thin_kept = sum(thin for image_id, thin, _, _ in rows if image_id in set(kept))
    area_kept = sum(area for image_id, _, area, _ in rows if image_id in set(kept))
    thin_all, area_all = sum(row[1] for row in rows), sum(row[2] for row in rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(kept))
    print(
        f"{len(kept)} of {len(rows)} records at >= {cut:.3f} thin px per spine px -> {args.output}; "
        f"thin share {thin_kept / area_kept:.3%} kept vs {thin_all / area_all:.3%} overall"
    )


if __name__ == "__main__":
    main()

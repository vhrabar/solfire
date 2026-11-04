"""
Do annotated filaments sit on polarity-inversion lines (PILs) of the matched GONG magnetogram?
"""

import argparse
import json

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from solfire import PROJECT_ROOT
from solfire.data import load_annotations
from solfire.experiments.data.magnetogram import (
    HA_SIZE,
    magnetogram_on_grid,
    magnetogram_path,
    mm_per_pixel,
    pil_map,
    read_magnetogram,
)

BZI = PROJECT_ROOT / "runs" / "external" / "gong_bzi"
GRID = 512
MM_PER_PIXEL = mm_per_pixel(GRID)


def spine_mask(annotations, size) -> np.ndarray:
    """
    Rasterise every annotator's filament spines onto the grid.
    :param annotations: Annotation records of one observation (all annotators)
    :param size: Grid edge length
    :return: Boolean spine mask
    """
    canvas = Image.new("1", (size, size), 0)
    draw = ImageDraw.Draw(canvas)
    scale = size / HA_SIZE
    for annotation in annotations:
        points = np.asarray(annotation.get("spine") or [], dtype=np.float64).reshape(-1, 2) * scale
        if len(points) >= 2:
            draw.line([tuple(point) for point in points], fill=1, width=1)
    return np.asarray(canvas, dtype=bool)


def summarise(distances, label) -> dict:
    """
    :param distances: Distances to the nearest PIL in grid pixels
    :param label: Row name
    :return: Summary row
    """
    distances = np.asarray(distances)
    row = {"label": label, "pixels": int(distances.size), "median_mm": round(float(np.median(distances)) * MM_PER_PIXEL, 1)}
    for limit in (5, 10, 20):
        row[f"within_{limit}mm"] = round(float((distances * MM_PER_PIXEL <= limit).mean()), 3)
    return row


def main() -> int:
    """
    Measure spine-to-PIL distances against both baselines for a few PIL settings.
    :return: Process exit code
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-correlation", type=float, default=0.3, help="drop pairs whose plage correlation is lower")
    parser.add_argument("--max-gap", type=float, default=60.0, help="drop pairs further apart in time, minutes")
    parser.add_argument("--min-gap", type=float, default=0.0, help="drop pairs closer in time, to test the far matches alone")
    parser.add_argument("--shuffles", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    manifest = json.loads((BZI / "manifest.json").read_text(encoding="utf-8"))
    usable = [
        entry["image"]
        for entry in manifest["images"]
        if entry["status"] == "ok"
        and args.min_gap <= entry["gap_minutes"] <= args.max_gap
        and entry["plage_correlation"] >= args.min_correlation
    ]
    images, by_image = load_annotations()
    by_file: dict[str, list] = {}
    for image in images:
        by_file.setdefault(image["file_name"], []).extend(by_image[image["id"]])
    usable = [name for name in usable if any(a.get("spine") for a in by_file.get(name, []))]
    print(
        f"{len(usable)} images with a usable magnetogram and spines ({sum(e['status'] == 'ok' for e in manifest['images'])} matched)"
    )

    fields, spines = {}, {}
    for name in usable:
        field, header = read_magnetogram(magnetogram_path(BZI / "fits", name))
        fields[name] = magnetogram_on_grid(field, header, GRID)
        spines[name] = spine_mask(by_file[name], GRID) & fields[name][1]

    rng = np.random.default_rng(args.seed)
    results = []
    for sigma, gradient in ((1.0, 2.0), (2.0, 1.0), (2.0, 2.0), (4.0, 0.5), (4.0, 1.0)):
        distance = {}
        for name in usable:
            pil = pil_map(*fields[name], sigma, gradient)
            distance[name] = ndimage.distance_transform_edt(~pil) if pil.any() else None
        names = [name for name in usable if distance[name] is not None]
        on_spine = np.concatenate([distance[n][spines[n]] for n in names])
        on_disk = np.concatenate([distance[n][fields[n][1]][::97] for n in names])
        shuffled = []
        for _ in range(args.shuffles):
            others = rng.permutation(names)
            shuffled.append(np.concatenate([distance[o][spines[n]] for n, o in zip(names, others) if o != n]))
        setting = f"sigma={sigma * MM_PER_PIXEL:.0f}Mm grad>={gradient}G/px"
        rows = [
            summarise(on_spine, "spines, own magnetogram"),
            summarise(on_disk, "random disk pixels"),
            summarise(np.concatenate(shuffled), "spines, other image's magnetogram"),
        ]
        print(f"\n{setting}  ({len(names)} images)")
        for row in rows:
            print("  " + "  ".join(f"{k}={v}" for k, v in row.items()))
        results.append({"setting": setting, "images": len(names), "rows": rows})

    field_strength = {
        "spine": np.concatenate([np.abs(ndimage.gaussian_filter(fields[n][0], 1))[spines[n]] for n in usable]),
        "disk": np.concatenate([np.abs(ndimage.gaussian_filter(fields[n][0], 1))[fields[n][1]][::97] for n in usable]),
    }
    print("\n|B| (smoothed 3 Mm) percentiles 50/90/99:")
    for key, values in field_strength.items():
        print(f"  {key:5s} " + " ".join(f"{np.percentile(values, q):6.1f}" for q in (50, 90, 99)) + " G")
    (BZI / "pil_enrichment.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

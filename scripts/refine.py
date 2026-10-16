"""
Crop refiner on the held-out maps of an ensemble.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_annotations, validation_split
from src.training.calibrate import cache_directory
from src.training.refine import collect_crops, load_refiner, refine_held_out, score_refined, train_refiner

STEPS = ("collect", "train", "refine", "score")


def held_out(checkpoints) -> tuple[dict, dict, dict]:
    """
    :param checkpoints: Fold checkpoints whose cached held-out maps are used
    :return: Tuple of (per file the cache directories, per file the annotations of each
        record, per file its fold)
    """
    images, by_image = load_annotations()
    file_names = {image["id"]: image["file_name"] for image in images}
    sources: dict[str, list[Path]] = {}
    records: dict[str, dict] = {}
    folds: dict[str, int] = {}
    for checkpoint in checkpoints:
        found = torch.load(checkpoint, map_location="cpu", weights_only=False)
        fold = found.get("fold") or 0
        _, val_ids = validation_split(
            images, found.get("folds") or 0, fold, found.get("val_fraction") or 0.2, found.get("seed", 1)
        )
        for image_id in val_ids:
            name = file_names[image_id]
            if cache_directory(checkpoint) not in sources.setdefault(name, []):
                sources[name].append(cache_directory(checkpoint))
            records.setdefault(name, {})[image_id] = by_image[image_id]
            if folds.setdefault(name, fold) != fold:
                raise SystemExit(f"{name} is held out by more than one fold")
    return sources, {name: list(value.values()) for name, value in records.items()}, folds


def main() -> int:
    """
    :return: Process exit code
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=(*STEPS, "all"))
    parser.add_argument("--calibration", type=Path, required=True, help="ensemble calibration JSON naming the checkpoints")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/refiner"))
    parser.add_argument("--name", default="r34", help="refiner run name under --output-dir")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    calibration = json.loads(args.calibration.read_text())
    threshold, offset = calibration["threshold"], calibration["offset"]
    sources, records, folds = held_out(calibration["checkpoints"])
    crops = args.output_dir / "crops"
    run = args.output_dir / args.name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    steps = STEPS if args.step == "all" else (args.step,)

    if "collect" in steps and not (crops / "meta.pkl").is_file():
        total = collect_crops(sources, records, folds, crops, offset=offset, workers=args.workers)
        print(f"wrote {total} samples to {crops}")
    if "train" in steps:
        for fold in args.folds:
            path = run / f"fold{fold}.pt"
            if not path.is_file():
                train_refiner(crops, fold, path, device, args.epochs, args.batch_size, args.lr, workers=args.workers)
    if "refine" in steps:
        models = {fold: load_refiner(run / f"fold{fold}.pt", device) for fold in sorted(set(folds.values()))}
        refine_held_out(models, sources, folds, run / "refined", device, threshold=threshold, offset=offset)
    if "score" in steps:
        report = score_refined(run / "refined", records, workers=args.workers)
        for reading, best in report.items():
            print(
                f"{reading}: pq {best['pq']:.4f} at threshold {best['threshold']}, min_score {best['min_score']} "
                f"(tp {best['tp']} fp {best['fp']} fn {best['fn']})"
            )
        (run / "score.json").write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

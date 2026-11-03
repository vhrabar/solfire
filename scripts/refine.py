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
from src.inference.refiner import CONTEXT, load_refiner
from src.training.calibrate import cache_directory
from src.training.refine import SCORE_TARGETS, collect_crops, record_weights, refine_held_out, score_refined, train_refiner

STEPS = ("collect", "train", "refine", "score")


def held_out(checkpoints, variants=("base",)) -> tuple[dict, dict, dict]:
    """
    :param checkpoints: Fold checkpoints whose cached held-out maps are used
    :param variants: Inference variants of every checkpoint averaged into the map, as the calibration lists them
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
            for variant in variants:
                if cache_directory(checkpoint, variant) not in sources.setdefault(name, []):
                    sources[name].append(cache_directory(checkpoint, variant))
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
    parser.add_argument("--seed", type=int, default=1, help="seed of the refiners' initialisation and sampling")
    parser.add_argument(
        "--lovasz-weight", type=float, default=0.0, help="weight of a per-record Lovász hinge in the refiner's mask loss"
    )
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--score-target",
        choices=SCORE_TARGETS,
        default="mean",
        help="earned IoU the score head learns: averaged over records, or on the record with the fewest filaments",
    )
    parser.add_argument("--crops", default="crops", help="sample directory under --output-dir")
    parser.add_argument("--context", type=float, default=CONTEXT, help="window edge per longer side of the proposal box")
    parser.add_argument(
        "--train-offsets",
        type=int,
        nargs="+",
        default=None,
        help="offsets the training proposals are cut with; default: the calibration's",
    )
    parser.add_argument(
        "--seed-from",
        default=None,
        help="refine step: start from the refined instances of this run under --output-dir instead of the "
        "thresholded ensemble map, i.e. refine a second time",
    )
    parser.add_argument("--seed-cut", type=float, default=0.45, help="probability the --seed-from masks are cut at")
    parser.add_argument(
        "--refiners",
        nargs="+",
        default=None,
        help="runs whose fold refiners are used, averaged when there are several; default: --name",
    )
    parser.add_argument("--encoder", default="resnet34", help="encoder of the refiner's U-Net, as smp spells it")
    parser.add_argument(
        "--encoder-init", type=Path, default=None, help="checkpoint the encoder starts from, e.g. a pretrain_mim encoder.pt"
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="refined calibration JSON; default: <calibration>-refined next to it"
    )
    args = parser.parse_args()

    calibration = json.loads(args.calibration.read_text())
    threshold, offset = calibration["threshold"], calibration["offset"]
    sources, records, folds = held_out(calibration["checkpoints"], calibration.get("variants") or ["base"])
    crops = args.output_dir / args.crops
    crop_settings = {"context": args.context, "offsets": args.train_offsets or [offset]}
    run = args.output_dir / args.name
    refiners = [args.output_dir / name for name in (args.refiners or [args.name])]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    steps = STEPS if args.step == "all" else (args.step,)

    if "collect" in steps and not (crops / "meta.pkl").is_file():
        total = collect_crops(
            sources, records, folds, crops, offsets=crop_settings["offsets"], workers=args.workers, context=args.context
        )
        (crops / "settings.json").write_text(json.dumps(crop_settings))
        print(f"wrote {total} samples to {crops}")
    stored = crops / "settings.json"
    found = json.loads(stored.read_text()) if stored.is_file() else {"context": CONTEXT, "offsets": [offset]}
    if found != crop_settings:
        raise SystemExit(f"{crops} was collected with {found}, not {crop_settings}; pick another --crops")
    if "train" in steps:
        for fold in args.folds:
            path = run / f"fold{fold}.pt"
            if not path.is_file():
                train_refiner(
                    crops,
                    fold,
                    path,
                    device,
                    args.epochs,
                    args.batch_size,
                    args.lr,
                    workers=args.workers,
                    seed=args.seed,
                    weights=record_weights(records, args.score_target),
                    lovasz_weight=args.lovasz_weight,
                    encoder=args.encoder,
                    encoder_init=args.encoder_init,
                )
    if "refine" in steps:
        models = {
            fold: [load_refiner(directory / f"fold{fold}.pt", device) for directory in refiners]
            for fold in sorted(set(folds.values()))
        }
        refine_held_out(
            models,
            sources,
            folds,
            run / "refined",
            device,
            threshold=threshold,
            offset=offset,
            context=args.context,
            seed_dir=args.output_dir / args.seed_from / "refined" if args.seed_from else None,
            seed_cut=args.seed_cut,
        )
    if "score" in steps:
        report = score_refined(run / "refined", records, folds, workers=args.workers)
        for reading, best in report.items():
            print(
                f"{reading}: pq {best['pq']:.4f} at threshold {best['threshold']}, min_score {best['min_score']} "
                f"(tp {best['tp']} fp {best['fp']} fn {best['fn']}); nested {best['nested']['pq']:.4f}"
            )
        (run / "score.json").write_text(json.dumps(report, indent=2))
        best = report["refined"]
        calibration["refiner"] = {
            "checkpoints": [str(directory / f"fold{fold}.pt") for directory in refiners for fold in sorted(set(folds.values()))],
            "proposal_threshold": threshold,
            "proposal_offset": offset,
            "context": args.context,
            "score_target": args.score_target,
            **({"lovasz_weight": args.lovasz_weight} if args.lovasz_weight else {}),
            "threshold": best["threshold"],
            "min_score": best["min_score"],
            "pq": best["pq"],
            "nested_pq": best["nested"]["pq"],
            **({"iterations": 2, "seed_cut": args.seed_cut} if args.seed_from else {}),
        }
        output = args.output or args.calibration.with_name(
            args.calibration.name.removesuffix(".calibration.json") + "-refined.calibration.json"
        )
        output.write_text(json.dumps(calibration, indent=2))
        print(f"refined calibration -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

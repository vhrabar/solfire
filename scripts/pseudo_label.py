"""
Soft pseudo-labels for the unlabeled GONG bursts, one set per cross-validation fold.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import TRAIN_IMAGES, load_annotations, load_image, validation_split
from src.inference.predict import load_model
from src.inference.tiling import predict_tiled

REPO = Path(__file__).resolve().parent.parent
GONG = REPO / "runs" / "external" / "gong"
DISTILL = REPO / "runs" / "distill" / "targets"


def fold_of(checkpoint) -> int:
    """
    The fold a checkpoint held out.
    :param checkpoint: Path to a checkpoint trained with ``--folds``
    :return: Fold index
    """
    found = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not found.get("folds"):
        raise SystemExit(f"{checkpoint} was not trained on a fold")
    return int(found["fold"])


def main() -> int:
    """
    Write ``targets/fold<k>/<centre frame>.png`` for every usable burst and fold.
    :return: Process exit code
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("teachers", nargs="+", type=Path, help="fold checkpoints; those of one fold are averaged")
    parser.add_argument("--manifest", type=Path, default=GONG / "manifest.json")
    parser.add_argument("--output", type=Path, default=None, help="default: the GONG targets, or runs/distill/targets")
    parser.add_argument(
        "--distill", action="store_true", help="label each fold's training observations instead of the GONG bursts"
    )
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--tile-batch", type=int, default=16)
    args = parser.parse_args()

    output_root = args.output or (DISTILL if args.distill else GONG / "targets")
    bursts = [] if args.distill else json.loads(args.manifest.read_text())["bursts"]
    by_fold: dict[int, list[Path]] = {}
    for checkpoint in args.teachers:
        by_fold.setdefault(fold_of(checkpoint), []).append(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"teachers per fold: { {k: len(v) for k, v in sorted(by_fold.items())} }")
    images = load_annotations()[0] if args.distill else []

    for fold, checkpoints in sorted(by_fold.items()):
        output = output_root / f"fold{fold}"
        output.mkdir(parents=True, exist_ok=True)
        if args.distill:
            train_ids, _ = validation_split(images, folds=5, fold=fold)
            names = {image["id"]: image["file_name"] for image in images}
            # a one-frame "burst" per training observation of this fold
            groups = [[Path(name).stem] for name in sorted({names[image_id] for image_id in train_ids})]
            frames = TRAIN_IMAGES
        else:
            groups, frames = bursts, GONG / "jpeg"
        todo = [burst for burst in groups if not (output / f"{burst[len(burst) // 2]}.png").is_file()]
        models = [load_model(checkpoint, device)[0] for checkpoint in checkpoints]
        for burst in tqdm(todo, desc=f"fold {fold}", mininterval=10.0):
            total = None
            for name in burst:
                image = load_image(frames / f"{name}.jpeg")
                for model in models:
                    probabilities = predict_tiled(
                        model, image, device, tile=args.tile, overlap=args.overlap, tta=False, batch_size=args.tile_batch
                    )
                    total = probabilities if total is None else total + probabilities
            assert total is not None
            target = (total / (len(burst) * len(models))).numpy()
            centre = burst[len(burst) // 2]
            partial = output / f"{centre}.part.png"
            Image.fromarray(np.round(target * 255).astype(np.uint8)).save(partial)
            partial.replace(output / f"{centre}.png")
        del models
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

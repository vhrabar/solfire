import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .data import FilamentDataset, TestImageDataset, load_annotations, train_val_split
from .inference import (
    SUBMISSION_SIZE,
    load_model,
    predict_dataset,
    write_detections,
    write_submission,
)
from .networks import ARCHITECTURES
from .training import (
    AugmentedDataset,
    BCEDiceLoss,
    build_optimizer,
    build_scheduler,
    estimate_pos_weight,
    fit,
)


def parse_args() -> argparse.Namespace:
    """
    Parse the ``train`` and ``predict`` command lines.
    :return: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Solar filament segmentation")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default=None, help="cpu, cuda,...")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="train the model on the annotated split")
    train.add_argument("--epochs", type=int, default=40)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--lr", type=float, default=3e-4)
    train.add_argument("--weight-decay", type=float, default=1e-2)
    train.add_argument("--size", type=int, default=256, help="edge length observations are resized to")
    train.add_argument("--arch", choices=sorted(ARCHITECTURES), default="unet")
    train.add_argument("--base-channels", type=int, default=32, help="unet only")
    train.add_argument("--depth", type=int, default=4, help="unet only")
    train.add_argument("--bilinear", action=argparse.BooleanOptionalAction, default=True, help="unet only")
    train.add_argument("--embed-dims", type=int, nargs="+", default=[32, 64, 160, 256], help="segformer only")
    train.add_argument("--depths", type=int, nargs="+", default=[2, 2, 2, 2], help="segformer only")
    train.add_argument("--num-heads", type=int, nargs="+", default=[1, 2, 5, 8], help="segformer only")
    train.add_argument("--decoder-dim", type=int, default=256, help="segformer only")
    train.add_argument("--dropout", type=float, default=0.0, help="segformer only")
    train.add_argument("--drop-path", type=float, default=0.1, help="segformer only")
    train.add_argument("--val-fraction", type=float, default=0.2)
    train.add_argument("--categories", type=int, nargs="*", default=[1, 2], help="annotation category ids to segment")
    train.add_argument(
        "--pos-weight", type=float, default=None, help="positive class weight in the BCE term (default: estimated)"
    )
    train.add_argument("--dice-weight", type=float, default=0.5)
    train.add_argument("--warmup-epochs", type=int, default=1)
    train.add_argument("--grad-clip", type=float, default=1.0)
    train.add_argument("--patience", type=int, default=0, help="0 disables early stopping")
    train.add_argument("--workers", type=int, default=4)
    train.add_argument("--amp", action="store_true", help="bfloat16 on cpu, float16 on cuda")
    train.add_argument("--no-augment", action="store_true")
    train.add_argument("--output-dir", default=None, help="default: runs/<arch>")

    predict = subparsers.add_parser("predict", help="write predictions for the test observations")
    predict.add_argument("--checkpoint", default="runs/unet/best.pt")
    predict.add_argument("--threshold", type=float, default=None, help="default: the value tuned on validation")
    predict.add_argument("--batch-size", type=int, default=4)
    predict.add_argument("--workers", type=int, default=4)
    predict.add_argument("--tta", action=argparse.BooleanOptionalAction, default=True)
    predict.add_argument("--min-area", type=int, default=32, help="drop predicted instances smaller than this many pixels")
    predict.add_argument("--category-id", type=int, default=1)
    predict.add_argument("--save-masks", action="store_true", help="also write mask PNGs")
    predict.add_argument("--output-dir", default="runs/unet/predictions")

    return parser.parse_args()


def resolve_device(name=None) -> torch.device:
    """
    Pick the device to run on.
    :param name: Explicit device string, or ``None`` to autodetect
    :return: Selected device
    """
    if name is not None:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model_config(args) -> dict:
    """
    Collect the constructor arguments of the requested architecture.
    :param args: Parsed ``train`` arguments
    :return: Keyword arguments for the matching entry of :data:`networks.ARCHITECTURES`
    """
    if args.arch == "unet":
        return {
            "in_channels": 1,
            "out_channels": 1,
            "base_channels": args.base_channels,
            "depth": args.depth,
            "bilinear": args.bilinear,
        }

    stages = len(args.embed_dims)
    return {
        "in_channels": 1,
        "out_channels": 1,
        "embed_dims": tuple(args.embed_dims),
        "depths": tuple(args.depths),
        "num_heads": tuple(args.num_heads),
        "reductions": tuple(max(1, 8 // 2**stage) for stage in range(stages)),
        "decoder_dim": args.decoder_dim,
        "dropout": args.dropout,
        "drop_path": args.drop_path,
    }


def run_training(args, device) -> None:
    """
    Train a segmentation model and checkpoint the best validation Dice.
    :param args: Parsed ``train`` arguments
    :param device: Device to run on
    :return: None
    """
    output_dir = args.output_dir or f"runs/{args.arch}"

    images, _ = load_annotations()
    train_ids, val_ids = train_val_split(images, val_fraction=args.val_fraction, seed=args.seed)
    categories = tuple(args.categories) if args.categories else None

    train_dataset = FilamentDataset(train_ids, size=args.size, category_ids=categories)
    valid_dataset = FilamentDataset(val_ids, size=args.size, category_ids=categories)
    print(f"train {len(train_dataset)} / val {len(valid_dataset)} observations")

    pos_weight = args.pos_weight
    if pos_weight is None:
        pos_weight = estimate_pos_weight(train_dataset)
        print(f"estimated pos_weight: {pos_weight:.1f}")

    train_data: Dataset = train_dataset
    if not args.no_augment:
        train_data = AugmentedDataset(train_dataset)

    loader_kwargs = {
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    model_config = build_model_config(args)
    model = ARCHITECTURES[args.arch](**model_config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{type(model).__name__}: {n_params / 1e6:.2f}M parameters")

    criterion = BCEDiceLoss(bce_weight=1.0 - args.dice_weight, dice_weight=args.dice_weight, pos_weight=pos_weight).to(device)
    optimizer = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, args.epochs, max(1, len(train_loader)), warmup_epochs=args.warmup_epochs)

    summary = fit(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        args.epochs,
        scheduler=scheduler,
        model_config=model_config,
        image_size=args.size,
        output_dir=output_dir,
        amp=args.amp,
        grad_clip=args.grad_clip,
        patience=args.patience,
    )
    print(
        f"best dice {summary['best_dice']:.4f} at epoch {summary['best_epoch']} "
        f"(threshold {summary['best_threshold']:.2f}) -> {output_dir}/best.pt"
    )


def run_prediction(args, device) -> None:
    """
    Write COCO-style detections for the unlabelled test observations.
    :param args: Parsed ``predict`` arguments
    :param device: Device to run on
    :return: None
    """
    model, checkpoint = load_model(args.checkpoint, device)
    size = checkpoint.get("image_size") or 256
    threshold = args.threshold if args.threshold is not None else checkpoint.get("threshold", 0.5)
    print(f"checkpoint epoch {checkpoint.get('epoch')}, size {size}, threshold {threshold:.2f}")

    dataset = TestImageDataset(size=size)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda"
    )
    print(f"{len(dataset)} test observations")

    output_dir = Path(args.output_dir)
    detections = predict_dataset(
        model,
        loader,
        device,
        threshold=threshold,
        tta=args.tta,
        original_size=SUBMISSION_SIZE,
        min_area=args.min_area,
        category_id=args.category_id,
        save_masks_dir=output_dir / "masks" if args.save_masks else None,
    )

    submission_path = output_dir / "submission.csv"
    write_submission(detections, submission_path)
    write_detections(detections, output_dir / "detections.json")

    n_images = len({d["image_id"] for d in detections})
    print(f"{len(detections)} filaments over {n_images}/{len(dataset)} observations -> {submission_path}")
    if not detections:
        print("warning: no filaments predicted; lower --threshold or train for longer")


def main() -> None:
    """
    Dispatch to the requested subcommand.
    :return: None
    """
    args = parse_args()
    set_seeds(args.seed)
    device = resolve_device(args.device)
    print(f"device: {device}")

    if args.command == "train":
        run_training(args, device)
    else:
        run_prediction(args, device)


def set_seeds(seed: int) -> None:
    """
    Sets random seeds for reproducibility
    :param seed: Seed value
    :return: None
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)

    np.random.seed(seed)  # noqa: NPY002

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()

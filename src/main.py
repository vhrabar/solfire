import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .data import (
    DATA_ROOT,
    RASTERS,
    TARGETS,
    FilamentCropDataset,
    FilamentDataset,
    TestImageDataset,
    load_annotations,
    validation_split,
)
from .inference import (
    DEFAULT_OVERLAP,
    MIN_AREA,
    SUBMISSION_SIZE,
    load_model,
    predict_dataset,
    write_detections,
    write_submission,
)
from .networks import ARCHITECTURES, DECODERS
from .training import (
    CALIBRATION_THRESHOLDS,
    DEFAULT_MIN_AREAS,
    DEFAULT_MIN_SCORES,
    DEFAULT_OFFSETS,
    KERNEL_SHAPES,
    AugmentedDataset,
    BCEDiceLoss,
    MorphologyLoss,
    build_criterion,
    build_optimizer,
    build_scheduler,
    cache_directory,
    cache_probabilities,
    calibrate,
    estimate_pos_weight,
    evaluate_tiled,
    fit,
    threshold_grid,
    write_calibration,
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
    train.add_argument(
        "--crop",
        type=int,
        default=512,
        help="edge length of the crops taken at the resolution the observations were annotated at; 0 resizes instead",
    )
    train.add_argument("--crops-per-image", type=int, default=4, help="crops drawn per observation and epoch")
    train.add_argument("--foreground-prob", type=float, default=0.7, help="fraction of crops drawn around an annotation")
    train.add_argument("--native-size", type=int, default=0, help="0 keeps the resolution the observations were annotated at")
    train.add_argument("--size", type=int, default=256, help="edge length observations are resized to when --crop is 0")
    train.add_argument("--arch", choices=sorted(ARCHITECTURES), default="unet")
    train.add_argument("--base-channels", type=int, default=32, help="unet/transunet only")
    train.add_argument("--depth", type=int, default=4, help="unet/transunet only")
    train.add_argument("--bilinear", action=argparse.BooleanOptionalAction, default=True, help="unet/transunet only")
    train.add_argument("--embed-dims", type=int, nargs="+", default=[32, 64, 160, 256], help="segformer only")
    train.add_argument("--depths", type=int, nargs="+", default=[2, 2, 2, 2], help="segformer only")
    train.add_argument("--num-heads", type=int, nargs="+", default=[1, 2, 5, 8], help="segformer only")
    train.add_argument("--decoder-dim", type=int, default=256, help="segformer only")
    train.add_argument("--blocks", type=int, default=4, help="transunet only: attention blocks at the bottleneck")
    train.add_argument("--heads", type=int, default=8, help="transunet only: attention heads")
    train.add_argument("--embed-dim", type=int, default=None, help="transunet only: token width (default: the bottleneck width)")
    train.add_argument("--mlp-ratio", type=float, default=2.0, help="transunet only")
    train.add_argument("--decoder", choices=sorted(DECODERS), default="unet", help="pretrained only")
    train.add_argument("--encoder", default="resnet34", help="pretrained only: smp encoder name, tu- prefixed for timm")
    train.add_argument(
        "--encoder-weights", default="imagenet", help="pretrained only: weights to start the encoder from; 'none' for random"
    )
    train.add_argument(
        "--encoder-lr-scale", type=float, default=1.0, help="pretrained only: encoder learning rate as a fraction of --lr"
    )
    train.add_argument("--dropout", type=float, default=0.0, help="segformer/transunet only")
    train.add_argument("--drop-path", type=float, default=0.1, help="segformer/transunet only")
    train.add_argument("--val-fraction", type=float, default=0.2)
    train.add_argument(
        "--folds", type=int, default=0, help="hold out one fold of a grouped k-fold instead of --val-fraction; 0 disables it"
    )
    train.add_argument("--fold", type=int, default=0, help="the fold held out when --folds is set")
    train.add_argument(
        "--raster", choices=RASTERS, default="coco", help="how annotation polygons are filled; coco matches the scored masks"
    )
    train.add_argument(
        "--targets",
        choices=TARGETS,
        default="each",
        help="each: one sample per annotator record; mean: one soft target per observation, averaged over its annotators",
    )
    train.add_argument("--val-tile", type=int, default=0, help="inference window used to validate; 0 follows --crop")
    train.add_argument("--val-overlap", type=float, default=0.0, help="overlap between the validation windows")
    train.add_argument("--val-limit", type=int, default=48, help="observations validated per epoch; 0 uses the whole split")
    train.add_argument("--val-min-area", type=int, default=MIN_AREA, help="instances dropped while validating")
    train.add_argument("--val-tta", action="store_true", help="average the validation windows over flips")
    train.add_argument(
        "--thresholds", type=float, nargs="+", default=None, help="default: a grid around the decision point --pos-weight implies"
    )
    train.add_argument(
        "--swa-epochs", type=int, default=0, help="average the weights of this many final epochs into swa.pt; 0 disables it"
    )
    train.add_argument("--tile-batch", type=int, default=8, help="inference windows evaluated per forward pass")
    train.add_argument("--categories", type=int, nargs="*", default=[1, 2, 3], help="annotation category ids to segment")
    train.add_argument(
        "--pos-weight", type=float, default=None, help="positive class weight in the BCE term (default: estimated)"
    )
    train.add_argument("--dice-weight", type=float, default=0.5)
    train.add_argument(
        "--morph-open-weight",
        type=float,
        default=0.0,
        help="weight of the opening residual term: penalises thin spurs and speckle the model predicts",
    )
    train.add_argument(
        "--morph-close-weight",
        type=float,
        default=0.0,
        help="weight of the closing residual term: penalises the small gaps that split one filament in two",
    )
    train.add_argument(
        "--morph-cldice-weight",
        type=float,
        default=0.0,
        help="weight of the soft centre-line Dice term: rewards keeping a filament connected end to end",
    )
    train.add_argument("--morph-kernel", choices=sorted(KERNEL_SHAPES), default="disk", help="structuring element shape")
    train.add_argument("--morph-kernel-size", type=int, default=3, help="structuring element edge length in pixels, odd")
    train.add_argument("--morph-skeleton-iters", type=int, default=5, help="erosion steps in the clDice skeleton")
    train.add_argument(
        "--morph-warmup-epochs",
        type=int,
        default=0,
        help="ramp the morphology terms in linearly over this many epochs; 0 applies them from the first step",
    )
    train.add_argument("--warmup-epochs", type=int, default=1)
    train.add_argument("--grad-clip", type=float, default=1.0)
    train.add_argument("--patience", type=int, default=0, help="0 disables early stopping")
    train.add_argument("--workers", type=int, default=4)
    train.add_argument("--amp", action="store_true", help="bfloat16 on cpu, float16 on cuda")
    train.add_argument("--no-augment", action="store_true")
    train.add_argument("--output-dir", default=None, help="default: runs/<arch>")
    train.add_argument(
        "--resume",
        action="store_true",
        help="continue from output-dir/last.pt where it left off instead of starting over; "
        "a no-op if it already reached --epochs",
    )

    predict = subparsers.add_parser("predict", help="write predictions for the test observations")
    predict.add_argument(
        "--checkpoint", nargs="+", default=["runs/unet/best.pt"], help="several checkpoints average their probabilities"
    )
    predict.add_argument(
        "--calibration", default=None, help="calibration JSON to take the post-processing from; default: the first checkpoint"
    )
    predict.add_argument("--threshold", type=float, default=None, help="default: the value tuned on validation")
    predict.add_argument("--batch-size", type=int, default=4)
    predict.add_argument("--workers", type=int, default=4)
    predict.add_argument("--tta", action=argparse.BooleanOptionalAction, default=True)
    predict.add_argument("--tile", type=int, default=0, help="inference window; 0 follows the checkpoint")
    predict.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP, help="overlap between the inference windows")
    predict.add_argument("--tile-batch", type=int, default=8, help="inference windows evaluated per forward pass")
    predict.add_argument("--offset", type=int, default=None, help="default: the value tuned on validation")
    predict.add_argument("--min-area", type=int, default=None, help="drop predicted instances smaller than this many pixels")
    predict.add_argument("--min-score", type=float, default=None, help="drop predicted instances with a lower mean probability")
    predict.add_argument("--category-id", type=int, default=1)
    predict.add_argument("--save-masks", action="store_true", help="also write mask PNGs")
    predict.add_argument("--output-dir", default="runs/unet/predictions")

    calibration = subparsers.add_parser(
        "calibrate", help="tune the decision threshold, the morphological offset and the minimum area on validation"
    )
    calibration.add_argument(
        "--checkpoint",
        nargs="+",
        default=["runs/unet/best.pt"],
        help="several checkpoints are averaged on the observations they all held out, and pooled across different splits",
    )
    calibration.add_argument("--thresholds", type=float, nargs="+", default=list(CALIBRATION_THRESHOLDS))
    calibration.add_argument("--offsets", type=int, nargs="+", default=list(DEFAULT_OFFSETS))
    calibration.add_argument("--min-areas", type=int, nargs="+", default=list(DEFAULT_MIN_AREAS))
    calibration.add_argument("--min-scores", type=float, nargs="+", default=list(DEFAULT_MIN_SCORES))
    calibration.add_argument("--tile", type=int, default=0, help="inference window; 0 follows the checkpoint")
    calibration.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP)
    calibration.add_argument("--tile-batch", type=int, default=8)
    calibration.add_argument("--tta", action=argparse.BooleanOptionalAction, default=True)
    calibration.add_argument("--workers", type=int, default=8, help="processes scoring observations")
    calibration.add_argument("--output", default=None, help="calibration JSON to write; required with several checkpoints")
    calibration.add_argument("--top", type=int, default=10, help="settings to print")

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
    if args.arch in {"unet", "transunet"}:
        config = {
            "in_channels": 1,
            "out_channels": 1,
            "base_channels": args.base_channels,
            "depth": args.depth,
            "bilinear": args.bilinear,
        }
        if args.arch == "transunet":
            config |= {
                "blocks": args.blocks,
                "num_heads": args.heads,
                "embed_dim": args.embed_dim,
                "mlp_ratio": args.mlp_ratio,
                "dropout": args.dropout,
                "drop_path": args.drop_path,
            }
        return config

    if args.arch == "pretrained":
        weights = None if args.encoder_weights.lower() == "none" else args.encoder_weights
        return {
            "in_channels": 1,
            "out_channels": 1,
            "decoder": args.decoder,
            "encoder": args.encoder,
            "encoder_weights": weights,
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


def build_datasets(args):
    """
    Build the training and validation datasets for the requested resolution mode.

    With ``--crop`` the observations are kept at the resolution they were annotated
    at and windows are drawn out of them, so an eleven pixel wide filament stays
    eleven pixels wide. Validation then runs on whole observations, which is the
    shape the submission is written in.
    :param args: Parsed ``train`` arguments
    :return: Tuple of (train dataset, validation dataset, validation batch size)
    """
    images, _ = load_annotations()
    train_ids, val_ids = validation_split(images, args.folds, args.fold, args.val_fraction, args.seed)
    categories = tuple(args.categories) if args.categories else None
    native_size = args.native_size or None

    if not args.crop:
        train_dataset: Dataset = FilamentDataset(train_ids, size=args.size, category_ids=categories, raster=args.raster)
        valid_dataset = FilamentDataset(val_ids, size=args.size, category_ids=categories, raster=args.raster)
        return train_dataset, valid_dataset, args.batch_size

    train_dataset = FilamentCropDataset(
        train_ids,
        crop=args.crop,
        native_size=native_size,
        repeat=args.crops_per_image,
        foreground_prob=args.foreground_prob,
        category_ids=categories,
        raster=args.raster,
        targets=args.targets,
    )
    valid_dataset = FilamentDataset(val_ids, size=native_size, category_ids=categories, raster=args.raster)
    return train_dataset, valid_dataset, 1


def run_training(args, device) -> None:
    """
    Train a segmentation model and checkpoint the best validation Panoptic Quality.
    :param args: Parsed ``train`` arguments
    :param device: Device to run on
    :return: None
    """
    output_dir = args.output_dir or f"runs/{args.arch}"
    categories = tuple(args.categories) if args.categories else None

    train_dataset, valid_dataset, val_batch_size = build_datasets(args)
    print(f"train {len(train_dataset)} crops / val {len(valid_dataset)} observations")

    pos_weight = args.pos_weight
    if pos_weight is None:
        pos_weight = estimate_pos_weight(train_dataset)
        print(f"estimated pos_weight: {pos_weight:.1f}")

    thresholds = tuple(args.thresholds) if args.thresholds else threshold_grid(pos_weight)
    print(
        f"validating at {' '.join(f'{t:.3f}' for t in thresholds)} "
        f"(pos_weight {pos_weight:.1f} puts the decision point at {pos_weight / (pos_weight + 1):.3f})"
    )

    train_data = train_dataset
    if not args.no_augment:
        train_data = AugmentedDataset(train_dataset)

    loader_kwargs = {
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(valid_dataset, batch_size=val_batch_size, shuffle=False, **loader_kwargs)

    model_config = build_model_config(args)
    model = ARCHITECTURES[args.arch](**model_config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{type(model).__name__}: {n_params / 1e6:.2f}M parameters")
    encoder_lr_scale = 1.0
    if args.arch == "pretrained":
        print(f"{args.decoder} on {args.encoder}, encoder weights {model_config['encoder_weights']}")
        encoder_lr_scale = args.encoder_lr_scale
        # the state dict carries the trained weights, so rebuilding must not fetch the pretrained ones again
        model_config = model_config | {"encoder_weights": None}

    base_criterion = BCEDiceLoss(bce_weight=1.0 - args.dice_weight, dice_weight=args.dice_weight, pos_weight=pos_weight)
    criterion = build_criterion(base_criterion, args).to(device)
    if isinstance(criterion, MorphologyLoss):
        terms = [
            f"{name} {weight:g}"
            for name, weight in (
                ("open", criterion.open_weight),
                ("close", criterion.close_weight),
                ("cldice", criterion.cldice_weight),
            )
            if weight > 0
        ]
        warmup = f", warmup {args.morph_warmup_epochs} epochs" if args.morph_warmup_epochs else ""
        print(f"morphology loss (serron, {args.morph_kernel} {args.morph_kernel_size}px): {', '.join(terms)}{warmup}")
    optimizer = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay, encoder_lr_scale=encoder_lr_scale)
    scheduler = build_scheduler(optimizer, args.epochs, max(1, len(train_loader)), warmup_epochs=args.warmup_epochs)

    tile = args.val_tile or args.crop or None
    extra = {
        "crop": args.crop or None,
        "native_size": args.native_size or None,
        "tile": tile,
        "overlap": args.val_overlap,
        "offset": 0,
        "min_area": args.val_min_area,
        "pos_weight": pos_weight,
        "val_fraction": args.val_fraction,
        "folds": args.folds,
        "fold": args.fold,
        "raster": args.raster,
        "targets": args.targets,
        "seed": args.seed,
        "categories": list(categories) if categories else None,
        "morph": {
            "open_weight": args.morph_open_weight,
            "close_weight": args.morph_close_weight,
            "cldice_weight": args.morph_cldice_weight,
            "kernel": args.morph_kernel,
            "kernel_size": args.morph_kernel_size,
            "skeleton_iters": args.morph_skeleton_iters,
            "warmup_epochs": args.morph_warmup_epochs,
        }
        if isinstance(criterion, MorphologyLoss)
        else None,
    }

    def validate_tiled(network):
        """
        Validate on whole observations, the way the submission is written.
        :param network: Model to score
        :return: Metrics dict
        """
        return evaluate_tiled(
            network,
            val_loader,
            device,
            thresholds=thresholds,
            min_area=args.val_min_area,
            tile=tile,
            overlap=args.val_overlap,
            tta=args.val_tta,
            batch_size=args.tile_batch,
            limit=args.val_limit or None,
        )

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
        image_size=None if args.crop else args.size,
        output_dir=output_dir,
        amp=args.amp,
        grad_clip=args.grad_clip,
        patience=args.patience,
        thresholds=thresholds,
        evaluate_fn=validate_tiled if args.crop else None,
        extra=extra,
        swa_epochs=args.swa_epochs,
        resume=args.resume,
    )
    print(
        f"best pq {summary['best_pq']:.4f} at epoch {summary['best_epoch']} "
        f"(threshold {summary['best_threshold']:.2f}) -> {output_dir}/best.pt"
    )
    name = "best.pt"
    if "swa_pq" in summary:
        name = "swa.pt" if summary["swa_pq"] > summary["best_pq"] else "best.pt"
        print(f"averaged {summary['swa_epochs']} epochs: pq {summary['swa_pq']:.4f}")
    if args.crop:
        print(f"now tune the post-processing: python -m src.main calibrate --checkpoint {output_dir}/{name}")


def checkpoint_setting(checkpoint, name, override, default=None):
    """
    Resolve a post-processing setting: the command line first, then the checkpoint.
    :param checkpoint: Checkpoint metadata
    :param name: Key to read from the checkpoint
    :param override: Value passed on the command line, or ``None``
    :param default: Value used when neither has it
    :return: The setting to run with
    """
    if override is not None:
        return override
    value = checkpoint.get(name)
    return default if value is None else value


def run_prediction(args, device) -> None:
    """
    Write COCO-style detections for the unlabelled test observations.
    :param args: Parsed ``predict`` arguments
    :param device: Device to run on
    :return: None
    """
    loaded = [load_model(path, device) for path in args.checkpoint]
    models = [model for model, _ in loaded]
    checkpoint = loaded[0][1]
    crop = checkpoint.get("crop")
    settings = json.loads(Path(args.calibration).read_text()) if args.calibration else checkpoint
    tile = args.tile or settings.get("tile") or (checkpoint.get("tile") if crop else None)
    size = checkpoint.get("native_size") if crop else (checkpoint.get("image_size") or 256)

    threshold = checkpoint_setting(settings, "threshold", args.threshold, 0.5)
    offset = checkpoint_setting(settings, "offset", args.offset, 0)
    min_area = checkpoint_setting(settings, "min_area", args.min_area, MIN_AREA)
    min_score = checkpoint_setting(settings, "min_score", args.min_score, 0.0)
    print(
        f"{len(models)} model(s), checkpoint epoch {checkpoint.get('epoch')}, size {size or 'native'}, "
        f"tile {tile or 'whole observation'}, threshold {threshold:.2f}, offset {offset:+d}, "
        f"min_area {min_area}, min_score {min_score:.2f}"
    )
    if args.calibration is None and (len(models) > 1 or checkpoint.get("calibration") is None):
        print("warning: no calibration for this set of models; run `calibrate` before submitting")

    dataset = TestImageDataset(size=size)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda"
    )
    print(f"{len(dataset)} test observations")

    output_dir = Path(args.output_dir)
    detections = predict_dataset(
        models,
        loader,
        device,
        threshold=threshold,
        tta=args.tta,
        original_size=SUBMISSION_SIZE,
        min_area=min_area,
        min_score=min_score,
        offset=offset,
        category_id=args.category_id,
        save_masks_dir=output_dir / "masks" if args.save_masks else None,
        tile=tile,
        overlap=args.overlap,
        tile_batch=args.tile_batch,
    )

    submission_path = output_dir / "submission.csv"
    write_submission(detections, submission_path)
    write_detections(detections, output_dir / "detections.json")

    n_images = len({d["image_id"] for d in detections})
    print(f"{len(detections)} filaments over {n_images}/{len(dataset)} observations -> {submission_path}")
    if not detections:
        print("warning: no filaments predicted; lower --threshold or train for longer")


def run_calibration(args, device) -> None:
    """
    Tune the post-processing on held-out observations, through the path a submission takes.

    Each checkpoint is run over the observations its own split held out, and the
    probabilities are cached next to it. Checkpoints that held out the same
    observation are averaged there, so one call calibrates a seed ensemble on its
    shared split as well as a set of fold models on their pooled out-of-fold maps.
    :param args: Parsed ``calibrate`` arguments
    :param device: Device to run on
    :return: None
    """
    if len(args.checkpoint) > 1 and not args.output:
        raise SystemExit("--output is required when calibrating several checkpoints")
    images, by_image = load_annotations()
    file_names = {image["id"]: image["file_name"] for image in images}

    sources: dict[str, list[Path]] = {}
    records: dict[str, dict] = {}
    tiles = set()
    for path in args.checkpoint:
        model, checkpoint = load_model(path, device)
        if not checkpoint.get("crop"):
            raise SystemExit(f"{path}: calibration needs a model trained on native resolution crops")
        tile = args.tile or checkpoint.get("tile") or checkpoint.get("crop")
        tiles.add(tile)
        _, val_ids = validation_split(
            images,
            checkpoint.get("folds") or 0,
            checkpoint.get("fold") or 0,
            checkpoint.get("val_fraction") or 0.2,
            checkpoint.get("seed", args.seed),
        )
        files = sorted({file_names[image_id] for image_id in val_ids})
        cache = cache_directory(path)
        print(f"{path}: {len(val_ids)} held-out records over {len(files)} observations, tile {tile}")
        cache_probabilities(
            model,
            files,
            cache,
            device,
            tile=tile,
            overlap=args.overlap,
            tta=args.tta,
            batch_size=args.tile_batch,
            identity={"epoch": checkpoint.get("epoch")},
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        for name in files:
            sources.setdefault(name, []).append(cache)
        for image_id in val_ids:
            records.setdefault(file_names[image_id], {})[image_id] = by_image[image_id]

    per_file = {name: list(found.values()) for name, found in records.items()}
    calibration = calibrate(
        sources,
        per_file,
        thresholds=tuple(args.thresholds),
        offsets=tuple(args.offsets),
        min_areas=tuple(args.min_areas),
        min_scores=tuple(args.min_scores),
        workers=args.workers,
    )
    calibration |= {
        "tile": tiles.pop() if len(tiles) == 1 else None,
        "overlap": args.overlap,
        "tta": args.tta,
        "checkpoints": [str(path) for path in args.checkpoint],
    }

    print(f"{'threshold':>10} {'offset':>7} {'min_area':>9} {'min_score':>10} {'pq':>7} {'tp':>6} {'fp':>6} {'fn':>6}")
    for row in calibration["table"][: args.top]:
        print(
            f"{row['threshold']:>10.3f} {row['offset']:>+7d} {row['min_area']:>9d} {row['min_score']:>10.2f} "
            f"{row['pq']:>7.4f} {row['tp']:>6d} {row['fp']:>6d} {row['fn']:>6d}"
        )
    print(
        f"best pq {calibration['pq']:.4f} over {calibration['records']} records / {calibration['files']} observations "
        f"at threshold {calibration['threshold']:.3f}, offset {calibration['offset']:+d}, "
        f"min_area {calibration['min_area']}, min_score {calibration['min_score']:.2f}"
    )
    if calibration["at_edge"]:
        print(
            f"warning: {', '.join(calibration['at_edge'])} landed on the edge of the grid; "
            f"widen it and run again, the optimum is outside what was searched"
        )

    if args.output:
        write_calibration(args.output, calibration)
        print(f"stored in {args.output}")
        return
    checkpoint_path = Path(args.checkpoint[0])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint |= {
        "threshold": calibration["threshold"],
        "offset": calibration["offset"],
        "min_area": calibration["min_area"],
        "min_score": calibration["min_score"],
        "tile": calibration["tile"],
        "overlap": args.overlap,
        "calibration": {key: value for key, value in calibration.items() if key != "table"},
    }
    torch.save(checkpoint, checkpoint_path)
    write_calibration(checkpoint_path.with_suffix(".calibration.json"), calibration)
    print(f"stored in {checkpoint_path} and {checkpoint_path.with_suffix('.calibration.json')}")


def main() -> None:
    """
    Dispatch to the requested subcommand.
    :return: None
    """
    args = parse_args()
    set_seeds(args.seed)
    device = resolve_device(args.device)
    print(f"device: {device}")
    print(f"data: {DATA_ROOT}")

    if args.command == "train":
        run_training(args, device)
    elif args.command == "calibrate":
        run_calibration(args, device)
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

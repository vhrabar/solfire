"""
The ``solfire`` command line: train, calibrate, evaluate and predict with the final method.

``solfire-lab`` (:mod:`solfire.experiments.main`) is the same command line with the dropped
experiments added through :class:`~solfire.extension.Extension`.
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import (
    DATA_ROOT,
    RASTERS,
    TARGETS,
    TEST_IMAGES,
    TRAIN_IMAGES,
    FilamentDataset,
    TestImageDataset,
    load_annotations,
    preprocessed_dir,
    validation_split,
)
from .extension import Extension
from .inference import (
    DEFAULT_OVERLAP,
    MIN_AREA,
    SUBMISSION_SIZE,
    load_model,
    predict_dataset,
    write_detections,
    write_submission,
)
from .inference.barbs import DEFAULT_BARBS
from .inference.refiner import load_refiner
from .networks import ARCHITECTURES, DECODERS
from .training import (
    CALIBRATION_THRESHOLDS,
    DEFAULT_MIN_AREAS,
    DEFAULT_MIN_SCORES,
    DEFAULT_OFFSETS,
    AugmentedDataset,
    build_optimizer,
    build_scheduler,
    cache_directory,
    cache_probabilities,
    calibrate,
    estimate_pos_weight,
    evaluate_tiled,
    fit,
    parse_variant,
    plot_evaluation,
    threshold_grid,
    write_calibration,
)
from .training.evaluate import evaluate

CORE = Extension()


def build_parser(extension: Extension = CORE) -> tuple[argparse.ArgumentParser, dict]:
    """
    Build the command line.
    :param extension: Adds options and subcommands
    :return: Tuple of (parser, subcommand name -> function of ``(args, device)`` for the extension's subcommands)
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
    train.add_argument(
        "--input-scale",
        type=float,
        default=1.0,
        help="train on observations enlarged by this factor; validation and inference enlarge them the same way",
    )
    train.add_argument("--size", type=int, default=256, help="edge length observations are resized to when --crop is 0")
    train.add_argument("--arch", choices=sorted(ARCHITECTURES), default="pretrained")
    train.add_argument("--decoder", choices=sorted(DECODERS), default="unet")
    train.add_argument("--encoder", default="resnet34", help="smp encoder name, tu- prefixed for timm")
    train.add_argument("--encoder-weights", default="imagenet", help="weights to start the encoder from; 'none' for random")
    train.add_argument("--encoder-lr-scale", type=float, default=1.0, help="encoder learning rate as a fraction of --lr")
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
    train.add_argument("--lovasz-weight", type=float, default=0.0, help="weight of the Lovász hinge term; 0 leaves it out")
    train.add_argument(
        "--init-checkpoint", default=None, help="start from this checkpoint's weights (same architecture), e.g. to fine-tune"
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
    # the final models read the observations as they are; solfire-lab adds --preprocess
    train.set_defaults(preprocess="none")

    predict = subparsers.add_parser("predict", help="write predictions for the test observations")
    predict.add_argument(
        "--checkpoint", nargs="+", default=["runs/pretrained/best.pt"], help="several checkpoints average their probabilities"
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
    predict.add_argument(
        "--precision",
        choices=("fp32", "fp16", "bf16"),
        default=None,
        help="autocast precision of the models and refiners; default: the calibration's, else fp32",
    )
    predict.add_argument(
        "--skip-dark",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="leave inference windows off the solar disk out; default: the calibration's, else off",
    )
    predict.add_argument(
        "--barbs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="grow the visible barbs into every instance (src/solfire/inference/barbs.py); default: the calibration's, else off",
    )
    predict.add_argument("--offset", type=int, default=None, help="default: the value tuned on validation")
    predict.add_argument("--min-area", type=int, default=None, help="drop predicted instances smaller than this many pixels")
    predict.add_argument("--min-score", type=float, default=None, help="drop predicted instances with a lower mean probability")
    predict.add_argument("--category-id", type=int, default=1)
    predict.add_argument("--save-masks", action="store_true", help="also write mask PNGs")
    predict.add_argument(
        "--refine",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use the crop refiner when the calibration names one (scripts/refine.py)",
    )
    predict.add_argument("--output-dir", default="runs/pretrained/predictions")

    calibration = subparsers.add_parser(
        "calibrate", help="tune the decision threshold, the morphological offset and the minimum area on validation"
    )
    calibration.add_argument(
        "--checkpoint",
        nargs="+",
        default=["runs/pretrained/best.pt"],
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
    calibration.add_argument(
        "--variants",
        nargs="+",
        default=["base"],
        help="inference variants averaged per checkpoint, each cached on its own: base, or '-'-joined parts "
        "turned|d4, s<scale>, o<overlap> (e.g. base turned s1.25); base+turned is the full D4 average",
    )
    calibration.add_argument("--workers", type=int, default=8, help="processes scoring observations")
    calibration.add_argument("--output", default=None, help="calibration JSON to write; required with several checkpoints")
    calibration.add_argument("--top", type=int, default=10, help="settings to print")

    evaluation = subparsers.add_parser(
        "evaluate", help="measure everything the final judging grades, on the held-out maps of the checkpoints"
    )
    evaluation.add_argument("--checkpoint", nargs="+", required=True, help="as for calibrate")
    evaluation.add_argument("--calibration", nargs="+", required=True, help="calibration JSONs whose post-processing is compared")
    evaluation.add_argument("--tile", type=int, default=0, help="inference window; 0 follows the checkpoint")
    evaluation.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP)
    evaluation.add_argument("--tile-batch", type=int, default=8)
    evaluation.add_argument("--tta", action=argparse.BooleanOptionalAction, default=True)
    evaluation.add_argument(
        "--variants",
        nargs="+",
        default=["base"],
        help="inference variants averaged per checkpoint, each cached on its own: base, or '-'-joined parts "
        "turned|d4, s<scale>, o<overlap> (e.g. base turned s1.25); base+turned is the full D4 average",
    )
    evaluation.add_argument("--workers", type=int, default=8)
    evaluation.add_argument("--output", default=None, help="JSON to write the summaries to")
    evaluation.add_argument("--plots", default=None, help="directory to write the distribution plots to")
    evaluation.add_argument(
        "--refined-dir",
        default=None,
        help="refined out-of-fold instances (scripts/refine.py refine); calibrations with a refiner block are "
        "scored on them, the others on the probabilities",
    )

    extension.add_arguments({"train": train, "predict": predict, "calibrate": calibration, "evaluate": evaluation})
    return parser, extension.add_commands(subparsers)


def parse_args(argv=None, extension: Extension = CORE) -> argparse.Namespace:
    """
    Parse the command line.
    :param argv: Arguments, or ``None`` for ``sys.argv``
    :param extension: Adds options and subcommands
    :return: Parsed arguments
    """
    parser, _ = build_parser(extension)
    return parser.parse_args(argv)


def resolve_device(name=None) -> torch.device:
    """
    Pick the device to run on.
    :param name: Explicit device string, or ``None`` to autodetect
    :return: Selected device
    """
    if name is not None:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model_config(args, extension: Extension = CORE) -> dict:
    """
    Collect the constructor arguments of the requested architecture.
    :param args: Parsed ``train`` arguments
    :param extension: Builds the architectures other than ``pretrained``, and sets the auxiliary channels
    :return: Keyword arguments for the matching entry of :data:`networks.ARCHITECTURES`
    """
    if args.arch != "pretrained":
        return extension.model_config(args)
    weights = None if args.encoder_weights.lower() == "none" else args.encoder_weights
    return {
        "in_channels": 1,
        "out_channels": 1,
        "decoder": args.decoder,
        "encoder": args.encoder,
        "encoder_weights": weights,
        "aux_channels": extension.aux_channels(args),
    }


def build_datasets(args, extension: Extension = CORE):
    """
    Build the training and validation datasets for the requested resolution mode.

    With ``--crop`` the observations are kept at the resolution they were annotated
    at and windows are drawn out of them, so an eleven pixel wide filament stays
    eleven pixels wide. Validation then runs on whole observations, which is the
    shape the submission is written in.
    :param args: Parsed ``train`` arguments
    :param extension: Builds the training crops and adds samples
    :return: Tuple of (train dataset, validation dataset, validation batch size)
    """
    images, _ = load_annotations()
    train_ids, val_ids = validation_split(images, args.folds, args.fold, args.val_fraction, args.seed)
    categories = tuple(args.categories) if args.categories else None
    native_size = args.native_size or None
    images_dir = require_images(preprocessed_dir(TRAIN_IMAGES, args.preprocess))
    common = {"category_ids": categories, "raster": args.raster, "images_dir": images_dir}

    if not args.crop:
        train_dataset = FilamentDataset(train_ids, size=args.size, **common)
        valid_dataset = FilamentDataset(val_ids, size=args.size, **common)
        return train_dataset, valid_dataset, args.batch_size

    options = {
        "crop": args.crop,
        "native_size": native_size,
        "repeat": args.crops_per_image,
        "foreground_prob": args.foreground_prob,
        "targets": args.targets,
        "input_scale": args.input_scale,
        **common,
    }
    train_dataset = extension.extend_train_dataset(args, extension.train_dataset(args, train_ids, options))
    valid_dataset = FilamentDataset(val_ids, size=native_size, **common)
    return train_dataset, valid_dataset, 1


def require_images(images_dir) -> Path:
    """
    Fail early when a preprocessed cache has not been built.
    :param images_dir: Directory the observations are read from
    :return: ``images_dir``
    """
    if not Path(images_dir).is_dir():
        raise SystemExit(f"{images_dir} does not exist; build it with `python scripts/experiments/preprocess.py`")
    return Path(images_dir)


def load_matching(model, state) -> None:
    """
    Load a state dict into a model whose output layer may have more channels.

    Tensors of equal shape are copied; an output layer that grew along its first axis
    gets the old channels in front and keeps its fresh initialisation for the new ones,
    so a one-channel checkpoint can start a model with auxiliary outputs.
    :param model: Model to load into
    :param state: State dict of the starting checkpoint
    :return: None
    """
    own = model.state_dict()
    for key, value in state.items():
        if key not in own:
            raise KeyError(f"{key} is not in the model")
        if own[key].shape == value.shape:
            own[key] = value
        elif own[key].shape[1:] == value.shape[1:] and own[key].shape[0] > value.shape[0]:
            own[key][: value.shape[0]] = value
        else:
            raise ValueError(f"{key}: checkpoint shape {tuple(value.shape)} does not fit {tuple(own[key].shape)}")
    model.load_state_dict(own)


def run_training(args, device, extension: Extension = CORE) -> None:
    """
    Train a segmentation model and checkpoint the best validation Panoptic Quality.
    :param args: Parsed ``train`` arguments
    :param device: Device to run on
    :param extension: Adds datasets, objectives and checks
    :return: None
    """
    extension.check(args)
    output_dir = args.output_dir or f"runs/{args.arch}"
    categories = tuple(args.categories) if args.categories else None

    train_dataset, valid_dataset, val_batch_size = build_datasets(args, extension)
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
    # --folds 0 --val-fraction 0 trains on every record and keeps the final epoch
    held_out = len(valid_dataset) > 0
    if not held_out and (args.swa_epochs or args.patience):
        raise SystemExit("--swa-epochs and --patience need a validation split")
    val_loader = DataLoader(valid_dataset, batch_size=val_batch_size, shuffle=False, **loader_kwargs) if held_out else None

    model_config = build_model_config(args, extension)
    model = ARCHITECTURES[args.arch](**model_config).to(device)
    if args.init_checkpoint:
        start = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        load_matching(model, start["model_state"])
        print(f"starting from {args.init_checkpoint} (epoch {start.get('epoch')})")
    extension.prepare_model(args, model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{type(model).__name__}: {n_params / 1e6:.2f}M parameters")
    encoder_lr_scale = 1.0
    if args.arch == "pretrained":
        print(f"{args.decoder} on {args.encoder}, encoder weights {model_config['encoder_weights']}")
        encoder_lr_scale = args.encoder_lr_scale
        # the state dict carries the trained weights, so rebuilding must not fetch the pretrained ones again
        model_config = model_config | {"encoder_weights": None}

    criterion = extension.wrap_criterion(args, extension.criterion(args, pos_weight)).to(device)
    optimizer = build_optimizer(
        model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        encoder_lr_scale=encoder_lr_scale,
        **extension.optimizer_options(args),
    )
    scheduler = build_scheduler(optimizer, args.epochs, max(1, len(train_loader)), warmup_epochs=args.warmup_epochs)

    tile = args.val_tile or args.crop or None
    extra = {
        "crop": args.crop or None,
        "native_size": args.native_size or None,
        "input_scale": args.input_scale,
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
        "preprocess": args.preprocess,
        "lovasz_weight": args.lovasz_weight,
        "init_checkpoint": args.init_checkpoint,
        "seed": args.seed,
        "categories": list(categories) if categories else None,
        **extension.checkpoint_metadata(args),
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
            scale=args.input_scale,
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
        evaluate_fn=validate_tiled if args.crop and held_out else None,
        extra=extra,
        swa_epochs=args.swa_epochs,
        resume=args.resume,
    )
    if not held_out:
        print(f"trained on every record; epoch {summary['best_epoch']} -> {output_dir}/best.pt")
        return
    print(
        f"best pq {summary['best_pq']:.4f} at epoch {summary['best_epoch']} "
        f"(threshold {summary['best_threshold']:.2f}) -> {output_dir}/best.pt"
    )
    name = "best.pt"
    if "swa_pq" in summary:
        name = "swa.pt" if summary["swa_pq"] > summary["best_pq"] else "best.pt"
        print(f"averaged {summary['swa_epochs']} epochs: pq {summary['swa_pq']:.4f}")
    if args.crop:
        print(f"now tune the post-processing: python -m solfire.main calibrate --checkpoint {output_dir}/{name}")


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


def run_prediction(args, device, extension: Extension = CORE) -> None:
    """
    Write COCO-style detections for the unlabelled test observations.
    :param args: Parsed ``predict`` arguments
    :param device: Device to run on
    :param extension: Loads a learned instance filter a calibration names
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
    instance_filter = extension.instance_filter(settings)
    precision = args.precision or settings.get("precision") or "fp32"
    skip_dark = settings.get("skip_dark", False) if args.skip_dark is None else args.skip_dark
    variant_names = settings.get("variants") or ["base"]
    variants = [parse_variant(name, tta=args.tta, overlap=args.overlap) for name in variant_names]
    # a calibration's "barbs" block overrides DEFAULT_BARBS; --barbs / --no-barbs switch it on or off
    barbs = settings.get("barbs")
    if args.barbs is not None:
        barbs = (barbs or {}) if args.barbs else None
    refiner_settings = settings.get("refiner") if args.refine else None
    refiners = [load_refiner(path, device) for path in refiner_settings["checkpoints"]] if refiner_settings else None
    print(
        f"{len(models)} model(s), checkpoint epoch {checkpoint.get('epoch')}, size {size or 'native'}, "
        f"tile {tile or 'whole observation'}, threshold {threshold:.2f}, offset {offset:+d}, "
        f"min_area {min_area}, min_score {min_score:.2f}, {precision}"
        + (", dark windows skipped" if skip_dark else "")
        + (f", variants {' '.join(variant_names)}" if variant_names != ["base"] else "")
        + (f", learned filter cut {instance_filter.cut:.2f}" if instance_filter else "")
        + (f", barbs grown ({', '.join(f'{k} {v}' for k, v in {**DEFAULT_BARBS, **barbs}.items())})" if barbs is not None else "")
    )
    if refiner_settings and refiners:
        print(
            f"{len(refiners)} crop refiner(s): proposals at {refiner_settings['proposal_threshold']:.3f} "
            f"offset {refiner_settings['proposal_offset']:+d}, refined masks at {refiner_settings['threshold']:.2f}, "
            f"min predicted IoU {refiner_settings['min_score']:.2f}; they replace the cuts above"
        )
    if args.calibration is None and (len(models) > 1 or checkpoint.get("calibration") is None):
        print("warning: no calibration for this set of models; run `calibrate` before submitting")

    methods = {found.get("preprocess") or "none" for _, found in loaded}
    if len(methods) > 1:
        raise SystemExit(f"the checkpoints were trained on different preprocessing ({', '.join(sorted(methods))})")
    dataset = TestImageDataset(size=size, images_dir=require_images(preprocessed_dir(TEST_IMAGES, methods.pop())))
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
        instance_filter=instance_filter,
        offset=offset,
        category_id=args.category_id,
        save_masks_dir=output_dir / "masks" if args.save_masks else None,
        tile=tile,
        overlap=args.overlap,
        tile_batch=args.tile_batch,
        refiners=refiners,
        refiner_settings=refiner_settings,
        scales=[found.get("input_scale") or 1.0 for _, found in loaded],
        variants=variants,
        precision=None if precision == "fp32" else precision,
        skip_dark=skip_dark,
        barbs=barbs,
    )

    submission_path = output_dir / "submission.csv"
    write_submission(detections, submission_path)
    write_detections(detections, output_dir / "detections.json")

    n_images = len({d["image_id"] for d in detections})
    print(f"{len(detections)} filaments over {n_images}/{len(dataset)} observations -> {submission_path}")
    if not detections:
        print("warning: no filaments predicted; lower --threshold or train for longer")


def held_out_maps(args, device) -> tuple[dict, dict, set, str]:
    """
    Cache every checkpoint's probabilities on the observations its split held out.

    Checkpoints that held out the same observation are averaged there, so a seed
    ensemble on a shared split and a set of fold models on their pooled
    out-of-fold maps are handled alike.
    :param args: Parsed ``calibrate`` or ``fit-filter`` arguments
    :param device: Device to run on
    :return: Tuple of (per file the cache directories, per file the annotations of each
        record, the inference windows used, the preprocessing the models share)
    """
    images, by_image = load_annotations()
    file_names = {image["id"]: image["file_name"] for image in images}

    sources: dict[str, list[Path]] = {}
    records: dict[str, dict] = {}
    tiles = set()
    methods = set()
    for path in args.checkpoint:
        model, checkpoint = load_model(path, device)
        if not checkpoint.get("crop"):
            raise SystemExit(f"{path}: calibration needs a model trained on native resolution crops")
        tile = args.tile or checkpoint.get("tile") or checkpoint.get("crop")
        tiles.add(tile)
        methods.add(checkpoint.get("preprocess") or "none")
        _, val_ids = validation_split(
            images,
            checkpoint.get("folds") or 0,
            checkpoint.get("fold") or 0,
            checkpoint.get("val_fraction") or 0.2,
            checkpoint.get("seed", args.seed),
        )
        files = sorted({file_names[image_id] for image_id in val_ids})
        print(f"{path}: {len(val_ids)} held-out records over {len(files)} observations, tile {tile}")
        caches = []
        for variant in args.variants:
            settings = parse_variant(variant, tta=args.tta, overlap=args.overlap)
            cache = cache_directory(path, variant)
            cache_probabilities(
                model,
                files,
                cache,
                device,
                tile=tile,
                overlap=settings["overlap"],
                tta=settings["tta"],
                scale=settings["scale"] * (checkpoint.get("input_scale") or 1.0),
                batch_size=args.tile_batch,
                identity={"epoch": checkpoint.get("epoch")},
                images_dir=require_images(preprocessed_dir(TRAIN_IMAGES, checkpoint.get("preprocess"))),
            )
            caches.append(cache)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        for name in files:
            sources.setdefault(name, []).extend(caches)
        for image_id in val_ids:
            records.setdefault(file_names[image_id], {})[image_id] = by_image[image_id]

    per_file = {name: list(found.values()) for name, found in records.items()}
    return sources, per_file, tiles, methods.pop() if len(methods) == 1 else "mixed"


def run_calibration(args, device) -> None:
    """
    Tune the post-processing on held-out observations, through the path a submission takes.

    Each checkpoint is run over the observations its own split held out, and the
    probabilities are cached next to it; see :func:`held_out_maps`.
    :param args: Parsed ``calibrate`` arguments
    :param device: Device to run on
    :return: None
    """
    if len(args.checkpoint) > 1 and not args.output:
        raise SystemExit("--output is required when calibrating several checkpoints")
    sources, per_file, tiles, _ = held_out_maps(args, device)
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
        "variants": args.variants,
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


def run_evaluation(args, device, extension: Extension = CORE) -> None:
    """
    Compare calibrations on every quantity the final judging grades.
    :param args: Parsed ``evaluate`` arguments
    :param device: Device to run on
    :param extension: Supplies the cross-validated filters of a calibration with a learned filter
    :return: None
    """
    sources, per_file, _, method = held_out_maps(args, device)
    images_dir = None if method == "mixed" else require_images(preprocessed_dir(TRAIN_IMAGES, method))
    cache: dict = {}
    summaries = {}
    for path in args.calibration:
        calibration = json.loads(Path(path).read_text())
        refined_dir = args.refined_dir if "refiner" in calibration else None
        filters = None
        if calibration.get("filter") and refined_dir is None:
            filters = extension.evaluation_filters(calibration, sources, per_file, images_dir, args.workers, cache)
        summary, overlaps = evaluate(
            sources,
            per_file,
            calibration,
            images_dir=images_dir or TRAIN_IMAGES,
            filters=filters,
            workers=args.workers,
            refined_dir=refined_dir,
        )
        summaries[str(path)] = summary
        if args.plots:
            target = Path(args.plots) / f"{Path(path).name.removesuffix('.json')}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            plot_evaluation([(torch.from_numpy(iou), torch.from_numpy(dice)) for iou, dice in overlaps], target)

    rows = (
        ("pq", "PQ"),
        ("sq", "SQ (matched IoU)"),
        ("rq", "RQ"),
        ("mean_dice", "mean Dice per record"),
        ("matched_dice", "Dice of matched pairs"),
        ("pair_iou", "IoU of overlapping pairs"),
        ("pair_dice", "Dice of overlapping pairs"),
        ("missed_share", "GT with no overlap"),
        ("fragmented_share", "GT meeting 2+ predictions"),
        ("spurious_share", "predictions with no overlap"),
        ("merged_share", "predictions meeting 2+ GT"),
        ("predictions_per_record", "predictions per record"),
    )
    names = [Path(path).name.removesuffix(".calibration.json").removesuffix(".json") for path in summaries]
    width = max(len(label) for _, label in rows)
    print(f"{'':<{width}}  " + "  ".join(f"{name:>16}" for name in names))
    for key, label in rows:
        print(f"{label:<{width}}  " + "  ".join(f"{summary[key]:>16.4f}" for summary in summaries.values()))
    for relation in ("gt_degree", "pred_degree"):
        print(f"{relation:<{width}}  " + "  ".join(f"{json.dumps(summary[relation]):>16}" for summary in summaries.values()))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        print(f"stored in {args.output}")


def main(argv=None, extension: Extension = CORE) -> None:
    """
    Dispatch to the requested subcommand.
    :param argv: Arguments, or ``None`` for ``sys.argv``
    :param extension: Adds options and subcommands; :mod:`solfire.experiments.main` passes the experiments'
    :return: None
    """
    parser, commands = build_parser(extension)
    args = parser.parse_args(argv)
    set_seeds(args.seed)
    device = resolve_device(args.device)
    print(f"device: {device}")
    print(f"data: {DATA_ROOT}")

    if args.command in commands:
        commands[args.command](args, device)
    elif args.command == "train":
        run_training(args, device, extension)
    elif args.command == "calibrate":
        run_calibration(args, device)
    elif args.command == "evaluate":
        run_evaluation(args, device, extension)
    else:
        run_prediction(args, device, extension)


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

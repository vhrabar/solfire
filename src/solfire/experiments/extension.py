"""
The experiments' side of :class:`~solfire.extension.Extension`: their ``train`` options, targets,
objectives, model initialisation and checkpoint settings, the learned instance filter at
``predict`` and ``evaluate``, and the ``fit-filter`` command.

Every option defaults to off, so ``solfire-lab`` without them trains exactly what ``solfire`` does.
"""

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import ConcatDataset

from solfire.data import TRAIN_IMAGES, preprocessed_dir, preprocessing_methods
from solfire.extension import Extension, TrainData
from solfire.inference import DEFAULT_OVERLAP
from solfire.main import held_out_maps, require_images
from solfire.training import write_calibration

from .data import ExperimentCropDataset, MagneticTargets, PseudoLabelledCropDataset
from .filter_fit import collect_instances, cross_validated_filters, filter_path, fit_instance_filter
from .instance_filter import InstanceFilter
from .losses import (
    KERNEL_SHAPES,
    BarbHeadLoss,
    CategoryHeadLoss,
    ExperimentPixelLoss,
    MagneticHeadLoss,
    MorphologyLoss,
    ThinHeadLoss,
    build_morphology_loss,
)

BARB_PRIOR_LOGIT = -4.0


def morphology_active(args) -> bool:
    """
    :param args: Parsed ``train`` arguments
    :return: Whether any morphology term has a weight
    """
    return max(args.morph_open_weight, args.morph_close_weight, args.morph_cldice_weight) > 0


class ExperimentsExtension(Extension):
    """
    Adds every dropped experiment to the command line.
    """

    def add_arguments(self, parsers: dict[str, argparse.ArgumentParser]) -> None:
        """
        The ``train`` options of the experiments.
        :param parsers: Subcommand name -> its parser
        :return: None
        """
        train = parsers["train"]
        train.add_argument("--base-channels", type=int, default=32, help="unet/transunet only")
        train.add_argument("--depth", type=int, default=4, help="unet/transunet only")
        train.add_argument("--bilinear", action=argparse.BooleanOptionalAction, default=True, help="unet/transunet only")
        train.add_argument("--embed-dims", type=int, nargs="+", default=[32, 64, 160, 256], help="segformer only")
        train.add_argument("--depths", type=int, nargs="+", default=[2, 2, 2, 2], help="segformer only")
        train.add_argument("--num-heads", type=int, nargs="+", default=[1, 2, 5, 8], help="segformer only")
        train.add_argument("--decoder-dim", type=int, default=256, help="segformer only")
        train.add_argument("--blocks", type=int, default=4, help="transunet only: attention blocks at the bottleneck")
        train.add_argument("--heads", type=int, default=8, help="transunet only: attention heads")
        train.add_argument(
            "--embed-dim", type=int, default=None, help="transunet only: token width (default: the bottleneck width)"
        )
        train.add_argument("--mlp-ratio", type=float, default=2.0, help="transunet only")
        train.add_argument("--dropout", type=float, default=0.0, help="segformer/transunet only")
        train.add_argument("--drop-path", type=float, default=0.1, help="segformer/transunet only")
        train.add_argument(
            "--pseudo-labels",
            default=None,
            help="directory of per-fold soft targets for unlabeled observations (scripts/experiments/pseudo_label.py); needs --folds",
        )
        train.add_argument(
            "--pseudo-images", default="runs/external/gong/jpeg", help="directory holding the pseudo-labelled observations"
        )
        train.add_argument("--pseudo-crops-per-image", type=int, default=1, help="crops drawn per pseudo-labelled observation")
        train.add_argument(
            "--preprocess",
            choices=preprocessing_methods(),
            default="none",
            help="read observations from a preprocessed cache; build it first with scripts/experiments/preprocess.py",
        )
        train.add_argument(
            "--category-head-weight",
            type=float,
            default=0.0,
            help="pretrained only: weight of an auxiliary head predicting the filament category (Left/Right/Unidentifiable); 0 disables it",
        )
        train.add_argument(
            "--magnetic-head-weight",
            type=float,
            default=0.0,
            help="pretrained only: weight of a training-only head regressing GONG magnetogram targets "
            "(PIL proximity, strong flux); 0 disables it",
        )
        train.add_argument(
            "--magnetic-targets",
            default="runs/external/gong_bzi/targets",
            help="directory of precomputed magnetic targets (scripts/experiments/magnetic_targets.py)",
        )
        train.add_argument(
            "--tversky-beta",
            type=float,
            default=0.5,
            help="weight of false negatives in the overlap term (Tversky); 0.5 is Dice, above it favours recall",
        )
        train.add_argument(
            "--thickness-weight-cap",
            type=float,
            default=0.0,
            help="weight filament pixels in the BCE and Lovász terms by 6 / local half-width, clamped to [1, cap], "
            "so barbs and thin ends count as much as the body; needs --crop; 0 disables it",
        )
        train.add_argument(
            "--barb-targets",
            default=None,
            help="directory of barb PNGs grown from the images outside the annotations (scripts/experiments/barb_targets.py); "
            "the target is raised to --barb-level there; needs --crop at the annotated resolution",
        )
        train.add_argument("--barb-level", type=float, default=1.0, help="target value at the --barb-targets pixels")
        train.add_argument(
            "--thin-head-weight",
            type=float,
            default=0.0,
            help="pretrained only: weight of a training-only head predicting the thin part of the annotation "
            "(opening residual, r=2) and the spine heatmap; 0 disables it",
        )
        train.add_argument(
            "--barb-head-weight",
            type=float,
            default=0.0,
            help="pretrained only: weight of a head predicting the thin part of the annotation of the records "
            "that draw barbs (--barb-records), kept in the checkpoint for scripts/experiments/barb_head.py; 0 disables it",
        )
        train.add_argument(
            "--head-lr-scale",
            type=float,
            default=1.0,
            help="pretrained only: learning rate of the output layer as a multiple of --lr, e.g. so a new barb head "
            "channel is learnt within a fine-tune",
        )
        train.add_argument(
            "--barb-records",
            default="runs/barb_drawers.json",
            help="JSON list of the image ids of the records that draw barbs (scripts/experiments/barb_drawers.py)",
        )
        train.add_argument(
            "--hard-negatives",
            default=None,
            help="JSON of per-file (x, y) look-alike points (scripts/experiments/hard_negatives.py)",
        )
        train.add_argument(
            "--hard-negative-prob",
            type=float,
            default=0.15,
            help="share of crops centred on a hard negative, taken from random crops",
        )
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

    def add_commands(self, subparsers) -> dict:
        """
        ``fit-filter``: learn which predicted instances to submit.
        :param subparsers: The ``add_subparsers`` action of the top-level parser
        :return: ``{"fit-filter": run_filter_fitting}``
        """
        fitting = subparsers.add_parser(
            "fit-filter", help="learn which predicted instances to submit, on the held-out maps of the checkpoints"
        )
        fitting.add_argument("--checkpoint", nargs="+", required=True, help="as for calibrate")
        fitting.add_argument("--threshold", type=float, default=0.5, help="probability threshold the instances are cut at")
        fitting.add_argument("--offset", type=int, default=0)
        fitting.add_argument("--tile", type=int, default=0, help="inference window; 0 follows the checkpoint")
        fitting.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP)
        fitting.add_argument("--tile-batch", type=int, default=8)
        fitting.add_argument("--tta", action=argparse.BooleanOptionalAction, default=True)
        fitting.add_argument(
            "--variants",
            nargs="+",
            default=["base"],
            help="inference variants averaged per checkpoint, each cached on its own: base, or '-'-joined parts "
            "turned|d4, s<scale>, o<overlap> (e.g. base turned s1.25); base+turned is the full D4 average",
        )
        fitting.add_argument("--workers", type=int, default=8)
        fitting.add_argument("--output", required=True, help="calibration JSON to write; the filter is pickled next to it")

        return {"fit-filter": run_filter_fitting}

    def check(self, args) -> None:
        """
        Reject the combinations of experiments that were never supported.
        :param args: Parsed ``train`` arguments
        :return: None
        :raises SystemExit: On such a combination
        """
        if args.barb_targets and not args.crop:
            raise SystemExit("--barb-targets needs --crop")
        thin_structure = args.thickness_weight_cap > 0 or args.thin_head_weight > 0
        if thin_structure:
            if not args.crop:
                raise SystemExit("--thickness-weight-cap and --thin-head-weight need --crop")
            if args.pseudo_labels:
                raise SystemExit("--thickness-weight-cap and --thin-head-weight do not support --pseudo-labels")
            if args.category_head_weight > 0 or args.magnetic_head_weight > 0:
                raise SystemExit("the thin-structure targets cannot be combined with the category or magnetic head")
            if morphology_active(args):
                raise SystemExit("the thin-structure targets cannot be combined with the morphology terms")
        if args.barb_head_weight > 0:
            if args.arch != "pretrained" or not args.crop:
                raise SystemExit("--barb-head-weight needs --arch pretrained and --crop")
            if thin_structure or args.pseudo_labels or args.category_head_weight > 0 or args.magnetic_head_weight > 0:
                raise SystemExit("--barb-head-weight cannot be combined with the other heads, pixel weights or pseudo-labels")
            if morphology_active(args):
                raise SystemExit("--barb-head-weight cannot be combined with the morphology terms")
        if args.head_lr_scale != 1.0 and args.arch != "pretrained":
            raise SystemExit("--head-lr-scale needs --arch pretrained")
        if args.thin_head_weight > 0 and args.arch != "pretrained":
            raise SystemExit("--thin-head-weight needs --arch pretrained")
        if not 0 < args.tversky_beta < 1:
            raise SystemExit("--tversky-beta must lie strictly between 0 and 1")
        if args.thickness_weight_cap and args.thickness_weight_cap < 1:
            raise SystemExit("--thickness-weight-cap must be at least 1")
        if args.category_head_weight > 0 and args.arch != "pretrained":
            raise SystemExit("--category-head-weight needs --arch pretrained")
        if args.magnetic_head_weight > 0:
            if args.arch != "pretrained":
                raise SystemExit("--magnetic-head-weight needs --arch pretrained")
            if args.category_head_weight > 0:
                raise SystemExit("--magnetic-head-weight and --category-head-weight cannot be combined")
            if args.pseudo_labels:
                raise SystemExit("--magnetic-head-weight does not support --pseudo-labels yet")
            if not args.crop:
                raise SystemExit("--magnetic-head-weight needs --crop")

    def model_config(self, args) -> dict:
        """
        Constructor arguments of the architectures trained from scratch.
        :param args: Parsed ``train`` arguments
        :return: Keyword arguments of ``unet``, ``transunet`` or ``segformer``
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
        if args.arch != "segformer":
            return super().model_config(args)
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

    def aux_channels(self, args) -> int:
        """
        :param args: Parsed ``train`` arguments
        :return: Channels of the category (3), magnetic or thin-structure (2) or barb (1) head
        """
        if args.category_head_weight > 0:
            return 3
        if args.magnetic_head_weight > 0 or args.thin_head_weight > 0:
            return 2
        return 1 if args.barb_head_weight > 0 else 0

    def train_dataset(self, args, image_ids, options: dict) -> TrainData:
        """
        Crops with the targets of the switched-on experiments.
        :param args: Parsed ``train`` arguments
        :param image_ids: Image ids of the training split
        :param options: Keyword arguments of the core crop dataset
        :return: An :class:`~solfire.experiments.data.ExperimentCropDataset`
        """
        return ExperimentCropDataset(
            image_ids,
            hard_negatives=json.loads(Path(args.hard_negatives).read_text()) if args.hard_negatives else None,
            hard_negative_prob=args.hard_negative_prob if args.hard_negatives else 0.0,
            categories=args.category_head_weight > 0,
            magnetic=MagneticTargets(args.magnetic_targets) if args.magnetic_head_weight > 0 else None,
            thickness_cap=args.thickness_weight_cap,
            thin_head=args.thin_head_weight > 0,
            barb_targets=args.barb_targets,
            barb_level=args.barb_level,
            barb_records=json.loads(Path(args.barb_records).read_text()) if args.barb_head_weight > 0 else None,
            **options,
        )

    def extend_train_dataset(self, args, dataset: TrainData) -> TrainData:
        """
        Add the pseudo-labelled GONG observations.
        :param args: Parsed ``train`` arguments
        :param dataset: The crops of the annotated observations
        :return: The dataset to train on
        """
        if not args.pseudo_labels:
            return dataset
        if not args.folds:
            raise SystemExit("--pseudo-labels needs --folds: the targets of fold k come from teachers that held fold k out")
        pseudo = PseudoLabelledCropDataset(
            args.pseudo_images,
            Path(args.pseudo_labels) / f"fold{args.fold}",
            crop=args.crop,
            repeat=args.pseudo_crops_per_image,
            foreground_prob=args.foreground_prob,
        )
        print(f"pseudo-labelled: {len(pseudo)} crops from {len(pseudo.targets)} observations")
        return ConcatDataset([dataset, pseudo])

    def criterion(self, args, pos_weight: float) -> nn.Module:
        """
        The pixel objective, with the Tversky term and the thin-structure pixel weights when asked for.
        :param args: Parsed ``train`` arguments
        :param pos_weight: Positive class weight of the BCE term
        :return: Loss taking ``(logits, targets)``
        """
        if args.tversky_beta == 0.5 and not args.thickness_weight_cap:
            return super().criterion(args, pos_weight)
        return ExperimentPixelLoss(
            bce_weight=1.0 - args.dice_weight,
            dice_weight=args.dice_weight,
            pos_weight=pos_weight,
            lovasz_weight=args.lovasz_weight,
            pixel_weights=args.thickness_weight_cap > 0,
            tversky_beta=args.tversky_beta,
        )

    def wrap_criterion(self, args, criterion: nn.Module) -> nn.Module:
        """
        Add the morphology terms, then the loss of whichever auxiliary head is on.
        :param args: Parsed ``train`` arguments
        :param criterion: The pixel objective
        :return: The objective to train with
        """
        criterion = build_morphology_loss(criterion, args)
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
        if args.thin_head_weight > 0:
            criterion = ThinHeadLoss(criterion, args.thin_head_weight)
        if args.barb_head_weight > 0:
            criterion = BarbHeadLoss(criterion, args.barb_head_weight)
        if args.category_head_weight > 0:
            criterion = CategoryHeadLoss(criterion, args.category_head_weight)
        if args.magnetic_head_weight > 0:
            criterion = MagneticHeadLoss(criterion, args.magnetic_head_weight)
        return criterion

    def prepare_model(self, args, model: nn.Module) -> None:
        """
        Start a new barb channel near "no barb".
        :param args: Parsed ``train`` arguments
        :param model: The model, after ``--init-checkpoint`` was loaded
        :return: None
        """
        if args.barb_head_weight <= 0:
            return
        # barbs cover well under one percent of a crop; a fresh channel at logit 0 says 0.5 everywhere
        # and a fine-tune learning rate takes far longer than 8 epochs to pull it down
        head = model.get_submodule("net.segmentation_head.0")
        assert isinstance(head, torch.nn.Conv2d) and head.bias is not None
        with torch.no_grad():
            head.weight[1:].mul_(0.1)
            head.bias[1:] = BARB_PRIOR_LOGIT

    def optimizer_options(self, args) -> dict:
        """
        :param args: Parsed ``train`` arguments
        :return: The output layer's learning rate scale
        """
        return {"head_lr_scale": args.head_lr_scale}

    def checkpoint_metadata(self, args) -> dict:
        """
        :param args: Parsed ``train`` arguments
        :return: The experiments' settings, as the checkpoints have always stored them
        """
        return {
            "tversky_beta": args.tversky_beta,
            "category_head_weight": args.category_head_weight,
            "magnetic_head_weight": args.magnetic_head_weight,
            "magnetic_targets": args.magnetic_targets if args.magnetic_head_weight > 0 else None,
            "thickness_weight_cap": args.thickness_weight_cap,
            "thin_head_weight": args.thin_head_weight,
            "barb_targets": args.barb_targets,
            "barb_level": args.barb_level,
            "barb_head_weight": args.barb_head_weight,
            "head_lr_scale": args.head_lr_scale,
            "barb_records": args.barb_records if args.barb_head_weight > 0 else None,
            "hard_negatives": args.hard_negatives,
            "hard_negative_prob": args.hard_negative_prob if args.hard_negatives else 0.0,
            "pseudo_labels": args.pseudo_labels,
            "pseudo_crops_per_image": args.pseudo_crops_per_image if args.pseudo_labels else 0,
            "morph": {
                "open_weight": args.morph_open_weight,
                "close_weight": args.morph_close_weight,
                "cldice_weight": args.morph_cldice_weight,
                "kernel": args.morph_kernel,
                "kernel_size": args.morph_kernel_size,
                "skeleton_iters": args.morph_skeleton_iters,
                "warmup_epochs": args.morph_warmup_epochs,
            }
            if morphology_active(args)
            else None,
        }

    def instance_filter(self, settings: dict) -> InstanceFilter | None:
        """
        :param settings: Calibration the post-processing is taken from
        :return: The pickled :class:`~solfire.experiments.instance_filter.InstanceFilter` it names, or ``None``
        """
        return InstanceFilter.load(settings["filter"]) if settings.get("filter") else None

    def evaluation_filters(self, calibration: dict, sources, per_file, images_dir, workers: int, cache: dict):
        """
        Refit the calibration's filter per group of observations, each without the group it scores.
        :param calibration: Calibration naming a learned ``filter``
        :param sources: Per observation file, the cache directories of its models
        :param per_file: Per observation file, the annotations of each record
        :param images_dir: Directory of the observations the filter reads, or ``None``
        :param workers: Processes working in parallel
        :param cache: Instance features per cut, shared between the calibrations of one run
        :return: ``(group_of, filters)``
        """
        if images_dir is None:
            raise SystemExit("the checkpoints were trained on different preprocessing; the filter reads one kind of image")
        cut_at = (float(calibration["threshold"]), int(calibration.get("offset", 0)))
        if cut_at not in cache:
            cache[cut_at] = collect_instances(sources, per_file, cut_at[0], cut_at[1], images_dir=images_dir, workers=workers)
        return cross_validated_filters(cache[cut_at], calibration)


EXPERIMENTS = ExperimentsExtension()


def run_filter_fitting(args, device) -> None:
    """
    Fit the learned instance filter on held-out maps and write a calibration that uses it.
    :param args: Parsed ``fit-filter`` arguments
    :param device: Device to run on
    :return: None
    """
    sources, per_file, tiles, method = held_out_maps(args, device)
    if method == "mixed":
        raise SystemExit("the checkpoints were trained on different preprocessing; the filter reads one kind of image")
    data = collect_instances(
        sources,
        per_file,
        args.threshold,
        args.offset,
        images_dir=require_images(preprocessed_dir(TRAIN_IMAGES, method)),
        workers=args.workers,
    )
    instance_filter, report = fit_instance_filter(data, args.threshold, args.offset)
    path = filter_path(args.output)
    instance_filter.save(path)
    print(
        f"out-of-fold pq {report['pq']:.4f} at cut {report['cut']:.2f}, keeping {report['kept']} of "
        f"{report['instances']} instances over {report['files']} observations -> {path}"
    )
    calibration = {
        "threshold": args.threshold,
        "offset": args.offset,
        "min_area": 0,
        "min_score": 0.0,
        "filter": str(path),
        "pq": report["pq"],
        "filter_report": report,
        "tile": tiles.pop() if len(tiles) == 1 else None,
        "overlap": args.overlap,
        "tta": args.tta,
        "checkpoints": [str(checkpoint) for checkpoint in args.checkpoint],
    }
    write_calibration(args.output, calibration)
    print(f"stored in {args.output}")

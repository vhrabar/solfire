"""
The points where research code plugs into the command line.

``solfire`` runs the final method with :class:`Extension` itself, whose hooks change nothing.
``solfire-lab`` (:mod:`solfire.experiments`) passes a subclass that adds the options, targets,
objectives and commands of the experiments that were tried and dropped, so they stay runnable
without the core knowing about any of them.
"""

import argparse
from collections.abc import Callable
from typing import Any

from torch import nn
from torch.utils.data import ConcatDataset

from .data import FilamentCropDataset
from .training import BCEDiceLoss

Command = Callable[[argparse.Namespace, object], None]
TrainData = FilamentCropDataset | ConcatDataset


class Extension:
    """
    Hooks the command line calls on its way; every default leaves the final method as it is.
    """

    def add_arguments(self, parsers: dict[str, argparse.ArgumentParser]) -> None:
        """
        Add options to the subcommands.
        :param parsers: Subcommand name -> its parser
        :return: None
        """

    def add_commands(self, subparsers) -> dict[str, Command]:
        """
        Add subcommands.
        :param subparsers: The ``add_subparsers`` action of the top-level parser
        :return: Subcommand name -> function of ``(args, device)`` running it
        """
        return {}

    def check(self, args: argparse.Namespace) -> None:
        """
        Reject option combinations before anything is loaded.
        :param args: Parsed ``train`` arguments
        :return: None
        :raises SystemExit: On a combination that cannot run
        """

    def model_config(self, args: argparse.Namespace) -> dict:
        """
        Constructor arguments of an architecture other than ``pretrained``.
        :param args: Parsed ``train`` arguments
        :return: Keyword arguments of the model class
        :raises SystemExit: The core only builds ``pretrained``
        """
        raise SystemExit(f"--arch {args.arch} is not part of the final method")

    def aux_channels(self, args: argparse.Namespace) -> int:
        """
        Output channels after the filament logit, for auxiliary heads.
        :param args: Parsed ``train`` arguments
        :return: Number of extra channels
        """
        return 0

    def train_dataset(self, args: argparse.Namespace, image_ids, options: dict) -> TrainData:
        """
        The training crops.
        :param args: Parsed ``train`` arguments
        :param image_ids: Image ids of the training split
        :param options: Keyword arguments of :class:`~solfire.data.dataset.FilamentCropDataset`
        :return: Dataset of ``(image, target)`` crops
        """
        return FilamentCropDataset(image_ids, **options)

    def extend_train_dataset(self, args: argparse.Namespace, dataset: TrainData) -> TrainData:
        """
        Add training samples from elsewhere.
        :param args: Parsed ``train`` arguments
        :param dataset: The crops of the annotated observations
        :return: The dataset to train on
        """
        return dataset

    def criterion(self, args: argparse.Namespace, pos_weight: float) -> nn.Module:
        """
        The pixel objective.
        :param args: Parsed ``train`` arguments
        :param pos_weight: Positive class weight of the BCE term
        :return: Loss taking ``(logits, targets)``
        """
        return BCEDiceLoss(
            bce_weight=1.0 - args.dice_weight,
            dice_weight=args.dice_weight,
            pos_weight=pos_weight,
            lovasz_weight=args.lovasz_weight,
        )

    def wrap_criterion(self, args: argparse.Namespace, criterion: nn.Module) -> nn.Module:
        """
        Add terms around the pixel objective.
        :param args: Parsed ``train`` arguments
        :param criterion: Result of :meth:`criterion`
        :return: The objective to train with
        """
        return criterion

    def prepare_model(self, args: argparse.Namespace, model: nn.Module) -> None:
        """
        Adjust a freshly built (and possibly initialised) model before training.
        :param args: Parsed ``train`` arguments
        :param model: The model
        :return: None
        """

    def optimizer_options(self, args: argparse.Namespace) -> dict:
        """
        :param args: Parsed ``train`` arguments
        :return: Extra keyword arguments of :func:`~solfire.training.optim.build_optimizer`
        """
        return {}

    def checkpoint_metadata(self, args: argparse.Namespace) -> dict:
        """
        :param args: Parsed ``train`` arguments
        :return: Extra settings stored in the checkpoint
        """
        return {}

    def instance_filter(self, settings: dict) -> Any | None:
        """
        The learned instance filter a calibration names.
        :param settings: Calibration (or checkpoint) the post-processing is taken from
        :return: Object with ``keep(labels, n_labels, probabilities, image)``, or ``None``
        :raises SystemExit: The calibration names a filter, which the final method does not use
        """
        if settings.get("filter"):
            raise SystemExit("this calibration uses a learned instance filter, an experiment; run it with solfire-lab")
        return None

    def evaluation_filters(
        self, calibration: dict, sources, per_file, images_dir, workers: int, cache: dict
    ) -> tuple[dict, dict] | None:
        """
        Cross-validated instance filters to score a calibration with.
        :param calibration: The calibration being evaluated
        :param sources: Per observation file, the cache directories of its models
        :param per_file: Per observation file, the annotations of each record
        :param images_dir: Directory of the observations the filters read, or ``None`` when the
            models disagree on the preprocessing
        :param workers: Processes working in parallel
        :param cache: Shared between the calibrations of one ``evaluate`` run
        :return: ``(group_of, filters)`` for :func:`~solfire.training.evaluate.evaluate`, or ``None``
        """
        return None

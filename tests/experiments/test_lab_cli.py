"""
``solfire-lab`` is ``solfire`` plus the experiments: without their options it trains the final method.
"""

import json

import numpy as np
import torch
from PIL import Image

import solfire.experiments  # noqa: F401  registers the architectures and preprocessing
from solfire import main as core
from solfire.data import FilamentCropDataset, preprocessing_methods
from solfire.experiments import main as lab
from solfire.experiments.data import ExperimentCropDataset
from solfire.experiments.extension import EXPERIMENTS
from solfire.networks import ARCHITECTURES
from solfire.training.losses import BCEDiceLoss

FINAL = ["train", "--arch", "pretrained", "--encoder", "resnet18", "--encoder-weights", "none", "--lovasz-weight", "0.5"]


def test_lab_registers_the_architectures_and_clahe():
    assert {"unet", "segformer", "transunet", "pretrained"} <= set(ARCHITECTURES)
    assert "clahe" in preprocessing_methods()


def test_lab_without_experiment_options_is_the_core():
    core_args, lab_args = core.parse_args(FINAL), lab.parse_args(FINAL)
    assert vars(core_args).items() <= vars(lab_args).items()
    assert core.build_model_config(core_args) == lab.build_model_config(lab_args)
    criterion = EXPERIMENTS.wrap_criterion(lab_args, EXPERIMENTS.criterion(lab_args, 5.0))
    assert type(criterion) is BCEDiceLoss and criterion.lovasz_weight == 0.5
    assert EXPERIMENTS.optimizer_options(lab_args) == {"head_lr_scale": 1.0}
    metadata = EXPERIMENTS.checkpoint_metadata(lab_args)
    assert metadata["morph"] is None and not any(metadata[key] for key in metadata if key.endswith("head_weight"))


def test_experiment_crops_default_to_the_core_crops(tmp_path):
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 255, (96, 96), dtype=np.uint8)).save(tmp_path / "a.png")
    annotations = {
        "images": [
            {"id": 1, "file_name": "a.png", "width": 96, "height": 96},
            {"id": 2, "file_name": "a.png", "width": 96, "height": 96},
        ],
        "annotations": [
            {"image_id": 1, "segmentation": [[8.0, 8.0, 60.0, 8.0, 60.0, 30.0, 8.0, 30.0]], "category_id": 1},
            {"image_id": 2, "segmentation": [[10.0, 6.0, 70.0, 6.0, 70.0, 28.0, 10.0, 28.0]], "category_id": 2},
        ],
    }
    (tmp_path / "annotations.json").write_text(json.dumps(annotations))
    common = {
        "crop": 48,
        "repeat": 6,
        "seed": 3,
        "targets": "mean",
        "images_dir": tmp_path,
        "annotations_path": tmp_path / "annotations.json",
    }
    plain, experiment = FilamentCropDataset(**common), ExperimentCropDataset(**common)
    for index in range(len(plain)):
        (image, target), (other_image, other_target) = plain[index], experiment[index]
        assert torch.equal(image, other_image) and torch.equal(target, other_target)

"""
Rasterisation of the annotations and the grouped folds.
"""

import json

import numpy as np
import pytest
from PIL import Image
from pycocotools import mask as coco_mask
from synthetic import filament_polygon

from src.data.dataset import FilamentCropDataset, kfold_split, polygon_to_mask, train_val_split


def test_coco_crop_equals_full_frame_fill(rng):
    """
    Filling a polygon inside a crop gives the same pixels as cropping the full-frame fill.
    """
    size, crop = 256, 64
    for _ in range(20):
        polygon = filament_polygon(rng, size)
        full = coco_mask.decode(coco_mask.frPyObjects([polygon], size, size))[:, :, 0].astype(bool)
        x, y = (int(v) for v in rng.integers(0, size - crop, 2))
        window = polygon_to_mask(polygon, crop, crop, origin=(x, y), raster="coco")
        assert (window != full[y : y + crop, x : x + crop]).sum() <= 2


def test_pil_fill_is_fatter_than_coco(rng):
    """
    The reason ``--raster coco`` exists: PIL fills more pixels than the scored masks.
    """
    polygon = filament_polygon(rng)
    pil = polygon_to_mask(polygon, 128, 128, raster="pil").sum()
    coco = polygon_to_mask(polygon, 128, 128, raster="coco").sum()
    assert pil > coco


def records(n_files=40, annotators=(1, 2, 3)):
    """
    Fake image records: several annotator records per observation file.
    """
    rng = np.random.default_rng(0)
    out = []
    for index in range(n_files):
        for annotator in range(int(rng.choice(annotators))):
            out.append({"id": f"{annotator}-{index}", "file_name": f"{index:04d}.jpeg"})
    return out


def test_folds_keep_observations_together():
    """
    Every record of one observation lands in the same fold, and the folds cover everything once.
    """
    images = records()
    by_id = {image["id"]: image["file_name"] for image in images}
    seen = []
    for fold in range(5):
        train, val = kfold_split(images, fold, 5)
        assert not {by_id[i] for i in train} & {by_id[i] for i in val}
        seen += val
    assert sorted(seen) == sorted(image["id"] for image in images)


def test_folds_are_deterministic_and_seed_independent_of_training():
    """
    The same call gives the same folds, so models trained with different seeds share them.
    """
    images = records()
    assert kfold_split(images, 2, 5) == kfold_split(images, 2, 5)


def test_random_split_is_grouped():
    """
    The single validation split also keeps an observation's records together.
    """
    images = records()
    by_id = {image["id"]: image["file_name"] for image in images}
    train, val = train_val_split(images, 0.2, seed=1)
    assert not {by_id[i] for i in train} & {by_id[i] for i in val}


def test_input_scale_enlarges_image_and_target(tmp_path):
    """
    At input scale 2 the observation is decoded at twice its size, so a filament covers
    four times as many target pixels and the image keeps its values.
    """
    Image.fromarray(np.full((64, 64), 200, dtype=np.uint8)).save(tmp_path / "a.png")
    square = [8.0, 8.0, 56.0, 8.0, 56.0, 56.0, 8.0, 56.0]
    annotations = {
        "images": [{"id": 1, "file_name": "a.png", "width": 64, "height": 64}],
        "annotations": [{"image_id": 1, "segmentation": [square], "category_id": 1}],
    }
    (tmp_path / "annotations.json").write_text(json.dumps(annotations))
    common = {"images_dir": tmp_path, "annotations_path": tmp_path / "annotations.json", "raster": "coco"}

    _, native = FilamentCropDataset(crop=64, **common)[0]
    image, zoomed = FilamentCropDataset(crop=128, input_scale=2.0, **common)[0]

    assert image.shape == (1, 128, 128) and abs(float(image.mean()) - 200 / 255) < 1e-3
    assert native.sum() == 48 * 48
    assert zoomed.sum() == 96 * 96


def test_barb_targets_raise_the_target_outside_the_annotation(tmp_path):
    """
    Barb pixels grown from the image join the target at the barb level; annotated pixels keep theirs.
    """
    Image.fromarray(np.full((64, 64), 200, dtype=np.uint8)).save(tmp_path / "a.png")
    square = [8.0, 8.0, 56.0, 8.0, 56.0, 56.0, 8.0, 56.0]
    annotations = {
        "images": [{"id": 1, "file_name": "a.png", "width": 64, "height": 64}],
        "annotations": [{"image_id": 1, "segmentation": [square], "category_id": 1}],
    }
    (tmp_path / "annotations.json").write_text(json.dumps(annotations))
    barbs = np.zeros((64, 64), dtype=np.uint8)
    barbs[2:8, 30] = 255
    (tmp_path / "barbs").mkdir()
    Image.fromarray(barbs).save(tmp_path / "barbs" / "a.png")
    common = {"images_dir": tmp_path, "annotations_path": tmp_path / "annotations.json", "raster": "coco"}

    _, target = FilamentCropDataset(crop=64, barb_targets=tmp_path / "barbs", barb_level=0.5, **common)[0]

    assert target.sum() == 48 * 48 + 6 * 0.5
    assert float(target[4, 30]) == 0.5 and float(target[20, 20]) == 1.0
    with pytest.raises(ValueError):
        FilamentCropDataset(crop=128, input_scale=2.0, barb_targets=tmp_path / "barbs", **common)


def test_barb_head_learns_only_from_the_records_that_draw_barbs(tmp_path):
    """
    The barb channel is the thin part of the barb-drawing record's mask, averaged over those records
    alone; a sample without such a record is marked invalid instead of taught that there are no barbs.
    """
    Image.fromarray(np.full((64, 64), 200, dtype=np.uint8)).save(tmp_path / "a.png")
    Image.fromarray(np.full((64, 64), 200, dtype=np.uint8)).save(tmp_path / "b.png")
    body = [16.0, 16.0, 48.0, 16.0, 48.0, 48.0, 16.0, 48.0]
    barb = [30.0, 4.0, 32.0, 4.0, 32.0, 16.0, 30.0, 16.0]
    annotations = {
        "images": [
            {"id": 1, "file_name": "a.png", "width": 64, "height": 64},
            {"id": 2, "file_name": "a.png", "width": 64, "height": 64},
            {"id": 3, "file_name": "b.png", "width": 64, "height": 64},
        ],
        "annotations": [
            {"image_id": 1, "segmentation": [body], "category_id": 1},
            {"image_id": 1, "segmentation": [barb], "category_id": 1},
            {"image_id": 2, "segmentation": [body], "category_id": 1},
            {"image_id": 3, "segmentation": [body], "category_id": 1},
        ],
    }
    (tmp_path / "annotations.json").write_text(json.dumps(annotations))
    common = {"images_dir": tmp_path, "annotations_path": tmp_path / "annotations.json", "raster": "coco", "targets": "mean"}
    dataset = FilamentCropDataset(crop=64, barb_records=[1], **common)

    _, drawn = dataset[0]
    _, undrawn = dataset[1]

    assert drawn.shape == (3, 64, 64)
    assert float(drawn[1, 10, 31]) == 1.0 and float(drawn[1, 32, 32]) == 0.0
    assert float(drawn[2].min()) == 1.0
    assert float(undrawn[1].max()) == 0.0 and float(undrawn[2].max()) == 0.0

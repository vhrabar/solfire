"""
Release of the final weights and staging of the Kaggle Model variations.
"""

import json

import torch

from scripts.release_weights import half, release_name
from scripts.upload_model import overview, stage, usage


def test_half_keeps_vectors_in_float32():
    """
    Weights with two or more dimensions become float16; biases, norms and integer buffers keep their type.
    """
    state = {
        "conv.weight": torch.ones(4, 3, 3, 3),
        "conv.bias": torch.ones(4),
        "norm.num_batches_tracked": torch.tensor(5),
    }
    halved = half(state)
    assert halved["conv.weight"].dtype == torch.float16
    assert halved["conv.bias"].dtype == torch.float32
    assert halved["norm.num_batches_tracked"].dtype == torch.int64


def test_release_names_stay_unique():
    """
    Members are named after their run, refiners after their set and fold, so presets can share one directory.
    """
    assert release_name("runs/full/c06_full-lovasz-hrnet32_seed1/best.pt") == "c06_full-lovasz-hrnet32_seed1-best.pt"
    assert release_name("runs/refiner/zoom/fold0.pt") == "refiner-zoom-fold0.pt"
    assert release_name("runs/refiner/r34/fold0.pt") != release_name("runs/refiner/zoom/fold0.pt")


def test_stage_copies_only_the_preset(tmp_path):
    """
    A variation holds the files its calibration loads, the calibration and a manifest of just those files.
    """
    weights = tmp_path / "final"
    weights.mkdir()
    for name in ("a-best.pt", "b-best.pt", "refiner-zoom-fold0.pt", "tiny.calibration.json"):
        (weights / name).write_text(name)
    tiny = {
        "name": "tiny",
        "summary": "one model",
        "calibration": "tiny.calibration.json",
        "files": ["b-best.pt"],
        "bytes": 2_000_000,
        "oof_pq_without_barbs": 0.4531,
        "oof_pq": None,
        "seconds_per_image": 1.5,
        "public_lb": None,
    }
    manifest = {
        "presets": {"tiny": tiny},
        "files": [{"file": name, "presets": []} for name in ("a-best.pt", "b-best.pt", "refiner-zoom-fold0.pt")],
    }
    (weights / "manifest.json").write_text(json.dumps(manifest))

    staged = tmp_path / "stage" / "tiny"
    stage(weights, tiny, staged)
    assert sorted(path.name for path in staged.iterdir()) == ["b-best.pt", "manifest.json", "tiny.calibration.json"]
    subset = json.loads((staged / "manifest.json").read_text())
    assert [entry["file"] for entry in subset["files"]] == ["b-best.pt"]
    assert list(subset["presets"]) == ["tiny"]

    text = overview(tiny)
    assert "0.4531 without barbs" in text and "not submitted" in text and "`b-best.pt`" in text
    assert "solfire-tiny/<version>" in usage(tiny, "owner", "model")

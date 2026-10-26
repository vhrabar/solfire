"""
Staging of the Kaggle Model files.
"""

import json

import pytest
import torch

from scripts.upload_model import stage


def test_stage_names_refiners_by_fold(tmp_path):
    """
    Ensemble members keep ``<run>-<stem>.pt``, refiners become ``refiner-fold<k>.pt``, and the
    manifest lists both; a training checkpoint passed as a refiner is refused.
    """
    member = tmp_path / "runs" / "c02_full" / "best.pt"
    member.parent.mkdir(parents=True)
    torch.save({"model_state": {}, "arch": "unet", "calibration": {"pq": 0.45}}, member)
    refiners = []
    for fold in (1, 0):
        path = tmp_path / "zoom" / f"fold{fold}.pt"
        path.parent.mkdir(exist_ok=True)
        torch.save({"state_dict": {}, "fold": fold, "epochs": 12, "size": 256}, path)
        refiners.append(path)
    extra = tmp_path / "b6.calibration.json"
    extra.write_text("{}")

    staged = tmp_path / "stage"
    assert stage([member], staged, [extra], refiners) == {"unet"}
    assert sorted(p.name for p in staged.iterdir()) == [
        "b6.calibration.json",
        "c02_full-best.pt",
        "manifest.json",
        "refiner-fold0.pt",
        "refiner-fold1.pt",
    ]
    manifest = json.loads((staged / "manifest.json").read_text())
    assert [entry.get("refiner", False) for entry in manifest] == [False, True, True]
    assert manifest[1]["fold"] == 1 and manifest[1]["epochs"] == 12

    with pytest.raises(SystemExit):
        stage([member], staged, refiners=[member])

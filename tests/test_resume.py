"""
Interrupted runs: checkpoints are written in one step, a cut-short ``last.pt`` is not fatal,
and the sweep runs a failed command again.
"""

import sys

import torch

from scripts.sweep import attempt
from src.training.engine import _load_resume_state, atomic_save


def test_atomic_save_leaves_no_partial_file(tmp_path):
    """
    The checkpoint appears under its own name and the staging file is gone.
    """
    path = tmp_path / "last.pt"
    atomic_save({"epoch": 3}, path)
    assert torch.load(path)["epoch"] == 3
    assert [p.name for p in tmp_path.iterdir()] == ["last.pt"]


def test_unreadable_last_checkpoint_starts_over(tmp_path):
    """
    A truncated ``last.pt`` gives no resume state rather than an exception.
    """
    path = tmp_path / "last.pt"
    atomic_save({"model_state": torch.nn.Linear(2, 2).state_dict(), "epoch": 1}, path)
    path.write_bytes(path.read_bytes()[:40])
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    assert _load_resume_state(path, model, optimizer, None, None, None, 4, torch.device("cpu")) is None


def test_attempt_retries_until_success(tmp_path, monkeypatch):
    """
    A command that fails once succeeds on the second attempt, and the failed log is kept.
    """
    marker = tmp_path / "ran"
    script = f"import pathlib, sys; p = pathlib.Path({str(marker)!r}); first = not p.exists(); p.touch(); sys.exit(int(first))"
    monkeypatch.setattr("scripts.sweep.REPO", tmp_path)
    log = tmp_path / "run.train.log"
    assert attempt([sys.executable, "-c", script], log, retries=1)
    assert log.is_file() and (tmp_path / "run.train.log.1").is_file()
    assert not attempt([sys.executable, "-c", "raise SystemExit(1)"], tmp_path / "bad.log", retries=0)

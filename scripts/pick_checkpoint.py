"""
Pick the better checkpoint of one training run by its validation Panoptic Quality.
"""

import sys
from pathlib import Path

import torch


def best_checkpoint(run_dir: Path) -> tuple[Path, float] | None:
    """
    The higher-scoring of ``swa.pt`` and ``best.pt`` under ``run_dir``.

    :param run_dir: A single training run's output directory
    :return: ``(path, best_pq)`` for the better checkpoint, or ``None`` when neither
        file exists or carries a ``metrics.best_pq``
    """
    candidates = []
    for name in ("swa.pt", "best.pt"):
        path = run_dir / name
        if not path.is_file():
            continue
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        pq = checkpoint.get("metrics", {}).get("best_pq")
        if pq is not None:
            candidates.append((pq, path))

    if not candidates:
        return None
    best_pq, best_path = max(candidates, key=lambda c: c[0])
    return best_path, best_pq


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} <run_dir>")

    result = best_checkpoint(Path(sys.argv[1]))
    if result is None:
        sys.exit(f"no scored checkpoint under {sys.argv[1]}")
    best_path, best_pq = result
    print(f"{best_path} {best_pq:.6f}")


if __name__ == "__main__":
    main()

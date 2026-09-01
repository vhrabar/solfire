"""
Pick the better checkpoint of one training run by its validation Panoptic Quality.
"""

import sys
from pathlib import Path

import torch


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} <run_dir>")
    run_dir = Path(sys.argv[1])

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
        sys.exit(f"no scored checkpoint under {run_dir}")

    best_pq, best_path = max(candidates, key=lambda c: c[0])
    print(f"{best_path} {best_pq:.6f}")


if __name__ == "__main__":
    main()

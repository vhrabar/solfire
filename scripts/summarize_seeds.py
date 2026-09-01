"""
Report the spread of a multi-seed run written by ``scripts/multi_seed.sh``.
"""

import argparse
import csv
import statistics
import sys
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    """
    Read the results CSV written by ``multi_seed.sh``.
    :param path: Path to ``results.csv``
    :return: One dict per seed, numeric fields converted from strings
    """
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("train_pq", "calibrated_pq", "threshold"):
            row[key] = float(row[key])
        for key in ("offset", "min_area"):
            row[key] = int(row[key])
    return rows


def summarize(values: list[float]) -> dict:
    """
    Mean, sample standard deviation, min and max of a column.
    :param values: Per-seed values
    :return: Dict of summary statistics; ``std`` is ``0.0`` for a single seed
    """
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_csv", type=Path, help="results.csv written by multi_seed.sh")
    args = parser.parse_args()

    if not args.results_csv.is_file():
        sys.exit(f"{args.results_csv} not found")
    rows = load_rows(args.results_csv)
    if not rows:
        sys.exit(f"{args.results_csv} has no rows yet")

    print(f"{'seed':>4} {'checkpoint':>10} {'train_pq':>9} {'calib_pq':>9} {'threshold':>9} {'offset':>7} {'min_area':>9}")
    for row in rows:
        checkpoint = Path(row["checkpoint"]).stem
        print(
            f"{row['seed']:>4} {checkpoint:>10} {row['train_pq']:>9.4f} {row['calibrated_pq']:>9.4f} "
            f"{row['threshold']:>9.3f} {row['offset']:>+7d} {row['min_area']:>9d}"
        )

    calibrated = [row["calibrated_pq"] for row in rows]
    trained = [row["train_pq"] for row in rows]
    thresholds = [row["threshold"] for row in rows]
    stats = summarize(calibrated)

    print()
    print(f"{len(rows)} seed(s)")
    print(f"calibrated pq  mean {stats['mean']:.4f}  std {stats['std']:.4f}  min {stats['min']:.4f}  max {stats['max']:.4f}")
    print(f"train-time pq  mean {summarize(trained)['mean']:.4f}  std {summarize(trained)['std']:.4f}")
    print(f"threshold      mean {summarize(thresholds)['mean']:.3f}  std {summarize(thresholds)['std']:.3f}")

    if len(rows) > 1 and summarize(thresholds)["std"] > 0.15:
        print(
            "warning: the calibrated threshold moved a lot between seeds; the grid search is "
            "probably fitting the validation split rather than finding a stable decision point"
        )
    if len(rows) == 1:
        print("note: one seed only tells you the score, not the noise around it; run with more seeds to see the spread")


if __name__ == "__main__":
    main()

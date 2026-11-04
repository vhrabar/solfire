"""
Build the preprocessed copies of the train and test observations.
"""

import argparse
import time

import solfire.experiments  # noqa: F401  registers clahe
from solfire.data import TEST_IMAGES, TRAIN_IMAGES, build_cache, preprocessing_methods


def main() -> int:
    """
    Preprocess every observation not cached yet.
    :return: Process exit code
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=[m for m in preprocessing_methods() if m != "none"], default="clahe")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    for images_dir in (TRAIN_IMAGES, TEST_IMAGES):
        started = time.perf_counter()
        output = build_cache(images_dir, args.method, workers=args.workers)
        count = sum(1 for _ in output.glob("*.png"))
        print(f"{images_dir} -> {output}: {count} observations in {time.perf_counter() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

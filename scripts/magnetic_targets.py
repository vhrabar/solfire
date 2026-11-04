"""
Precompute the auxiliary magnetic training targets from the matched GONG magnetograms.
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solfire.data.magnetogram import TARGET_GRID, magnetic_targets, magnetogram_path, read_magnetogram

BZI = Path(__file__).resolve().parent.parent / "runs" / "external" / "gong_bzi"


def build(name, output, settings) -> str:
    """
    Write the targets of one observation.
    :param name: H-alpha file name
    :param output: Directory to write to
    :param settings: Keyword arguments of :func:`magnetic_targets`
    :return: ``name``
    """
    field, header = read_magnetogram(magnetogram_path(BZI / "fits", name))
    targets = magnetic_targets(field, header, **settings)
    np.save(Path(output) / f"{Path(name).stem}.npy", np.round(targets * 255).astype(np.uint8))
    return name


def main() -> int:
    """
    Build the targets of every usable matched magnetogram.
    :return: Process exit code
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default=str(BZI / "targets"))
    parser.add_argument("--max-gap", type=float, default=60.0, help="drop pairs further apart in time, minutes")
    parser.add_argument("--min-correlation", type=float, default=0.15, help="drop pairs that fail the registration check")
    parser.add_argument("--grid", type=int, default=TARGET_GRID)
    parser.add_argument("--sigma-mm", type=float, default=12.0, help="smoothing before tracing inversion lines")
    parser.add_argument("--gradient", type=float, default=1.0, help="least gradient across a line, gauss per grid pixel")
    parser.add_argument("--length-mm", type=float, default=10.0, help="decay length of the PIL proximity")
    parser.add_argument("--strong-gauss", type=float, default=100.0, help="field scale of the strong-flux target")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    manifest = json.loads((BZI / "manifest.json").read_text(encoding="utf-8"))
    names = [
        entry["image"]
        for entry in manifest["images"]
        if entry["status"] == "ok" and entry["gap_minutes"] <= args.max_gap and entry["plage_correlation"] >= args.min_correlation
    ]
    settings = {
        "size": args.grid,
        "sigma_mm": args.sigma_mm,
        "gradient": args.gradient,
        "length_mm": args.length_mm,
        "strong_gauss": args.strong_gauss,
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(args.workers) as pool:
        done = list(pool.map(partial(build, output=output, settings=settings), names))
    meta = settings | {"max_gap": args.max_gap, "min_correlation": args.min_correlation, "images": len(done)}
    (output / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"{len(done)} of {len(manifest['images'])} images -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

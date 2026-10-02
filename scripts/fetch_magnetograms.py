"""
Download the GONG site magnetogram ("bzi") closest in time to each MAGFiLO training image.
"""

import argparse
import datetime as dt
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fetch_gong

from src.data import TRAIN_IMAGES
from src.data.magnetogram import magnetogram_path, plage_correlation, read_magnetogram

ARCHIVE = "https://nispdata.nso.edu/ftp/vmb/archive/bzi"
OUTPUT = fetch_gong.REPO / "runs" / "external" / "gong_bzi"
SITES = {"B": "BB", "C": "CT", "L": "LE", "M": "ML", "T": "TD", "U": "UD"}
FRAME = re.compile(r'href="(\w\wbzi(\d{6}t\d{4})\.fits\.gz)"')


def site_frames(site, day) -> dict[dt.datetime, str]:
    """
    The magnetograms one site's day directory holds.
    :param site: Two-letter site code
    :param day: Directory date; a site's observing day can straddle UT midnight
    :return: Observation minute -> URL
    """
    url = f"{ARCHIVE}/{day:%Y%m}/{site}{day:%y%m%d}.mag/"
    try:
        listing = fetch_gong.fetch(url).decode("utf-8", "replace")
    except OSError:
        return {}
    return {dt.datetime.strptime(stamp, "%y%m%dt%H%M"): url + name for name, stamp in set(FRAME.findall(listing))}


def match(image_name, directory, tolerance) -> dict:
    """
    Find, download and check the magnetogram for one H-alpha image.
    :param image_name: MAGFiLO file name such as ``20110109104734Ch.jpeg``
    :param directory: Where the magnetogram goes
    :param tolerance: Largest accepted time gap in minutes
    :return: Manifest entry
    """
    moment = dt.datetime.strptime(image_name[:14], "%Y%m%d%H%M%S")
    site = SITES[image_name[14]]
    target = magnetogram_path(directory, image_name)
    entry = {"image": image_name}
    if not target.is_file():
        frames: dict[dt.datetime, str] = {}
        nearest, gap = None, float("inf")
        # the UT day first; a neighbouring day only when it holds nothing close enough
        for offset in (0, -1, 1):
            frames.update(site_frames(site, (moment + dt.timedelta(days=offset)).date()))
            if frames:
                nearest = min(frames, key=lambda stamp: abs((stamp - moment).total_seconds()))
                gap = abs((nearest - moment).total_seconds()) / 60
            if gap <= tolerance:
                break
        if nearest is None:
            return entry | {"status": "no frames"}
        if gap > tolerance:
            return entry | {"status": "too far", "gap_minutes": round(gap, 2)}
        partial = target.with_suffix(".part")
        partial.write_bytes(fetch_gong.fetch(frames[nearest]))
        partial.replace(target)
        entry["source"] = frames[nearest].rsplit("/", 1)[1]
    field, header = read_magnetogram(target)
    observed = dt.datetime.strptime(f"{header['DATE-OBS'].strip()} {header['TIME-OBS'].strip()[:8]}", "%Y/%m/%d %H:%M:%S")
    entry["gap_minutes"] = round(abs((observed - moment).total_seconds()) / 60, 2)
    entry["plage_correlation"] = round(plage_correlation(field, header, TRAIN_IMAGES / image_name), 4)
    return entry | {"status": "ok"}


def _safe_match(image_name, directory, tolerance) -> dict:
    """
    :func:`match`, reporting a failed download or unreadable file instead of raising.
    :param image_name: MAGFiLO file name
    :param directory: Where the magnetogram goes
    :param tolerance: Largest accepted time gap in minutes
    :return: Manifest entry
    """
    try:
        return match(image_name, directory, tolerance)
    except (OSError, KeyError, ValueError) as error:
        return {"image": image_name, "status": f"failed ({type(error).__name__}: {error})"}


def main() -> int:
    """
    Match every training image and write the manifest.
    :return: Process exit code
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tolerance", type=float, default=60.0, help="largest accepted time gap, minutes")
    parser.add_argument("--workers", type=int, default=2, help="parallel downloads; keep it low, the archive rate-limits")
    parser.add_argument(
        "--delay", type=float, default=fetch_gong.REQUEST_DELAY, help="seconds each worker waits before a request"
    )
    parser.add_argument("--interface", default=None, help="network interface to send requests through")
    parser.add_argument("--limit", type=int, default=None, help="only the first N images, for a dry run")
    args = parser.parse_args()
    fetch_gong.REQUEST_DELAY = args.delay
    fetch_gong.use_interface(args.interface)

    frames = OUTPUT / "fits"
    frames.mkdir(parents=True, exist_ok=True)
    names = sorted(path.name for path in TRAIN_IMAGES.glob("*.jpeg"))[: args.limit]
    entries = []
    with ThreadPoolExecutor(args.workers) as pool:
        for index, entry in enumerate(pool.map(lambda name: _safe_match(name, frames, args.tolerance), names), 1):
            entries.append(entry)
            if index % 50 == 0 or index == len(names):
                ok = sum(item["status"] == "ok" for item in entries)
                print(f"  {index}/{len(names)} images, {ok} matched", flush=True)

    statuses: dict[str, int] = {}
    for entry in entries:
        key = entry["status"].split(" (")[0]
        statuses[key] = statuses.get(key, 0) + 1
    manifest = {"archive": ARCHIVE, "tolerance_minutes": args.tolerance, "statuses": statuses, "images": entries}
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"{statuses} -> {OUTPUT / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Download unlabeled GONG H-alpha bursts from outside the MAGFiLO period and convert them like MAGFiLO.

"""

import argparse
import datetime as dt
import http.client
import json
import random
import re
import socket
import subprocess
import tempfile
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
from astropy.io import fits
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
ARCHIVE = "https://gong2.nso.edu/HA/haf"
CONVERTOR = REPO / "external" / "mleco-fits2jpegconvertor" / "src"
OUTPUT = REPO / "runs" / "external" / "gong"
FRAME = re.compile(r'href="(\d{14})([A-Z])h\.fits\.fz"')
OFFSETS = (-4, -2, 0, 2, 4)
JPEG_QUALITY = 92

INNER_RADIUS = 800
BRIGHTNESS = (0.485, 0.555)
MAX_DARK_SHARE = 0.015
DARK_LEVEL = 0.3


REQUEST_DELAY = 1.0


def use_interface(device) -> None:
    """
    Send every request through one network interface, e.g. a phone hotspot.

    Binding the socket to the device, as ``curl --interface`` does, and not only to its
    address: with a default route on another interface the kernel would otherwise still
    send the packets there.
    :param device: Interface name such as ``wlp9s0``, or ``None`` for the system's choice
    :return: None
    """
    if not device:
        return

    class DeviceHTTPSConnection(http.client.HTTPSConnection):
        def connect(self):
            address = socket.getaddrinfo(self.host, self.port, socket.AF_INET, socket.SOCK_STREAM)[0][4]
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BINDTODEVICE, device.encode())
            sock.settimeout(self.timeout)
            sock.connect(address)
            self.sock = self._context.wrap_socket(sock, server_hostname=self.host)

    class DeviceHTTPSHandler(urllib.request.HTTPSHandler):
        def https_open(self, req):
            return self.do_open(DeviceHTTPSConnection, req)

    urllib.request.install_opener(urllib.request.build_opener(DeviceHTTPSHandler()))


def fetch(url, attempts=4) -> bytes:
    """
    Download one URL politely, retrying on transient failures.
    :param url: Address to fetch
    :param attempts: Tries before giving up
    :return: The response body
    """
    for attempt in range(attempts):
        time.sleep(REQUEST_DELAY)
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                return response.read()
        except OSError:
            if attempt == attempts - 1:
                raise
            time.sleep(2 * (attempt + 1))
    raise AssertionError("unreachable")


def day_frames(day) -> dict[str, list[dt.datetime]]:
    """
    The frames one day of the archive holds, per site.
    :param day: Date to list
    :return: Site letter -> sorted observation times
    """
    url = f"{ARCHIVE}/{day:%Y%m}/{day:%Y%m%d}/"
    try:
        listing = fetch(url).decode("utf-8", "replace")
    except OSError:
        return {}
    sites: dict[str, list[dt.datetime]] = {}
    for stamp, site in set(FRAME.findall(listing)):
        sites.setdefault(site, []).append(dt.datetime.strptime(stamp, "%Y%m%d%H%M%S"))
    return {site: sorted(times) for site, times in sites.items()}


def pick_burst(times, rng) -> list[dt.datetime] | None:
    """
    A centre frame whose site also observed about each offset away from it.
    :param times: Sorted observation times of one site on one day
    :param rng: Random source
    :return: Frame times in the order of :data:`OFFSETS`, or ``None``
    """
    for _ in range(20):
        centre = rng.choice(times)
        burst = []
        for minutes in OFFSETS:
            target = centre + dt.timedelta(minutes=minutes)
            nearest = min(times, key=lambda t, target=target: abs((t - target).total_seconds()))
            if abs((nearest - target).total_seconds()) > 40:
                break
            burst.append(nearest)
        if len(burst) == len(OFFSETS) and len(set(burst)) == len(OFFSETS):
            return burst
    return None


def frame_name(moment, site) -> str:
    """
    MAGFiLO-style name of one frame.
    :param moment: Observation time
    :param site: Site letter
    :return: ``YYYYMMDDHHMMSS<site>h``
    """
    return f"{moment:%Y%m%d%H%M%S}{site}h"


def convert(name, directory) -> tuple[str, bool, str]:
    """
    Download one frame and convert it the way the MAGFiLO JPEGs were made.
    :param name: Frame name from :func:`frame_name`
    :param directory: Where the JPEG goes
    :return: Tuple of (name, whether it passed the quality check, reason when it did not)
    """
    target = Path(directory) / f"{name}.jpeg"
    if not target.is_file():
        url = f"{ARCHIVE}/{name[:6]}/{name[:8]}/{name}.fits.fz"
        with tempfile.TemporaryDirectory() as scratch:
            work = Path(scratch)
            (work / "frame.fits.fz").write_bytes(fetch(url))
            with fits.open(work / "frame.fits.fz") as hdus:
                fits.PrimaryHDU(data=hdus[1].data, header=hdus[1].header).writeto(work / "frame.fits")
            subprocess.run(
                [str(CONVERTOR / "wavelet"), str(work / "frame.fits"), str(work / "wavelet.fits")],
                check=True,
                capture_output=True,
                cwd=CONVERTOR,
            )
            subprocess.run(
                [
                    str(CONVERTOR / "fits2ppmColorscale" / "fits2ppmColorscale"),
                    str(work / "wavelet.fits"),
                    "grey",
                    str(work / "wavelet.ppm"),
                    "640",
                    "5335",
                ],
                check=True,
                capture_output=True,
                cwd=CONVERTOR,
            )
            partial = target.with_suffix(".part.jpeg")
            Image.open(work / "wavelet.ppm").convert("L").save(partial, quality=JPEG_QUALITY)
            partial.replace(target)
    with Image.open(target) as handle:
        image = np.asarray(handle.convert("L"), dtype=np.float32) / 255
    size = image.shape[0]
    rows, columns = np.ogrid[:size, :size]
    inner = image[(rows - size / 2) ** 2 + (columns - size / 2) ** 2 < INNER_RADIUS**2]
    if not BRIGHTNESS[0] <= float(np.median(inner)) <= BRIGHTNESS[1]:
        return name, False, "disk brightness"
    if float((inner < DARK_LEVEL).mean()) > MAX_DARK_SHARE:
        return name, False, "dark patches"
    return name, True, ""


def main() -> int:
    """
    Sample bursts, download and convert them, and write a manifest of the usable ones.
    :return: Process exit code
    """
    global REQUEST_DELAY
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bursts", type=int, default=1000, help="bursts to sample")
    parser.add_argument("--start", default="2023-01-01", help="first day to sample, after the MAGFiLO period")
    parser.add_argument("--end", default="2026-09-27")
    parser.add_argument("--workers", type=int, default=2, help="parallel downloads; keep it low, the archive rate-limits")
    parser.add_argument("--delay", type=float, default=REQUEST_DELAY, help="seconds each worker waits before a request")
    parser.add_argument("--interface", default=None, help="network interface to send requests through, e.g. a hotspot's")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    REQUEST_DELAY = args.delay
    use_interface(args.interface)

    rng = random.Random(args.seed)
    start, end = dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end)
    days = [start + dt.timedelta(days=offset) for offset in range((end - start).days + 1)]
    frames = OUTPUT / "jpeg"
    frames.mkdir(parents=True, exist_ok=True)

    # one burst per sampled day, from a random site that observed that day
    chosen = sorted(rng.sample(days, min(len(days), int(args.bursts * 1.3))))
    with ThreadPoolExecutor(args.workers) as pool:
        listings = list(pool.map(day_frames, chosen))
    bursts = []
    for listing in listings:
        if not listing:
            continue
        site = rng.choice(sorted(listing))
        picked = pick_burst(listing[site], rng)
        if picked:
            bursts.append([frame_name(moment, site) for moment in picked])
        if len(bursts) == args.bursts:
            break
    print(f"{len(bursts)} bursts of {len(OFFSETS)} frames from {len(chosen)} sampled days")

    names = [name for burst in bursts for name in burst]
    passed: dict[str, bool] = {}
    reasons: dict[str, int] = {}
    with ProcessPoolExecutor(args.workers, initializer=_set_delay, initargs=(args.delay, args.interface)) as pool:
        for index, (name, ok, reason) in enumerate(pool.map(partial(_safe_convert, directory=frames), names), 1):
            passed[name] = ok
            if not ok:
                reasons[reason] = reasons.get(reason, 0) + 1
            if index % 250 == 0 or index == len(names):
                print(f"  {index}/{len(names)} frames, rejected {sum(reasons.values())} {reasons}", flush=True)

    usable = [burst for burst in bursts if all(passed.get(name) for name in burst)]
    manifest = {"offsets_minutes": list(OFFSETS), "bursts": usable, "rejected": reasons}
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"{len(usable)} usable bursts -> {OUTPUT / 'manifest.json'}")
    return 0


def _set_delay(delay, interface) -> None:
    """
    Carry the request settings into a worker process.
    :param delay: Seconds to wait before each request
    :param interface: Network interface to send requests through, or ``None``
    :return: None
    """
    global REQUEST_DELAY
    REQUEST_DELAY = delay
    use_interface(interface)


def _safe_convert(name, directory) -> tuple[str, bool, str]:
    """
    :func:`convert`, reporting a failed download or conversion as a rejection.
    :param name: Frame name
    :param directory: Where the JPEG goes
    :return: As :func:`convert`
    """
    try:
        return convert(name, directory)
    except (OSError, subprocess.CalledProcessError, IndexError, ValueError) as error:
        return name, False, f"failed ({type(error).__name__})"


if __name__ == "__main__":
    raise SystemExit(main())

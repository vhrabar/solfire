"""
Download the extra GONG data for the Kaggle dataset, slowly enough not to get blocked.

Four stages, each resumable

- ``channels``: continuum intensity (izi) and Doppler velocity (vzi) frames taken with each matched bzi magnetogram
- ``bursts``: the bzi magnetogram closest to the centre frame of every unlabelled H-alpha burst
- ``haf``: the raw H-alpha FITS behind every unlabelled JPEG
- ``synoptic``: the hourly zero-point-corrected synoptic magnetogram (mrzqs) closest to every observation
"""

import argparse
import datetime as dt
import json
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fetch_gong

from src.data import TRAIN_IMAGES
from src.data.magnetogram import plage_correlation, read_magnetogram

EXTERNAL = fetch_gong.REPO / "runs" / "external"
HALPHA_JPEG = EXTERNAL / "gong" / "jpeg"
HALPHA_FITS = EXTERNAL / "gong" / "fits"
BURSTS = EXTERNAL / "gong" / "manifest.json"
MAGFILO_BZI = EXTERNAL / "gong_bzi" / "fits"
BURST_OUTPUT = EXTERNAL / "gong_bursts"
SYNOPTIC_OUTPUT = EXTERNAL / "gong_synoptic"

VMB = "https://nispdata.nso.edu/ftp/vmb/archive"
HAF = "https://gong2.nso.edu/HA/haf"
ZQS = "https://gong2.nso.edu/oQR/zqs"
CHANNELS = {"izi": "int", "vzi": "vel"}
SITES = {"B": "BB", "C": "CT", "L": "LE", "M": "ML", "T": "TD", "U": "UD"}
BZI_FRAME = re.compile(r'href="(\w\wbzi(\d{6}t\d{4})\.fits\.gz)"')
ZQS_FRAME = re.compile(r'href="(mrzqs(\d{6}t\d{4})c\d+_\d+\.fits\.gz)"')

HEADERS = {"User-Agent": "gong-fetch/1.0 (research download; one connection, paced)"}
# a FITS file, gzipped FITS, or an Apache index page; anything else is an error page
MAGIC = (b"SIMPLE", b"\x1f\x8b")
MISSING = {400, 404, 410}
SAVE_EVERY = 25
LIMIT = None


def log(message) -> None:
    """
    Print one timestamped line.
    :param message: Text to print
    :return: None
    """
    print(f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} {message}", flush=True)


class Blocked(RuntimeError):
    """The host kept refusing for longer than the give-up time."""


class Host:
    """
    One archive server: one request at a time, a jittered pause between requests and an
    exponential cool-down while it refuses.
    """

    def __init__(self, name, delay, give_up_hours):
        """
        :param name: Label for the log
        :param delay: Smallest pause between two requests, seconds; each pause adds up to half again
        :param give_up_hours: How long to keep cooling down before :class:`Blocked` is raised
        """
        self.name, self.delay, self.give_up = name, delay, give_up_hours * 3600
        self.lock = threading.Lock()
        self.last = 0.0
        self.requests = 0
        self.listings: dict[str, dict[str, str]] = {}

    def fetch(self, url, listing=False) -> bytes | None:
        """
        Download one URL, waiting out refusals.
        :param url: Address to fetch
        :param listing: Whether the answer is a directory listing rather than a FITS file
        :return: The body, or ``None`` when the server says the file does not exist
        """
        refused_since, failures = None, 0
        while True:
            with self.lock:
                wait = self.last + self.delay * random.uniform(1.0, 1.5) - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                retry_after = None
                try:
                    request = urllib.request.Request(url, headers=HEADERS)
                    with urllib.request.urlopen(request, timeout=120) as response:
                        body = response.read()
                    if (b"Index of" in body[:2000]) if listing else body.startswith(MAGIC):
                        return body
                    reason = f"unexpected answer {body[:60]!r}"
                except urllib.error.HTTPError as error:
                    if error.code in MISSING:
                        return None
                    retry_after = error.headers.get("Retry-After")
                    reason = f"HTTP {error.code}"
                except OSError as error:
                    reason = f"{type(error).__name__}: {error}"
                finally:
                    self.last = time.monotonic()
                    self.requests += 1
            failures += 1
            refused_since = refused_since or time.monotonic()
            if time.monotonic() - refused_since > self.give_up:
                raise Blocked(f"{self.name} refused for {self.give_up / 3600:.1f} h, last: {reason}")
            pause = min(30 * 2**failures, 1800)
            if retry_after and retry_after.strip().isdigit():
                pause = max(pause, int(retry_after))
            log(f"[{self.name}] {reason} on {url}; cooling down {pause} s")
            time.sleep(pause)

    def listing(self, url, pattern) -> dict[str, str]:
        """
        File names in one directory that match a name pattern, cached for the run.
        :param url: Directory address ending in ``/``
        :param pattern: Regex with groups (file name, ``%y%m%dt%H%M`` stamp)
        :return: Stamp -> file URL
        """
        if url not in self.listings:
            body = self.fetch(url, listing=True)
            text = body.decode("utf-8", "replace") if body else ""
            self.listings[url] = {stamp: url + name for name, stamp in set(pattern.findall(text))}
        return self.listings[url]

    def download(self, url, target) -> bool:
        """
        Save one file, skipping it when it is already there.
        :param url: Address to fetch
        :param target: Where it goes
        :return: Whether the file is now on disk
        """
        if target.is_file():
            return True
        body = self.fetch(url)
        if body is None:
            return False
        partial = target.with_name(target.name + ".part")
        partial.write_bytes(body)
        partial.replace(target)
        return True


class Manifest:
    """Entries of one stage, keyed by name, loaded on resume and saved as the stage goes."""

    def __init__(self, path, **fields):
        """
        :param path: JSON file
        :param fields: Extra top-level fields to store
        """
        self.path, self.fields = Path(path), fields
        self.entries = {}
        if self.path.is_file():
            self.entries = {entry["name"]: entry for entry in json.loads(self.path.read_text())["entries"]}

    def done(self, name) -> bool:
        """
        :param name: Entry name
        :return: Whether a previous run finished it, successfully or with a final verdict
        """
        return self.entries.get(name, {}).get("status", "failed").split(" ")[0] in ("ok", "no", "too")

    def add(self, entry) -> None:
        """
        Record one entry and save every few.
        :param entry: Dictionary with at least ``name`` and ``status``
        :return: None
        """
        self.entries[entry["name"]] = entry
        if len(self.entries) % SAVE_EVERY == 0:
            self.save()

    def save(self) -> None:
        """
        Write the manifest atomically.
        :return: None
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        statuses: dict[str, int] = {}
        for entry in self.entries.values():
            key = entry["status"].split(" (")[0]
            statuses[key] = statuses.get(key, 0) + 1
        data = self.fields | {"statuses": statuses, "entries": sorted(self.entries.values(), key=lambda e: e["name"])}
        partial = self.path.with_suffix(".part")
        partial.write_text(json.dumps(data, indent=1), encoding="utf-8")
        partial.replace(self.path)


def observed(header) -> dt.datetime:
    """
    :param header: GONG site-image header
    :return: Observation time
    """
    stamp = f"{header['DATE-OBS'].strip()} {header['TIME-OBS'].strip()[:8]}"
    return dt.datetime.strptime(stamp, "%Y/%m/%d %H:%M:%S")


def burst_centres() -> list[str]:
    """
    :return: Name of the middle frame of every usable H-alpha burst, e.g. ``20240118183542Ch``
    """
    bursts = json.loads(BURSTS.read_text(encoding="utf-8"))["bursts"]
    return [burst[len(burst) // 2] for burst in bursts]


def halpha_site(name) -> str | None:
    """
    Two-letter site code of one unlabelled H-alpha frame: from its FITS header when the ``haf``
    stage fetched it, for the letters the MAGFiLO mapping lacks (A, Z).
    :param name: Frame name such as ``20240118183542Ch``
    :return: Site code such as ``CT``, or ``None`` when unknown
    """
    path = HALPHA_FITS / f"{name}.fits.fz"
    if name[14] not in SITES and path.is_file():
        try:
            return fits.getheader(path, 1)["SITE"].strip().upper()
        except OSError, KeyError, IndexError:
            pass
    return SITES.get(name[14])


def channels(host, bzi_directory, manifest_path) -> None:
    """
    Fetch the izi and vzi frames sharing each bzi magnetogram's site and minute.
    :param host: The NISP host
    :param bzi_directory: Folder of bzi files named after their H-alpha frame
    :param manifest_path: Where the stage's manifest goes
    :return: None
    """
    manifest = Manifest(manifest_path, archive=VMB, products=list(CHANNELS))
    files = sorted(bzi_directory.glob("*.fits.gz"))[:LIMIT]
    log(f"channels: {len(files)} magnetograms in {bzi_directory}")
    for index, path in enumerate(files, 1):
        name = path.name.removesuffix(".fits.gz")
        if manifest.done(name):
            continue
        try:
            header = fits.getheader(path)
            if "FITSNAME" in header:
                source = header["FITSNAME"].strip().removesuffix(".fits")  # e.g. ctbzi110109t1049
            else:
                # newer files drop FITSNAME; the archive name is the observation time cut to the minute
                source = f"{header['SITE'].strip().lower()}bzi{observed(header):%y%m%dt%H%M}"
            site, stamp = source[:2].upper(), source[5:]
            moment = dt.datetime.strptime(stamp, "%y%m%dt%H%M")
            entry = {"name": name, "bzi": source}
            for product, suffix in CHANNELS.items():
                target = path.parent.parent / product / path.name
                target.parent.mkdir(parents=True, exist_ok=True)
                # the day directory is the site's observing day, which can straddle UT midnight
                for offset in (0, -1, 1):
                    day = moment + dt.timedelta(days=offset)
                    url = f"{VMB}/{product}/{day:%Y%m}/{site}{day:%y%m%d}.{suffix}/{source[:2]}{product}{stamp}.fits.gz"
                    if host.download(url, target):
                        entry[product] = url.rsplit("/", 1)[1]
                        break
            missing = [product for product in CHANNELS if product not in entry]
            entry["status"] = "ok" if not missing else f"no {'+'.join(missing)}"
        except (OSError, KeyError, ValueError) as error:
            entry = {"name": name, "status": f"failed ({type(error).__name__}: {error})"}
        manifest.add(entry)
        if index % 50 == 0 or index == len(files):
            log(f"channels: {index}/{len(files)}, {host.requests} requests to {host.name}")
    manifest.save()


def bursts(host, tolerance) -> None:
    """
    Fetch the same-site bzi magnetogram closest to the centre frame of each H-alpha burst.
    :param host: The NISP host
    :param tolerance: Largest accepted time gap, minutes
    :return: None
    """
    manifest = Manifest(BURST_OUTPUT / "manifest.json", archive=f"{VMB}/bzi", tolerance_minutes=tolerance)
    directory = BURST_OUTPUT / "fits"
    directory.mkdir(parents=True, exist_ok=True)
    centres = burst_centres()[:LIMIT]
    log(f"bursts: {len(centres)} burst centres")
    for index, name in enumerate(centres, 1):
        if manifest.done(name):
            continue
        entry = {"name": name}
        try:
            site = halpha_site(name)
            if site is None:
                manifest.add(entry | {"status": "no site"})
                continue
            entry["site"] = site
            moment = dt.datetime.strptime(name[:14], "%Y%m%d%H%M%S")
            target = directory / f"{name}.fits.gz"
            frames: dict[dt.datetime, str] = {}
            nearest, gap = None, float("inf")
            for offset in (0, -1, 1):
                day = moment + dt.timedelta(days=offset)
                listing = host.listing(f"{VMB}/bzi/{day:%Y%m}/{site}{day:%y%m%d}.mag/", BZI_FRAME)
                frames |= {dt.datetime.strptime(stamp, "%y%m%dt%H%M"): url for stamp, url in listing.items()}
                if frames:
                    nearest = min(frames, key=lambda stamp: abs((stamp - moment).total_seconds()))
                    gap = abs((nearest - moment).total_seconds()) / 60
                if gap <= tolerance:
                    break
            if nearest is None:
                manifest.add(entry | {"status": "no frames"})
                continue
            if gap > tolerance:
                manifest.add(entry | {"status": "too far", "gap_minutes": round(gap, 2)})
                continue
            if not host.download(frames[nearest], target):
                manifest.add(entry | {"status": "failed (vanished)"})
                continue
            field, header = read_magnetogram(target)
            entry["source"] = frames[nearest].rsplit("/", 1)[1]
            entry["gap_minutes"] = round(abs((observed(header) - moment).total_seconds()) / 60, 2)
            entry["plage_correlation"] = round(plage_correlation(field, header, HALPHA_JPEG / f"{name}.jpeg"), 4)
            entry["status"] = "ok"
        except (OSError, KeyError, ValueError) as error:
            entry["status"] = f"failed ({type(error).__name__}: {error})"
        manifest.add(entry)
        if index % 50 == 0 or index == len(centres):
            log(f"bursts: {index}/{len(centres)}, {host.requests} requests to {host.name}")
    manifest.save()


def haf(host) -> None:
    """
    Fetch the raw H-alpha FITS of every unlabelled JPEG, burst centres first.
    :param host: The GONG2 host
    :return: None
    """
    HALPHA_FITS.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(HALPHA_FITS.parent / "fits_manifest.json", archive=HAF)
    centres = burst_centres()
    rest = sorted({path.stem for path in HALPHA_JPEG.glob("*.jpeg")} - set(centres))
    names = (centres + rest)[:LIMIT]
    log(f"haf: {len(names)} frames")
    for index, name in enumerate(names, 1):
        if manifest.done(name):
            continue
        url = f"{HAF}/{name[:6]}/{name[:8]}/{name}.fits.fz"
        try:
            ok = host.download(url, HALPHA_FITS / f"{name}.fits.fz")
            entry = {"name": name, "status": "ok" if ok else "no file"}
        except OSError as error:
            entry = {"name": name, "status": f"failed ({type(error).__name__}: {error})"}
        manifest.add(entry)
        if index % 50 == 0 or index == len(names):
            log(f"haf: {index}/{len(names)}, {host.requests} requests to {host.name}")
    manifest.save()


def synoptic(host, tolerance) -> None:
    """
    Fetch the hourly mrzqs synoptic map closest to every MAGFiLO training image and burst centre.
    :param host: The GONG2 host
    :param tolerance: Largest accepted time gap, minutes
    :return: None
    """
    manifest = Manifest(SYNOPTIC_OUTPUT / "manifest.json", archive=ZQS, tolerance_minutes=tolerance)
    directory = SYNOPTIC_OUTPUT / "zqs"
    directory.mkdir(parents=True, exist_ok=True)
    names = sorted(path.stem for path in TRAIN_IMAGES.glob("*.jpeg")) + burst_centres()
    names = names[:LIMIT]
    log(f"synoptic: {len(names)} observations")
    for index, name in enumerate(names, 1):
        if manifest.done(name):
            continue
        entry = {"name": name}
        try:
            moment = dt.datetime.strptime(name[:14], "%Y%m%d%H%M%S")
            maps: dict[dt.datetime, str] = {}
            nearest, gap = None, float("inf")
            for offset in (0, -1, 1):
                day = moment + dt.timedelta(days=offset)
                listing = host.listing(f"{ZQS}/{day:%Y%m}/mrzqs{day:%y%m%d}/", ZQS_FRAME)
                maps |= {dt.datetime.strptime(stamp, "%y%m%dt%H%M"): url for stamp, url in listing.items()}
                if maps:
                    nearest = min(maps, key=lambda stamp: abs((stamp - moment).total_seconds()))
                    gap = abs((nearest - moment).total_seconds()) / 60
                if gap <= tolerance:
                    break
            if nearest is None:
                entry["status"] = "no maps"
            elif gap > tolerance:
                entry |= {"status": "too far", "gap_minutes": round(gap, 2)}
            else:
                file = maps[nearest].rsplit("/", 1)[1]
                ok = host.download(maps[nearest], directory / file)
                entry |= {"map": file, "gap_minutes": round(gap, 2), "status": "ok" if ok else "failed (vanished)"}
        except (OSError, ValueError) as error:
            entry["status"] = f"failed ({type(error).__name__}: {error})"
        manifest.add(entry)
        if index % 50 == 0 or index == len(names):
            log(f"synoptic: {index}/{len(names)}, {host.requests} requests to {host.name}")
    manifest.save()


def run_queue(host, steps) -> None:
    """
    Run one host's stages in order, stopping that host only if it blocks for good.
    :param host: The host every step uses
    :param steps: (label, callable) pairs
    :return: None
    """
    for label, step in steps:
        log(f"[{host.name}] start {label}")
        try:
            step()
        except Blocked as error:
            log(f"[{host.name}] giving up: {error}; rerun later to resume")
            return
        log(f"[{host.name}] done {label}")


def main() -> int:
    """
    Run the selected stages, one thread per archive host.
    :return: Process exit code
    """
    stages = ("channels", "bursts", "haf", "synoptic")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stages", nargs="+", choices=stages, default=list(stages))
    parser.add_argument("--delay", type=float, default=3.0, help="smallest pause between requests to one host, seconds")
    parser.add_argument("--give-up-hours", type=float, default=8.0, help="cool-down time before a host is left alone")
    parser.add_argument("--tolerance", type=float, default=60.0, help="largest accepted bzi time gap, minutes")
    parser.add_argument("--synoptic-tolerance", type=float, default=360.0, help="largest accepted synoptic map gap, minutes")
    parser.add_argument("--interface", default=None, help="network interface to send requests through")
    parser.add_argument("--limit", type=int, default=None, help="only the first N items per stage, for a dry run")
    args = parser.parse_args()
    global LIMIT
    LIMIT = args.limit
    fetch_gong.use_interface(args.interface)

    nisp = Host("nispdata", args.delay, args.give_up_hours)
    gong2 = Host("gong2", args.delay, args.give_up_hours)
    # bursts read each centre's site from its haf header, which the gong2 queue fetches first
    queues = {
        nisp: [
            ("channels (MAGFiLO)", lambda: channels(nisp, MAGFILO_BZI, MAGFILO_BZI.parent / "channels.json")),
            ("bursts", lambda: bursts(nisp, args.tolerance)),
            ("channels (bursts)", lambda: channels(nisp, BURST_OUTPUT / "fits", BURST_OUTPUT / "channels.json")),
        ],
        gong2: [("haf", lambda: haf(gong2)), ("synoptic", lambda: synoptic(gong2, args.synoptic_tolerance))],
    }
    wanted = {"channels (MAGFiLO)": "channels", "channels (bursts)": "channels", "bursts": "bursts"}
    threads = []
    for host, steps in queues.items():
        steps = [(label, step) for label, step in steps if wanted.get(label, label) in args.stages]
        if steps:
            threads.append(threading.Thread(target=run_queue, args=(host, steps), name=host.name))
            threads[-1].start()
    for thread in threads:
        thread.join()
    log("all queues finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

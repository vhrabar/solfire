"""
Rebuild ``docs/submissions.csv``: every Kaggle submission with its validation, public and private
score and the measured prediction time.
"""

import argparse
import csv
import json
import re
import sys
import tempfile
import time
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.submit import COMPETITION, REPO, TEST_OBSERVATIONS, calibration_of, predict, write_manifest

LEDGER = REPO / "docs" / "submissions.csv"
MANIFEST = "submission.json"
COLUMNS = [
    "ref",
    "date",
    "description",
    "models",
    "val_pq",
    "public",
    "private",
    "seconds_per_image",
    "predict_seconds",
    "gpu",
    "status",
    "prediction_dir",
]
# every submission was predicted on this machine; manifests only exist for the ones submit.py timed
GPU = "NVIDIA GeForce RTX 5070 Ti"
# descriptions as they should read in the ledger, for submissions whose Kaggle description is wrong
RENAMED = {"test?best": "base_best"}
ENSEMBLE = re.compile(r"^(?P<parent>[^/ ]+)/(?P<stem>\S+) \((?P<models>\d+) models\)")
SINGLE = re.compile(r"^(?P<sweep>[^/ ]+)/(?P<run>\S+) (?P<checkpoint>\S+) \| val pq")
VAL_PQ = re.compile(r"\| val pq (?P<pq>[0-9.]+)")


def prediction_dir(description: str) -> Path | None:
    """
    Find where ``submit.py`` wrote the predictions a submission was made from.
    :param description: The Kaggle submission description ``submit.py`` set
    :return: The predictions directory, or ``None`` for submissions not made by ``submit.py``
    """
    if match := ENSEMBLE.match(description):
        return REPO / "runs" / match["parent"] / f"predictions_{match['stem']}"
    if match := SINGLE.match(description):
        return REPO / "runs" / match["sweep"] / match["run"] / f"predictions_{match['checkpoint']}"
    return None


def model_count(description: str) -> str:
    """
    :param description: The Kaggle submission description
    :return: Number of models as text, or ``""`` when the description does not say
    """
    if match := ENSEMBLE.match(description):
        return match["models"]
    return "1" if SINGLE.match(description) else ""


def submissions(api: KaggleApi) -> list:
    """
    :param api: Authenticated ``KaggleApi``
    :return: Every submission to the competition, newest first
    """
    found: list = []
    for page in range(1, 11):
        batch = api.competition_submissions(COMPETITION, page_number=page, page_size=50) or []
        found += [s for s in batch if s is not None]
        if len(batch) < 50:
            break
    return found


def row(submission) -> dict:
    """
    One ledger row for a submission.
    :param submission: Kaggle API submission object
    :return: Values keyed by :data:`COLUMNS`
    """
    description = submission.description or ""
    directory = prediction_dir(description)
    manifest_path = directory / MANIFEST if directory is not None else None
    manifest = json.loads(manifest_path.read_text()) if manifest_path and manifest_path.is_file() else {}
    val_pq = VAL_PQ.search(description)
    return {
        "ref": submission.ref,
        "date": str(submission.date)[:19],
        "description": RENAMED.get(description, description),
        "models": manifest.get("models", model_count(description)),
        "val_pq": val_pq["pq"] if val_pq else "",
        "public": submission.public_score or "",
        "private": submission.private_score or "",
        "seconds_per_image": manifest.get("seconds_per_image", ""),
        "predict_seconds": manifest.get("predict_seconds", ""),
        "gpu": manifest.get("gpu") or GPU,
        "status": str(submission.status).removeprefix("SubmissionStatus."),
        "prediction_dir": directory.relative_to(REPO) if directory is not None and directory.is_dir() else "",
    }


def sources(description: str) -> tuple[list[Path], dict, Path | None] | None:
    """
    Recover what a ``submit.py`` submission was predicted from.
    :param description: The Kaggle submission description
    :return: Tuple of (checkpoints, calibration, calibration file or ``None`` when it lives in the
        checkpoint), or ``None`` when the inputs no longer exist
    """
    if match := ENSEMBLE.match(description):
        path = REPO / "runs" / match["parent"] / f"{match['stem']}.calibration.json"
        if not path.is_file():
            return None
        calibration = json.loads(path.read_text())
        checkpoints = [REPO / checkpoint for checkpoint in calibration.get("checkpoints", [])]
        if not checkpoints or not all(checkpoint.is_file() for checkpoint in checkpoints):
            return None
        return checkpoints, calibration, path
    if match := SINGLE.match(description):
        checkpoint = REPO / "runs" / match["sweep"] / match["run"] / f"{match['checkpoint']}.pt"
        stored = calibration_of(checkpoint) if checkpoint.is_file() else None
        return ([checkpoint], stored, None) if stored is not None else None
    return None


def same_rows(first: Path, second: Path) -> bool:
    """
    :param first: A ``submission.csv``
    :param second: Another ``submission.csv``
    :return: Whether both hold the same rows, in any order
    """
    with open(first, encoding="utf-8") as a, open(second, encoding="utf-8") as b:
        return sorted(a.read().splitlines()) == sorted(b.read().splitlines())


def measured(manifest: Path) -> bool:
    """
    :param manifest: A submission's timing manifest
    :return: Whether it holds a measured prediction time, rather than none or an estimate backfilled
        from a log's progress bar, which leaves out model loading
    """
    if not manifest.is_file():
        return False
    return not json.loads(manifest.read_text()).get("note", "").startswith("backfilled")


def retime(api: KaggleApi) -> None:
    """
    Predict again every ``submit.py`` submission without a measured time, and record the time.

    Predictions go to a scratch directory, so the submitted ``submission.csv`` is never touched; the
    manifest notes whether the new output is identical to it.
    :param api: Authenticated ``KaggleApi``
    :return: None
    """
    todo = []
    for submission in reversed(submissions(api)):
        description = submission.description or ""
        directory = prediction_dir(description)
        if directory is None or not (directory / "submission.csv").is_file() or measured(directory / MANIFEST):
            continue
        found = sources(description)
        if found is None:
            print(f"retime: skipping '{description}', its checkpoints or calibration are gone")
            continue
        todo.append((description, directory, *found))
    print(f"retime: {len(todo)} submission(s) to predict again")
    for index, (description, directory, checkpoints, calibration, calibration_path) in enumerate(todo, 1):
        print(f"\n[{index}/{len(todo)}] {description}")
        with tempfile.TemporaryDirectory() as scratch:
            seconds = predict(checkpoints, Path(scratch), calibration, calibration_path)
            identical = same_rows(Path(scratch) / "submission.csv", directory / "submission.csv")
        write_manifest(directory, description, checkpoints, calibration_path, seconds)
        manifest = json.loads((directory / MANIFEST).read_text())
        manifest["note"] = "retimed after submission; output identical to the submitted CSV: " + str(identical).lower()
        (directory / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"  {seconds:.0f} s ({seconds / TEST_OBSERVATIONS:.2f} s per observation), identical to submitted: {identical}")


def refresh(api: KaggleApi, output: Path = LEDGER, wait_seconds: float = 0) -> None:
    """
    Write the ledger, oldest submission first.
    :param api: Authenticated ``KaggleApi``
    :param output: Where to write it
    :param wait_seconds: How long to wait for pending submissions to be scored first
    :return: None
    """
    found = submissions(api)
    deadline = time.monotonic() + wait_seconds
    while any("PENDING" in str(s.status) for s in found) and time.monotonic() < deadline:
        time.sleep(20)
        found = submissions(api)
    # submissions without a description were not made by submit.py and carry nothing to trace them by
    rows = [row(submission) for submission in reversed(found) if submission.description]
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    timed = sum(1 for r in rows if r["seconds_per_image"] != "")
    pending = sum(1 for r in rows if r["status"] == "PENDING")
    note = f", {pending} still pending; run scripts/ledger.py again later" if pending else ""
    print(f"ledger: {len(rows)} submissions ({timed} timed{note}) -> {output.relative_to(REPO)}")


def main() -> int:
    """
    Fetch the submissions and write the ledger.
    :return: Process exit code
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=LEDGER, help="where to write the ledger")
    parser.add_argument("--wait", type=float, default=0, help="seconds to wait for pending submissions to be scored")
    parser.add_argument(
        "--retime", action="store_true", help="predict again every submission without a measured time, then write the ledger"
    )
    args = parser.parse_args()
    api = KaggleApi()
    api.authenticate()
    if args.retime:
        retime(api)
    refresh(api, args.output.resolve(), args.wait)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

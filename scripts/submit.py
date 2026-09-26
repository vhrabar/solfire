"""
Predict the test observations with each selected model and submit the result to Kaggle.
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

REPO = Path(__file__).resolve().parent.parent
COMPETITION = "filament-segmentation-2026"
TEST_OBSERVATIONS = 180
PYTHON = ["uv", "run", "--group", "cuda-132", "python"]


def calibration_of(checkpoint: Path) -> dict | None:
    """
    Load the calibration ``calibrate`` wrote next to a checkpoint.
    :param checkpoint: ``*.pt`` path
    :return: The calibration dict, or ``None`` when the checkpoint was never calibrated
    """
    path = checkpoint.with_suffix(".calibration.json")
    return json.loads(path.read_text()) if path.is_file() else None


def resolve(model: Path) -> Path:
    """
    Turn a run directory into the checkpoint that was calibrated in it.
    :param model: Checkpoint file or run directory
    :return: Checkpoint path
    :raises SystemExit: When nothing calibrated is found
    """
    if model.is_file():
        return model.resolve()
    calibrated = [path for path in model.glob("*.pt") if calibration_of(path) is not None]
    if not calibrated:
        sys.exit(f"{model}: no calibrated checkpoint; run `python -m src.main calibrate` first")
    return max(calibrated, key=lambda path: calibration_of(path)["pq"]).resolve()


def model_name(checkpoint: Path) -> str:
    """
    Name a model the way its submission is described: ``<sweep>/<run> <checkpoint stem>``.
    :param checkpoint: Checkpoint path under ``runs/``
    :return: The name
    """
    run = checkpoint.parent
    prefix = f"{run.parent.name}/{run.name}" if run.parent.name != "runs" else run.name
    return f"{prefix} {checkpoint.stem}"


def predict(checkpoint: Path, output_dir: Path, calibration: dict) -> None:
    """
    Write ``submission.csv`` for the test observations with the calibrated post-processing.

    The window and overlap are the ones the calibration was scored with; ``predict``
    would otherwise fall back to its own default overlap.
    :param checkpoint: Checkpoint to run
    :param output_dir: Where ``submission.csv`` goes
    :param calibration: The checkpoint's calibration
    :return: None
    :raises SystemExit: When prediction fails
    """
    command = [
        *PYTHON,
        "-m",
        "src.main",
        "predict",
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(output_dir),
        "--tile",
        str(calibration["tile"]),
        "--overlap",
        str(calibration["overlap"]),
        "--batch-size",
        "1",
        "--workers",
        "4",
        "--tile-batch",
        "8",
    ]
    if not calibration.get("tta", True):
        command.append("--no-tta")
    print("  " + " ".join(command[len(PYTHON) :]))
    if subprocess.run(command, cwd=REPO).returncode:
        sys.exit(f"prediction failed for {checkpoint}")


def check_submission(path: Path) -> tuple[int, int]:
    """
    Count what a submission covers.
    :param path: ``submission.csv``
    :return: Tuple of (rows, test observations with at least one filament)
    """
    with open(path, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    observations = {row["filament_id"].rsplit("_", 1)[0] for row in rows}
    return len(rows), len(observations)


def submitted_names(api: KaggleApi) -> list[str]:
    """
    Descriptions of the submissions already made to the competition.
    :param api: Authenticated ``KaggleApi``
    :return: Descriptions, newest first
    """
    descriptions: list[str] = []
    for page in range(1, 11):
        submissions = api.competition_submissions(COMPETITION, page_number=page, page_size=50) or []
        descriptions += [s.description or "" for s in submissions if s is not None]
        if len(submissions) < 50:
            break
    return descriptions


def main() -> int:
    """
    Predict and submit every selected model.
    :return: Process exit code
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="+", type=Path, help="checkpoint files or run directories")
    parser.add_argument("--force-predict", action="store_true", help="predict again even if submission.csv exists")
    parser.add_argument("--resubmit", action="store_true", help="submit even if a submission with this name exists")
    parser.add_argument("--no-submit", action="store_true", help="predict only")
    args = parser.parse_args()

    checkpoints = list(dict.fromkeys(resolve(model) for model in args.models))

    api = KaggleApi()
    api.authenticate()
    existing = [] if args.no_submit else submitted_names(api)

    for checkpoint in checkpoints:
        name = model_name(checkpoint)
        calibration = calibration_of(checkpoint)
        if calibration is None:
            sys.exit(f"{checkpoint} was never calibrated; run `python -m src.main calibrate` first")
        description = f"{name} | val pq {calibration['pq']:.4f}"
        output_dir = checkpoint.parent / f"predictions_{checkpoint.stem}"
        submission = output_dir / "submission.csv"
        print(f"\n{name}  (val pq {calibration['pq']:.4f})")

        fresh = submission.is_file() and submission.stat().st_mtime > checkpoint.stat().st_mtime
        if fresh and not args.force_predict:
            print(f"  reusing {submission.relative_to(REPO)}")
        else:
            predict(checkpoint, output_dir, calibration)

        rows, observations = check_submission(submission)
        print(f"  {rows} filaments over {observations}/{TEST_OBSERVATIONS} test observations")
        if observations < TEST_OBSERVATIONS * 0.9:
            print("  warning: many test observations have no filament at all")

        if args.no_submit:
            continue
        if not args.resubmit and any(d.startswith(name + " ") or d == name for d in existing):
            print(f"  already submitted as '{name}', skipping (pass --resubmit to submit again)")
            continue
        api.competition_submit(str(submission), description, COMPETITION)
        print(f"  submitted as '{description}'")

    if not args.no_submit:
        print(f"\nscores appear at https://www.kaggle.com/competitions/{COMPETITION}/submissions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

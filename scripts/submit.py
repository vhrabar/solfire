"""
Predict the test observations with each selected model and submit the result to Kaggle.
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

REPO = Path(__file__).resolve().parent.parent
COMPETITION = "filament-segmentation-2026"
TEST_OBSERVATIONS = 180
PYTHON = ["uv", "run", "--group", "cuda-134", "python"]


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
        sys.exit(f"{model}: no calibrated checkpoint; run `python -m solfire.main calibrate` first")
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


def predict(checkpoints: list[Path], output_dir: Path, calibration: dict, calibration_path: Path | None = None) -> float:
    """
    Write ``submission.csv`` for the test observations with the calibrated post-processing.

    The window and overlap are the ones the calibration was scored with; ``predict``
    would otherwise fall back to its own default overlap.
    :param checkpoints: Checkpoints to run; several are averaged
    :param output_dir: Where ``submission.csv`` goes
    :param calibration: The calibration to predict with
    :param calibration_path: File holding ``calibration`` when it is not stored in the checkpoint
    :return: Wall-clock seconds the prediction took, model loading included
    :raises SystemExit: When prediction fails
    """
    command = [
        *PYTHON,
        "-m",
        "solfire.main",
        "predict",
        "--checkpoint",
        *(str(checkpoint) for checkpoint in checkpoints),
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
    if calibration_path is not None:
        command += ["--calibration", str(calibration_path)]
    if not calibration.get("tta", True):
        command.append("--no-tta")
    print("  " + " ".join(command[len(PYTHON) :]))
    start = time.perf_counter()
    if subprocess.run(command, cwd=REPO).returncode:
        sys.exit(f"prediction failed for {' '.join(str(checkpoint) for checkpoint in checkpoints)}")
    return time.perf_counter() - start


def gpu_name() -> str:
    """
    :return: The first GPU's name as ``nvidia-smi`` reports it, or ``""`` without one
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, check=True
        )
    except OSError, subprocess.CalledProcessError:
        return ""
    return result.stdout.splitlines()[0].strip() if result.stdout.strip() else ""


def write_manifest(
    output_dir: Path, description: str, checkpoints: list[Path], calibration_path: Path | None, seconds: float
) -> None:
    """
    Record what a ``submission.csv`` was predicted from and how long it took, for ``scripts/ledger.py``.
    :param output_dir: Directory holding ``submission.csv``
    :param description: The submission description
    :param checkpoints: Checkpoints that were averaged
    :param calibration_path: Calibration file, or ``None`` when it came from the checkpoint
    :param seconds: Wall-clock prediction time
    :return: None
    """
    manifest = {
        "description": description,
        "models": len(checkpoints),
        "checkpoints": [str(checkpoint.relative_to(REPO)) for checkpoint in checkpoints],
        "calibration": str(calibration_path.relative_to(REPO)) if calibration_path else None,
        "predict_seconds": round(seconds, 1),
        "seconds_per_image": round(seconds / TEST_OBSERVATIONS, 2),
        "gpu": gpu_name(),
        "predicted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (output_dir / "submission.json").write_text(json.dumps(manifest, indent=2) + "\n")


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


def submit_one(
    api: KaggleApi | None,
    existing: list[str],
    name: str,
    checkpoints: list[Path],
    calibration: dict,
    output_dir: Path,
    args: argparse.Namespace,
    calibration_path: Path | None = None,
) -> None:
    """
    Predict with one model or ensemble, check the result and submit it.
    :param api: Authenticated ``KaggleApi``, or ``None`` when only predicting
    :param existing: Descriptions of the submissions already made
    :param name: Name the submission is described by
    :param checkpoints: Checkpoints to run; several are averaged
    :param calibration: The calibration to predict with
    :param output_dir: Where ``submission.csv`` goes
    :param args: Parsed command line
    :param calibration_path: File holding ``calibration`` when it is not stored in the checkpoint
    :return: None
    """
    # a calibration with crop refiners is judged by what the refined instances score
    pq = calibration.get("refiner", calibration)["pq"]
    description = f"{name} | val pq {pq:.4f}"
    submission = output_dir / "submission.csv"
    print(f"\n{name}  (val pq {pq:.4f})")

    newest_input = max(path.stat().st_mtime for path in [*checkpoints, *([calibration_path] if calibration_path else [])])
    fresh = submission.is_file() and submission.stat().st_mtime > newest_input
    if fresh and not args.force_predict:
        print(f"  reusing {submission.relative_to(REPO)}")
    else:
        seconds = predict(checkpoints, output_dir, calibration, calibration_path)
        write_manifest(output_dir, description, checkpoints, calibration_path, seconds)
        print(f"  predicted in {seconds:.0f} s ({seconds / TEST_OBSERVATIONS:.2f} s per observation)")

    rows, observations = check_submission(submission)
    print(f"  {rows} filaments over {observations}/{TEST_OBSERVATIONS} test observations")
    if observations < TEST_OBSERVATIONS * 0.9:
        print("  warning: many test observations have no filament at all")

    if api is None:
        return
    if not args.resubmit and any(d.startswith(name + " ") or d == name for d in existing):
        print(f"  already submitted as '{name}', skipping (pass --resubmit to submit again)")
        return
    api.competition_submit(str(submission), description, COMPETITION)
    print(f"  submitted as '{description}'")


def main() -> int:
    """
    Predict and submit every selected model, or all of them as one ensemble.
    :return: Process exit code
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="+", type=Path, help="checkpoint files or run directories")
    parser.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="calibration JSON from `calibrate --output`; the models are then submitted together as one ensemble",
    )
    parser.add_argument("--force-predict", action="store_true", help="predict again even if submission.csv exists")
    parser.add_argument("--resubmit", action="store_true", help="submit even if a submission with this name exists")
    parser.add_argument("--no-submit", action="store_true", help="predict only")
    args = parser.parse_args()

    api = None
    existing: list[str] = []
    if not args.no_submit:
        api = KaggleApi()
        api.authenticate()
        existing = submitted_names(api)

    if args.calibration is not None:
        checkpoints = [model.resolve() if model.is_file() else resolve(model) for model in args.models]
        args.calibration = args.calibration.resolve()
        calibration = json.loads(args.calibration.read_text())
        stem = args.calibration.name.removesuffix(".calibration.json").removesuffix(".json")
        name = f"{args.calibration.parent.name}/{stem} ({len(checkpoints)} models)"
        output_dir = args.calibration.parent / f"predictions_{stem}"
        submit_one(api, existing, name, checkpoints, calibration, output_dir, args, calibration_path=args.calibration)
    else:
        for checkpoint in dict.fromkeys(resolve(model) for model in args.models):
            calibration = calibration_of(checkpoint)
            if calibration is None:
                sys.exit(f"{checkpoint} was never calibrated; run `python -m solfire.main calibrate` first")
            output_dir = checkpoint.parent / f"predictions_{checkpoint.stem}"
            submit_one(api, existing, model_name(checkpoint), [checkpoint], calibration, output_dir, args)

    if api is not None:
        sys.path.insert(0, str(REPO))
        from scripts.ledger import refresh

        refresh(api, wait_seconds=900)
        print(f"\nscores appear at https://www.kaggle.com/competitions/{COMPETITION}/submissions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

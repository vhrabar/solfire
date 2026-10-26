"""
Publish trained checkpoints to Kaggle as a private Model, so a notebook can attach them.
"""

import argparse
import json
import re
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from kaggle.api.kaggle_api_extended import KaggleApi
from requests import HTTPError

REPO = Path(__file__).resolve().parent.parent

MODEL_SLUG = "filament-segmentation-2026"
MODEL_TITLE = "Solar Filament Segmentation 2026"
FRAMEWORK = "pyTorch"
LICENSE = "Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)"
# checkpoint keys that describe how it was trained and how its output is post-processed
MANIFEST_KEYS = (
    "arch",
    "model_config",
    "epoch",
    "crop",
    "native_size",
    "tile",
    "overlap",
    "threshold",
    "offset",
    "min_area",
    "pos_weight",
    "val_fraction",
    "seed",
    "categories",
)


def slugify(text: str) -> str:
    """
    Turn free text into a valid Kaggle slug: lowercase letters, digits and hyphens.
    :param text: Any string
    :return: The slug
    """
    return re.sub(r"[^a-z0-9-]+", "-", text.lower()).strip("-")


def recipe_slug(checkpoint: dict) -> str:
    """
    Name the recipe a checkpoint was trained with.
    :param checkpoint: Loaded checkpoint
    :return: Variation slug
    """
    arch = checkpoint.get("arch", "unet")
    if arch != "pretrained":
        return slugify(arch)
    config = checkpoint.get("model_config") or {}
    encoder = str(config.get("encoder", "encoder")).removeprefix("tu-")
    return slugify(f"{config.get('decoder', 'unet')}-{encoder}")


def staged_name(path: Path) -> str:
    """
    A file name that stays unique when several runs' ``best.pt`` are staged together.
    :param path: Checkpoint path, ``runs/<sweep>/<run>/<stem>.pt``
    :return: ``<run>-<stem>`` or just ``<stem>`` when the file sits at the top level
    """
    run = path.parent.name
    return f"{run}-{path.stem}" if run and path.parent != REPO else path.stem


def stage(
    checkpoints: list[Path], stage_dir: Path, extras: list[Path] | None = None, refiners: list[Path] | None = None
) -> set[str]:
    """
    Copy the checkpoints and their calibration into a fresh ``stage_dir``, with a manifest.
    :param checkpoints: Checkpoint files to publish
    :param stage_dir: Directory to upload from; wiped first
    :param extras: Further files to publish as they are, e.g. an ensemble's calibration
        and learned filter
    :param refiners: Crop refiner checkpoints, one per fold; staged as ``refiner-fold<k>.pt``
        so the notebook finds them and does not mistake them for ensemble members
    :return: The recipe slugs of the staged checkpoints
    """
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)

    recipes = set()
    manifest = []
    for path in checkpoints:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if "model_state" not in checkpoint:
            sys.exit(f"{path} has no model_state; is it a training checkpoint?")
        if checkpoint.get("calibration") is None:
            print(f"  warning: {path} was never calibrated; run `calibrate` before publishing it")
        recipes.add(recipe_slug(checkpoint))

        name = staged_name(path)
        shutil.copy2(path, stage_dir / f"{name}.pt")
        calibration = path.with_suffix(".calibration.json")
        if calibration.is_file():
            shutil.copy2(calibration, stage_dir / f"{name}.calibration.json")

        pq = (checkpoint.get("calibration") or {}).get("pq")
        manifest.append(
            {
                "file": f"{name}.pt",
                "source": str(path.relative_to(REPO) if path.is_relative_to(REPO) else path),
                "calibrated_pq": pq,
                **{key: checkpoint.get(key) for key in MANIFEST_KEYS},
            }
        )
        pq_text = f"calibrated pq {pq:.4f}" if pq is not None else "uncalibrated"
        print(f"  staged {name}.pt  ({path.stat().st_size / 1e6:.1f} MB, {recipe_slug(checkpoint)}, {pq_text})")

    folds = set()
    for path in refiners or []:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if "state_dict" not in checkpoint or "fold" not in checkpoint:
            sys.exit(f"{path} has no state_dict and fold; is it a crop refiner checkpoint?")
        fold = int(checkpoint["fold"])
        if fold in folds:
            sys.exit(f"two refiners of fold {fold}")
        folds.add(fold)
        name = f"refiner-fold{fold}.pt"
        shutil.copy2(path, stage_dir / name)
        manifest.append(
            {
                "file": name,
                "source": str(path.relative_to(REPO) if path.is_relative_to(REPO) else path),
                "refiner": True,
                **{key: checkpoint.get(key) for key in ("fold", "epochs", "size")},
            }
        )
        print(f"  staged {name}  ({path.stat().st_size / 1e6:.1f} MB, crop refiner)")

    for extra in extras or []:
        shutil.copy2(extra, stage_dir / extra.name)
        print(f"  staged {extra.name}  ({extra.stat().st_size / 1e6:.1f} MB)")
    (stage_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return recipes


def exists(call, *args) -> bool:
    """
    Whether a Kaggle lookup succeeds; a missing model comes back as 403 or 404, not as ``None``.
    :param call: Bound API method
    :param args: Its arguments
    :return: ``True`` when the call returned without an HTTP error
    """
    try:
        call(*args)
    except HTTPError as error:
        if error.response is not None and error.response.status_code in (403, 404):
            return False
        raise
    return True


def latest_version(api, instance: str, newer_than=0, tries=24) -> int | None:
    """
    Highest version of one variation.

    A version that was just uploaded can take a while to be listed, so after an
    upload pass the version seen before it: returning the old maximum would point
    the notebook at stale weights.
    :param api: Authenticated ``KaggleApi``
    :param instance: ``owner/model/framework/variation``
    :param newer_than: Wait until a version above this one is listed
    :param tries: Polls, five seconds apart
    :return: The version number, or ``None`` if nothing newer showed up
    """
    best = None
    for attempt in range(tries):
        response = api.model_instance_versions_list(instance)
        versions = response.version_list.versions if response.version_list else []
        numbers = [v.version_number for v in versions or [] if v is not None]
        best = max(numbers, default=None)
        if best is not None and best > newer_than:
            return best
        if attempt + 1 < tries:
            time.sleep(5)
    return best if newer_than == 0 else None


def write_json(path: Path, data: dict) -> None:
    """
    :param path: Destination file
    :param data: JSON-serialisable dict
    :return: None
    """
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def upload(api, owner: str, stage_dir: Path, variation: str, notes: str, dry_run: bool) -> int:
    """
    Create the model and the variation if needed, then push the staged files as a new version.
    :param api: Authenticated ``KaggleApi``
    :param owner: Kaggle user the model belongs to
    :param stage_dir: Directory holding the staged files
    :param variation: Variation slug
    :param notes: Version notes
    :param dry_run: Only report what would be done
    :return: Process exit code
    """
    model = f"{owner}/{MODEL_SLUG}"
    instance = f"{model}/{FRAMEWORK}/{variation}"

    if not exists(api.model_get, model):
        print(f"{model} not found - creating it (private)")
        model_dir = stage_dir.parent / f"{stage_dir.name}-model"
        if model_dir.exists():
            shutil.rmtree(model_dir)
        model_dir.mkdir()
        write_json(
            model_dir / api.MODEL_METADATA_FILE,
            {
                "ownerSlug": owner,
                "title": MODEL_TITLE,
                "slug": MODEL_SLUG,
                "isPrivate": True,
                "description": "",
            },
        )
        if not dry_run:
            api.model_create_new(str(model_dir))

    if exists(api.model_instance_get, instance):
        before = latest_version(api, instance, tries=1) or 0
        print(f"{instance}: adding version {before + 1}")
        if dry_run:
            print("dry run: nothing uploaded")
            return 0
        api.model_instance_version_create(instance, str(stage_dir), version_notes=notes)
    else:
        print(f"{instance}: creating the variation (version 1)")
        write_json(
            stage_dir / api.MODEL_INSTANCE_METADATA_FILE,
            {
                "ownerSlug": owner,
                "modelSlug": MODEL_SLUG,
                "instanceSlug": variation,
                "framework": FRAMEWORK,
                "licenseName": LICENSE,
                "fineTunable": False,
                "trainingData": [],
                "modelInstanceType": "Unspecified",
            },
        )
        if dry_run:
            print("dry run: nothing uploaded")
            return 0
        before = 0
        api.model_instance_create(str(stage_dir))

    version = latest_version(api, instance, newer_than=before)
    if version is None:
        print(f"uploaded, but {instance} has not listed the new version yet; check it on kaggle.com")
        return 0
    print(f"published {instance}/{version}")
    print(f"attach in a notebook as model source: {instance}/{version}")
    print(f"mounted at /kaggle/input/{MODEL_SLUG}/{FRAMEWORK.lower()}/{variation}/{version}/")
    return 0


def main() -> int:
    """
    Stage the checkpoints named on the command line and publish them.
    :return: Process exit code
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoints", nargs="+", type=Path, help="checkpoint files, e.g. runs/<sweep>/<run>/best.pt")
    parser.add_argument("-m", "--message", help="version notes (default: UTC timestamp)")
    parser.add_argument("--variation", help="variation slug (default: derived from the checkpoints' recipe)")
    parser.add_argument(
        "--extra", nargs="*", type=Path, default=[], help="further files to publish, e.g. ensemble calibration and filter"
    )
    parser.add_argument(
        "--refiner", nargs="*", type=Path, default=[], help="crop refiner checkpoints, e.g. runs/refiner/zoom/fold*.pt"
    )
    parser.add_argument("--stage", type=Path, default=REPO / "build" / "kaggle-model", help="staging dir to upload from")
    parser.add_argument("--dry-run", action="store_true", help="stage and report, upload nothing")
    args = parser.parse_args()

    checkpoints = [path.resolve() for path in args.checkpoints]
    missing = [str(path) for path in checkpoints if not path.is_file()]
    if missing:
        sys.exit(f"not found: {', '.join(missing)}")

    stage_dir = args.stage.resolve()
    extras = [path.resolve() for path in args.extra]
    refiners = [path.resolve() for path in args.refiner]
    missing_extras = [str(path) for path in [*extras, *refiners] if not path.is_file()]
    if missing_extras:
        sys.exit(f"not found: {', '.join(missing_extras)}")
    recipes = stage(checkpoints, stage_dir, extras, refiners)
    if args.variation:
        variation = slugify(args.variation)
    elif len(recipes) == 1:
        variation = recipes.pop()
    else:
        sys.exit(f"the checkpoints come from different recipes {sorted(recipes)}; pass --variation")

    api = KaggleApi()
    api.authenticate()
    owner = api.get_config_value("username")
    notes = args.message or f"checkpoints {datetime.now(UTC):%Y-%m-%d %H:%M UTC}"
    print(f"\n{len(checkpoints)} checkpoint(s) staged in {stage_dir} -> {owner}/{MODEL_SLUG}/{FRAMEWORK}/{variation}")
    return upload(api, owner, stage_dir, variation, notes, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())

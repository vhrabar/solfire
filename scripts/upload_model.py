"""
Publish the presets of ``weights/final`` (``scripts/release_weights.py``) to Kaggle Models, one variation each.
"""

import argparse
import json
import os
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi
from requests import HTTPError

REPO = Path(__file__).resolve().parent.parent
CARD_DIR = REPO / "kaggle" / "model"
FRAMEWORK = "pyTorch"
LICENSE = "Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)"
NOTEBOOK = "notebook/Final pipeline.ipynb"


def variation_slug(preset: str) -> str:
    """
    :param preset: Preset name, e.g. ``final``
    :return: Its variation slug, ``solfire-<preset>``
    """
    return f"solfire-{preset}"


def overview(preset: dict) -> str:
    """
    The variation's overview: what it holds and how good and fast it is.
    :param preset: Manifest entry of the preset
    :return: Markdown
    """
    oof = f"{preset['oof_pq_without_barbs']:.4f} without barbs"
    if preset.get("oof_pq") is not None:
        oof = f"{preset['oof_pq']:.4f} ({oof})"
    lb = f"{preset['public_lb']:.2f}" if preset.get("public_lb") is not None else "not submitted"
    files = "\n".join(f"- `{name}`" for name in preset["files"])
    return (
        f"**SOLFIRE `{preset['name']}`**: {preset['summary']}.\n\n"
        f"| Out-of-fold PQ | Public LB | s / image (RTX 5070 Ti, fp16) | Size |\n|---|---|---|---|\n"
        f"| {oof} | {lb} | {preset['seconds_per_image']} | {preset['bytes'] / 1e6:.0f} MB |\n\n"
        f"Out-of-fold PQ is that of the matching five-fold models through the same cut-offs; the published models "
        f"are their full-data counterparts. Weights are float16.\n\n"
        f"Files:\n{files}\n- `{preset['calibration']}`: cut-offs and inference settings\n"
        f"- `manifest.json`: source, SHA-256 and training settings of each file\n"
    )


def usage(preset: dict, owner: str, model_slug: str) -> str:
    """
    How to run the variation.
    :param preset: Manifest entry of the preset
    :param owner: Kaggle user
    :param model_slug: Model slug
    :return: Markdown
    """
    handle = f"{owner}/{model_slug}/{FRAMEWORK}/{variation_slug(preset['name'])}"
    calibration = preset["calibration"]
    return (
        f"In a Kaggle notebook, attach this variation and run `{NOTEBOOK}` from the repository with\n\n"
        f'```python\nMODEL_HANDLE = "{handle}/<version>"\nCALIBRATION_NAME = "{calibration}"\n```\n\n'
        f"Locally, from a checkout of the repository with the `solfire` package installed, and these files in "
        f"`weights/final/` (where `git lfs pull` puts them; the calibration names its checkpoints there):\n\n"
        f"```bash\nsolfire predict --calibration weights/final/{calibration} \\\n"
        f"  --checkpoint {' '.join(f'weights/final/{name}' for name in preset['files'] if not name.startswith('refiner-'))}\n```\n"
    )


def stage(weights: Path, preset: dict, target: Path) -> None:
    """
    Copy one preset's files into a fresh directory, with a manifest of just those files.
    :param weights: Release directory, ``weights/final``
    :param preset: Manifest entry of the preset
    :param target: Directory to upload from; wiped first
    :return: None
    """
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    manifest = json.loads((weights / "manifest.json").read_text(encoding="utf-8"))
    for name in [*preset["files"], preset["calibration"]]:
        try:
            os.link(weights / name, target / name)
        except OSError:
            shutil.copy2(weights / name, target / name)
    files = [entry for entry in manifest["files"] if entry["file"] in preset["files"]]
    subset = {"presets": {preset["name"]: preset}, "files": files}
    write_json(target / "manifest.json", subset)


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
        # the listing holds every variation of the model, not just this one
        variation = instance.rsplit("/", 1)[-1]
        numbers = [v.version_number for v in versions or [] if v is not None and v.variation_slug == variation]
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


def model_metadata(owner: str) -> dict:
    """
    The model-level metadata, with the model card as its description.
    :param owner: Kaggle user
    :return: Contents of ``model-metadata.json``
    """
    metadata = json.loads((CARD_DIR / "model-metadata.json").read_text(encoding="utf-8"))
    metadata["ownerSlug"] = owner
    metadata["description"] = (CARD_DIR / "README.md").read_text(encoding="utf-8")
    return metadata


def push_model(api, metadata: dict, folder: Path, dry_run: bool) -> None:
    """
    Create the model, or update its title, subtitle, card and provenance.
    :param api: Authenticated ``KaggleApi``
    :param metadata: Model-level metadata
    :param folder: Scratch directory for the metadata file
    :param dry_run: Only report what would be done
    :return: None
    """
    model = f"{metadata['ownerSlug']}/{metadata['slug']}"
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    write_json(folder / api.MODEL_METADATA_FILE, metadata)
    found = exists(api.model_get, model)
    print(f"{model}: {'updating the card' if found else 'creating it (private)'}")
    if dry_run:
        return
    if found:
        api.model_update(str(folder))
    else:
        api.model_create_new(str(folder))


def push_variation(api, metadata: dict, preset: dict, folder: Path, notes: str, dry_run: bool) -> int:
    """
    Create the variation, or update its overview and add a version.
    :param api: Authenticated ``KaggleApi``
    :param metadata: Model-level metadata
    :param preset: Manifest entry of the preset
    :param folder: Staged files of the preset
    :param notes: Version notes
    :param dry_run: Only report what would be done
    :return: Process exit code
    """
    owner, model_slug = metadata["ownerSlug"], metadata["slug"]
    variation = variation_slug(preset["name"])
    instance = f"{owner}/{model_slug}/{FRAMEWORK}/{variation}"
    instance_metadata = {
        "ownerSlug": owner,
        "modelSlug": model_slug,
        "instanceSlug": variation,
        "framework": FRAMEWORK,
        "overview": overview(preset),
        "usage": usage(preset, owner, model_slug),
        "licenseName": LICENSE,
        "fineTunable": False,
        "trainingData": [],
        "modelInstanceType": "Unspecified",
    }
    size = sum(path.stat().st_size for path in folder.iterdir())
    found = exists(api.model_instance_get, instance)
    before = (latest_version(api, instance, tries=1) or 0) if found else 0
    print(f"{instance}: {f'version {before + 1}' if found else 'new variation (version 1)'}, {size / 1e6:.0f} MB")
    if dry_run:
        write_json(folder.with_name(f"{folder.name}.instance-metadata.json"), instance_metadata)
        return 0

    if found:
        meta_dir = folder.with_name(f"{folder.name}-metadata")
        if meta_dir.exists():
            shutil.rmtree(meta_dir)
        meta_dir.mkdir()
        write_json(meta_dir / api.MODEL_INSTANCE_METADATA_FILE, instance_metadata)
        api.model_instance_update(str(meta_dir))
        api.model_instance_version_create(instance, str(folder), version_notes=notes)
    else:
        write_json(folder / api.MODEL_INSTANCE_METADATA_FILE, instance_metadata)
        api.model_instance_create(str(folder))

    version = latest_version(api, instance, newer_than=before)
    if version is None:
        print(f"uploaded, but {instance} has not listed the new version yet; check it on kaggle.com")
        return 0
    print(f'published {instance}/{version}: MODEL_HANDLE = "{instance}/{version}"')
    return 0


def main() -> int:
    """
    Stage the requested presets and publish them.
    :return: Process exit code
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("presets", nargs="*", help="presets to publish (default: every preset in the manifest)")
    parser.add_argument("-m", "--message", help="version notes (default: UTC timestamp)")
    parser.add_argument("--weights", type=Path, default=REPO / "weights" / "final", help="release directory")
    parser.add_argument("--stage", type=Path, default=REPO / "build" / "kaggle-model", help="staging dir to upload from")
    parser.add_argument("--dry-run", action="store_true", help="stage and report, upload nothing")
    args = parser.parse_args()

    weights = args.weights.resolve()
    manifest_path = weights / "manifest.json"
    if not manifest_path.is_file():
        sys.exit(f"{manifest_path} not found; build the release with scripts/release_weights.py")
    presets = json.loads(manifest_path.read_text(encoding="utf-8"))["presets"]
    names = args.presets or list(presets)
    unknown = sorted(set(names) - set(presets))
    if unknown:
        sys.exit(f"unknown presets {unknown}; the release has {sorted(presets)}")

    api = KaggleApi()
    api.authenticate()
    owner = api.get_config_value("username")
    if not owner:
        sys.exit("no Kaggle username configured; set up ~/.kaggle/kaggle.json")
    metadata = model_metadata(owner)
    notes = args.message or f"release {datetime.now(UTC):%Y-%m-%d %H:%M UTC}"
    stage_root = args.stage.resolve()
    push_model(api, metadata, stage_root / "model", args.dry_run)
    for name in names:
        folder = stage_root / name
        stage(weights, presets[name], folder)
        if push_variation(api, metadata, presets[name], folder, notes, args.dry_run):
            return 1
    if args.dry_run:
        print(f"dry run: staged in {stage_root}, nothing uploaded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

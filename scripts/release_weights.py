"""
Build ``weights/final``: the shipped weights in float16, one crop refiner per preset, and a calibration per preset.
"""

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from solfire import PROJECT_ROOT
from solfire.data import TEST_IMAGES, TRAIN_ANNOTATIONS

REPO = PROJECT_ROOT
ENSEMBLES = Path("runs/ensembles")
# inference settings every preset runs with; the cut-offs stay those of the source calibration
RUNTIME = {"tile": 512, "overlap": 0.25, "tta": True, "precision": "fp16", "skip_dark": True}
BARBS = {"contrast": 0.05, "ridge": 0.025, "band": 20, "min_ratio": 4.0, "max_width": 4.0}
# s/img: 6 test observations on an RTX 5070 Ti, batch 1, fp16 (2026-11-04)
PRESETS = {
    "final": {
        "source": ENSEMBLES / "b6-zoom-ssl-tta32-barbs.calibration.json",
        "summary": "4 models with TTA (o0.5 + turned), crop refiner, barbs",
        "seconds_per_image": 12.9,
        "public_lb": 0.40,
    },
    "fast": {
        "source": ENSEMBLES / "b6-zoom-ssl-refined.calibration.json",
        "summary": "the same 4 models without TTA, crop refiner, barbs",
        "seconds_per_image": 5.1,
        "public_lb": 0.40,
    },
    "lite": {
        "source": ENSEMBLES / "b6-full-fp16-refined.calibration.json",
        "summary": "3 models (ImageNet ResNet34 U-Net++, MiT-B2, HRNet-W32), crop refiner, barbs",
        "seconds_per_image": 2.7,
        "public_lb": 0.40,
    },
    "tiny": {
        "source": ENSEMBLES / "ctl-hrnet-oof.calibration.json",
        # the cut-offs come from the five fold HRNets at their final epoch, the same recipe as the full-data one
        "checkpoints": ["runs/full/c06_full-lovasz-hrnet32_seed1/best.pt"],
        "summary": "the full-data HRNet-W32 U-Net alone, barbs",
        "seconds_per_image": 1.5,
        "public_lb": None,
    },
}
# measured cost of the barbs on the refined out-of-fold instances of `final` (0.4692 -> 0.4668)
FINAL_BARBS_PQ = 0.4668


def half(state: dict) -> dict:
    """
    Store the weights of a state dict as float16.
    :param state: Parameter and buffer tensors
    :return: The state dict with every floating tensor of two or more dimensions cast to float16
    """
    return {
        key: value.half() if torch.is_tensor(value) and value.is_floating_point() and value.dim() >= 2 else value
        for key, value in state.items()
    }


def release_name(source: str) -> str:
    """
    File name of a checkpoint in ``weights/final``.
    :param source: ``runs/<sweep>/<run>/best.pt`` or ``runs/refiner/<set>/fold<k>.pt``
    :return: ``<run>-best.pt`` or ``refiner-<set>-fold<k>.pt``
    """
    path = Path(source)
    if path.parts[:2] == ("runs", "refiner"):
        return f"refiner-{path.parent.name}-{path.stem}.pt"
    return f"{path.parent.name}-{path.stem}.pt"


def sha256(path: Path) -> str:
    """
    :param path: File to hash
    :return: Hex SHA-256 of its bytes
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def preset_calibration(name: str, spec: dict, refiner_fold: int) -> tuple[dict, dict]:
    """
    The calibration of one preset with paths into ``runs/``, and the same one pointing at ``weights/final``.
    :param name: Preset name
    :param spec: Its entry in ``PRESETS``
    :param refiner_fold: Which of the five fold refiners to keep
    :return: Tuple of (calibration on the source checkpoints, release calibration)
    """
    calibration = json.loads((REPO / spec["source"]).read_text(encoding="utf-8"))
    calibration.pop("table", None)
    if "checkpoints" in spec:
        calibration["checkpoints"] = list(spec["checkpoints"])
    calibration.update(RUNTIME)
    calibration.setdefault("variants", ["base"])
    calibration["barbs"] = dict(BARBS)
    pq_without_barbs = calibration["pq"]
    refiner = calibration.get("refiner")
    if refiner:
        sources = refiner["checkpoints"]
        kept = [path for path in sources if Path(path).stem == f"fold{refiner_fold}"]
        if len(kept) != 1:
            sys.exit(f"{spec['source']}: no refiner of fold {refiner_fold} in {sources}")
        refiner["checkpoints"] = kept
        refiner.pop("table", None)
        pq_without_barbs = refiner.get("pq_without_barbs", refiner["pq"])
    calibration["preset"] = {
        "name": name,
        "summary": spec["summary"],
        "source_calibration": str(spec["source"]),
        # pooled out-of-fold PQ of the source fold models through these cut-offs; the barbs lower it slightly
        "oof_pq_without_barbs": round(pq_without_barbs, 4),
        "oof_pq": FINAL_BARBS_PQ if name == "final" else None,
        "seconds_per_image": spec["seconds_per_image"],
        "public_lb": spec["public_lb"],
        "refiner_fold": refiner_fold if refiner else None,
    }

    released = copy.deepcopy(calibration)
    released["checkpoints"] = [f"weights/final/{release_name(path)}" for path in calibration["checkpoints"]]
    if refiner:
        released["refiner"]["checkpoints"] = [
            f"weights/final/{release_name(path)}" for path in calibration["refiner"]["checkpoints"]
        ]
    return calibration, released


def files_of(calibration: dict) -> list[str]:
    """
    :param calibration: A calibration
    :return: Every checkpoint it loads, ensemble members first
    """
    return [*calibration["checkpoints"], *(calibration.get("refiner") or {}).get("checkpoints", [])]


def write_half(source: Path, target: Path) -> dict:
    """
    Save a float16 copy of a checkpoint.
    :param source: Segmentation (``model_state``) or crop refiner (``state_dict``) checkpoint
    :param target: Where to write the copy
    :return: Its manifest entry
    """
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    key = "model_state" if "model_state" in checkpoint else "state_dict"
    if key not in checkpoint:
        sys.exit(f"{source} holds neither model_state nor state_dict")
    checkpoint[key] = half(checkpoint[key])
    checkpoint["weights_dtype"] = "float16"
    torch.save(checkpoint, target)
    entry = {"file": target.name, "source": str(source.relative_to(REPO)), "refiner": key == "state_dict"}
    if key == "model_state":
        entry.update({name: checkpoint.get(name) for name in ("arch", "model_config", "epoch", "input_scale", "seed")})
    else:
        entry.update({name: checkpoint.get(name) for name in ("fold", "epochs", "size")})
    return entry


def build(stage: Path, refiner_fold: int) -> dict[str, dict]:
    """
    Write the release into ``stage``.
    :param stage: Empty directory
    :param refiner_fold: Which fold refiner every preset keeps
    :return: Per preset the calibration on the source checkpoints
    """
    sources: dict[str, dict] = {}
    released: dict[str, dict] = {}
    for name, spec in PRESETS.items():
        sources[name], released[name] = preset_calibration(name, spec, refiner_fold)

    entries = {}
    for path in dict.fromkeys(path for calibration in sources.values() for path in files_of(calibration)):
        target = stage / release_name(path)
        entries[target.name] = write_half(REPO / path, target)
        print(f"  {target.name}  {(REPO / path).stat().st_size / 1e6:.0f} -> {target.stat().st_size / 1e6:.0f} MB")

    presets: dict[str, dict] = {}
    for name, calibration in released.items():
        (stage / f"{name}.calibration.json").write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
        files = [Path(path).name for path in files_of(calibration)]
        size = sum((stage / file).stat().st_size for file in files)
        presets[name] = {**calibration["preset"], "calibration": f"{name}.calibration.json", "files": files, "bytes": size}
        print(f"  {name}.calibration.json  {len(files)} files, {size / 1e6:.0f} MB")
    for entry in entries.values():
        entry["sha256"] = sha256(stage / entry["file"])
        entry["bytes"] = (stage / entry["file"]).stat().st_size
        entry["presets"] = [name for name, preset in presets.items() if entry["file"] in preset["files"]]
    manifest = {"presets": presets, "files": list(entries.values())}
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return sources


def predict(calibration: Path, checkpoints: list[str], data_root: Path, output: Path) -> str:
    """
    Run ``solfire predict`` on a small data root.
    :param calibration: Calibration JSON
    :param checkpoints: Ensemble members, in the calibration's order
    :param data_root: Directory with ``train/<annotations>`` and ``test/test_images``
    :param output: Output directory
    :return: The submission CSV text
    """
    command = [
        *(sys.executable, "-m", "solfire.main", "predict"),
        *("--checkpoint", *checkpoints, "--calibration", str(calibration)),
        *("--batch-size", "1", "--workers", "2", "--output-dir", str(output)),
    ]
    environment = {**os.environ, "FILAMENT_DATA_ROOT": str(data_root)}
    result = subprocess.run(command, cwd=REPO, env=environment, capture_output=True, text=True)
    if result.returncode:
        sys.exit(f"predict failed:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}")
    return (output / "submission.csv").read_text(encoding="utf-8")


def check(stage: Path, sources: dict[str, dict], images: int, scratch: Path) -> bool:
    """
    Predict a few test observations with the float32 sources and with the release, per preset.
    :param stage: Built release
    :param sources: Per preset the calibration on the source checkpoints
    :param images: Test observations to use
    :param scratch: Directory for the data root and the outputs
    :return: Whether every preset's submission is identical
    """
    data_root = scratch / "data"
    (data_root / "train").mkdir(parents=True)
    (data_root / "test" / "test_images").mkdir(parents=True)
    (data_root / "train" / TRAIN_ANNOTATIONS.name).symlink_to(TRAIN_ANNOTATIONS)
    for image in sorted(TEST_IMAGES.iterdir())[:images]:
        (data_root / "test" / "test_images" / image.name).symlink_to(image)

    identical = True
    for name, calibration in sources.items():
        source = scratch / f"{name}-source.calibration.json"
        source.write_text(json.dumps(calibration), encoding="utf-8")
        released = json.loads((stage / f"{name}.calibration.json").read_text(encoding="utf-8"))
        released["checkpoints"] = [str(stage / Path(path).name) for path in released["checkpoints"]]
        if released.get("refiner"):
            released["refiner"]["checkpoints"] = [str(stage / Path(path).name) for path in released["refiner"]["checkpoints"]]
        release = scratch / f"{name}-release.calibration.json"
        release.write_text(json.dumps(released), encoding="utf-8")

        expected = predict(source, calibration["checkpoints"], data_root, scratch / f"{name}-source")
        actual = predict(release, released["checkpoints"], data_root, scratch / f"{name}-release")
        same = expected == actual
        identical &= same
        rows = expected.count("\n") - 1
        print(f"  {name}: {'identical' if same else 'DIFFERENT'} submission ({rows} filaments on {images} images)")
    return identical


def main() -> int:
    """
    :return: Process exit code
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=REPO / "weights" / "final", help="release directory to replace")
    parser.add_argument("--refiner-fold", type=int, default=0, help="the fold refiner each preset keeps")
    parser.add_argument("--check", type=int, default=3, help="test observations to compare source and release on; 0 skips")
    args = parser.parse_args()

    output = args.output.resolve()
    stage = output.with_name(f".{output.name}.new")
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    print(f"building {stage}")
    sources = build(stage, args.refiner_fold)

    if args.check:
        print(f"checking on {args.check} test observations")
        with tempfile.TemporaryDirectory() as scratch:
            if not check(stage, sources, args.check, Path(scratch)):
                print(f"the release differs from its sources; left in {stage}, {output} untouched")
                return 1

    if output.exists():
        shutil.rmtree(output)
    stage.rename(output)
    total = sum(path.stat().st_size for path in output.iterdir())
    print(f"released {output} ({total / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

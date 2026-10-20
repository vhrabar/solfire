"""
Assemble the public GONG Kaggle dataset from the files under ``runs/external``.
"""

import argparse
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXTERNAL = REPO / "runs" / "external"
SOURCE = REPO / "kaggle" / "gong-dataset"

# Published name -> (source directory, glob) for folders, or source file for single files
FOLDERS = {
    "halpha_2023_2025": (EXTERNAL / "gong" / "jpeg", "*.jpeg"),
    "bzi_fits": (EXTERNAL / "gong_bzi" / "fits", "*.fits.gz"),
    "bzi_targets": (EXTERNAL / "gong_bzi" / "targets", "*.npy"),
}
FILES = {
    "halpha_bursts.json": EXTERNAL / "gong" / "manifest.json",
    "bzi_matches.json": EXTERNAL / "gong_bzi" / "manifest.json",
}


def link(source, target) -> None:
    """
    Symlink one file, replacing an older link.
    :param source: Existing file
    :param target: Link to create
    """
    target.unlink(missing_ok=True)
    target.symlink_to(source.resolve())


def main() -> int:
    """
    Rebuild the upload and settings folders.
    :return: Process exit code
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default=str(REPO / "dataset"), help="upload folder")
    parser.add_argument("--settings", default=str(REPO / "dataset_meta"), help="cover image and metadata folder")
    args = parser.parse_args()

    output, settings = Path(args.output), Path(args.settings)
    for folder in (output, settings):
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)

    for name, (source, pattern) in FOLDERS.items():
        files = sorted(source.glob(pattern))
        if not files:
            raise SystemExit(f"no {pattern} under {source}")
        (output / name).mkdir()
        for file in files:
            link(file, output / name / file.name)
        print(f"{name}: {len(files)} files")
    for name, source in FILES.items():
        link(source, output / name)

    metadata = json.loads((SOURCE / "dataset-metadata.json").read_text(encoding="utf-8"))
    metadata["description"] = (SOURCE / "README.md").read_text(encoding="utf-8")
    if "<" in metadata["description"]:
        raise SystemExit("README.md contains '<', which Kaggle treats as an HTML tag and cuts the description")
    text = json.dumps(metadata, indent=2, ensure_ascii=False)
    for folder in (output, settings):
        (folder / "dataset-metadata.json").write_text(text, encoding="utf-8")
    shutil.copy2(SOURCE / "dataset-cover-image.png", settings)
    print(f"{metadata['id']} -> {output}, settings -> {settings}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

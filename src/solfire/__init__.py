"""
SOLFIRE: SOLar FIlament Refined Ensemble.

Automatic Segmentation of Solar Filaments using Computer Vision algorithms (with or without fitting) from GONG's H-Alpha Observations
"""

import os
from pathlib import Path


def _project_root() -> Path:
    """
    Directory ``data/`` and ``runs/`` are resolved against.

    ``FILAMENT_PROJECT_ROOT`` wins; otherwise the source checkout when the package is
    imported from one (editable install), else the working directory (wheel install).

    :return: The project root
    """
    override = os.environ.get("FILAMENT_PROJECT_ROOT")
    if override:
        return Path(override)
    checkout = Path(__file__).resolve().parents[2]
    if (checkout / "pyproject.toml").is_file():
        return checkout
    return Path.cwd()


PROJECT_ROOT = _project_root()

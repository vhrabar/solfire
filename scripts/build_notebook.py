"""
Build ``Final pipeline.ipynb`` from ``notebook/pipeline.py``.

"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SOURCE = REPO / "notebook" / "pipeline.py"
NOTEBOOK = REPO / "Final pipeline.ipynb"
MARKDOWN = "# %% [markdown]"
CODE = "# %%"


def split_cells(text: str) -> list[tuple[str, str]]:
    """
    :param text: Percent-format source
    :return: (cell type, source) per cell, in order, without the cell markers
    """
    cells: list[tuple[str, list[str]]] = []
    for line in text.splitlines():
        if line.strip() == MARKDOWN:
            cells.append(("markdown", []))
        elif line.strip() == CODE:
            cells.append(("code", []))
        elif cells:
            cells[-1][1].append(line)
        elif line.strip():
            raise ValueError(f"{SOURCE.name}: text before the first cell marker: {line!r}")
    result = []
    for kind, lines in cells:
        if kind == "markdown":
            lines = [line[2:] if line.startswith("# ") else line.removeprefix("#") for line in lines]
        body = "\n".join(lines).strip("\n")
        if body:
            result.append((kind, body))
    return result


def cell(kind: str, body: str, index: int) -> dict:
    """
    :param kind: ``code`` or ``markdown``
    :param body: Cell source
    :param index: Position, for a stable cell id
    :return: Notebook cell, without outputs
    """
    entry = {"cell_type": kind, "id": f"cell-{index:03d}", "metadata": {}, "source": body.splitlines(keepends=True)}
    if kind == "code":
        entry |= {"execution_count": None, "outputs": []}
    return entry


def build() -> dict:
    """
    :return: Notebook JSON assembled from :data:`SOURCE`
    """
    cells = split_cells(SOURCE.read_text(encoding="utf-8"))
    return {
        "cells": [cell(kind, body, index) for index, (kind, body) in enumerate(cells)],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def render(notebook: dict) -> str:
    """
    :param notebook: Notebook JSON
    :return: The file contents, formatted the way Jupyter writes them
    """
    return json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"


def main() -> int:
    """
    Write the notebook, or with ``--check`` verify the committed one is current.
    :return: Process exit code
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="fail if the notebook differs from what would be built")
    args = parser.parse_args()
    text = render(build())
    if args.check:
        if not NOTEBOOK.is_file() or NOTEBOOK.read_text(encoding="utf-8") != text:
            print(f"{NOTEBOOK.name} is out of date; run `python scripts/build_notebook.py`", file=sys.stderr)
            return 1
        print(f"{NOTEBOOK.name} is up to date")
        return 0
    NOTEBOOK.write_text(text, encoding="utf-8")
    print(f"wrote {NOTEBOOK.name}: {len(build()['cells'])} cells from {SOURCE.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

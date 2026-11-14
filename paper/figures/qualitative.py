"""
Qualitative figure of the report: annotators against the ``final`` variant on held-out observations.
"""

import json
import multiprocessing
import os
import pickle
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
import matplotlib.colors

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import ndimage

import solfire
from solfire.data import TRAIN_IMAGES, load_annotations, load_image
from solfire.data.preprocess import observation_path
from solfire.inference import refiner as refiner_module
from solfire.inference.barbs import DEFAULT_BARBS, grow_barbs
from solfire.inference.postprocess import SUBMISSION_SIZE
from solfire.inference.refiner import assemble
from solfire.training.metrics import (
    annotation_pixels,
    instance_iou,
    pq_counts,
    pq_from_counts,
)

# the refined instances were pickled before the package was renamed from ``src`` to ``solfire``
sys.modules.update(
    {
        "src": solfire,
        "src.inference": solfire.inference,
        "src.inference.refiner": refiner_module,
    }
)

REFINED = Path("runs/refiner/zoom-ssl-tta-rt/refined")
OUTPUT = Path(os.environ.get("QUALITATIVE_OUTPUT", "paper/figures/qualitative.pdf"))
MASK_CUT, IOU_CUT = 0.45, 0.32
PERCENTILES = (("High", 90), ("Median", 50), ("Low", 10))
MIN_RECORDS, MIN_FILAMENTS = 2, 4
# rows of the ``--barbs`` figure, chosen for the grown barbs: among observations with at least two annotator
# records, the ones whose barb pixels fall mostly inside the annotators' masks and read clearly at 12x
SHOWN = ("20150427162314Mh.jpeg", "20150527161214Mh.jpeg", "20110826074214Th.jpeg")
CROP = 768
# Okabe-Ito colours, which stay apart under colour blindness and on a grey background
ANNOTATOR_COLOURS = ("#56B4E9", "#F0E442", "#CC79A7")
MODEL_COLOUR, BARB_COLOUR = "#009E73", "#D55E00"
ZOOM = 64


def observation_records() -> dict[str, list[list[dict]]]:
    """
    :return: Per observation file, the annotations of each annotator record
    """
    images, by_image = load_annotations()
    records = defaultdict(list)
    for image in images:
        records[image["file_name"]].append(by_image[image["id"]])
    return dict(records)


def refined_labels(name) -> tuple[np.ndarray, int]:
    """
    :param name: Observation file
    :return: Tuple of (``(H, W)`` labels of the held-out refined instances, number of labels)
    """
    with open(REFINED / f"{Path(name).stem}.pkl", "rb") as handle:
        refined = pickle.load(handle)["refined"]
    labels, n_labels, _ = assemble(refined, SUBMISSION_SIZE, SUBMISSION_SIZE, threshold=MASK_CUT, min_score=IOU_CUT)
    return labels, n_labels


def observation_pq(labels, n_labels, annotations) -> float:
    """
    :param labels: ``(H, W)`` predicted labels
    :param n_labels: Number of labels
    :param annotations: Annotations of each record of the observation
    :return: PQ of the observation, pooled over its records
    """
    total = np.zeros(4)
    for record in annotations:
        total += pq_counts(instance_iou(annotation_pixels(record), labels, n_labels))
    return float(pq_from_counts(*total))


def score_job(job) -> tuple[str, float]:
    """
    :param job: Tuple of (observation file, annotations of each record)
    :return: Tuple of (observation file, its PQ without barbs)
    """
    name, annotations = job
    return name, observation_pq(*refined_labels(name), annotations)


def select(records) -> list[tuple[str, str]]:
    """
    :param records: Output of :func:`observation_records`
    :return: Per row, (row label, observation file)
    """
    jobs = [
        (name, annotations)
        for name, annotations in sorted(records.items())
        if len(annotations) >= MIN_RECORDS and min(len(record) for record in annotations) >= MIN_FILAMENTS
    ]
    with ProcessPoolExecutor(4, mp_context=multiprocessing.get_context("fork")) as pool:
        scores = sorted(pool.map(score_job, jobs), key=lambda item: item[1])
    print(f"{len(scores)} observations qualify")
    return [(label, scores[round(percentile / 100 * (len(scores) - 1))][0]) for label, percentile in PERCENTILES]


def date_label(name) -> str:
    """
    :param name: Observation file, named by its UTC time stamp
    :return: Its date as ``YYYY-MM-DD``
    """
    return f"{name[:4]}-{name[4:6]}-{name[6:8]}"


def annotation_mask(record) -> np.ndarray:
    """
    :param record: Annotations of one record
    :return: ``(H, W)`` boolean union of its filaments
    """
    mask = np.zeros(SUBMISSION_SIZE**2, dtype=bool)
    for pixels in annotation_pixels(record):
        mask[pixels] = True
    return mask.reshape(SUBMISSION_SIZE, SUBMISSION_SIZE)


def densest_window(mask, side=CROP) -> tuple[slice, slice]:
    """
    :param mask: ``(H, W)`` boolean map of what the figure should show
    :param side: Edge of the window
    :return: Tuple of (rows, columns) of the window holding the most of ``mask``
    """
    density = ndimage.uniform_filter(mask.astype(np.float32), side, mode="constant")
    half = side // 2
    limited = density[half : SUBMISSION_SIZE - half + 1, half : SUBMISSION_SIZE - half + 1]
    row, column = np.unravel_index(limited.argmax(), limited.shape)
    return slice(int(row), int(row) + side), slice(int(column), int(column) + side)


def barb_window(added, side=ZOOM) -> tuple[slice, slice]:
    """
    :param added: ``(H, W)`` boolean map of the grown barb pixels
    :param side: Edge of the window
    :return: Tuple of (rows, columns) of the window holding the most barb pixels
    """
    density = ndimage.uniform_filter(added.astype(np.float32), side, mode="constant")
    half = side // 2
    limited = density[half : SUBMISSION_SIZE - half + 1, half : SUBMISSION_SIZE - half + 1]
    row, column = np.unravel_index(limited.argmax(), limited.shape)
    inside = added[row : row + side, column : column + side]
    centre_row, centre_column = (np.argwhere(inside).mean(axis=0) + np.array([row, column])).round().astype(int)
    row = int(np.clip(centre_row - side // 2, 0, SUBMISSION_SIZE - side))
    column = int(np.clip(centre_column - side // 2, 0, SUBMISSION_SIZE - side))
    return slice(row, row + side), slice(column, column + side)


def overlay(axis, mask, thick) -> None:
    """
    :param axis: Axis to draw on
    :param mask: Boolean mask of the shown window
    :param thick: Pixels the mask is widened by so that it survives downsampling
    :return: None
    """
    if thick:
        mask = ndimage.binary_dilation(mask, iterations=thick)
    rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
    rgba[mask] = (*matplotlib.colors.to_rgb(BARB_COLOUR), 1.0)
    axis.imshow(rgba, interpolation="nearest")


def outline(axis, mask, colour) -> None:
    """
    :param axis: Axis to draw on
    :param mask: Boolean mask of the shown window
    :param colour: Line colour
    :return: None
    """
    if mask.any():
        axis.contour(mask.astype(np.uint8), levels=[0.5], colors=[colour], linewidths=0.45)


def draw(rows, records, zoom_column=False) -> list[dict]:
    """
    :param rows: Output of :func:`select`
    :param records: Output of :func:`observation_records`
    :param zoom_column: Whether to add a fourth column that zooms on the grown barbs
    :return: What each row shows
    """
    # TrueType, as ACM rejects the Type 3 fonts matplotlib embeds by default
    plt.rcParams.update({"font.size": 6, "pdf.fonttype": 42})
    columns = 4 if zoom_column else 3
    figure, axes = plt.subplots(len(rows), columns, figsize=(3.33, 2.75 if zoom_column else 3.45), squeeze=False)
    shown = []
    for row, (label, name) in zip(axes, rows):
        image = load_image(observation_path(TRAIN_IMAGES, name)).numpy()[0]
        labels, n_labels = refined_labels(name)
        grown = grow_barbs(labels, n_labels, image, DEFAULT_BARBS)
        annotations = [annotation_mask(record) for record in records[name]]
        window = densest_window(np.logical_or.reduce([*annotations, grown > 0]))
        crop = image[window]
        low, high = np.percentile(crop[crop > 0.04], (0.5, 99.5))
        every_barb = (grown > 0) & (labels == 0)
        zoom = barb_window(every_barb) if every_barb.any() else window
        for axis in row[:3]:
            axis.imshow(crop, cmap="gray", vmin=low, vmax=high, interpolation="lanczos")
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_linewidth(0.4)
        for mask, colour in zip(annotations, ANNOTATOR_COLOURS):
            outline(row[1], mask[window], colour)
        body, added = labels[window] > 0, (grown[window] > 0) & (labels[window] == 0)
        for number in np.unique(labels[window]):
            if number:
                outline(row[2], labels[window] == number, MODEL_COLOUR)
        overlay(row[2], added, 1)
        if zoom_column:
            zoom_image = image[zoom]
            floor = zoom_image[zoom_image > 0.04]
            low, high = np.percentile(floor if floor.size else zoom_image, (0.5, 99.5))
            row[3].imshow(zoom_image, cmap="gray", vmin=low, vmax=high, interpolation="lanczos")
            for number in np.unique(labels[zoom]):
                if number:
                    outline(row[3], labels[zoom] == number, MODEL_COLOUR)
            overlay(row[3], every_barb[zoom], 0)
            row[3].set_xticks([])
            row[3].set_yticks([])
            for spine in row[3].spines.values():
                spine.set_linewidth(0.4)
        score = observation_pq(grown, n_labels, records[name])
        row[0].set_ylabel(f"{label}\nPQ {score:.2f}" if zoom_column else f"{label} (PQ {score:.2f})", fontsize=6, labelpad=1.5)
        shown.append(
            {
                "row": label,
                "file": name,
                "pq": score,
                "records": len(annotations),
                "body_px": int(body.sum()),
                "barb_px": int(added.sum()),
                "zoom_barb_px": int(every_barb[zoom].sum()),
            }
        )
    titles = ("H$\\alpha$ observation", "Annotators", "final", "barbs, 12$\\times$")[:columns]
    for axis, title in zip(axes[0], titles):
        axis.set_title(title, fontsize=6, pad=1.5)
    handles = [
        *[
            Line2D([], [], color=colour, lw=0.8, label=f"annotator {number}")
            for number, colour in enumerate(ANNOTATOR_COLOURS, 1)
        ],
        Line2D([], [], color=MODEL_COLOUR, lw=0.8, label="final instance"),
        Patch(color=BARB_COLOUR, label="grown barb"),
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        fontsize=5,
        frameon=False,
        handlelength=1.1,
        columnspacing=0.8,
        handletextpad=0.3,
    )
    figure.subplots_adjust(left=0.065, right=0.998, top=0.965, bottom=0.05, wspace=0.02, hspace=0.02)
    figure.savefig(OUTPUT, dpi=300)
    plt.close(figure)
    return shown


def main() -> None:
    """
    :return: None
    """
    barbs = "--barbs" in sys.argv[1:]
    files = [argument for argument in sys.argv[1:] if argument != "--barbs"]
    records = observation_records()
    if files:
        rows = [(date_label(name), name) for name in files]
    elif barbs:
        rows = [(date_label(name), name) for name in SHOWN]
    else:
        rows = select(records)
    print(json.dumps(draw(rows, records, zoom_column=barbs), indent=2))


if __name__ == "__main__":
    main()

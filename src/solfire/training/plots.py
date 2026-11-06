from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def plot_distribution(axes, matrices, title, x_label) -> None:
    """
    Histogram of the pairwise overlap scores between annotated and predicted filaments.

    Pairs that do not touch carry no information and would swamp the zero bin,
    so only positive scores are kept.
    :param axes: Axes to draw on
    :param matrices: Per-observation ``(n_gt, n_pred)`` score matrices
    :param title: Plot title
    :param x_label: Label of the x axis
    :return: None
    """
    scores = np.concatenate([np.asarray(matrix, dtype=float).ravel() for matrix in matrices] or [np.zeros(0)])
    scores = scores[scores > 0]

    counts, edges = np.histogram(scores, bins=np.linspace(0.0, 1.0, 51))
    probabilities = counts / counts.sum() if counts.sum() > 0 else counts
    centers = (edges[:-1] + edges[1:]) / 2.0

    axes.bar(centers, probabilities, width=0.02, edgecolor="black", alpha=0.7, color="navy")
    axes.grid(axis="y", linestyle="--")

    if scores.size > 0:
        axes.axvline(scores.mean(), color="crimson", linestyle="--", linewidth=1.5, label=f"Mean = {scores.mean():.3f}")
        axes.legend(fontsize=8)
    axes.set_xlabel(x_label)
    axes.set_ylabel("Probability")
    axes.set_title(title)


def plot_matching_cases(axes, matrices, title="Many to One and One to Many Distribution") -> None:
    """
    Frequency of annotations and predictions that touch more than one partner.

    Annotations are counted to the right of the axis and predictions to the
    left, so fragmentation and over-merging can be read off directly.
    :param axes: Axes to draw on
    :param matrices: Per-observation ``(n_gt, n_pred)`` IoU matrices
    :param title: Plot title
    :return: None
    """
    offset = 0.5
    gt_match_counts: list[int] = []
    pred_match_counts: list[int] = []

    for iou in matrices:
        hits = iou > 0
        gt_match_counts.extend(hits.sum(dim=1).tolist())
        pred_match_counts.extend(hits.sum(dim=0).tolist())

    gt_degrees = Counter(gt_match_counts)
    pred_degrees = Counter(pred_match_counts)
    gt_misses = gt_degrees.get(0, 0)
    gt_hits = sum(count for degree, count in gt_degrees.items() if degree > 0)
    pred_misses = pred_degrees.get(0, 0)
    pred_hits = sum(count for degree, count in pred_degrees.items() if degree > 0)

    colors = {"gt_miss": "crimson", "gt_hit": "forestgreen", "pred_miss": "darkorange", "pred_hit": "steelblue"}
    positions = []

    for degree, count in sorted(gt_degrees.items()):
        color = colors["gt_miss"] if degree == 0 else colors["gt_hit"]
        axes.bar(degree + offset, count, width=0.8, color=color, edgecolor="black", alpha=0.7)
        positions.append(degree + offset)

    for degree, count in sorted(pred_degrees.items()):
        color = colors["pred_miss"] if degree == 0 else colors["pred_hit"]
        axes.bar(-(degree + offset), count, width=0.8, color=color, edgecolor="black", alpha=0.7)
        positions.append(-(degree + offset))

    axes.axvline(0, color="black", linewidth=0.8)
    axes.grid(axis="y", linestyle="--")
    axes.set_xlabel("Matching Case")
    axes.set_ylabel("Frequency")
    axes.set_title(title)

    labels = [f"{round(abs(p) - offset)}:1" if p < 0 else f"1:{round(p - offset)}" for p in positions]
    axes.set_xticks(positions, labels, rotation=45, fontsize=9)
    axes.set_xlim(-max([abs(p) for p in positions], default=offset) - 0.8, max(positions, default=offset) + 0.8)

    axes.legend(
        handles=[
            Patch(facecolor=colors["gt_hit"], edgecolor="black", alpha=0.7, label=f"GT Hits (1:n, n>0): {gt_hits}"),
            Patch(facecolor=colors["gt_miss"], edgecolor="black", alpha=0.7, label=f"GT Misses (1:0): {gt_misses}"),
            Patch(facecolor=colors["pred_miss"], edgecolor="black", alpha=0.7, label=f"Prediction Misses (0:1): {pred_misses}"),
            Patch(facecolor=colors["pred_hit"], edgecolor="black", alpha=0.7, label=f"Prediction Hits (n:1, n>0): {pred_hits}"),
        ],
        fontsize=9,
    )


def plot_evaluation(overlaps, path) -> None:
    """
    Write the IoU, Dice and matching-case figures of one validation pass.
    :param overlaps: Per-observation ``(iou, dice)`` matrices, as returned in the metrics
    :param path: Destination image file
    :return: None
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    plot_distribution(axes[0], [iou for iou, _ in overlaps], "IoU Distribution", "IoU")
    plot_distribution(axes[1], [dice for _, dice in overlaps], "Dice Distribution", "Dice")
    plot_matching_cases(axes[2], [iou for iou, _ in overlaps])
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)

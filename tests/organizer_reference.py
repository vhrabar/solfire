"""
The organizers' Panoptic Quality, kept as close to their code as possible.

Ported from the competition's ``Self_Evaluation_Notebook``
(https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook)
"""

import torch


def get_overlap_matrices(gt_layers: torch.Tensor, pred_layers: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pairwise IoU and Dice between ground-truth and predicted layers of one image.
    :param gt_layers: ``(n_gt, H, W)`` binary layers
    :param pred_layers: ``(n_pred, H, W)`` binary layers
    :return: Tuple of ``(n_gt, n_pred)`` IoU and Dice matrices
    """
    n_gt, height, width = gt_layers.shape
    n_pred = pred_layers.shape[0]
    gt_flat = gt_layers.reshape(n_gt, height * width)
    pred_flat = pred_layers.reshape(n_pred, height * width)
    intersection = torch.matmul(gt_flat, pred_flat.t())
    gt_areas = gt_flat.sum(dim=1).view(-1, 1)
    pred_areas = pred_flat.sum(dim=1).view(1, -1)
    union = gt_areas + pred_areas - intersection
    iou_matrix = torch.where(union == 0, torch.tensor(0.0), intersection / union)
    dice_matrix = torch.where(union == 0, torch.tensor(0.0), 2 * intersection / (gt_areas + pred_areas))
    return iou_matrix, dice_matrix


def get_pq_score(entries: list[tuple[torch.Tensor, int, int]]) -> float:
    """
    Panoptic Quality over all annotator-image entries.
    :param entries: One ``(iou_matrix, n_gt, n_pred)`` per annotator record
    :return: PQ in ``[0, 1]``
    """
    iou_threshold = 0.5
    tp_iou_scores: list[float] = []
    fp_count = 0
    fn_count = 0
    for iou_matrix, n_gt, n_pred in entries:
        if n_gt == 0:
            fp_count += n_pred
            continue
        if n_pred == 0:
            fn_count += n_gt
            continue
        hit_matrix = iou_matrix > iou_threshold
        tp_iou_scores.extend(iou_matrix[hit_matrix].tolist())
        fp_count += int((hit_matrix.sum(dim=0) == 0).sum().item())
        fn_count += int((hit_matrix.sum(dim=1) == 0).sum().item())
    denominator = len(tp_iou_scores) + 0.5 * fp_count + 0.5 * fn_count
    return sum(tp_iou_scores) / denominator if denominator > 0 else 0.0

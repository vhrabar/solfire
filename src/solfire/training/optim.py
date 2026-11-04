"""
Optimiser and learning rate schedule construction.
"""

import torch
from torch import nn


def build_optimizer(model, lr=3e-4, weight_decay=1e-2, encoder_lr_scale=1.0, head_lr_scale=1.0) -> torch.optim.Optimizer:
    """
    Build AdamW with decay applied only to weight matrices.

    :param model: Model whose parameters are optimised
    :param lr: Peak learning rate
    :param weight_decay: Decay applied to the convolution and projection weights
    :param encoder_lr_scale: Learning rate of ``model.encoder`` as a fraction of ``lr``;
        ignored at ``1``
    :param head_lr_scale: Learning rate of ``model.net.segmentation_head`` as a multiple of
        ``lr``, so a freshly added output channel can be learnt in a short fine-tune; ignored at ``1``
    :return: Configured optimiser
    """
    scales: dict[int, float] = {}
    if encoder_lr_scale != 1.0:
        scales |= {id(param): encoder_lr_scale for param in model.encoder.parameters()}
    if head_lr_scale != 1.0:
        scales |= {id(param): head_lr_scale for param in model.net.segmentation_head.parameters()}

    groups: dict[tuple[bool, float], list[nn.Parameter]] = {}
    norms = nn.modules.batchnorm._BatchNorm | nn.LayerNorm | nn.GroupNorm
    for module in model.modules():
        for name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            decays = not (name == "bias" or isinstance(module, norms))
            groups.setdefault((decays, scales.get(id(param), 1.0)), []).append(param)

    return torch.optim.AdamW(
        [
            {"params": params, "weight_decay": weight_decay if decays else 0.0, "lr": lr * scale}
            for (decays, scale), params in groups.items()
        ],
        lr=lr,
        betas=(0.9, 0.999),
    )


def build_scheduler(optimizer, epochs, steps_per_epoch, warmup_epochs=1, min_lr_factor=0.01):
    """
    Linear warmup followed by cosine decay, stepped once per optimiser step.

    :param optimizer: Optimiser to schedule
    :param epochs: Total number of epochs
    :param steps_per_epoch: Optimiser steps per epoch
    :param warmup_epochs: Epochs spent ramping the learning rate up from zero
    :param min_lr_factor: Fraction of the peak learning rate reached at the end
    :return: Per-step learning rate scheduler
    """

    total_steps = max(2, epochs * steps_per_epoch)
    warmup_steps = min(total_steps - 1, max(1, warmup_epochs * steps_per_epoch))
    peak_lr = max(group["lr"] for group in optimizer.param_groups)

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0 / warmup_steps,
        end_factor=1.0,
        total_iters=warmup_steps - 1,
    )
    decay = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=min_lr_factor * peak_lr,
    )
    return torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, decay], milestones=[warmup_steps])

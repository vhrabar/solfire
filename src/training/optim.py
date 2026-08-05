"""
Optimiser and learning rate schedule construction.
"""

import torch
from torch import nn


def build_optimizer(model, lr=3e-4, weight_decay=1e-2) -> torch.optim.Optimizer:
    """
    Build AdamW with decay applied only to weight matrices.

    :param model: Model whose parameters are optimised
    :param lr: Peak learning rate
    :param weight_decay: Decay applied to the convolution and projection weights
    :return: Configured optimiser
    """
    decay = []
    no_decay = []
    norms = nn.modules.batchnorm._BatchNorm | nn.LayerNorm | nn.GroupNorm
    for module in model.modules():
        for name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            if name == "bias" or isinstance(module, norms):
                no_decay.append(param)
            else:
                decay.append(param)

    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
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

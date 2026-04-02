from __future__ import annotations

from typing import Literal, Optional

import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import LRScheduler

SchedulerName = Literal["none", "step", "cosine"]


def build_optimizer(
    model: nn.Module,
    lr: float,
    *,
    weight_decay: float,
    use_l2_via_weight_decay: bool,
) -> optim.SGD:
    """SGD with optional L2 via weight decay (when order == 2 in the original script)."""
    if use_l2_via_weight_decay:
        return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optim.SGD(model.parameters(), lr=lr)


def build_lr_scheduler(
    optimizer: optim.Optimizer,
    name: SchedulerName,
    *,
    epochs: int,
    step_size: int = 30,
    gamma: float = 0.1,
) -> Optional[LRScheduler]:
    """Optional learning-rate schedule (constant LR when name is ``"none"``)."""
    if name == "none":
        return None
    if name == "step":
        return optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs)
        )
    raise ValueError(f"unknown scheduler {name!r}")


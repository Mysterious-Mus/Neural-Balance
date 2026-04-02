from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class TrainingHistory:
    train_loss: List[float] = field(default_factory=list)
    test_loss: List[float] = field(default_factory=list)
    test_acc: List[float] = field(default_factory=list)


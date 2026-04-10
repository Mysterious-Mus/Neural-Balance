from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class TrainingHistory:
    train_loss: List[float] = field(default_factory=list)
    """Mean training loss over batches (train mode)."""
    train_eval_loss: List[float] = field(default_factory=list)
    """Cross-entropy on full training set in eval mode (comparable to test loss)."""
    train_acc: List[float] = field(default_factory=list)
    """Accuracy on training set in eval mode (percentage)."""
    test_loss: List[float] = field(default_factory=list)
    test_acc: List[float] = field(default_factory=list)


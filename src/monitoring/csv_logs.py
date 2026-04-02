from __future__ import annotations

import csv
from pathlib import Path

from src.monitoring.history import TrainingHistory


def write_history_csv(history: TrainingHistory, path: Path) -> None:
    """Write one row per epoch: epoch, test_accuracy, train_loss, test_loss."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["", "epoch", "test_accuracy", "train_loss", "test_loss"])
        for epoch, (tl, vl, va) in enumerate(
            zip(history.train_loss, history.test_loss, history.test_acc)
        ):
            w.writerow([epoch, epoch, va, tl, vl])


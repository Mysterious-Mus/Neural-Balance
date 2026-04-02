from __future__ import annotations

from dataclasses import dataclass

from src.monitoring.history import TrainingHistory


@dataclass
class TrainingDynamicsMonitor:
    """Collects per-epoch metrics and optional console logging."""

    history: TrainingHistory

    def on_epoch_end(
        self,
        epoch: int,
        num_epochs: int,
        train_loss: float,
        test_loss: float,
        test_acc: float,
        *,
        verbose: bool = True,
    ) -> None:
        self.history.train_loss.append(train_loss)
        self.history.test_loss.append(test_loss)
        self.history.test_acc.append(test_acc)
        if verbose:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], train loss: {train_loss:.4f} | "
                f"test acc: {test_acc:.2f}% | test loss: {test_loss:.4f}"
            )


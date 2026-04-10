from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Optional

from src.monitoring.history import TrainingHistory


def write_history_csv(history: TrainingHistory, path: Path) -> None:
    """Write one row per epoch with metrics aligned to the evaluation table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "epoch",
                "train_accuracy",
                "train_loss",
                "train_eval_loss",
                "test_accuracy",
                "test_loss",
            ]
        )
        for epoch, row in enumerate(
            zip(
                history.train_loss,
                history.train_eval_loss,
                history.train_acc,
                history.test_loss,
                history.test_acc,
            )
        ):
            tl, tel, ta, vl, va = row
            w.writerow([epoch, ta, tl, tel, va, vl])


def _first_epoch_1indexed_reaching(test_accs: list[float], tau: float) -> Optional[int]:
    """1-based epoch index when test accuracy first reaches tau (matches human epoch count)."""
    for i, a in enumerate(test_accs):
        if a >= tau:
            return i + 1
    return None


def write_run_summary(
    history: TrainingHistory,
    path: Path | str,
    *,
    target_tau_test_acc_pct: float,
    seed: int,
) -> None:
    """Write a JSON summary with table-aligned fields (single run; σ_test is null)."""
    if not history.test_acc:
        return

    path = Path(path)

    final_te = history.train_eval_loss[-1]
    final_ta = history.train_acc[-1]
    final_tl = history.train_loss[-1]
    final_va = history.test_acc[-1]
    final_vl = history.test_loss[-1]
    best_va = max(history.test_acc)
    gap = final_ta - final_va
    epochs_tau = _first_epoch_1indexed_reaching(history.test_acc, target_tau_test_acc_pct)

    payload: dict[str, Any] = {
        "target_tau_test_acc_pct": target_tau_test_acc_pct,
        "epochs_at_tau": epochs_tau,
        "final_train_acc_pct": final_ta,
        "final_train_loss_batch_mean": final_tl,
        "final_train_loss_eval": final_te,
        "final_test_acc_pct": final_va,
        "final_test_loss": final_vl,
        "best_test_acc_pct": best_va,
        "generalization_gap_train_minus_test_ppts": gap,
        "sigma_final_test_acc_pct_across_seeds": None,
        "seed": seed,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def write_run_summary_from_history_csv(
    history_csv_path: Path | str,
    summary_json_path: Path | str,
    *,
    target_tau_test_acc_pct: float,
    seed: int,
) -> None:
    """Rebuild summary.json from an existing per-epoch history CSV."""
    history_csv_path = Path(history_csv_path)
    summary_json_path = Path(summary_json_path)

    test_acc: list[float] = []
    train_acc: list[float] = []
    train_loss: list[float] = []
    train_eval_loss: list[float] = []
    test_loss: list[float] = []

    with history_csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            train_acc.append(float(row["train_accuracy"]))
            train_loss.append(float(row["train_loss"]))
            train_eval_loss.append(float(row["train_eval_loss"]))
            test_acc.append(float(row["test_accuracy"]))
            test_loss.append(float(row["test_loss"]))

    if not test_acc:
        raise ValueError(f"no rows found in history CSV: {history_csv_path}")

    epochs_tau = _first_epoch_1indexed_reaching(test_acc, target_tau_test_acc_pct)
    payload: dict[str, Any] = {
        "target_tau_test_acc_pct": target_tau_test_acc_pct,
        "epochs_at_tau": epochs_tau,
        "final_train_acc_pct": train_acc[-1],
        "final_train_loss_batch_mean": train_loss[-1],
        "final_train_loss_eval": train_eval_loss[-1],
        "final_test_acc_pct": test_acc[-1],
        "final_test_loss": test_loss[-1],
        "best_test_acc_pct": max(test_acc),
        "generalization_gap_train_minus_test_ppts": train_acc[-1] - test_acc[-1],
        "sigma_final_test_acc_pct_across_seeds": None,
        "seed": seed,
    }

    summary_json_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

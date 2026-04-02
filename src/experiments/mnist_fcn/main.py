from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]

# Allow running this file directly (`python src/experiments/.../main.py ...`).
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from torch import nn

from src.datasets.mnist import get_train_and_test_loaders
from src.models.fcn import build_fcn
from src.monitoring.csv_logs import write_history_csv
from src.monitoring.dynamics import TrainingDynamicsMonitor
from src.monitoring.history import TrainingHistory
from src.training.loop import apply_neural_balance, evaluate, full_balance_at_start, train_epoch
from src.training.schedulers import build_lr_scheduler, build_optimizer
from src.utils.repro import set_seed

_DEFAULT_DATA_ROOT = _REPO_ROOT / "data"
_DEFAULT_HIST_ROOT = _REPO_ROOT / "MNIST-FCN" / "hist"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MNIST fully-connected training")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--model",
        type=str,
        default="small_fcn",
        choices=["small_fcn", "medium_fcn", "large_fcn", "xlarge_fcn"],
    )
    p.add_argument("--dataset", type=str, default="mnist", choices=["mnist"])
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--batchsize", type=int, default=256)
    p.add_argument("--l2_weight", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--neural_balance", type=int, default=0)
    p.add_argument("--neural_balance_epoch", type=int, default=1)
    p.add_argument("--order", type=int, default=2)
    p.add_argument("--neuralFullBalanceAtStart", type=int, default=0)
    p.add_argument("--trainDataFrac", type=float, default=1.0)
    p.add_argument("--reversed", type=int, default=0)

    p.add_argument(
        "--data_root",
        type=str,
        default=str(_DEFAULT_DATA_ROOT),
        help="MNIST download directory (default: <repo>/data)",
    )
    p.add_argument(
        "--hist_root",
        type=str,
        default=str(_DEFAULT_HIST_ROOT),
        help="root directory for CSV logs (default: MNIST-FCN/hist)",
    )
    p.add_argument(
        "--foldername",
        type=str,
        default=None,
        help="subfolder under hist_root for CSV output",
    )
    p.add_argument(
        "--filename",
        type=str,
        default=None,
        help="CSV basename without extension",
    )

    p.add_argument(
        "--activation",
        type=str,
        default="relu",
        choices=["relu", "tanh"],
        help="hidden activation (tanh matches legacy Mnist_tanh)",
    )
    p.add_argument(
        "--tanh_on_output",
        type=int,
        default=0,
        help="if 1, apply tanh to the final logits as well",
    )

    p.add_argument(
        "--lr_scheduler",
        type=str,
        default="none",
        choices=["none", "step", "cosine"],
        help="optional PyTorch LR schedule (default: constant LR)",
    )
    p.add_argument("--scheduler_step_size", type=int, default=30)
    p.add_argument("--scheduler_gamma", type=float, default=0.1)

    return p


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def run_training(args: argparse.Namespace) -> TrainingHistory:
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    trainloader, testloader = get_train_and_test_loaders(
        args.batchsize,
        args.seed,
        train_fraction=args.trainDataFrac,
        data_root=args.data_root,
    )

    model = build_fcn(
        args.model,
        activation=args.activation,
        tanh_on_output=bool(args.tanh_on_output),
    ).to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()

    # Legacy mapping:
    # - order==2 + l2_weight>0 used optimizer weight_decay (L2 via SGD weight decay).
    # - order==1 + l2_weight>0 used L1-like term in the training loop.
    use_wd = args.order == 2 and args.l2_weight > 0
    if args.l2_weight > 0:
        print(
            f"regularization weight = {args.l2_weight} "
            f"({'WD (order==2)' if use_wd else 'L1 term (order==1)'})"
        )

    optimizer = build_optimizer(
        model,
        args.lr,
        weight_decay=args.l2_weight,
        use_l2_via_weight_decay=use_wd,
    )
    scheduler = build_lr_scheduler(
        optimizer,
        args.lr_scheduler,
        epochs=args.epochs,
        step_size=args.scheduler_step_size,
        gamma=args.scheduler_gamma,
    )

    reversed_layers = args.reversed != 0
    if args.neuralFullBalanceAtStart == 1:
        full_balance_at_start(model, args.order, reversed_layers)

    history = TrainingHistory()
    monitor = TrainingDynamicsMonitor(history)

    for epoch in range(args.epochs):
        train_loss = train_epoch(
            model,
            trainloader,
            criterion,
            optimizer,
            device,
            l2_weight=args.l2_weight,
            order=args.order,
        )

        if args.neural_balance == 1 and epoch % args.neural_balance_epoch == 0:
            apply_neural_balance(model, args.order, reversed_layers)

        test_loss, test_acc = evaluate(model, testloader, criterion, device)
        monitor.on_epoch_end(
            epoch,
            args.epochs,
            train_loss,
            test_loss,
            test_acc,
            verbose=False,
        )

        print(f"Epoch [{epoch + 1}/{args.epochs}], Loss: {train_loss:.4f}")
        print(f"Accuracy of the network on the 10000 test images: {test_acc:.2f}%")
        print(f"Loss of the network on the 10000 test images: {test_loss:.4f}")

        if scheduler is not None:
            scheduler.step()

    if args.foldername is not None and args.filename is not None:
        out = Path(args.hist_root) / args.foldername / f"{args.filename}.csv"
        write_history_csv(history, out)
        print(f"wrote {out}")

    return history


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()


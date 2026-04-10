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
from src.monitoring.csv_logs import write_history_csv, write_run_summary
from src.monitoring.dynamics import TrainingDynamicsMonitor
from src.monitoring.history import TrainingHistory
from src.training.loop import apply_neural_balance, evaluate, full_balance_at_start, train_epoch
from src.training.schedulers import build_lr_scheduler, build_optimizer
from src.utils.parsing import str2bool
from src.utils.repro import set_seed

_DEFAULT_DATA_ROOT = _REPO_ROOT / "data"


def _coerce_bool_flags(ns: argparse.Namespace) -> None:
    """Normalize bool-like values (YAML may pass 0/1)."""
    for name in (
        "do_neural_balance",
        "full_balance_at_start",
        "reverse_balance_layer_order",
        "tanh_on_output",
    ):
        setattr(ns, name, str2bool(getattr(ns, name)))


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
    p.add_argument(
        "--do-neural-balance",
        dest="do_neural_balance",
        type=str2bool,
        default=False,
        help="if true, apply periodic neural balance during training",
    )
    p.add_argument(
        "--neural-balance-epoch",
        dest="neural_balance_epoch",
        type=int,
        default=1,
        help="when do_neural_balance is true, balance every this many epochs (ignored otherwise)",
    )
    p.add_argument("--order", type=int, default=2)
    p.add_argument(
        "--full-balance-at-start",
        dest="full_balance_at_start",
        type=str2bool,
        default=False,
        help="if true, run full neural balance once before training",
    )
    p.add_argument("--trainDataFrac", type=float, default=1.0)
    p.add_argument(
        "--reverse-balance-layer-order",
        dest="reverse_balance_layer_order",
        type=str2bool,
        default=False,
        help="if true, traverse layers from output toward input when balancing",
    )

    p.add_argument(
        "--data_root",
        type=str,
        default=str(_DEFAULT_DATA_ROOT),
        help="MNIST download directory (default: <repo>/data)",
    )
    p.add_argument(
        "--metrics-csv",
        dest="metrics_csv",
        type=str,
        default=None,
        help="path to write per-epoch metrics CSV (omit to skip writing)",
    )
    p.add_argument(
        "--summary-json",
        dest="summary_json",
        type=str,
        default=None,
        help="path for table-aligned run summary JSON; default: <metrics_csv_dir>/summary.json",
    )
    p.add_argument(
        "--target-tau",
        dest="target_tau",
        type=float,
        default=95.0,
        help="target test accuracy (%%) τ for Epochs@τ in summary.json (first epoch reaching ≥τ)",
    )

    p.add_argument(
        "--activation",
        type=str,
        default="relu",
        choices=["relu", "tanh"],
        help="hidden activation (tanh matches legacy Mnist_tanh)",
    )
    p.add_argument(
        "--tanh-on-output",
        dest="tanh_on_output",
        type=str2bool,
        default=False,
        help="if true, apply tanh to the final logits as well",
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
    _coerce_bool_flags(args)
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
        tanh_on_output=args.tanh_on_output,
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

    reversed_layers = args.reverse_balance_layer_order
    if args.full_balance_at_start:
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

        if args.do_neural_balance and epoch % args.neural_balance_epoch == 0:
            apply_neural_balance(model, args.order, reversed_layers)

        train_eval_loss, train_acc = evaluate(model, trainloader, criterion, device)
        test_loss, test_acc = evaluate(model, testloader, criterion, device)
        monitor.on_epoch_end(
            epoch,
            args.epochs,
            train_loss,
            train_eval_loss,
            train_acc,
            test_loss,
            test_acc,
            verbose=False,
        )

        print(f"Epoch [{epoch + 1}/{args.epochs}], Loss: {train_loss:.4f}")
        print(f"Accuracy of the network on the 10000 test images: {test_acc:.2f}%")
        print(f"Loss of the network on the 10000 test images: {test_loss:.4f}")

        if scheduler is not None:
            scheduler.step()

    if args.metrics_csv is not None:
        out = Path(args.metrics_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        write_history_csv(history, out)
        print(f"wrote {out}")

        summary_path = (
            Path(args.summary_json)
            if args.summary_json is not None
            else out.parent / "summary.json"
        )
        write_run_summary(
            history,
            summary_path,
            target_tau_test_acc_pct=args.target_tau,
            seed=args.seed,
        )
        print(f"wrote {summary_path}")

    return history


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()


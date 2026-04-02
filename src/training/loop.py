from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.balance import neural_balance
from src.models.fcn import FCN


def full_balance_at_start(
    model: FCN, order: int, reversed_layers: bool
) -> None:
    """Iteratively balance adjacent linear layers until scale metrics stabilize."""
    print("full balancing at start")
    lay = list(model.layers)
    while True:
        restart = False
        indices = range(len(lay) - 1)
        if reversed_layers:
            indices = reversed(range(len(lay) - 1))
        for i in indices:
            lay1, lay2 = lay[i], lay[i + 1]
            incoming = torch.linalg.norm(lay1.weight, dim=1, ord=order)
            outgoing = torch.linalg.norm(lay2.weight, dim=0, ord=order)
            optimal_l = torch.sqrt(outgoing / incoming).sum() / incoming.shape[0]
            print(optimal_l)
            if optimal_l > 1.001 or optimal_l < 0.999:
                restart = True
            neural_balance(lay1, lay2, order=order)
        if not restart:
            break


def apply_neural_balance(
    model: FCN, order: int, reversed_layers: bool
) -> None:
    print()
    print("performing neural balance")
    print()
    lay = list(model.layers)
    indices = range(len(lay) - 1)
    if reversed_layers:
        indices = reversed(range(len(lay) - 1))
    for i in indices:
        neural_balance(lay[i], lay[i + 1], order=order)


def train_epoch(
    model: FCN,
    trainloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    l2_weight: float,
    order: int,
) -> float:
    model.train()
    train_loss_sum = 0.0
    for images, labels in tqdm(trainloader):
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        # Legacy behavior: if order==1, they used an L1-like regularizer via torch.norm(..., 1).
        if l2_weight > 0 and order == 1:
            params = torch.cat([x.view(-1) for x in model.parameters()])
            loss = loss + l2_weight * torch.norm(params, 1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item()

    return train_loss_sum / len(trainloader)


@torch.no_grad()
def evaluate(
    model: FCN,
    testloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    for images, labels in testloader:
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        test_loss += criterion(outputs, labels).item()

    avg_loss = test_loss / len(testloader)
    acc_pct = 100.0 * correct / total
    return avg_loss, acc_pct


import torch
import torch.nn as nn


def neural_balance(
    in_layer: nn.Linear, out_layer: nn.Linear, *, order: int
) -> None:
    incoming = torch.linalg.norm(in_layer.weight, dim=1, ord=order)
    outgoing = torch.linalg.norm(out_layer.weight, dim=0, ord=order)
    optimal_l = torch.sqrt(outgoing / incoming)
    in_layer.weight.data *= optimal_l.unsqueeze(1)
    out_layer.weight.data /= optimal_l


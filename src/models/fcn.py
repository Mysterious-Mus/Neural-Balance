from __future__ import annotations

from typing import Dict, List, Literal, Sequence

import torch
import torch.nn as nn

ActivationName = Literal["relu", "tanh"]

# Hidden widths for each preset name (input 784, output 10).
_PRESETS: Dict[str, List[int]] = {
    "small_fcn": [256],
    "medium_fcn": [256, 128],
    "large_fcn": [512, 256, 128, 64],
    "xlarge_fcn": [4096, 2048, 1024, 512, 256, 128, 64, 32, 16],
}


def list_model_names() -> List[str]:
    return list(_PRESETS.keys())


class FCN(nn.Module):
    """Fully-connected network for flattened MNIST (784 -> ... -> num_classes)."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        num_classes: int,
        *,
        activation: ActivationName = "relu",
        tanh_on_output: bool = False,
    ) -> None:
        super().__init__()
        if activation == "relu":
            self._hidden_act = nn.ReLU()
        else:
            self._hidden_act = nn.Tanh()
        self._tanh_on_output = tanh_on_output

        layers: List[nn.Module] = [nn.Linear(input_size, hidden_sizes[0])]
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
        layers.append(nn.Linear(hidden_sizes[-1], num_classes))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = self._hidden_act(layer(x))
        x = self.layers[-1](x)
        # Legacy `Mnist_tanh.py` applied tanh to the logits as well.
        if self._tanh_on_output:
            x = self._hidden_act(x)
        return x


def build_fcn(
    name: str, *, activation: ActivationName = "relu", tanh_on_output: bool = False
) -> FCN:
    if name not in _PRESETS:
        raise ValueError(
            f"unknown model {name!r}; choose one of {list(_PRESETS.keys())}"
        )
    hidden = _PRESETS[name]
    return FCN(
        784, hidden, 10, activation=activation, tanh_on_output=tanh_on_output
    )


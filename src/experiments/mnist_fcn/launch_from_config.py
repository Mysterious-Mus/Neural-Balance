"""Launch MNIST FCN training from a YAML config next to experiment logs.

Run from the repository root::

    python -m src.experiments.mnist_fcn.launch_from_config path/to/experiment_dir

Or from inside the experiment directory (must contain ``config.yaml``)::

    cd path/to/experiment_dir
    python -m src.experiments.mnist_fcn.launch_from_config .

Paths in the config file that are not absolute are resolved relative to the
experiment directory (except ``data_root`` defaults to ``<repo>/data`` when omitted).
If ``metrics_csv`` is omitted, metrics are written to ``<experiment_dir>/logs/train.csv``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping

_REPO_ROOT = Path(__file__).resolve().parents[3]

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import yaml
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "launch_from_config requires PyYAML. Install with: pip install pyyaml"
    ) from e

from src.experiments.mnist_fcn.main import _coerce_bool_flags, build_parser, run_training


# YAML keys may use snake_case aliases; values map onto argparse ``dest`` names.
_YAML_ALIASES: dict[str, str] = {
    "train_data_frac": "trainDataFrac",
    "neural_full_balance_at_start": "full_balance_at_start",
    "neuralFullBalanceAtStart": "full_balance_at_start",
    "do_neural_balance": "do_neural_balance",
    "neural_balance": "do_neural_balance",
    "neural_balance_epoch": "neural_balance_epoch",
    "reverse_balance_layer_order": "reverse_balance_layer_order",
    "reversed": "reverse_balance_layer_order",
    "tanh_on_output": "tanh_on_output",
    "lr_scheduler": "lr_scheduler",
    "scheduler_step_size": "scheduler_step_size",
    "scheduler_gamma": "scheduler_gamma",
    "l2_weight": "l2_weight",
    "data_root": "data_root",
    "metrics_csv": "metrics_csv",
    "summary_json": "summary_json",
    "target_tau": "target_tau",
    # Backward compatibility with older configs:
    "convergence_tau": "target_tau",
}


def _parser_destinations(parser: argparse.ArgumentParser) -> set[str]:
    out: set[str] = set()
    for action in parser._actions:
        if action.dest and action.dest != "help":
            out.add(action.dest)
    return out


def _normalize_config(raw: Mapping[str, Any], allowed: set[str]) -> dict[str, Any]:
    """Map YAML keys to argparse dest names and drop unknown keys."""
    normalized: dict[str, Any] = {}
    for key, value in raw.items():
        if key.startswith("_"):
            continue
        dest = _YAML_ALIASES.get(key, key)
        if dest not in allowed:
            raise ValueError(
                f"Unknown config key {key!r} (resolved to {dest!r}). "
                f"Allowed keys match argparse destinations; see --help on main.py."
            )
        normalized[dest] = value
    return normalized


def _resolve_paths(
    experiment_dir: Path,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Fill defaults for data/metrics paths and resolve relative paths."""
    out = dict(cfg)

    if out.get("data_root") is None:
        out["data_root"] = str(_REPO_ROOT / "data")

    if out.get("metrics_csv") is None:
        out["metrics_csv"] = str((experiment_dir / "logs" / "train.csv").resolve())

    for name in ("data_root", "metrics_csv"):
        val = out.get(name)
        if val is None:
            continue
        p = Path(str(val))
        if not p.is_absolute():
            out[name] = str((experiment_dir / p).resolve())

    return out


def load_config_and_args(
    experiment_dir: Path,
    config_name: str = "config.yaml",
) -> argparse.Namespace:
    experiment_dir = experiment_dir.resolve()
    config_path = experiment_dir / config_name
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config file: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise TypeError(f"Config root must be a mapping, got {type(raw)}")

    parser = build_parser()
    allowed = _parser_destinations(parser)
    normalized = _normalize_config(raw, allowed)
    merged = _resolve_paths(experiment_dir, normalized)

    parser.set_defaults(**merged)
    args = parser.parse_args([])
    _coerce_bool_flags(args)
    return args


def main() -> None:
    p = argparse.ArgumentParser(description="Run MNIST FCN training from a YAML config.")
    p.add_argument(
        "experiment_dir",
        nargs="?",
        default=Path("."),
        type=Path,
        help="Directory containing config.yaml and (by default) logs/ output",
    )
    p.add_argument(
        "--config",
        default="config.yaml",
        help="Config filename inside the experiment directory",
    )
    launch_args = p.parse_args()

    args = load_config_and_args(launch_args.experiment_dir, launch_args.config)
    run_training(args)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.experiments.mnist_fcn.launch_from_config import load_config_and_args
from src.monitoring.csv_logs import write_run_summary_from_history_csv


def _resolve_seed(summary_path: Path, fallback_seed: int) -> int:
    if not summary_path.is_file():
        return fallback_seed
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return fallback_seed
    seed = data.get("seed", fallback_seed)
    try:
        return int(seed)
    except Exception:
        return fallback_seed


def main() -> None:
    p = argparse.ArgumentParser(
        description="Refresh summary.json from existing logs/train.csv without retraining."
    )
    p.add_argument(
        "experiment_dir",
        nargs="?",
        default=".",
        help="Experiment directory containing config.yaml and logs/",
    )
    p.add_argument(
        "--config",
        default="config.yaml",
        help="Config filename inside experiment_dir",
    )
    p.add_argument(
        "--history-csv",
        default=None,
        help="Override history CSV path (default: from config metrics_csv / logs/train.csv)",
    )
    p.add_argument(
        "--summary-json",
        default=None,
        help="Override summary JSON path (default: from config summary_json / logs/summary.json)",
    )
    args = p.parse_args()

    experiment_dir = Path(args.experiment_dir).resolve()
    cfg = load_config_and_args(experiment_dir, args.config)

    history_csv = (
        Path(args.history_csv).resolve()
        if args.history_csv is not None
        else Path(cfg.metrics_csv).resolve()
    )
    summary_json = (
        Path(args.summary_json).resolve()
        if args.summary_json is not None
        else (
            Path(cfg.summary_json).resolve()
            if cfg.summary_json is not None
            else history_csv.parent / "summary.json"
        )
    )

    seed = _resolve_seed(summary_json, int(cfg.seed))
    write_run_summary_from_history_csv(
        history_csv,
        summary_json,
        target_tau_test_acc_pct=float(cfg.target_tau),
        seed=seed,
    )
    print(f"refreshed {summary_json} from {history_csv} (target_tau={cfg.target_tau})")


if __name__ == "__main__":
    main()

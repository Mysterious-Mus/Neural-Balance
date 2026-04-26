#!/usr/bin/env python3
"""Run remaining regularization BO experiments sequentially.

This script launches method-specific BO meta experiments one by one and writes a
manifest that records which config knobs were changed/tuned for each method.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Allow running this script directly from any cwd, e.g.:
#   python /workspaces/Neural-Balance/scripts/run_experiment.py
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.experiments.mnist_fcn.config_loading import load_resolved_config
from src.utils.paths import repo_root


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    experiment_dir: str
    config_path: str


METHODS: tuple[MethodSpec, ...] = (
    # MethodSpec(
    #     key="l1_penalty",
    #     label="L1 penalty",
    #     experiment_dir="experiments/mnist_fcn_regscomp/l1_reg_BO",
    #     config_path="configs/mnist_fcn_regscomp/meta_bo_l1.yaml",
    # ),
    # MethodSpec(
    #     key="dropout",
    #     label="Dropout",
    #     experiment_dir="experiments/mnist_fcn_regscomp/dropout_BO",
    #     config_path="configs/mnist_fcn_regscomp/meta_bo_dropout.yaml",
    # ),
    # MethodSpec(
    #     key="early_stopping",
    #     label="Early stopping",
    #     experiment_dir="experiments/mnist_fcn_regscomp/early_stopping_BO",
    #     config_path="configs/mnist_fcn_regscomp/meta_bo_early_stopping.yaml",
    # ),
    MethodSpec(
        key="synaptic_balance_l1",
        label="Synaptic balance (L1)",
        experiment_dir="experiments/mnist_fcn_regscomp/synaptic_balance_l1_BO",
        config_path="configs/mnist_fcn_regscomp/meta_bo_synaptic_balance_l1.yaml",
    ),
    MethodSpec(
        key="synaptic_balance_l2",
        label="Synaptic balance (L2)",
        experiment_dir="experiments/mnist_fcn_regscomp/synaptic_balance_l2_BO",
        config_path="configs/mnist_fcn_regscomp/meta_bo_synaptic_balance_l2.yaml",
    ),
)

# The plain baseline not include in this file now.

# Disabled by default to avoid re-running your ongoing/finished L2 BO job.
# Re-enable by uncommenting this block and setting L2_METHOD to MethodSpec(...).
# L2_METHOD = MethodSpec(
#     key="l2_weight_decay",
#     label="L2 weight decay",
#     experiment_dir="experiments/mnist_fcn_regscomp/l2_reg_BO",
#     config_path="configs/mnist_fcn_regscomp/meta_bo_l2.yaml",
# )
L2_METHOD: MethodSpec | None = None


def _load_yaml(path: Path) -> dict[str, Any]:
    cfg, _ = load_resolved_config(path)
    if not isinstance(cfg, dict):
        raise TypeError(f"resolved config must be a mapping: {path}")
    return dict(cfg)


def _extract_method_tracking(meta_cfg: dict[str, Any], plain_cfg: dict[str, Any]) -> dict[str, Any]:
    static_overrides = meta_cfg.get("static_overrides", {})
    if not isinstance(static_overrides, dict):
        static_overrides = {}

    search_space = meta_cfg.get("search_space", [])
    tuned_keys: list[str] = []
    if isinstance(search_space, list):
        for dim in search_space:
            if isinstance(dim, dict) and "name" in dim:
                tuned_keys.append(str(dim["name"]))

    changed_from_plain = {
        k: v
        for k, v in static_overrides.items()
        if plain_cfg.get(k) != v
    }

    return {
        "static_overrides": static_overrides,
        "tuned_keys": tuned_keys,
        "changed_from_plain": changed_from_plain,
        "bo": meta_cfg.get("bo", {}),
        "objective": meta_cfg.get("objective"),
        "seeds": meta_cfg.get("seeds", []),
        "validation_split_pct": meta_cfg.get("validation_split_pct"),
    }


def run(args: argparse.Namespace) -> int:
    root = repo_root()
    plain_cfg = _load_yaml(root / "configs" / "mnist_fcn_regscomp" / "plain.yaml")

    selected: list[MethodSpec] = list(METHODS)
    if args.include_l2:
        if L2_METHOD is None:
            print("--include-l2 requested, but L2_METHOD is disabled in this file; skipping L2.")
        else:
            selected = [L2_METHOD] + selected

    if args.methods:
        wanted = {x.strip() for x in args.methods.split(",") if x.strip()}
        selected = [m for m in selected if m.key in wanted]
        known_keys = {m.key for m in METHODS}
        if L2_METHOD is not None:
            known_keys.add(L2_METHOD.key)
        unknown = sorted(wanted - known_keys)
        if unknown:
            raise ValueError(f"unknown method keys: {unknown}")

    if not selected:
        print("No methods selected.")
        return 0

    manifest_path = root / "experiments" / "mnist_fcn_regscomp" / "run_experiment_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "started_at_unix": time.time(),
        "python_executable": args.python_executable,
        "dry_run": args.dry_run,
        "methods": [],
    }

    for method in selected:
        cfg_path = root / method.config_path
        meta_cfg = _load_yaml(cfg_path)
        tracking = _extract_method_tracking(meta_cfg, plain_cfg)

        cmd = [
            args.python_executable,
            "-m",
            "src.experiments.mnist_fcn.run_bo_meta_experiment",
            method.experiment_dir,
            "--config-path",
            method.config_path,
        ]

        print()
        print(f"=== {method.label} ({method.key}) ===")
        print("Command:", " ".join(cmd))

        method_record: dict[str, Any] = {
            "key": method.key,
            "label": method.label,
            "experiment_dir": method.experiment_dir,
            "config_path": method.config_path,
            "command": cmd,
            "config_tracking": tracking,
            "status": "pending",
        }

        t0 = time.time()
        if args.dry_run:
            method_record["status"] = "dry_run"
            method_record["duration_sec"] = 0.0
        else:
            result = subprocess.run(cmd, cwd=str(root))
            method_record["return_code"] = int(result.returncode)
            method_record["duration_sec"] = round(time.time() - t0, 2)
            if result.returncode == 0:
                method_record["status"] = "completed"
            else:
                method_record["status"] = "failed"
                manifest["methods"].append(method_record)
                manifest["finished_at_unix"] = time.time()
                manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
                print(f"Failed: {method.key} (rc={result.returncode})")
                if not args.continue_on_error:
                    print(f"Wrote manifest: {manifest_path}")
                    return int(result.returncode)

        manifest["methods"].append(method_record)
        manifest["finished_at_unix"] = time.time()
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print()
    print(f"Done. Wrote manifest: {manifest_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run remaining regularization BO experiments sequentially "
            "(L1, dropout, early stopping, synaptic balance L1/L2)."
        )
    )
    p.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python executable used to launch runs (default: current interpreter)",
    )
    p.add_argument(
        "--include-l2",
        action="store_true",
        help="Also run L2 BO (useful if you want to re-run l2_reg_BO in this batch)",
    )
    p.add_argument(
        "--methods",
        default="",
        help=(
            "Optional comma-separated method keys to run. "
            "Keys: l1_penalty, dropout, early_stopping, synaptic_balance_l1, "
            "synaptic_balance_l2, l2_weight_decay"
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and write manifest without launching training",
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with next method when one method fails",
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()

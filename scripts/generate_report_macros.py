#!/usr/bin/env python3
"""Generate LaTeX macros from configs/experiment_config.yaml and experiment outputs.

Writes REPORT/generated/report_values.tex.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: pyyaml. Install with `pip install pyyaml`."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "experiment_config.yaml"
OUTPUT_PATH = REPO_ROOT / "REPORT" / "generated" / "report_values.tex"
EXPERIMENTS_ROOT = REPO_ROOT / "experiments" / "mnist_fcn_regscomp"


class RawLatex(str):
    """Marker type for values that should be emitted without escaping."""


def latex_escape(value: str) -> str:
    """Very small LaTeX escaper for macro text values."""
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def macro_line(name: str, value: object) -> str:
    if isinstance(value, RawLatex):
        return f"\\newcommand{{\\{name}}}{{{value}}}"
    if isinstance(value, float):
        text = f"{value:g}"
    else:
        text = str(value)
    return f"\\newcommand{{\\{name}}}{{{latex_escape(text)}}}"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def format_metric(value: float | int | None, *, decimals: int = 2, integer: bool = False) -> object:
    if value is None:
        return RawLatex(r"\tbd")
    if integer:
        return str(int(round(float(value))))
    if decimals == 0:
        return str(int(round(float(value))))

    text = f"{float(value):.{decimals}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def format_epoch_metric(value: float | None) -> object:
    if value is None:
        return RawLatex(r"\tbd")
    rounded = round(float(value), 1)
    if abs(rounded - round(rounded)) < 1e-9:
        return str(int(round(rounded)))
    return f"{rounded:.1f}"


METHOD_SPECS = [
    {
        "prefix": "RegMnistPlain",
        "source": "plain",
        "summary_path": EXPERIMENTS_ROOT / "plain" / "logs" / "summary.json",
        "is_balance": False,
    },
    {
        "prefix": "RegMnistLTwo",
        "source": "bo",
        "bo_summary_path": EXPERIMENTS_ROOT / "l2_reg_BO" / "bo_summary.json",
        "is_balance": False,
    },
    {
        "prefix": "RegMnistLOne",
        "source": "bo",
        "bo_summary_path": EXPERIMENTS_ROOT / "l1_reg_BO" / "bo_summary.json",
        "is_balance": False,
    },
    {
        "prefix": "RegMnistDropout",
        "source": "bo",
        "bo_summary_path": EXPERIMENTS_ROOT / "dropout_BO" / "bo_summary.json",
        "is_balance": False,
    },
    {
        "prefix": "RegMnistEarlyStop",
        "source": "bo",
        "bo_summary_path": EXPERIMENTS_ROOT / "early_stopping_BO" / "bo_summary.json",
        "is_balance": False,
    },
    {
        "prefix": "RegMnistSynBalLOne",
        "source": "bo",
        "bo_summary_path": EXPERIMENTS_ROOT / "synaptic_balance_l1_BO" / "bo_summary.json",
        "is_balance": True,
    },
    {
        "prefix": "RegMnistSynBalLTwo",
        "source": "bo",
        "bo_summary_path": EXPERIMENTS_ROOT / "synaptic_balance_l2_BO" / "bo_summary.json",
        "is_balance": True,
    },
]


def summarize_plain(summary_path: Path) -> dict[str, float | int | None]:
    if not summary_path.exists():
        return {
            "epochs_at_tau": None,
            "final_train_acc_pct": None,
            "final_train_loss_batch_mean": None,
            "final_test_acc_pct": None,
            "final_test_loss": None,
            "best_val_acc_pct": None,
            "generalization_gap_train_minus_test_ppts": None,
            "sigma_final_test_acc_pct": None,
            "seed_count": 0,
        }

    payload = load_json(summary_path)
    return {
        "epochs_at_tau": payload.get("epochs_at_tau"),
        "final_train_acc_pct": payload.get("final_train_acc_pct"),
        "final_train_loss_batch_mean": payload.get("final_train_loss_batch_mean"),
        "final_test_acc_pct": payload.get("final_test_acc_pct"),
        "final_test_loss": payload.get("final_test_loss"),
        "best_val_acc_pct": payload.get("best_val_acc_pct", payload.get("best_test_acc_pct")),
        "generalization_gap_train_minus_test_ppts": payload.get("generalization_gap_train_minus_test_ppts"),
        "sigma_final_test_acc_pct": payload.get("sigma_final_test_acc_pct_across_seeds"),
        "seed_count": 1,
    }


def summarize_bo(bo_summary_path: Path) -> dict[str, float | int | None]:
    if not bo_summary_path.exists():
        return {
            "epochs_at_tau": None,
            "final_train_acc_pct": None,
            "final_train_loss_batch_mean": None,
            "final_test_acc_pct": None,
            "final_test_loss": None,
            "best_val_acc_pct": None,
            "generalization_gap_train_minus_test_ppts": None,
            "sigma_final_test_acc_pct": None,
            "seed_count": 0,
        }

    bo_summary = load_json(bo_summary_path)
    trial_name = bo_summary["best_trial"]["trial"]
    trial_summary_path = bo_summary_path.parent / "trials" / trial_name / "trial_summary.json"
    trial_summary = load_json(trial_summary_path)
    per_seed = trial_summary.get("per_seed_metrics", [])

    def values_for(key: str) -> list[float]:
        return [float(item[key]) for item in per_seed if item.get(key) is not None]

    return {
        "epochs_at_tau": mean(values_for("epochs_at_tau")),
        "final_train_acc_pct": mean(values_for("final_train_acc_pct")),
        "final_train_loss_batch_mean": mean(values_for("final_train_loss_batch_mean")),
        "final_test_acc_pct": mean(values_for("final_test_acc_pct")),
        "final_test_loss": mean(values_for("final_test_loss")),
        "best_val_acc_pct": mean(values_for("best_val_acc_pct")),
        "generalization_gap_train_minus_test_ppts": mean(
            values_for("generalization_gap_train_minus_test_ppts")
        ),
        "sigma_final_test_acc_pct": trial_summary.get("sigma_final_test_acc_pct_across_seeds"),
        "seed_count": len(per_seed),
    }


def collect_regularization_table_values() -> dict[str, object]:
    values: dict[str, object] = {}
    summaries: dict[str, dict[str, float | int | None]] = {}
    non_balance_final_test: dict[str, float] = {}

    for spec in METHOD_SPECS:
        if spec["source"] == "plain":
            summary = summarize_plain(spec["summary_path"])
        else:
            summary = summarize_bo(spec["bo_summary_path"])
        prefix = spec["prefix"]
        summaries[prefix] = summary

        values[f"{prefix}EpochsAtTau"] = format_epoch_metric(summary["epochs_at_tau"])
        values[f"{prefix}FinalTrainAccPct"] = format_metric(summary["final_train_acc_pct"], decimals=2)
        values[f"{prefix}FinalTrainLoss"] = format_metric(summary["final_train_loss_batch_mean"], decimals=3)
        values[f"{prefix}FinalTestAccPct"] = format_metric(summary["final_test_acc_pct"], decimals=2)
        values[f"{prefix}FinalTestLoss"] = format_metric(summary["final_test_loss"], decimals=3)
        values[f"{prefix}BestValAccPct"] = format_metric(summary["best_val_acc_pct"], decimals=2)
        values[f"{prefix}GapPpts"] = format_metric(
            summary["generalization_gap_train_minus_test_ppts"], decimals=2
        )
        values[f"{prefix}SigmaFinalTestAccPct"] = format_metric(
            summary["sigma_final_test_acc_pct"], decimals=2
        )
        values[f"{prefix}SeedCount"] = str(int(summary["seed_count"] or 0))

        final_test = summary["final_test_acc_pct"]
        if not spec["is_balance"] and final_test is not None:
            non_balance_final_test[prefix] = float(final_test)

    best_non_balance = max(non_balance_final_test.values()) if non_balance_final_test else None
    if best_non_balance is not None:
        values["RegMnistBestNonBalanceFinalTestAccPct"] = format_metric(best_non_balance, decimals=2)
        synbal_l1 = summaries["RegMnistSynBalLOne"]["final_test_acc_pct"]
        synbal_l2 = summaries["RegMnistSynBalLTwo"]["final_test_acc_pct"]
        values["RegMnistSynBalLOneGainOverBestNonBalancePpts"] = format_metric(
            (float(synbal_l1) - best_non_balance) if synbal_l1 is not None else None,
            decimals=2,
        )
        values["RegMnistSynBalLTwoGainOverBestNonBalancePpts"] = format_metric(
            (float(synbal_l2) - best_non_balance) if synbal_l2 is not None else None,
            decimals=2,
        )
    else:
        values["RegMnistBestNonBalanceFinalTestAccPct"] = RawLatex(r"\tbd")
        values["RegMnistSynBalLOneGainOverBestNonBalancePpts"] = RawLatex(r"\tbd")
        values["RegMnistSynBalLTwoGainOverBestNonBalancePpts"] = RawLatex(r"\tbd")

    return values


def main() -> int:
    if not CONFIG_PATH.exists():
        raise SystemExit(f"Config not found: {CONFIG_PATH}")

    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    report = cfg.get("report", {})
    bo = report.get("bo", {})
    training = report.get("training", {})
    metrics = report.get("metrics", {})
    dataset = report.get("dataset", {})

    seeds = report.get("seed_list", [])
    if not isinstance(seeds, list):
        raise SystemExit("report.seed_list must be a list.")

    values = {
        "ValSplitPct": report.get("validation_split_pct", 10),
        "NumSeeds": len(seeds),
        "SeedList": ", ".join(str(x) for x in seeds),
        "BOBudgetTotal": bo.get("total_candidates", 12),
        "BOBudgetInit": bo.get("init_random", 4),
        "BOBudgetGuided": bo.get("guided", 8),
        "MaxEpochs": training.get("max_epochs", 150),
        "LearningRate": training.get("learning_rate", 0.001),
        "BatchSize": training.get("batch_size", 256),
        "TargetTauPct": metrics.get("target_tau_pct", 95.0),
        "MnistTrainSize": dataset.get("mnist_train_size", 60000),
        "MnistTestSize": dataset.get("mnist_test_size", 10000),
    }
    values.update(collect_regularization_table_values())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "%% AUTO-GENERATED FILE. DO NOT EDIT BY HAND.",
        f"%% Source: {CONFIG_PATH.relative_to(REPO_ROOT)}",
        "",
    ]
    for key in sorted(values):
        lines.append(macro_line(key, values[key]))
    lines.append("")

    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

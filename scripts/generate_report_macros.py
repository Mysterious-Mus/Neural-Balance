#!/usr/bin/env python3
"""Generate LaTeX macros from configs/experiment_config.yaml.

Writes REPORT/generated/report_values.tex.
"""

from __future__ import annotations

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
    if isinstance(value, float):
        text = f"{value:g}"
    else:
        text = str(value)
    return f"\\newcommand{{\\{name}}}{{{latex_escape(text)}}}"


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

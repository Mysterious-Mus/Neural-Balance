from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from src.experiments.mnist_fcn.config_loading import dump_logged_config, load_resolved_config
from src.experiments.mnist_fcn.launch_from_config import _resolve_config_source_path
from src.utils.paths import repo_root

DimensionType = Literal["float", "int", "categorical", "bool"]


@dataclass(frozen=True)
class SearchDimension:
    name: str
    kind: DimensionType
    low: float | None = None
    high: float | None = None
    log_scale: bool = False
    categories: tuple[Any, ...] = ()

    def sample(self, rng: random.Random) -> Any:
        if self.kind == "float":
            assert self.low is not None and self.high is not None
            if self.log_scale:
                return 10 ** rng.uniform(math.log10(self.low), math.log10(self.high))
            return rng.uniform(self.low, self.high)
        if self.kind == "int":
            assert self.low is not None and self.high is not None
            if self.log_scale:
                return int(round(10 ** rng.uniform(math.log10(self.low), math.log10(self.high))))
            return rng.randint(int(self.low), int(self.high))
        if self.kind == "bool":
            return bool(rng.randint(0, 1))
        if self.kind == "categorical":
            return rng.choice(self.categories)
        raise ValueError(f"unsupported dimension kind {self.kind!r}")

    def encode(self, value: Any) -> float:
        if self.kind == "float":
            x = float(value)
            return math.log10(x) if self.log_scale else x
        if self.kind == "int":
            x = int(value)
            return math.log10(float(x)) if self.log_scale else float(x)
        if self.kind == "bool":
            return 1.0 if bool(value) else 0.0
        if self.kind == "categorical":
            return float(self.categories.index(value))
        raise ValueError(f"unsupported dimension kind {self.kind!r}")


@dataclass(frozen=True)
class BOSpec:
    total_candidates: int
    init_random: int
    guided: int
    acquisition_samples: int
    seed: int
    xi: float


@dataclass(frozen=True)
class RuntimeSpec:
    parallel_seeds: int
    gpu_pool: tuple[str, ...]
    python_executable: str


def _build_space(raw_space: Sequence[Mapping[str, Any]]) -> list[SearchDimension]:
    out: list[SearchDimension] = []
    for item in raw_space:
        name = str(item["name"])
        kind = str(item["type"]).lower()
        if kind in ("float", "int"):
            out.append(
                SearchDimension(
                    name=name,
                    kind=kind,  # type: ignore[arg-type]
                    low=float(item["low"]),
                    high=float(item["high"]),
                    log_scale=bool(item.get("log_scale", False)),
                )
            )
        elif kind == "categorical":
            categories = item.get("categories")
            if not isinstance(categories, list) or not categories:
                raise ValueError(f"dimension {name!r} needs non-empty categories")
            out.append(SearchDimension(name=name, kind="categorical", categories=tuple(categories)))
        elif kind == "bool":
            out.append(SearchDimension(name=name, kind="bool"))
        else:
            raise ValueError(f"unsupported dimension type {kind!r} for {name!r}")
    return out


def _encode_candidate(dims: Sequence[SearchDimension], candidate: Mapping[str, Any]) -> list[float]:
    return [dim.encode(candidate[dim.name]) for dim in dims]


def _candidate_key(candidate: Mapping[str, Any]) -> str:
    return json.dumps(candidate, sort_keys=True, separators=(",", ":"))


def _discrete_space_size(dims: Sequence[SearchDimension]) -> int | None:
    total = 1
    for dim in dims:
        if dim.kind == "bool":
            total *= 2
        elif dim.kind == "categorical":
            total *= len(dim.categories)
        elif dim.kind == "int":
            assert dim.low is not None and dim.high is not None
            total *= max(0, int(dim.high) - int(dim.low) + 1)
        else:
            # Any continuous dimension makes the space effectively unbounded.
            return None
    return total


def _sample_unique_candidate(
    dims: Sequence[SearchDimension],
    rng: random.Random,
    seen: set[str],
    *,
    max_tries: int = 5000,
) -> dict[str, Any]:
    for _ in range(max_tries):
        cand = {dim.name: dim.sample(rng) for dim in dims}
        key = _candidate_key(cand)
        if key not in seen:
            return cand
    raise RuntimeError("failed to sample a unique candidate")


def _expected_improvement(
    gp: GaussianProcessRegressor,
    xs: np.ndarray,
    best_y: float,
    *,
    xi: float,
) -> np.ndarray:
    mu, sigma = gp.predict(xs, return_std=True)
    sigma = np.maximum(sigma, 1e-12)
    improvement = mu - best_y - xi
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    ei[sigma <= 1e-12] = 0.0
    return ei


def _propose_guided_candidate(
    dims: Sequence[SearchDimension],
    evaluated_candidates: Sequence[Mapping[str, Any]],
    y_values: Sequence[float],
    rng: random.Random,
    *,
    acquisition_samples: int,
    xi: float,
) -> dict[str, Any]:
    seen = {_candidate_key(c) for c in evaluated_candidates}
    x_train = np.array([_encode_candidate(dims, c) for c in evaluated_candidates], dtype=float)
    y_train = np.array(y_values, dtype=float)
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=np.ones(x_train.shape[1]),
        nu=2.5,
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        random_state=rng.randint(0, 2**32 - 1),
        n_restarts_optimizer=2,
    )
    gp.fit(x_train, y_train)
    max_discrete = _discrete_space_size(dims)
    if max_discrete is not None:
        remaining = max_discrete - len(seen)
        if remaining <= 0:
            raise RuntimeError("search space exhausted: no unseen candidates remain")
        sample_budget = min(acquisition_samples, remaining)
    else:
        sample_budget = acquisition_samples

    pool: list[dict[str, Any]] = []
    attempts = 0
    max_attempts = max(1000, sample_budget * 100)
    while len(pool) < sample_budget and attempts < max_attempts:
        attempts += 1
        cand = {dim.name: dim.sample(rng) for dim in dims}
        key = _candidate_key(cand)
        if key in seen:
            continue
        pool.append(cand)
        seen.add(key)
    if not pool:
        raise RuntimeError(
            "failed to build guided candidate pool; try reducing bo.acquisition_samples "
            "or enlarging the search space"
        )
    x_pool = np.array([_encode_candidate(dims, c) for c in pool], dtype=float)
    ei = _expected_improvement(gp, x_pool, best_y=max(y_train), xi=xi)
    return pool[int(np.argmax(ei))]


def _read_bo_spec(raw: Mapping[str, Any]) -> BOSpec:
    bo = raw.get("bo", {})
    total_candidates = int(bo.get("total_candidates", 12))
    init_random = int(bo.get("init_random", 4))
    guided = int(bo.get("guided", total_candidates - init_random))
    if total_candidates != init_random + guided:
        raise ValueError("bo.total_candidates must equal bo.init_random + bo.guided")
    return BOSpec(
        total_candidates=total_candidates,
        init_random=init_random,
        guided=guided,
        acquisition_samples=int(bo.get("acquisition_samples", 2048)),
        seed=int(bo.get("seed", 1234)),
        xi=float(bo.get("xi", 0.01)),
    )


def _read_runtime_spec(raw: Mapping[str, Any], static_overrides: Mapping[str, Any]) -> RuntimeSpec:
    runtime = raw.get("runtime", {})
    if not isinstance(runtime, dict):
        raise TypeError("runtime must be a mapping when provided")
    gpu_pool_raw = runtime.get("gpu_pool")
    if gpu_pool_raw is None:
        static_gpu = static_overrides.get("gpu")
        if isinstance(static_gpu, list):
            gpu_pool = tuple(str(x) for x in static_gpu)
        elif static_gpu is None:
            gpu_pool = ("cpu",)
        else:
            gpu_pool = (str(static_gpu),)
    else:
        if not isinstance(gpu_pool_raw, list) or not gpu_pool_raw:
            raise ValueError("runtime.gpu_pool must be a non-empty list when provided")
        gpu_pool = tuple(str(x) for x in gpu_pool_raw)
    parallel_default = len(gpu_pool)
    parallel_seeds = max(1, int(runtime.get("parallel_seeds", parallel_default)))
    python_executable = str(runtime.get("python_executable", sys.executable))
    return RuntimeSpec(
        parallel_seeds=parallel_seeds,
        gpu_pool=gpu_pool,
        python_executable=python_executable,
    )


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values))


def _stdev(values: Sequence[float]) -> float:
    mu = _mean(values)
    return float(math.sqrt(sum((v - mu) ** 2 for v in values) / max(1, len(values) - 1)))


def _load_summary(summary_path: Path) -> dict[str, Any]:
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _run_seed_job(seed_dir: Path, python_executable: str) -> dict[str, Any]:
    cmd = [
        python_executable,
        "-m",
        "src.experiments.mnist_fcn.launch_from_config",
        str(seed_dir),
        "--config",
        "config.yaml",
    ]
    logs_dir = seed_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    run_log = logs_dir / "run.log"
    with run_log.open("w", encoding="utf-8") as logf:
        result = subprocess.run(
            cmd,
            cwd=str(repo_root()),
            stdout=logf,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"seed run failed in {seed_dir}\n"
            f"command: {' '.join(cmd)}\n"
            f"see log: {run_log}"
        )
    summary_path = seed_dir / "logs" / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"missing summary output: {summary_path}")
    payload = _load_summary(summary_path)
    payload["seed_dir"] = str(seed_dir)
    return payload


def _prepare_seed_configs(
    trial_dir: Path,
    seeds: Sequence[int],
    base_config: Mapping[str, Any],
    resolved_overrides: Mapping[str, Any],
    runtime: RuntimeSpec,
) -> list[tuple[int, str, Path]]:
    out: list[tuple[int, str, Path]] = []
    for i, seed in enumerate(seeds):
        gpu = runtime.gpu_pool[i % len(runtime.gpu_pool)]
        seed_dir = trial_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        cfg = dict(base_config)
        cfg.update(resolved_overrides)
        cfg["seed"] = int(seed)
        cfg["gpu"] = gpu
        dump_logged_config(seed_dir / "config.yaml", cfg)
        out.append((seed, gpu, seed_dir))
    return out


def _resolve_base_config(
    raw_meta_cfg: Mapping[str, Any],
    experiment_dir: Path,
) -> tuple[dict[str, Any], Path | None, str]:
    inline_base_cfg = raw_meta_cfg.get("base_config")
    if inline_base_cfg is not None:
        if not isinstance(inline_base_cfg, dict):
            raise TypeError("base_config must be a mapping when provided")
        return dict(inline_base_cfg), None, "inline.base_config"

    base_config_path = raw_meta_cfg.get("base_config_path")
    if base_config_path is None:
        raise ValueError("meta config must define either base_config or base_config_path")
    base_config_name = str(raw_meta_cfg.get("base_config_name", "config.yaml"))
    resolved_base_config_path = _resolve_config_source_path(
        experiment_dir,
        base_config_name,
        base_config_path,
    )
    base_cfg, resolved_base_source_path = load_resolved_config(resolved_base_config_path)
    return dict(base_cfg), resolved_base_source_path, "base_config_path"


def run_meta_experiment(
    experiment_dir: Path,
    config_name: str,
    *,
    config_path: str | Path | None = None,
    dry_run: bool = False,
) -> None:
    experiment_dir = experiment_dir.resolve()
    source_meta_config_path = _resolve_config_source_path(experiment_dir, config_name, config_path)
    raw, resolved_meta_source_path = load_resolved_config(source_meta_config_path)

    base_cfg, resolved_base_source_path, base_cfg_source_mode = _resolve_base_config(raw, experiment_dir)
    logged_meta_config_path = experiment_dir / "config.yaml"
    logged_meta_cfg = dict(raw)
    # Make logged meta config self-contained for easy tweaking/smoke edits.
    logged_meta_cfg["base_config"] = base_cfg
    dump_logged_config(logged_meta_config_path, logged_meta_cfg)

    trials_dir = experiment_dir / "trials"
    print(f"meta experiment dir: {experiment_dir}")
    print(f"meta config source: {resolved_meta_source_path}")
    print(f"logged meta config: {logged_meta_config_path}")
    print(
        "base config source: "
        + (
            str(resolved_base_source_path)
            if resolved_base_source_path is not None
            else "inline.base_config"
        )
    )
    print(f"base config resolution mode: {base_cfg_source_mode}")
    print(f"trials dir: {trials_dir}")
    if dry_run:
        print("dry-run enabled: wrote only meta config, no BO trials launched.")
        return

    seeds_raw = raw.get("seeds")
    if not isinstance(seeds_raw, list) or not seeds_raw:
        raise ValueError("meta config must define non-empty seeds list")
    seeds = [int(s) for s in seeds_raw]

    raw_space = raw.get("search_space")
    if not isinstance(raw_space, list) or not raw_space:
        raise ValueError("meta config must define non-empty search_space list")
    dims = _build_space(raw_space)

    static_overrides = raw.get("static_overrides", {})
    if not isinstance(static_overrides, dict):
        raise TypeError("static_overrides must be a mapping when provided")
    runtime = _read_runtime_spec(raw, static_overrides)
    bo_spec = _read_bo_spec(raw)
    objective_name = str(raw.get("objective", "mean_final_val_acc_pct"))
    if objective_name not in {"mean_final_val_acc_pct", "mean_final_test_acc_pct"}:
        raise ValueError("objective must be mean_final_val_acc_pct or mean_final_test_acc_pct")

    max_discrete = _discrete_space_size(dims)
    effective_total_candidates = bo_spec.total_candidates
    if max_discrete is not None and bo_spec.total_candidates > max_discrete:
        effective_total_candidates = max_discrete
        print(
            "warning: requested bo.total_candidates exceeds finite search space; "
            f"capping trials to {effective_total_candidates} (requested {bo_spec.total_candidates})"
        )

    rng = random.Random(bo_spec.seed)
    trials_dir.mkdir(parents=True, exist_ok=True)

    evaluated_candidates: list[dict[str, Any]] = []
    objective_values: list[float] = []
    trial_rows: list[dict[str, Any]] = []

    for trial_idx in range(effective_total_candidates):
        seen = {_candidate_key(c) for c in evaluated_candidates}
        if trial_idx < bo_spec.init_random or len(evaluated_candidates) < 2:
            candidate = _sample_unique_candidate(dims, rng, seen)
            source = "random_init" if trial_idx < bo_spec.init_random else "random_fallback"
        else:
            candidate = _propose_guided_candidate(
                dims,
                evaluated_candidates,
                objective_values,
                rng,
                acquisition_samples=bo_spec.acquisition_samples,
                xi=bo_spec.xi,
            )
            source = "gp_ei"

        trial_name = f"trial_{trial_idx:03d}"
        trial_dir = trials_dir / trial_name
        trial_dir.mkdir(parents=True, exist_ok=True)

        resolved = dict(static_overrides)
        resolved.update(candidate)
        seed_jobs = _prepare_seed_configs(
            trial_dir,
            seeds,
            base_cfg,
            resolved,
            runtime,
        )

        (trial_dir / "candidate.json").write_text(
            json.dumps(
                {
                    "trial": trial_name,
                    "proposal_source": source,
                    "candidate": candidate,
                    "resolved_overrides": resolved,
                    "seed_gpu_assignments": {str(seed): gpu for seed, gpu, _ in seed_jobs},
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        per_seed_metrics: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=runtime.parallel_seeds) as pool:
            futures = [
                pool.submit(_run_seed_job, seed_dir, runtime.python_executable)
                for _, _, seed_dir in seed_jobs
            ]
            for fut in as_completed(futures):
                per_seed_metrics.append(fut.result())
        per_seed_metrics.sort(key=lambda x: int(x.get("seed", -1)))

        mean_final_val = _mean([float(m["final_val_acc_pct"]) for m in per_seed_metrics])
        mean_final_test = _mean([float(m["final_test_acc_pct"]) for m in per_seed_metrics])
        std_final_test = _stdev([float(m["final_test_acc_pct"]) for m in per_seed_metrics])
        objective_value = mean_final_val if objective_name == "mean_final_val_acc_pct" else mean_final_test

        trial_summary = {
            "trial": trial_name,
            "proposal_source": source,
            "candidate": candidate,
            "resolved_overrides": resolved,
            "mean_final_val_acc_pct": mean_final_val,
            "mean_final_test_acc_pct": mean_final_test,
            "sigma_final_test_acc_pct_across_seeds": std_final_test,
            "objective_name": objective_name,
            "objective_value": objective_value,
            "per_seed_metrics": per_seed_metrics,
        }
        (trial_dir / "trial_summary.json").write_text(
            json.dumps(trial_summary, indent=2) + "\n",
            encoding="utf-8",
        )

        evaluated_candidates.append(candidate)
        objective_values.append(objective_value)
        trial_rows.append(
            {
                "trial": trial_name,
                "proposal_source": source,
                "objective_name": objective_name,
                "objective_value": objective_value,
                "mean_final_val_acc_pct": mean_final_val,
                "mean_final_test_acc_pct": mean_final_test,
                "sigma_final_test_acc_pct_across_seeds": std_final_test,
                **candidate,
            }
        )
        print(
            f"[{trial_name}] source={source} objective={objective_value:.4f} "
            f"mean_final_test_acc={mean_final_test:.4f}"
        )

    best_idx = int(np.argmax(np.array(objective_values)))
    best_trial = trial_rows[best_idx]

    with (experiment_dir / "bo_trials.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(trial_rows[0].keys()))
        writer.writeheader()
        for row in trial_rows:
            writer.writerow(row)

    summary = {
        "meta_experiment_dir": str(experiment_dir),
        "meta_config_source_path": str(resolved_meta_source_path),
        "base_config_source_mode": base_cfg_source_mode,
        "base_config_source_path": (
            str(resolved_base_source_path)
            if resolved_base_source_path is not None
            else None
        ),
        "seeds": seeds,
        "runtime": {
            "parallel_seeds": runtime.parallel_seeds,
            "gpu_pool": list(runtime.gpu_pool),
            "python_executable": runtime.python_executable,
        },
        "bo": {
            "requested_total_candidates": bo_spec.total_candidates,
            "executed_total_candidates": len(trial_rows),
            "total_candidates": bo_spec.total_candidates,
            "init_random": bo_spec.init_random,
            "guided": bo_spec.guided,
            "acquisition_samples": bo_spec.acquisition_samples,
            "seed": bo_spec.seed,
            "xi": bo_spec.xi,
        },
        "objective": objective_name,
        "search_space": raw_space,
        "static_overrides": static_overrides,
        "best_trial": best_trial,
        "all_trials_csv": str((experiment_dir / "bo_trials.csv").resolve()),
    }
    (experiment_dir / "bo_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"BO complete. Best trial: {best_trial['trial']}")
    print(f"wrote {(experiment_dir / 'bo_summary.json').resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BO meta-experiment over regularization hyperparameters.")
    parser.add_argument("experiment_dir", nargs="?", default=Path("."), type=Path, help="Meta-experiment output directory")
    parser.add_argument("--config", default="config.yaml", help="Meta-experiment config filename inside experiment_dir (ignored when --config-path is provided)")
    parser.add_argument("--config-path", default=None, help="Authoritative meta config path relative to repo root or absolute (recommended: configs/...)")
    parser.add_argument("--dry-run", action="store_true", help="Resolve and log meta config only; skip BO and training")
    args = parser.parse_args()
    run_meta_experiment(args.experiment_dir, args.config, config_path=args.config_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

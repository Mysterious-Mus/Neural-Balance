from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.utils.paths import repo_root

_REPO_ROOT = repo_root()

_KEY_CONSUME = "_consume"
_KEY_SELECT = "_select"
_KEY_OVERRIDES = "_overrides"


def _load_yaml_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise TypeError(f"YAML root must be a mapping, got {type(raw)!r} in {path}")
    return raw


def _resolve_nested_path(obj: Mapping[str, Any], select: str, source: Path) -> dict[str, Any]:
    cur: Any = obj
    for part in select.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            raise KeyError(f"missing key path {select!r} in {source}")
        cur = cur[part]
    if not isinstance(cur, dict):
        raise TypeError(f"selected value at {select!r} is not a mapping in {source}")
    return dict(cur)


def load_resolved_config(config_path: Path) -> tuple[dict[str, Any], Path]:
    """Load config file with optional `_consume` indirection.

    If `_consume` and `_select` are present, data is loaded from the consumed file
    and selected nested mapping, then merged with optional `_overrides` and any
    non-private keys from the wrapper config.
    """
    config_path = config_path.resolve()
    raw = _load_yaml_file(config_path)

    consume = raw.get(_KEY_CONSUME)
    select = raw.get(_KEY_SELECT)
    if consume is None or select is None:
        return raw, config_path

    consume_path = Path(str(consume))
    if not consume_path.is_absolute():
        consume_path = (_REPO_ROOT / consume_path).resolve()
    if not consume_path.is_file():
        raise FileNotFoundError(f"Missing consumed config file: {consume_path}")

    consumed_root = _load_yaml_file(consume_path)
    resolved = _resolve_nested_path(consumed_root, str(select), consume_path)

    overrides = raw.get(_KEY_OVERRIDES, {})
    if overrides is not None:
        if not isinstance(overrides, dict):
            raise TypeError(f"{_KEY_OVERRIDES} must be a mapping in {config_path}")
        resolved.update(overrides)

    inline_overrides = {
        k: v for k, v in raw.items() if not str(k).startswith("_")
    }
    resolved.update(inline_overrides)

    return resolved, consume_path


def dump_logged_config(path: Path, cfg: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            deepcopy(dict(cfg)),
            f,
            sort_keys=False,
            default_flow_style=False,
        )

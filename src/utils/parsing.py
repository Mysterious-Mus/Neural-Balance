"""Parsing helpers for CLI and YAML experiment configs."""

from __future__ import annotations


def str2bool(v: object) -> bool:
    """Parse bool from YAML/CLI (``true``/``false``, ``0``/``1``, etc.)."""
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        if v == 0:
            return False
        if v == 1:
            return True
        raise ValueError(f"expected 0 or 1 for boolean, got {v!r}")
    s = str(v).strip().lower()
    if s in ("yes", "true", "t", "1", "y"):
        return True
    if s in ("no", "false", "f", "0", "n"):
        return False
    raise ValueError(f"invalid boolean value: {v!r}")

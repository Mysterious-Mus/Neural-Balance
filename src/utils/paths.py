from __future__ import annotations

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def repo_root() -> Path:
    """Return repository root by scanning upward from this file."""
    here = Path(__file__).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "src").is_dir() and (candidate / "configs").is_dir():
            return candidate
    raise RuntimeError("Could not locate repository root from src/utils/paths.py")


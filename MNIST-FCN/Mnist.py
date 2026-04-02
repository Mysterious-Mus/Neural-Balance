"""Legacy CLI entry (ReLU FCN). Implementation lives in ``src/``."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.experiments.mnist_fcn.main import main

if __name__ == "__main__":
    main()

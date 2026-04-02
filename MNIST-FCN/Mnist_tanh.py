"""Legacy CLI entry (tanh FCN, legacy ``Mnist_tanh`` behavior). Implementation lives in ``src/``."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.experiments.mnist_fcn.main import build_parser, run_training


def main() -> None:
    p = build_parser()
    p.set_defaults(activation="tanh", tanh_on_output=1)
    args = p.parse_args()
    run_training(args)


if __name__ == "__main__":
    main()

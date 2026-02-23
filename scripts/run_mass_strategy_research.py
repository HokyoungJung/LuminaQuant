"""Alias wrapper for strategy-factory research pipeline."""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    target = Path(__file__).resolve().parent / "run_strategy_factory_pipeline.py"
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()

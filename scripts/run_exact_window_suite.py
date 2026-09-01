"""Thin wrapper around `uv run lq exact-window`."""

from __future__ import annotations

import sys

from lumina_quant.cli.exact_window import main as exact_window_main


def main(argv: list[str] | None = None) -> int:
    return int(exact_window_main(argv if argv is not None else sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())

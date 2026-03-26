from __future__ import annotations

import sys

RETIREMENT_MESSAGE = """\
LuminaQuant Dashboard Not Included In public/main

The interactive dashboard runtime is maintained only on private/main.

Public/main supports the CLI workflows and the sample public strategy surface.
Use `uv run lq backtest`, `uv run lq optimize`, and `uv run lq live` from this repo.
"""


def main() -> int:
    print(RETIREMENT_MESSAGE, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

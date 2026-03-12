from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

from lumina_quant.cli import backtest, dashboard, data, exact_window, live, optimize

Handler = Callable[[list[str] | None], int]


def _run_handler(handler: Handler, argv: list[str]) -> int:
    try:
        return int(handler(argv))
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        return int(code)


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    commands: dict[str, Handler] = {
        "backtest": backtest.main,
        "optimize": optimize.main,
        "live": live.main,
        "dashboard": dashboard.main,
        "data": data.main,
        "exact-window": exact_window.main,
    }

    parser = argparse.ArgumentParser(
        prog="lq",
        description="LuminaQuant command-line interface.",
    )
    parser.add_argument("command", nargs="?", choices=sorted(commands.keys()))
    parser.add_argument("command_args", nargs=argparse.REMAINDER)
    parsed = parser.parse_args(args)

    if not parsed.command:
        parser.print_help()
        return 0

    return _run_handler(commands[parsed.command], list(parsed.command_args or []))


if __name__ == "__main__":
    raise SystemExit(main())

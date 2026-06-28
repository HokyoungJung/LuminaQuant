from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from importlib import import_module

Handler = Callable[[list[str] | None], int]


COMMAND_MODULES = {
    "backtest": "backtest",
    "optimize": "optimize",
    "live": "live",
    "dashboard": "dashboard",
    "data": "data",
    "deep_learning": "deep_learning",
    "exact_window": "exact_window",
    "autonomous_research": "autonomous_research",
    "config": "config",
    "registry": "registry",
}


def _load_handler(module_name: str, attr: str = "main") -> Handler:
    module = import_module(f"lumina_quant.cli.{module_name}")
    handler = getattr(module, attr)
    return handler


def _run_handler(handler: Handler, argv: list[str]) -> int:
    try:
        return int(handler(argv))
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        return int(code)


def __getattr__(name: str):
    module_name = COMMAND_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = import_module(f"lumina_quant.cli.{module_name}")
    globals()[name] = module
    return module


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    commands = {
        "backtest": COMMAND_MODULES["backtest"],
        "optimize": COMMAND_MODULES["optimize"],
        "live": COMMAND_MODULES["live"],
        "dashboard": COMMAND_MODULES["dashboard"],
        "data": COMMAND_MODULES["data"],
        "exact-window": COMMAND_MODULES["exact_window"],
        "autonomous-research": COMMAND_MODULES["autonomous_research"],
        "autonomous_research": COMMAND_MODULES["autonomous_research"],
        "config": COMMAND_MODULES["config"],
        "deep-learning": COMMAND_MODULES["deep_learning"],
        "deep_learning": COMMAND_MODULES["deep_learning"],
        "registry": COMMAND_MODULES["registry"],
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

    handler = _load_handler(commands[parsed.command])
    return _run_handler(handler, list(parsed.command_args or []))


if __name__ == "__main__":
    raise SystemExit(main())

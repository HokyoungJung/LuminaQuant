"""Backtest workflow adapter."""

from lumina_quant.cli.backtest import main as _main


def run(argv: list[str] | None = None) -> int:
    return int(_main(argv))

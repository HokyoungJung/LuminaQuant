"""Live portfolio boundary exports.

The live runtime currently reuses the backtesting portfolio implementation, but
this module keeps that dependency behind the live package boundary so callers do
not need to import the backtesting module directly.
"""

from __future__ import annotations

from typing import Any


def get_live_portfolio_cls():
    from lumina_quant.backtesting.portfolio_backtest import Portfolio as _Portfolio

    return _Portfolio


def __getattr__(name: str) -> Any:
    if name in {"LivePortfolio", "Portfolio"}:
        return get_live_portfolio_cls()
    raise AttributeError(name)

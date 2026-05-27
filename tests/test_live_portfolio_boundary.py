from __future__ import annotations

import importlib
import sys


def test_live_portfolio_boundary_defers_backtesting_import_until_needed():
    sys.modules.pop("lumina_quant.backtesting.portfolio_backtest", None)
    sys.modules.pop("lumina_quant.live.portfolio", None)
    sys.modules.pop("lumina_quant.system_assembly", None)

    system_assembly = importlib.import_module("lumina_quant.system_assembly")

    assert "lumina_quant.backtesting.portfolio_backtest" not in sys.modules

    live_portfolio_cls = system_assembly.LivePortfolio

    assert "lumina_quant.backtesting.portfolio_backtest" in sys.modules
    assert live_portfolio_cls.__name__ == "Portfolio"


def test_live_portfolio_module_exports_lazy_boundary_aliases():
    sys.modules.pop("lumina_quant.backtesting.portfolio_backtest", None)
    sys.modules.pop("lumina_quant.live.portfolio", None)

    live_portfolio_module = importlib.import_module("lumina_quant.live.portfolio")

    assert "lumina_quant.backtesting.portfolio_backtest" not in sys.modules

    live_portfolio_cls = live_portfolio_module.LivePortfolio

    assert "lumina_quant.backtesting.portfolio_backtest" in sys.modules
    assert live_portfolio_module.Portfolio is live_portfolio_cls

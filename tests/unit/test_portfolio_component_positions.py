from types import SimpleNamespace

from lumina_quant.backtesting.portfolio_backtest import Portfolio
from lumina_quant.core.events import FillEvent, SignalEvent


class _BarsStub:
    symbol_list = ["BNB/USDT"]

    @staticmethod
    def get_latest_bar_value(symbol, field):
        _ = symbol, field
        return 100.0


class _ConfigStub:
    INITIAL_CAPITAL = 10_000.0
    MAX_DAILY_LOSS_PCT = 0.05
    RISK_PER_TRADE = 0.005
    MAX_SYMBOL_EXPOSURE_PCT = 0.25
    MAX_ORDER_VALUE = 5000.0
    DEFAULT_STOP_LOSS_PCT = 0.01
    MIN_TRADE_QTY = 0.001
    TARGET_ALLOCATION = 0.10
    SYMBOL_LIMITS = {}


def test_exit_signal_uses_component_scoped_position_not_total_symbol_position() -> None:
    portfolio = Portfolio(
        _BarsStub(), SimpleNamespace(put=lambda event: None), "2026-04-17T00:00:00Z", _ConfigStub()
    )

    portfolio.update_fill(
        FillEvent(
            timeindex="2026-04-17T00:00:00Z",
            symbol="BNB/USDT",
            exchange="SIM",
            quantity=1.0,
            direction="BUY",
            fill_cost=100.0,
            commission=0.0,
            metadata={"component_id": "comp-a"},
        )
    )
    portfolio.update_fill(
        FillEvent(
            timeindex="2026-04-17T00:01:00Z",
            symbol="BNB/USDT",
            exchange="SIM",
            quantity=1.0,
            direction="BUY",
            fill_cost=100.0,
            commission=0.0,
            metadata={"component_id": "comp-b"},
        )
    )

    order = portfolio.generate_order_from_signal(
        SignalEvent(
            strategy_id="portfolio-mode",
            symbol="BNB/USDT",
            datetime="2026-04-17T00:02:00Z",
            signal_type="EXIT",
            metadata={"component_id": "comp-a"},
        )
    )

    assert portfolio.current_positions["BNB/USDT"] == 2.0
    assert portfolio.component_positions["comp-a"]["BNB/USDT"] == 1.0
    assert portfolio.component_positions["comp-b"]["BNB/USDT"] == 1.0
    assert order is not None
    assert order.quantity == 1.0

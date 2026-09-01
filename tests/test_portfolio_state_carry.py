from __future__ import annotations

from lumina_quant.backtesting.portfolio_backtest import Portfolio


class _Bars:
    symbol_list = ["BTC/USDT"]


class _Events:
    def put(self, item):
        self.item = item


class _Config:
    INITIAL_CAPITAL = 10_000.0


def test_portfolio_state_carries_liquidation_events():
    portfolio = Portfolio(
        bars=_Bars(),
        events=_Events(),
        start_date=0,
        config=_Config,
        record_history=False,
        track_metrics=False,
        record_trades=False,
    )
    portfolio.liquidation_events = [{"symbol": "BTC/USDT", "quantity": 1.0}]

    restored = Portfolio(
        bars=_Bars(),
        events=_Events(),
        start_date=0,
        config=_Config,
        record_history=False,
        track_metrics=False,
        record_trades=False,
    )
    restored.set_state(portfolio.get_state())

    assert restored.liquidation_events == portfolio.liquidation_events

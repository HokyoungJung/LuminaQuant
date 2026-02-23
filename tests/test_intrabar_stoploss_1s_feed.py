from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
from lumina_quant.backtesting.backtest import Backtest
from lumina_quant.backtesting.data import HistoricCSVDataHandler
from lumina_quant.backtesting.execution_sim import SimulatedExecutionHandler
from lumina_quant.backtesting.portfolio_backtest import Portfolio
from lumina_quant.events import SignalEvent
from lumina_quant.strategy import Strategy


class _OneShotLongStrategy(Strategy):
    def __init__(self, bars, events, stop_loss=99.0):
        self.bars = bars
        self.events = events
        self.symbol = next(iter(self.bars.symbol_list))
        self.stop_loss = float(stop_loss)
        self.sent = False

    def calculate_signals(self, event):
        if self.sent:
            return
        if getattr(event, "type", None) != "MARKET":
            return
        if getattr(event, "symbol", None) != self.symbol:
            return
        self.sent = True
        self.events.put(
            SignalEvent(
                strategy_id="oneshot",
                symbol=self.symbol,
                datetime=event.time,
                signal_type="LONG",
                strength=1.0,
                stop_loss=self.stop_loss,
            )
        )


def _build_1s_df():
    start = datetime(2026, 1, 1, 0, 0, 0)
    datetimes = []
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []

    price = 100.0
    for sec in range(180):
        t = start + timedelta(seconds=sec)
        o = price
        h = price + 0.05
        low_price = price - 0.05
        c = price

        # Force an intrabar stop-loss breach inside the first 1m strategy candle
        # after entry fill (entry expected on t=1s, breach at t=30s).
        if sec == 30:
            low_price = 98.5
            c = 99.0
        elif sec > 30:
            c = 100.5
            h = 100.7
            low_price = 100.3

        datetimes.append(t)
        opens.append(float(o))
        highs.append(float(max(h, o, c)))
        lows.append(float(min(low_price, o, c)))
        closes.append(float(c))
        volumes.append(100000.0)
        price = c

    return pl.DataFrame(
        {
            "datetime": datetimes,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    )


def test_intrabar_stoploss_triggers_with_1s_source_feed():
    symbol = "BTC/USDT"
    data_dict = {symbol: _build_1s_df()}
    start_date = datetime(2026, 1, 1, 0, 0, 0)

    backtest = Backtest(
        csv_dir="data",
        symbol_list=[symbol],
        start_date=start_date,
        data_handler_cls=HistoricCSVDataHandler,
        execution_handler_cls=SimulatedExecutionHandler,
        portfolio_cls=Portfolio,
        strategy_cls=_OneShotLongStrategy,
        strategy_params={"stop_loss": 99.0},
        data_dict=data_dict,
        record_history=False,
        track_metrics=True,
        record_trades=True,
        strategy_timeframe="1m",
    )

    backtest.simulate_trading(output=False)

    # Entry + stop-loss exit should both occur inside simulation.
    assert int(backtest.portfolio.trade_count) >= 2
    assert abs(float(backtest.portfolio.current_positions[symbol])) < 1e-9

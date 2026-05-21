"""Minimal backtesting engine for the public sample pipeline."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

from lumina_quant.models import Bar, EquityPoint, TargetPosition, Trade
from lumina_quant.sample_strategy import MovingAverageCrossStrategy


@dataclass(frozen=True, slots=True)
class BacktestResult:
    initial_cash: float
    final_equity: float
    total_return: float
    max_drawdown: float
    trade_count: int
    equity_curve: list[EquityPoint]
    trades: list[Trade]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["equity_curve"] = [asdict(point) for point in self.equity_curve]
        payload["trades"] = [asdict(trade) for trade in self.trades]
        return payload


def _max_drawdown(values: list[float]) -> float:
    peak = values[0]
    worst = 0.0
    for value in values:
        peak = max(peak, value)
        if peak > 0.0:
            worst = max(worst, (peak - value) / peak)
    return worst


def run_backtest(
    bars: Iterable[Bar],
    strategy: MovingAverageCrossStrategy,
    *,
    initial_cash: float = 10_000.0,
    fee_bps: float = 1.0,
) -> BacktestResult:
    """Run a deterministic long/flat backtest over local bars."""
    cash = float(initial_cash)
    quantity = 0.0
    fee_rate = float(fee_bps) / 10_000.0
    trades: list[Trade] = []
    equity_curve: list[EquityPoint] = []

    for bar in bars:
        signal = strategy.on_bar(bar)
        price = float(bar.close)
        if signal.target is TargetPosition.LONG and quantity == 0.0:
            gross_quantity = cash / price
            fee = cash * fee_rate
            quantity = max(0.0, (cash - fee) / price)
            cash = 0.0
            trades.append(Trade(bar.timestamp, "buy", gross_quantity, price, fee))
        elif signal.target is TargetPosition.FLAT and quantity > 0.0:
            gross_value = quantity * price
            fee = gross_value * fee_rate
            cash = gross_value - fee
            trades.append(Trade(bar.timestamp, "sell", quantity, price, fee))
            quantity = 0.0
        equity_curve.append(EquityPoint(bar.timestamp, cash + quantity * price))

    if not equity_curve:
        raise ValueError("backtest requires at least one bar")
    final_equity = equity_curve[-1].equity
    equity_values = [point.equity for point in equity_curve]
    return BacktestResult(
        initial_cash=float(initial_cash),
        final_equity=final_equity,
        total_return=(final_equity / float(initial_cash)) - 1.0,
        max_drawdown=_max_drawdown(equity_values),
        trade_count=len(trades),
        equity_curve=equity_curve,
        trades=trades,
    )

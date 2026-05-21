"""Generic metrics and scoring for the public sample pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt

from lumina_quant.backtesting import BacktestResult


@dataclass(frozen=True, slots=True)
class MetricsSummary:
    total_return: float
    max_drawdown: float
    trade_count: int
    mean_bar_return: float
    volatility: float
    sharpe_like: float
    score: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _equity_returns(equity_values: list[float]) -> list[float]:
    returns: list[float] = []
    for previous, current in zip(equity_values, equity_values[1:], strict=False):
        if previous > 0.0:
            returns.append((current / previous) - 1.0)
    return returns


def compute_metrics(result: BacktestResult) -> MetricsSummary:
    """Compute generic, strategy-agnostic performance metrics."""
    equity_values = [point.equity for point in result.equity_curve]
    returns = _equity_returns(equity_values)
    mean_return = sum(returns) / len(returns) if returns else 0.0
    if len(returns) > 1:
        variance = sum((item - mean_return) ** 2 for item in returns) / (len(returns) - 1)
        volatility = sqrt(max(0.0, variance))
    else:
        volatility = 0.0
    sharpe_like = mean_return / volatility * sqrt(len(returns)) if volatility > 0.0 else 0.0
    score = result.total_return - (2.0 * result.max_drawdown) - (0.0001 * result.trade_count)
    return MetricsSummary(
        total_return=result.total_return,
        max_drawdown=result.max_drawdown,
        trade_count=result.trade_count,
        mean_bar_return=mean_return,
        volatility=volatility,
        sharpe_like=sharpe_like,
        score=score,
    )

"""Paper-live replay pipeline.

This module simulates order handling locally from sample bars. It has no real
broker, network, credential, or exchange integration.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from lumina_quant.backtesting import BacktestResult, run_backtest
from lumina_quant.models import Bar
from lumina_quant.strategy_loader import StrategyProtocol


@dataclass(frozen=True, slots=True)
class PaperLiveResult:
    mode: str
    order_execution_enabled: bool
    backtest_equivalent: BacktestResult

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "order_execution_enabled": self.order_execution_enabled,
            "backtest_equivalent": self.backtest_equivalent.to_dict(),
        }


def run_paper_live(
    bars: Iterable[Bar],
    strategy: StrategyProtocol,
    *,
    initial_cash: float = 10_000.0,
    fee_bps: float = 1.0,
) -> PaperLiveResult:
    result = run_backtest(bars, strategy, initial_cash=initial_cash, fee_bps=fee_bps)
    return PaperLiveResult(
        mode="paper_replay_only",
        order_execution_enabled=False,
        backtest_equivalent=result,
    )


def paper_summary(result: PaperLiveResult) -> dict[str, object]:
    payload = result.to_dict()
    payload["safety"] = {
        "real_order_routing": False,
        "uses_only_local_sample_data": True,
        "credentials_required": False,
    }
    return payload

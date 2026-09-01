from __future__ import annotations

from types import SimpleNamespace

from lumina_quant.strategies.taker_flow_exhaustion_reversal import (
    TakerFlowExhaustionReversalStrategy,
)


class _Queue:
    def __init__(self) -> None:
        self.items = []

    def put(self, item) -> None:
        self.items.append(item)


class _Bars:
    symbol_list = ["ETH/USDT"]

    def get_latest_feature_value(self, symbol, field):
        _ = symbol, field
        return None

    def get_latest_bar_value(self, symbol, field):
        _ = symbol, field
        return None


def _event(idx: int, close: float, *, hour: int = 14, buy: float = 100.0, sell: float = 100.0):
    return SimpleNamespace(
        type="MARKET",
        time=f"2026-05-02T{hour:02d}:{idx:02d}:00Z",
        symbol="ETH/USDT",
        close=close,
        funding_rate=0.00005,
        taker_buy_quote_volume=buy,
        taker_sell_quote_volume=sell,
    )


def test_taker_sell_exhaustion_emits_long_with_risk_metadata() -> None:
    queue = _Queue()
    strategy = TakerFlowExhaustionReversalStrategy(
        _Bars(),
        queue,
        flow_lookback_bars=4,
        momentum_lookback_bars=4,
        volatility_lookback_bars=4,
        evaluation_cadence_bars=1,
        flow_imbalance_min=0.20,
        price_extension_min=0.005,
        max_realized_volatility=0.20,
        entry_hours_utc="14",
    )

    for idx, close in enumerate([100.0, 100.0, 100.0, 100.0, 99.0], start=1):
        strategy.calculate_signals(_event(idx, close, buy=100.0, sell=900.0))

    assert queue.items
    signal = queue.items[0]
    assert signal.signal_type == "LONG"
    assert signal.metadata["reason"] == "taker_sell_exhaustion_reversal_long"
    assert signal.metadata["flow_imbalance"] <= -0.20
    assert signal.metadata["target_allocation"] <= 0.008
    assert signal.metadata["feature_coverage"]["taker_flow"] is True


def test_entry_hour_filter_blocks_other_sessions() -> None:
    queue = _Queue()
    strategy = TakerFlowExhaustionReversalStrategy(
        _Bars(),
        queue,
        flow_lookback_bars=4,
        momentum_lookback_bars=4,
        volatility_lookback_bars=4,
        evaluation_cadence_bars=1,
        flow_imbalance_min=0.20,
        price_extension_min=0.005,
        max_realized_volatility=0.20,
        entry_hours_utc="14",
    )

    for idx, close in enumerate([100.0, 100.0, 100.0, 100.0, 99.0], start=1):
        strategy.calculate_signals(_event(idx, close, hour=3, buy=100.0, sell=900.0))

    assert queue.items == []


def test_state_roundtrip_preserves_flow_state() -> None:
    queue = _Queue()
    strategy = TakerFlowExhaustionReversalStrategy(
        _Bars(), queue, flow_lookback_bars=4, momentum_lookback_bars=4, cooldown_bars=3
    )
    strategy.calculate_signals(_event(1, 100.0, buy=800.0, sell=200.0))
    strategy._state.cooldown_remaining = 2
    state = strategy.get_state()

    restored = TakerFlowExhaustionReversalStrategy(_Bars(), _Queue())
    restored.set_state(state)

    restored_state = restored.get_state()["state"]
    assert restored_state["closes"] == [100.0]
    assert restored_state["taker_buy_quote_volume"] == [800.0]
    assert restored_state["has_taker_flow"] is True
    assert restored_state["cooldown_remaining"] == 2

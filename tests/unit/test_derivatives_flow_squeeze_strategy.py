from __future__ import annotations

from types import SimpleNamespace

from lumina_quant.strategies.derivatives_flow_squeeze import DerivativesFlowSqueezeStrategy


class _Queue:
    def __init__(self) -> None:
        self.items = []

    def put(self, item) -> None:
        self.items.append(item)


class _Bars:
    symbol_list = ["BTC/USDT"]

    def get_latest_feature_value(self, symbol, field):
        _ = symbol, field
        return None

    def get_latest_bar_value(self, symbol, field):
        _ = symbol, field
        return None


def _event(idx: int, close: float, **features):
    return SimpleNamespace(
        type="MARKET",
        time=f"2026-05-02T00:{idx:02d}:00Z",
        symbol="BTC/USDT",
        open=close * 0.999,
        high=close * 1.001,
        low=close * 0.998,
        close=close,
        volume=1000.0,
        **features,
    )


def test_flow_continuation_emits_long_with_oi_and_taker_alignment() -> None:
    queue = _Queue()
    strategy = DerivativesFlowSqueezeStrategy(
        _Bars(),
        queue,
        evaluation_cadence_bars=1,
        momentum_lookback_bars=3,
        flow_lookback_bars=3,
        oi_lookback_bars=3,
        volatility_lookback_bars=4,
        continuation_momentum_min=0.0005,
        flow_imbalance_min=0.05,
        oi_delta_min=0.0,
        oi_delta_z_min=-10.0,
        target_allocation=0.02,
        max_order_value=500.0,
    )

    for idx, close in enumerate([100.0, 100.1, 100.25, 100.55, 100.90], start=1):
        strategy.calculate_signals(
            _event(
                idx,
                close,
                funding_rate=0.0001,
                open_interest=1_000_000.0 + idx * 2_000.0,
                taker_buy_quote_volume=700_000.0,
                taker_sell_quote_volume=300_000.0,
            )
        )

    assert queue.items
    signal = queue.items[0]
    assert signal.signal_type == "LONG"
    assert signal.metadata["reason"] == "flow_continuation_long"
    assert 0.0 < signal.metadata["target_allocation"] <= 0.02
    assert signal.metadata["feature_coverage"]["taker_flow"] is True


def test_liquidation_exhaustion_emits_long_after_forced_sell_reclaim() -> None:
    queue = _Queue()
    strategy = DerivativesFlowSqueezeStrategy(
        _Bars(),
        queue,
        evaluation_cadence_bars=1,
        enable_continuation=False,
        enable_exhaustion=True,
        momentum_lookback_bars=6,
        short_reclaim_bars=1,
        flow_lookback_bars=3,
        oi_lookback_bars=2,
        liquidation_window_bars=8,
        liquidation_z_min=1.5,
        liquidation_notional_min=1_000.0,
        price_shock_min=0.003,
        reclaim_min=0.001,
    )
    closes = [100.0, 99.8, 99.5, 99.1, 98.8, 98.4, 98.0, 98.35]
    for idx, close in enumerate(closes, start=1):
        strategy.calculate_signals(
            _event(
                idx,
                close,
                funding_rate=0.0001,
                open_interest=1_000_000.0 + idx * 100.0,
                liquidation_long_notional=1_000_000.0 if idx == len(closes) else 0.0,
                liquidation_short_notional=0.0,
                taker_buy_quote_volume=600_000.0,
                taker_sell_quote_volume=400_000.0,
            )
        )

    assert queue.items
    signal = queue.items[0]
    assert signal.signal_type == "LONG"
    assert signal.metadata["reason"] == "liquidation_exhaustion_long"
    assert signal.metadata["liquidation_long_z"] >= 1.5
    assert signal.metadata["feature_coverage"]["liquidation"] is True


def test_volatility_governor_scales_target_allocation() -> None:
    queue = _Queue()
    strategy = DerivativesFlowSqueezeStrategy(
        _Bars(),
        queue,
        evaluation_cadence_bars=1,
        momentum_lookback_bars=3,
        flow_lookback_bars=3,
        oi_lookback_bars=3,
        volatility_lookback_bars=5,
        continuation_momentum_min=0.0001,
        flow_imbalance_min=0.01,
        oi_delta_min=0.0,
        oi_delta_z_min=-10.0,
        volatility_target_per_bar=0.001,
        min_volatility_multiplier=0.20,
        max_volatility_multiplier=1.0,
        volatility_hard_cap=0.20,
        target_allocation=0.05,
    )
    for idx, close in enumerate([100.0, 104.0, 101.0, 107.0, 111.0], start=1):
        strategy.calculate_signals(
            _event(
                idx,
                close,
                funding_rate=0.0001,
                open_interest=1_000_000.0 + idx * 5_000.0,
                taker_buy_quote_volume=800_000.0,
                taker_sell_quote_volume=200_000.0,
            )
        )

    assert queue.items
    target = queue.items[0].metadata["target_allocation"]
    assert 0.0 < target < 0.05
    assert queue.items[0].metadata["volatility_multiplier"] < 1.0


def test_state_roundtrip_preserves_symbol_state() -> None:
    queue = _Queue()
    strategy = DerivativesFlowSqueezeStrategy(
        _Bars(),
        queue,
        evaluation_cadence_bars=1,
        momentum_lookback_bars=2,
        flow_lookback_bars=2,
        oi_lookback_bars=2,
    )
    strategy.calculate_signals(
        _event(
            1,
            100.0,
            funding_rate=0.0001,
            open_interest=1_000_000.0,
            taker_buy_quote_volume=500_000.0,
            taker_sell_quote_volume=500_000.0,
        )
    )
    state = strategy.get_state()

    restored = DerivativesFlowSqueezeStrategy(_Bars(), _Queue())
    restored.set_state(state)
    restored_state = restored.get_state()["symbol_state"]["BTC/USDT"]
    assert restored_state["closes"] == [100.0]
    assert restored_state["funding_rate"] == [0.0001]

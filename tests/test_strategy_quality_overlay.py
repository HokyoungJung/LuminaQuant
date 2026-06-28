from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from lumina_quant.configuration.loader import build_runtime_config
from lumina_quant.backtesting._config_view import BacktestConfigView
from lumina_quant.core.events import SignalEvent
from lumina_quant.portfolio.strategy_quality import StrategyQualityOverlay


@dataclass
class _Bars:
    closes: list[float]
    volumes: list[float]

    def get_latest_bars_values(self, symbol, val_type, N=1):
        del symbol
        source = self.volumes if val_type == "volume" else self.closes
        return list(source)[-N:]

    def get_latest_bars(self, symbol, N=1):
        del symbol
        now = datetime(2026, 1, 1, tzinfo=UTC)
        rows = []
        for idx, close in enumerate(self.closes[-N:]):
            rows.append(
                (
                    now + timedelta(minutes=idx),
                    close * 0.999,
                    close * 1.002,
                    close * 0.998,
                    close,
                    self.volumes[-N + idx] if len(self.volumes) >= N else 1.0,
                )
            )
        return rows


def _config(**overrides):
    data = {
        "strategy_quality": {
            "enabled": True,
            "allow_unknown_edge": False,
            "min_expected_edge_bps": 10.0,
            "edge_cost_buffer_bps": 0.0,
            "target_vol_per_bar": 0.001,
            "max_daily_turnover_pct": 3.0,
            **overrides,
        },
        "execution": {
            "maker_fee_rate": 0.0,
            "taker_fee_rate": 0.0,
            "spread_rate": 0.0,
        },
        "backtest": {"slippage_rate": 0.0},
    }
    return BacktestConfigView(build_runtime_config(data, env={}))


def _signal(strategy_id="rsi", signal_type="LONG", metadata=None):
    return SignalEvent(
        strategy_id=strategy_id,
        symbol="BTC/USDT",
        datetime=datetime(2026, 1, 1, tzinfo=UTC),
        signal_type=signal_type,
        strength=1.0,
        metadata=dict(metadata or {}),
    )


def test_strategy_quality_edge_gate_blocks_low_edge_signal():
    overlay = StrategyQualityOverlay(_config())
    bars = _Bars(closes=[100.0, 100.1, 100.2, 100.3, 100.4], volumes=[10.0] * 5)

    decision = overlay.apply(
        _signal(metadata={"expected_edge_bps": 5.0}),
        bars=bars,
        current_price=100.4,
        current_equity=10_000.0,
    )

    assert decision.signal is None
    assert decision.blocked_reason == "edge_below_cost_floor"


def test_strategy_quality_regime_router_blocks_reversion_in_strong_trend():
    overlay = StrategyQualityOverlay(_config(min_expected_edge_bps=1.0, regime_lookback_bars=5))
    bars = _Bars(closes=[100.0, 101.0, 102.0, 103.0, 104.0, 105.0], volumes=[10.0] * 6)

    decision = overlay.apply(
        _signal(strategy_id="rsi", metadata={"expected_edge_bps": 50.0}),
        bars=bars,
        current_price=105.0,
        current_equity=10_000.0,
    )

    assert decision.signal is None
    assert decision.blocked_reason == "regime_router_block"


def test_strategy_quality_scales_position_and_attaches_exit_overlay():
    overlay = StrategyQualityOverlay(
        _config(
            min_expected_edge_bps=1.0,
            allow_unknown_edge=True,
            trend_return_bps=10_000.0,
        )
    )
    bars = _Bars(
        closes=[100.0, 101.0, 100.5, 101.5, 101.0, 102.0, 101.8, 102.2],
        volumes=[10.0] * 8,
    )

    decision = overlay.apply(
        _signal(
            strategy_id="rolling_breakout",
            metadata={"expected_edge_bps": 80.0, "target_allocation": 0.10},
        ),
        bars=bars,
        current_price=102.2,
        current_equity=10_000.0,
    )

    assert decision.signal is not None
    metadata = decision.signal.metadata or {}
    assert 0.0 < metadata["target_allocation"] <= 0.10
    assert metadata["strategy_quality"]["strategy_quality_scale"] <= 1.0
    assert decision.signal.stop_loss is not None
    assert decision.signal.take_profit is not None
    assert decision.signal.trailing_percent is not None


def test_strategy_quality_blocks_pair_correlation_breakdown():
    overlay = StrategyQualityOverlay(_config(min_expected_edge_bps=1.0))
    bars = _Bars(closes=[100.0, 100.1, 100.2, 100.3, 100.4], volumes=[10.0] * 5)

    decision = overlay.apply(
        _signal(
            strategy_id="pair_zscore",
            metadata={"expected_edge_bps": 50.0, "correlation": 0.1},
        ),
        bars=bars,
        current_price=100.4,
        current_equity=10_000.0,
    )

    assert decision.signal is None
    assert decision.blocked_reason == "pair_correlation_breakdown"


def test_strategy_quality_runtime_config_round_trip():
    runtime = build_runtime_config(
        {
            "strategy_quality": {
                "enabled": True,
                "models": "ignored",
                "min_expected_edge_bps": -5,
                "max_daily_turnover_pct": 2.5,
            }
        },
        env={"LQ__STRATEGY_QUALITY__PAIR_MIN_CORRELATION": "0.45"},
    )

    assert runtime.strategy_quality.enabled is True
    assert runtime.strategy_quality.min_expected_edge_bps == 0.0
    assert runtime.strategy_quality.max_daily_turnover_pct == 2.5
    assert runtime.strategy_quality.pair_min_correlation == 0.45

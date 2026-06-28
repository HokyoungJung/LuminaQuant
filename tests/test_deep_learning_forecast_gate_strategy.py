from __future__ import annotations

import queue
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from lumina_quant.core.events import MarketEvent
from lumina_quant.data.deep_learning_forecasts import DeepLearningForecastStore
from lumina_quant.strategies.deep_learning_forecast_gate import DeepLearningForecastGateStrategy
from lumina_quant.strategies.registry import get_strategy_param_schema, resolve_strategy_class


@dataclass
class _BarStore:
    symbol_list: list[str]
    close: float = 100.0

    def get_latest_bar_value(self, symbol: str, value_type: str) -> float:
        _ = symbol, value_type
        return float(self.close)


def _drain(events: queue.Queue) -> list:
    out = []
    while not events.empty():
        out.append(events.get())
    return out


def _write_deep_learning_exports(path, *, returns: dict[str, float], pred_date: datetime) -> None:
    rows = ["model_code,dbcode,pred_date,Date,value,confidence"]
    target = pred_date + timedelta(hours=1)
    for model, ret in returns.items():
        value = 100.0 * (1.0 + float(ret))
        rows.append(
            f"{model}_BTC_close_multi_1h_scale_120_1_flow0,BTC_close,"
            f"{pred_date.isoformat()},{target.isoformat()},{value},0.90"
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_deep_learning_forecast_store_parses_cmamba_patchtst_exports(tmp_path) -> None:
    pred_date = datetime(2026, 1, 1, tzinfo=UTC)
    artifact = tmp_path / "predictions.csv"
    _write_deep_learning_exports(
        artifact,
        returns={
            "FITS": 0.004,
            "CycleNet": 0.005,
            "CMamba": 0.006,
            "PatchTST": 0.0045,
        },
        pred_date=pred_date,
    )

    store = DeepLearningForecastStore(artifact)
    snapshot = store.snapshot(
        "BTC/USDT",
        pred_date + timedelta(minutes=5),
        current_price=100.0,
        return_threshold=0.001,
        max_age_seconds=3600,
        horizon_seconds=3600,
    )

    assert snapshot is not None
    assert snapshot.symbol == "BTC/USDT"
    assert set(snapshot.model_returns) == {"FITS", "CycleNet", "CMamba", "PatchTST"}
    assert snapshot.long_vote_fraction == 1.0
    assert snapshot.short_vote_fraction == 0.0
    assert snapshot.mean_return > 0.004


def test_deep_learning_forecast_gate_strategy_emits_consensus_long(tmp_path) -> None:
    pred_date = datetime(2026, 1, 1, tzinfo=UTC)
    artifact = tmp_path / "predictions.csv"
    _write_deep_learning_exports(
        artifact,
        returns={
            "FITS": 0.004,
            "CycleNet": 0.005,
            "CMamba": 0.006,
            "PatchTST": 0.0045,
        },
        pred_date=pred_date,
    )
    events: queue.Queue = queue.Queue()
    strategy = DeepLearningForecastGateStrategy(
        _BarStore(["BTC/USDT"]),
        events,
        forecast_path=str(artifact),
        entry_threshold_bps=10.0,
        min_model_agreement=0.75,
        max_dispersion_bps=50.0,
        target_allocation=0.20,
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
    )

    strategy.calculate_signals(
        MarketEvent(
            time=pred_date + timedelta(minutes=10),
            symbol="BTC/USDT",
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=10.0,
        )
    )
    signals = _drain(events)

    assert [signal.signal_type for signal in signals] == ["LONG"]
    signal = signals[0]
    assert signal.position_side == "LONG"
    assert signal.stop_loss == 98.0
    assert signal.take_profit == 104.0
    assert signal.metadata["strategy"] == "DeepLearningForecastGateStrategy"
    assert set(signal.metadata["models"]) == {"FITS", "CycleNet", "CMamba", "PatchTST"}
    assert 0.0 < signal.metadata["target_allocation"] <= 0.20


def test_deep_learning_forecast_gate_strategy_blocks_high_dispersion(tmp_path) -> None:
    pred_date = datetime(2026, 1, 1, tzinfo=UTC)
    artifact = tmp_path / "predictions.csv"
    _write_deep_learning_exports(
        artifact,
        returns={
            "FITS": 0.010,
            "CycleNet": 0.009,
            "CMamba": -0.008,
            "PatchTST": -0.009,
        },
        pred_date=pred_date,
    )
    events: queue.Queue = queue.Queue()
    strategy = DeepLearningForecastGateStrategy(
        _BarStore(["BTC/USDT"]),
        events,
        forecast_path=str(artifact),
        entry_threshold_bps=10.0,
        min_model_agreement=0.75,
        max_dispersion_bps=20.0,
    )

    strategy.calculate_signals(
        MarketEvent(
            time=pred_date + timedelta(minutes=10),
            symbol="BTC/USDT",
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=10.0,
        )
    )

    assert _drain(events) == []


def test_deep_learning_forecast_gate_strategy_is_registry_discoverable() -> None:
    assert (
        resolve_strategy_class("DeepLearningForecastGateStrategy")
        is DeepLearningForecastGateStrategy
    )
    schema = get_strategy_param_schema("DeepLearningForecastGateStrategy")
    assert "forecast_path" in schema
    assert "entry_threshold_bps" in schema

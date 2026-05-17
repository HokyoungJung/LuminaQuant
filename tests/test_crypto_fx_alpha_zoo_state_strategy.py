from __future__ import annotations

import importlib.util
import json
import queue
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from lumina_quant.core.events import MarketBatchEvent, MarketEvent
from lumina_quant.core.market_window_contract import build_market_window_event
from lumina_quant.strategies.crypto_fx_alpha_zoo_state import CryptoFxAlphaZooStateStrategy
from lumina_quant.strategies.registry import get_strategy_map, get_strategy_tier

_REPLAY_SPEC = importlib.util.spec_from_file_location("replay_crypto_fx_alpha_zoo_state", Path("scripts/research/replay_crypto_fx_alpha_zoo_state.py"))
_REPLAY_MODULE = importlib.util.module_from_spec(_REPLAY_SPEC)
assert _REPLAY_SPEC.loader is not None
sys.modules[_REPLAY_SPEC.name] = _REPLAY_MODULE
_REPLAY_SPEC.loader.exec_module(_REPLAY_MODULE)
_GridSpec = _REPLAY_MODULE._GridSpec
_build_trades = _REPLAY_MODULE._build_trades
_liquidation_lanes = _REPLAY_MODULE._liquidation_lanes
_paper_forward_diagnostics = _REPLAY_MODULE._paper_forward_diagnostics
replay_frame = _REPLAY_MODULE.replay_frame

_SUMMARY_SPEC = importlib.util.spec_from_file_location(
    "write_crypto_fx_alpha_zoo_real_data_summary", Path("scripts/research/write_crypto_fx_alpha_zoo_real_data_summary.py")
)
_SUMMARY_MODULE = importlib.util.module_from_spec(_SUMMARY_SPEC)
assert _SUMMARY_SPEC.loader is not None
sys.modules[_SUMMARY_SPEC.name] = _SUMMARY_MODULE
_SUMMARY_SPEC.loader.exec_module(_SUMMARY_MODULE)
build_summary_payload = _SUMMARY_MODULE.build_summary_payload


@dataclass(slots=True)
class _Bars:
    symbol_list: list[str]


def _batch(ts: int, eth_mult: float = 1.0, sol_mult: float = 1.0) -> MarketBatchEvent:
    btc = 100.0 * (1.0 + 0.001 * ts)
    eth = 100.0 * (1.0 + 0.006 * ts) * eth_mult
    sol = 100.0 * (1.0 - 0.004 * ts) * sol_mult
    eurusd = 1.10 * (1.0 - 0.0001 * ts)
    gbpusd = 1.25 * (1.0 - 0.0001 * ts)
    audusd = 0.65 * (1.0 + 0.0002 * ts)
    usdjpy = 145.0 * (1.0 + 0.0001 * ts)
    bars = []
    for symbol, close, volume in (
        ("BTC/USDT", btc, 1000.0 + ts),
        ("ETH/USDT", eth, 1200.0 + 20.0 * ts),
        ("SOL/USDT", sol, 1100.0 + 5.0 * ts),
        ("EURUSD", eurusd, 0.0),
        ("GBPUSD", gbpusd, 0.0),
        ("AUDUSD", audusd, 0.0),
        ("USDJPY", usdjpy, 0.0),
    ):
        bars.append(MarketEvent(ts, symbol, close * 0.999, close * 1.002, close * 0.998, close, volume))
    return MarketBatchEvent(time=ts, bars=tuple(bars))


def _window_from_batch(batch: MarketBatchEvent):
    bars_1s = {
        bar.symbol: (
            (
                int(batch.time),
                float(bar.open),
                float(bar.high),
                float(bar.low),
                float(bar.close),
                float(bar.volume),
            ),
        )
        for bar in batch.bars
    }
    return build_market_window_event(
        time=batch.time,
        window_seconds=3600,
        bars_1s=bars_1s,
        event_time_watermark_ms=int(batch.time),
        commit_id=None,
        lag_ms=0,
        is_stale=False,
        emit_metrics=False,
    )


def _run_strategy(*, require_edge: bool = True, calibrated_edges: dict[str, float] | None = None):
    events: queue.Queue = queue.Queue()
    strategy = CryptoFxAlphaZooStateStrategy(
        _Bars(["BTC/USDT", "ETH/USDT", "SOL/USDT", "EURUSD", "GBPUSD", "AUDUSD", "USDJPY"]),
        events,
        fast_lookback_bars=2,
        slow_lookback_bars=8,
        history_window=16,
        entry_threshold=0.15,
        exit_threshold=0.02,
        use_fx_filter=False,
        require_calibrated_edge=require_edge,
        calibrated_edges=calibrated_edges or {},
        max_longs=1,
        max_shorts=1,
    )
    for ts in range(30):
        strategy.calculate_signals(_batch(ts))
    signals = []
    while not events.empty():
        signals.append(events.get())
    return strategy, signals


def test_strategy_is_registered_live_opt_in_and_calendar_safe() -> None:
    strategy_map = get_strategy_map()
    assert strategy_map["CryptoFxAlphaZooStateStrategy"] is CryptoFxAlphaZooStateStrategy
    assert get_strategy_tier("CryptoFxAlphaZooStateStrategy") == "live_opt_in"
    assert CryptoFxAlphaZooStateStrategy.strategy_validity["calendar_primary"] is False
    assert CryptoFxAlphaZooStateStrategy.strategy_validity["locked_oos_role"] == "gate_report_only"


def test_strategy_requires_calibrated_edge_for_entries() -> None:
    _, blocked = _run_strategy(require_edge=True, calibrated_edges={})
    assert [signal for signal in blocked if signal.signal_type in {"LONG", "SHORT"}] == []

    _, allowed = _run_strategy(require_edge=True, calibrated_edges={"default:LONG": 5.0, "default:SHORT": 5.0})
    assert any(signal.signal_type in {"LONG", "SHORT"} for signal in allowed)
    first_entry = next(signal for signal in allowed if signal.signal_type in {"LONG", "SHORT"})
    assert first_entry.symbol.endswith("USDT")
    assert first_entry.metadata["uses_locked_oos_for_selection"] is False
    assert first_entry.metadata["calibrated_lower_bound_edge_bps"] == 5.0
    assert first_entry.metadata["dominant_factor_family"]
    assert first_entry.metadata["factor_family_scores"]


def test_strategy_state_roundtrip_preserves_signal_sequence() -> None:
    events_full: queue.Queue = queue.Queue()
    full = CryptoFxAlphaZooStateStrategy(
        _Bars(["BTC/USDT", "ETH/USDT", "SOL/USDT", "EURUSD", "GBPUSD", "AUDUSD", "USDJPY"]),
        events_full,
        fast_lookback_bars=2,
        slow_lookback_bars=8,
        history_window=16,
        entry_threshold=0.15,
        use_fx_filter=False,
        calibrated_edges={"default:LONG": 5.0, "default:SHORT": 5.0},
    )
    for ts in range(30):
        full.calculate_signals(_batch(ts))
    full_signals = [(item.datetime, item.symbol, item.signal_type) for item in list(events_full.queue)]

    events_split: queue.Queue = queue.Queue()
    split_a = CryptoFxAlphaZooStateStrategy(
        _Bars(["BTC/USDT", "ETH/USDT", "SOL/USDT", "EURUSD", "GBPUSD", "AUDUSD", "USDJPY"]),
        events_split,
        fast_lookback_bars=2,
        slow_lookback_bars=8,
        history_window=16,
        entry_threshold=0.15,
        use_fx_filter=False,
        calibrated_edges={"default:LONG": 5.0, "default:SHORT": 5.0},
    )
    for ts in range(15):
        split_a.calculate_signals(_batch(ts))
    state = split_a.get_state()
    split_b = CryptoFxAlphaZooStateStrategy(
        _Bars(["BTC/USDT", "ETH/USDT", "SOL/USDT", "EURUSD", "GBPUSD", "AUDUSD", "USDJPY"]),
        events_split,
        fast_lookback_bars=2,
        slow_lookback_bars=8,
        history_window=16,
        entry_threshold=0.15,
        use_fx_filter=False,
        calibrated_edges={"default:LONG": 5.0, "default:SHORT": 5.0},
    )
    split_b.set_state(state)
    for ts in range(15, 30):
        split_b.calculate_signals(_batch(ts))
    split_signals = [(item.datetime, item.symbol, item.signal_type) for item in list(events_split.queue)]
    assert split_signals == full_signals


def test_market_window_live_path_matches_batch_path_for_hourly_alpha_zoo_decisions() -> None:
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "EURUSD", "GBPUSD", "AUDUSD", "USDJPY"]
    params = {
        "fast_lookback_bars": 2,
        "slow_lookback_bars": 8,
        "history_window": 16,
        "entry_threshold": 0.15,
        "exit_threshold": 0.02,
        "use_fx_filter": False,
        "calibrated_edges": {"default:LONG": 5.0, "default:SHORT": 5.0},
        "max_longs": 1,
        "max_shorts": 1,
        "decision_cadence_seconds": 3600,
    }
    batch_events: queue.Queue = queue.Queue()
    window_events: queue.Queue = queue.Queue()
    batch_strategy = CryptoFxAlphaZooStateStrategy(_Bars(symbols), batch_events, **params)
    window_strategy = CryptoFxAlphaZooStateStrategy(_Bars(symbols), window_events, **params)

    for ts in range(30):
        batch = _batch(ts)
        batch_strategy.calculate_signals(MarketBatchEvent(time=int(batch.time) * 1000, bars=batch.bars))
        window_strategy.calculate_signals_window(_window_from_batch(batch), None)

    batch_signals = [
        (item.datetime, item.symbol, item.signal_type, round(float(item.price), 8))
        for item in list(batch_events.queue)
    ]
    window_signals = [
        (item.datetime, item.symbol, item.signal_type, round(float(item.price), 8))
        for item in list(window_events.queue)
    ]

    assert batch_strategy.decision_cadence_seconds == 3600
    assert window_strategy.decision_cadence_seconds == 3600
    assert window_signals == batch_signals


def test_replay_grid_hides_locked_oos_until_after_train_validation_selection() -> None:
    rows = []
    for ts in range(60):
        timestamp = pd.Timestamp("2026-01-01") + pd.Timedelta(hours=ts)
        split = "train" if ts < 30 else "validation" if ts < 45 else "locked_oos"
        for symbol, drift in (("BTC/USDT", 0.001), ("ETH/USDT", 0.004), ("SOL/USDT", -0.002)):
            close = 100.0 * (1.0 + drift * ts)
            rows.append(
                {
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "open": close * 0.999,
                    "high": close * 1.002,
                    "low": close * 0.998,
                    "close": close,
                    "volume": 1000.0 + ts,
                    "split": split,
                }
            )
    payload = replay_frame(
        pd.DataFrame(rows),
        require_calibrated_edge=True,
        calibrated_edges={"default:LONG": 5.0, "default:SHORT": 5.0},
        strategy_params={"fast_lookback_bars": 2, "slow_lookback_bars": 8, "history_window": 16, "entry_threshold": 0.15},
        grid_specs=[
            _GridSpec("low_threshold", "unit", {"entry_threshold": 0.15}),
            _GridSpec("higher_threshold", "unit", {"entry_threshold": 0.30}),
        ],
    )
    grid = payload["candidate_selection_grid"]
    assert grid["uses_locked_oos_for_selection"] is False
    assert grid["locked_oos_calibration_record_count"] == 0
    for row in grid["rows"]:
        assert set(row["selection_metrics"]) == {"train", "validation"}
        assert row["locked_oos_metrics_visible_during_selection"] is False
        assert row["uses_locked_oos_for_selection"] is False
    assert payload["selection_provenance"]["candidate_freeze_before_locked_oos_gate"] is True


def test_return_mdd_reference_hurdle_is_diagnostic_not_promotion_gate() -> None:
    data = pd.DataFrame(columns=["timestamp", "symbol", "open", "high", "low", "close", "volume", "split"])
    trades = [
        {
            "symbol": "BTC/USDT",
            "side": "LONG",
            "entry_time": pd.Timestamp("2026-01-01T00:00:00Z"),
            "exit_time": pd.Timestamp("2026-01-01T01:00:00Z"),
            "entry_price": 100.0,
            "entry_split": "locked_oos",
            "gross_return": gross_return,
        }
        for gross_return in (0.20, -0.05, 0.02)
    ]

    lanes = _liquidation_lanes(data, trades, allocation_fraction=1.0, max_leverage=1)
    strict_lane = lanes["strict_zero_liquidation_lane"]
    promoted = strict_lane["promoted_candidate"]

    assert promoted["strict_safe"] is True
    assert promoted["performance_gates"]["oos_return_beats_current_base"] is True
    assert "oos_return_mdd_beats_current_base" not in promoted["performance_gates"]
    assert promoted["performance_diagnostics"]["oos_return_mdd_beats_current_base"] is False
    assert promoted["performance_diagnostics"]["return_mdd_hurdle_required"] is False
    assert promoted["performance_diagnostics"]["return_mdd_role"] == "diagnostic_report_only"
    assert promoted["deployable_success"] is True


def test_replay_preserves_exit_reason_from_strategy_metadata() -> None:
    data = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-01-01T00:00:00Z"),
                "symbol": "BTC/USDT",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1.0,
                "split": "locked_oos",
            },
            {
                "timestamp": pd.Timestamp("2026-01-01T01:00:00Z"),
                "symbol": "BTC/USDT",
                "open": 105.0,
                "high": 106.0,
                "low": 104.0,
                "close": 105.0,
                "volume": 1.0,
                "split": "locked_oos",
            },
        ]
    )
    signals = [
        {
            "datetime": pd.Timestamp("2026-01-01T00:00:00Z").isoformat(),
            "symbol": "BTC/USDT",
            "signal_type": "LONG",
            "price": 100.0,
            "metadata": {"fx_risk_state": "risk_on", "dominant_factor_family": "crypto_residual_momentum"},
        },
        {
            "datetime": pd.Timestamp("2026-01-01T01:00:00Z").isoformat(),
            "symbol": "BTC/USDT",
            "signal_type": "EXIT",
            "price": 105.0,
            "metadata": {"exit_reason_detail": "take_profit"},
        },
    ]

    trades = _build_trades(data, signals)

    assert trades[0]["exit_reason"] == "take_profit"
    assert trades[0]["exit_metadata"]["exit_reason_detail"] == "take_profit"


def test_paper_forward_diagnostics_report_breakdowns_and_cost_sensitivity() -> None:
    trades = [
        {
            "symbol": "BTC/USDT",
            "side": "LONG",
            "entry_time": pd.Timestamp("2026-01-01T00:00:00Z"),
            "exit_time": pd.Timestamp("2026-01-01T12:00:00Z"),
            "entry_split": "train",
            "gross_return": 0.03,
            "exit_reason": "take_profit",
            "entry_metadata": {
                "fx_risk_state": "risk_on",
                "dominant_factor_family": "crypto_residual_momentum",
                "factor_family_scores": {"crypto_residual_momentum": 1.2},
            },
        },
        {
            "symbol": "ETH/USDT",
            "side": "SHORT",
            "entry_time": pd.Timestamp("2026-01-02T00:00:00Z"),
            "exit_time": pd.Timestamp("2026-01-02T06:00:00Z"),
            "entry_split": "validation",
            "gross_return": -0.01,
            "exit_reason": "stop_loss",
            "entry_metadata": {
                "fx_risk_state": "risk_off",
                "dominant_factor_family": "breakout_failure",
                "factor_family_scores": {"breakout_failure": -1.0},
            },
        },
        {
            "symbol": "SOL/USDT",
            "side": "SHORT",
            "entry_time": pd.Timestamp("2026-01-03T00:00:00Z"),
            "exit_time": pd.Timestamp("2026-01-04T00:00:00Z"),
            "entry_split": "locked_oos",
            "gross_return": 0.04,
            "exit_reason": "time_exit",
            "entry_metadata": {
                "fx_risk_state": "risk_off",
                "dominant_factor_family": "trend_efficiency",
                "factor_family_scores": {"trend_efficiency": -0.8},
            },
        },
    ]

    diagnostics = _paper_forward_diagnostics(trades, leverage=6.0, allocation_fraction=0.10, candidate_name="unit")
    breakdowns = diagnostics["breakdowns"]

    assert diagnostics["promotion_allowed"] is False
    assert "risk_off" in breakdowns["by_regime"]["groups"]
    assert "SOL/USDT" in breakdowns["by_symbol"]["groups"]
    assert "SHORT" in breakdowns["by_side"]["groups"]
    assert "trend_efficiency" in breakdowns["by_factor_family"]["groups"]
    assert "time_exit" in breakdowns["by_exit_reason"]["groups"]

    slippage_rows = diagnostics["slippage_sensitivity"]["rows"]
    funding_rows = diagnostics["funding_cost_sensitivity"]["rows"]
    assert slippage_rows[-1]["locked_oos"]["total_return"] < slippage_rows[0]["locked_oos"]["total_return"]
    assert funding_rows[-1]["locked_oos"]["total_return"] < funding_rows[0]["locked_oos"]["total_return"]


def test_real_data_summary_fails_closed_when_return_mdd_is_a_strict_gate(tmp_path: Path) -> None:
    screen_path = tmp_path / "screen.json"
    calibration_path = tmp_path / "calibration.json"
    replay_path = tmp_path / "replay.json"
    screen_path.write_text(
        json.dumps(
            {
                "factor_count": 1,
                "row_count": 1,
                "calendar_primary": False,
                "uses_locked_oos_for_selection": False,
                "screen": {"selected_factors": []},
                "source_coverage": {"input": {"symbols": ["BTC/USDT"]}},
                "candidate_outcome_ledger": {"record_count": 1, "train_validation_record_count": 1, "locked_oos_record_count": 0},
            }
        ),
        encoding="utf-8",
    )
    calibration_path.write_text(
        json.dumps(
            {
                "calibration_policy": "physical_train_validation_record_filter_before_bucket_estimation",
                "input_record_count": 1,
                "calibration_record_count": 1,
                "locked_oos_calibration_record_count": 0,
                "calibrated_edges_for_strategy": {"default:LONG": 1.0},
            }
        ),
        encoding="utf-8",
    )
    replay_path.write_text(
        json.dumps(
            {
                "promotion_policy": {"return_mdd_hurdle_required": True, "return_mdd_role": "strict_promotion_gate"},
                "integer_grid_results": [
                    {
                        "performance_gates": {
                            "oos_return_beats_current_base": True,
                            "oos_return_mdd_beats_current_base": False,
                        },
                        "performance_diagnostics": {
                            "oos_return_mdd_beats_current_base": False,
                            "return_mdd_hurdle_required": True,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    try:
        build_summary_payload(
            screen_path=screen_path,
            calibration_path=calibration_path,
            replay_path=replay_path,
            output_json_path=tmp_path / "summary.json",
            output_md_path=tmp_path / "summary.md",
        )
    except ValueError as exc:
        assert "return/MDD" in str(exc)
    else:
        raise AssertionError("summary writer must fail closed if return/MDD is a strict promotion gate")

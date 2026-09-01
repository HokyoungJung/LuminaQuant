"""Deterministic unit tests for the micro-signal-informed >=30m alpha sleeves.

These tests exercise the four sleeves entirely through the local
MARKET_WINDOW event / bar contract (no backtest, no data fetch):

- registry presence for all four classes;
- each sleeve pins ``decision_cadence_seconds == 1800``;
- #1 IntradayFlowPressureRiderStrategy goes LONG on persistent positive taker
  flow and SHORT on persistent negative flow (taker volumes stubbed via
  ``bars.get_latest_feature_value``), then rides;
- #2 VolOfVolRegimeTrendGateStrategy up-sizes a clean low-GK trend and
  down-sizes/vetoes a high-GK choppy trend (allocation differs by regime);
- #5 VWAPCompressionReversionStrategy only acts when the compression regime is
  active and fades the VWAP deviation;
- #6 SeasonalMicroBreakoutRiderStrategy requires a range breakout, favorable
  one-bar-deferred intraday slot drift, and fresh micro confirmation;
- candidate-wiring asserts ``decision_cadence_seconds == 1800`` and timeframes
  are a subset of {30m, 1h, 4h, 1d}.
"""

from __future__ import annotations

import math
import queue
from dataclasses import dataclass, field
from typing import Any

from lumina_quant.core.market_window_contract import build_market_window_event
from lumina_quant.strategies.micro_signal_alpha_sleeves import (
    IntradayFlowPressureRiderStrategy,
    SeasonalMicroBreakoutRiderStrategy,
    VolOfVolRegimeTrendGateStrategy,
    VWAPCompressionReversionStrategy,
)
from lumina_quant.strategies.registry import get_strategy_map
from lumina_quant.strategy_factory.candidate_library import _CandidateBuildContext
import lumina_quant.strategy_factory.candidate_library as candidate_library


MICRO_STRATEGIES = (
    "IntradayFlowPressureRiderStrategy",
    "VolOfVolRegimeTrendGateStrategy",
    "VWAPCompressionReversionStrategy",
    "SeasonalMicroBreakoutRiderStrategy",
)

_TF = {"30m", "1h", "4h", "1d"}


@dataclass(slots=True)
class _Bars:
    """Minimal bars stub: symbol_list + optional latched feature values."""

    symbol_list: list[str]
    feature_values: dict[tuple[str, str], float] = field(default_factory=dict)

    def get_latest_feature_value(self, symbol: str, field_name: str) -> float | None:
        return self.feature_values.get((str(symbol), str(field_name)))


_WINDOW_SECONDS = 1800
_STEP_MS = _WINDOW_SECONDS * 1000


def _ohlc_rows(
    ts: int,
    *,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: float,
    sub_closes: list[float] | None = None,
) -> tuple[tuple[Any, ...], ...]:
    """Build 1s rows for one window. When ``sub_closes`` is given, emit one row
    per sub-tick (a richer micro tape for tick-agreement / VWAP reads)."""
    if sub_closes is None:
        return (
            (
                int(ts),
                float(open_),
                float(high),
                float(low),
                float(close),
                float(volume),
            ),
        )
    rows: list[tuple[Any, ...]] = []
    n = len(sub_closes)
    per_vol = float(volume) / float(max(1, n))
    prev = float(open_)
    for i, sc in enumerate(sub_closes):
        rows.append(
            (
                int(ts) + i,
                float(prev),
                float(max(prev, sc)),
                float(min(prev, sc)),
                float(sc),
                per_vol,
            )
        )
        prev = float(sc)
    return tuple(rows)


def _window(
    ts: int,
    rows_by_symbol: dict[str, tuple[tuple[Any, ...], ...]],
) -> Any:
    return build_market_window_event(
        time=ts,
        window_seconds=_WINDOW_SECONDS,
        bars_1s=rows_by_symbol,
        event_time_watermark_ms=int(ts),
        commit_id=None,
        lag_ms=0,
        is_stale=False,
        emit_metrics=False,
    )


def _drain(events: queue.Queue) -> list[Any]:
    out: list[Any] = []
    while not events.empty():
        out.append(events.get())
    return out


# --------------------------------------------------------------------------- #
# registry + cadence
# --------------------------------------------------------------------------- #
def test_registry_presence() -> None:
    strategy_map = get_strategy_map()
    for name in MICRO_STRATEGIES:
        assert name in strategy_map, f"{name} missing from registry"


def test_decision_cadence_is_1800() -> None:
    for cls in (
        IntradayFlowPressureRiderStrategy,
        VolOfVolRegimeTrendGateStrategy,
        VWAPCompressionReversionStrategy,
        SeasonalMicroBreakoutRiderStrategy,
    ):
        assert cls.decision_cadence_seconds == 1800
        assert cls.preferred_contract == "market_window"
        assert cls.uses_timeframe_aggregator is False


# --------------------------------------------------------------------------- #
# #1 IntradayFlowPressureRiderStrategy
# --------------------------------------------------------------------------- #
def _run_flow_rider(*, flow_sign: float) -> list[Any]:
    symbol = "BTC/USDT"
    bars = _Bars(symbol_list=[symbol])
    events: queue.Queue = queue.Queue()
    strat = IntradayFlowPressureRiderStrategy(
        bars,
        events,
        flow_z_window=6,
        entry_z=1.0,
        pyramid_z=1.5,
        tick_agree_frac=0.5,
        require_vwap_edge=False,
        atr_period=5,
        vol_window=8,
        min_price=0.0,
        max_hold_bars=200,
        add_step_atr=0.5,
    )
    price = 100.0
    ts = _STEP_MS
    emitted: list[Any] = []
    for step in range(40):
        # Persistent directional drift so the micro tick-agreement confirms flow.
        drift = 0.4 * flow_sign
        sub = [price + drift * (k + 1) / 6.0 for k in range(6)]
        close = sub[-1]
        rows = _ohlc_rows(
            ts,
            open_=price,
            high=max(price, close) + 0.05,
            low=min(price, close) - 0.05,
            close=close,
            volume=1000.0,
            sub_closes=sub,
        )
        # Stub window-latched taker quote volumes consistent with flow_sign.
        buy = 800.0 if flow_sign > 0 else 200.0
        sell = 200.0 if flow_sign > 0 else 800.0
        bars.feature_values[(symbol, "taker_buy_quote_volume")] = buy
        bars.feature_values[(symbol, "taker_sell_quote_volume")] = sell
        strat.calculate_signals_window(_window(ts, {symbol: rows}), None)
        emitted.extend(_drain(events))
        price = close
        ts += _STEP_MS
    return emitted


def test_flow_rider_long_on_positive_flow() -> None:
    signals = _run_flow_rider(flow_sign=+1.0)
    entries = [s for s in signals if s.signal_type == "LONG"]
    assert entries, "expected at least one LONG entry on persistent positive flow"
    assert not any(s.signal_type == "SHORT" for s in signals)
    first = entries[0]
    assert first.metadata.get("target_allocation", 0.0) > 0.0
    assert first.metadata.get("flow_zscore") is not None


def test_flow_rider_short_on_negative_flow() -> None:
    signals = _run_flow_rider(flow_sign=-1.0)
    entries = [s for s in signals if s.signal_type == "SHORT"]
    assert entries, "expected at least one SHORT entry on persistent negative flow"
    assert not any(s.signal_type == "LONG" for s in signals)


def test_flow_rider_rides_with_pyramiding() -> None:
    signals = _run_flow_rider(flow_sign=+1.0)
    # A persistent winner should produce at least one pyramid add beyond entry.
    adds = [s for s in signals if s.metadata.get("action") == "pyramid_add"]
    assert adds, "expected a pyramid add on a sustained flow trend"


# --------------------------------------------------------------------------- #
# #2 VolOfVolRegimeTrendGateStrategy
# --------------------------------------------------------------------------- #
# The candidate-library DEFAULT params for VolOfVolRegimeTrendGateStrategy. The
# relative GK/rv governor is calibration-correct ONLY against these defaults, so
# the no-trade-regression guard MUST construct the strategy with them and must NOT
# loosen the gk_rv_clean_rel / gk_rv_veto_rel thresholds.
_VOL_OF_VOL_DEFAULTS: dict[str, Any] = {
    "trend_z_window": 48,
    "trend_score_min": 0.10,
    "gk_window": 20,
    "rv_window": 20,
    "ker_period": 10,
    "clean_threshold": 0.40,
    "choppy_threshold": 0.20,
    "gk_rv_history_window": 32,
    "gk_rv_veto_rel": 1.6,
    "gk_rv_clean_rel": 1.0,
    "downsize_floor": 0.40,
    "upsize_cap": 1.5,
    "trail_atr_mult": 3.0,
    "atr_period": 14,
    "max_adds": 0,
    "add_step_atr": 1.0,
    "vol_window": 48,
    "target_vol": 0.020,
    "max_hold_bars": 300,
    "allow_short": True,
    "add_alloc_fraction": 0.5,
    "min_price": 0.0,
}


def _run_vol_of_vol(*, regime: str) -> list[Any]:
    """Drive the sleeve on a clean persistent uptrend with REALISTIC intrabar ranges.

    Garman-Klass range vol is structurally ~2-2.5x close-to-close realized vol on
    normal bars, so the intrabar high-low range here is comparable to / larger than
    the close-to-close move (GK/rv lands well above 1.0 -- a NORMAL bar, not stress).
    The strategy is constructed with the CANDIDATE-LIBRARY DEFAULT params so the
    relative-to-median governor is exercised exactly as it ships.

    - ``regime="clean"``: range gently tightens, so the latest GK/rv sits at/below
      its trailing median (relative <= gk_rv_clean_rel) -> up-size on a clean trend.
    - ``regime="stress"``: a clean median is established, then the intrabar range
      spikes hard (hidden intrabar skids) so GK/rv is elevated RELATIVE to this
      symbol's norm -> the governor down-sizes / vetoes.
    """
    symbol = "ETH/USDT"
    bars = _Bars(symbol_list=[symbol])
    events: queue.Queue = queue.Queue()
    strat = VolOfVolRegimeTrendGateStrategy(bars, events, **_VOL_OF_VOL_DEFAULTS)
    price = 100.0
    ts = _STEP_MS
    emitted: list[Any] = []
    for step in range(90):
        # A noisy persistent uptrend (real series carry close-to-close dispersion;
        # a perfectly linear ramp degenerates the realized-vol denominator).
        noise = 0.5 * math.sin(step * 1.7)
        close = price + 0.8 + noise
        move = abs(close - price)
        if regime == "clean":
            # Realistic intrabar range (~1.5x the close-to-close move) that gently
            # tightens over time -> latest GK/rv at/below its trailing median.
            range_mult = max(1.0, 1.6 - step * 0.004)
            half_range = max(0.5, range_mult * move) * 0.5
        elif regime == "stress":
            # Clean range establishes a median, then a hard intrabar-range spike
            # (hidden skids/gaps) elevates GK/rv vs the symbol's own norm.
            base_half = max(0.5, 1.5 * move) * 0.5
            half_range = base_half if step < 60 else base_half * 4.0
        else:  # pragma: no cover - guard against typos in the test
            raise AssertionError(f"unknown regime {regime!r}")
        rows = _ohlc_rows(
            ts,
            open_=price,
            high=max(price, close) + half_range,
            low=min(price, close) - half_range,
            close=close,
            volume=1000.0,
        )
        strat.calculate_signals_window(_window(ts, {symbol: rows}), None)
        emitted.extend(_drain(events))
        price = close
        ts += _STEP_MS
    return emitted


def test_vol_of_vol_directional_long_on_uptrend() -> None:
    # No-trade-regression guard: with the SHIPPING default thresholds and a clean
    # persistent uptrend carrying NORMAL (GK/rv ~2-2.5x) intrabar ranges, the
    # relative governor must NOT silently veto -- a LONG entry has to fire.
    signals = _run_vol_of_vol(regime="clean")
    longs = [s for s in signals if s.signal_type == "LONG"]
    assert longs, "expected a LONG entry on a clean uptrend with default thresholds"


def test_vol_of_vol_downsizes_high_gk_regime() -> None:
    clean = _run_vol_of_vol(regime="clean")
    stressed = _run_vol_of_vol(regime="stress")
    clean_entries = [s for s in clean if s.signal_type == "LONG"]
    stressed_entries = [s for s in stressed if s.signal_type == "LONG"]
    assert clean_entries, "clean regime should produce a LONG entry"
    clean_alloc = clean_entries[0].metadata.get("target_allocation", 0.0)
    clean_mult = clean_entries[0].metadata.get("size_multiplier", 1.0)
    # Clean regime up-sizes (>1x governor); stressed regime must down-size or veto.
    assert clean_mult > 1.0, "clean low-relative-GK trend should up-size"
    if stressed_entries:
        stressed_alloc = stressed_entries[0].metadata.get("target_allocation", 0.0)
        stressed_mult = stressed_entries[0].metadata.get("size_multiplier", 1.0)
        assert stressed_mult < clean_mult
        assert stressed_alloc < clean_alloc
    else:
        # A full veto under hidden intrabar stress is the strongest down-size.
        assert clean_alloc > 0.0


# --------------------------------------------------------------------------- #
# #5 VWAPCompressionReversionStrategy
# --------------------------------------------------------------------------- #
def _run_vwap_compression(*, compressed: bool) -> list[Any]:
    symbol = "SOL/USDT"
    bars = _Bars(symbol_list=[symbol])
    events: queue.Queue = queue.Queue()
    strat = VWAPCompressionReversionStrategy(
        bars,
        events,
        vwap_window=12,
        z_window=24,
        bandwidth_window=12,
        percentile_window=60,
        compression_percentile=0.40,
        compression_vol_ratio=1.20,
        entry_z=1.5,
        exit_z=0.0,
        min_dispersion=0.0,
        max_hold_bars=50,
        stop_loss_pct=0.0,
        min_price=0.0,
    )
    base = 100.0
    ts = _STEP_MS
    emitted: list[Any] = []

    def feed(close: float, half_range: float, volume: float) -> None:
        nonlocal ts
        rows = _ohlc_rows(
            ts,
            open_=close,
            high=close + half_range,
            low=close - half_range,
            close=close,
            volume=volume,
        )
        strat.calculate_signals_window(_window(ts, {symbol: rows}), None)
        emitted.extend(_drain(events))
        ts += _STEP_MS

    if compressed:
        # Phase A: higher-vol churn sets a high bandwidth baseline.
        for step in range(80):
            feed(base + 1.5 * math.sin(step / 2.0), 1.5, 1000.0)
        # Phase B: contract into a tight coil (bandwidth -> bottom percentile).
        for step in range(40):
            feed(base + 0.05 * math.sin(step / 3.0), 0.05, 800.0)
        # Phase C: a gradual drift below the coiled VWAP anchor (low vol but a
        # growing negative deviation) -> active compression + deviation fade.
        last = base + 0.05 * math.sin(39 / 3.0)
        for step in range(5):
            last -= 0.04
            feed(last, 0.05, 800.0)
    else:
        # Persistent wide swings -> vol-compression never activates.
        for step in range(125):
            feed(base + 4.0 * math.sin(step / 2.0), 4.0, 1000.0)
        last = base
        for step in range(5):
            last -= 0.04
            feed(last, 4.0, 1000.0)
    return emitted


def test_vwap_compression_acts_only_when_compressed() -> None:
    compressed = _run_vwap_compression(compressed=True)
    expanded = _run_vwap_compression(compressed=False)
    compressed_entries = [s for s in compressed if s.signal_type in {"LONG", "SHORT"}]
    expanded_entries = [s for s in expanded if s.signal_type in {"LONG", "SHORT"}]
    assert compressed_entries, "expected an entry in the active compression regime"
    assert not expanded_entries, "must not act when vol-compression is inactive"


def test_vwap_compression_fades_negative_deviation_long() -> None:
    signals = _run_vwap_compression(compressed=True)
    longs = [s for s in signals if s.signal_type == "LONG"]
    assert longs, "a negative VWAP deviation under compression should fade LONG"
    entry = longs[0]
    assert entry.metadata.get("deviation_z") is not None
    assert entry.metadata.get("deviation_z") <= 0.0
    assert entry.metadata.get("target_allocation", 0.0) > 0.0


# --------------------------------------------------------------------------- #
# #6 SeasonalMicroBreakoutRiderStrategy
# --------------------------------------------------------------------------- #
def _run_seasonal_micro_breakout(*, micro_agrees: bool, min_slot_obs: int = 4) -> list[Any]:
    symbol = "BTC/USDT"
    bars = _Bars(symbol_list=[symbol])
    events: queue.Queue = queue.Queue()
    strat = SeasonalMicroBreakoutRiderStrategy(
        bars,
        events,
        slot_minutes=1440,
        seasonal_decay=0.35,
        min_slot_observations=min_slot_obs,
        slot_t_threshold=0.2,
        breakout_window=5,
        breakout_buffer_atr_mult=0.0,
        trend_lookback=3,
        tick_agree_frac=0.60,
        require_micro_vwap_edge=True,
        atr_period=3,
        vol_window=8,
        max_adds=1,
        max_hold_bars=80,
        min_price=0.0,
    )
    price = 100.0
    ts = _STEP_MS
    emitted: list[Any] = []
    for step in range(28):
        drift = 0.25 if step < 12 else 0.85
        close = price + drift
        if micro_agrees:
            sub = [price + drift * (idx + 1) / 6.0 for idx in range(6)]
        else:
            # Same positive decision-bar breakout, but the last-window tape sells
            # down into the close, so tick agreement and micro-VWAP edge veto.
            sub = [close + 0.35 - idx * 0.07 for idx in range(6)]
            sub[-1] = close
        rows = _ohlc_rows(
            ts,
            open_=price,
            high=max(price, close) + 0.05,
            low=min(price, close) - 0.05,
            close=close,
            volume=1000.0,
            sub_closes=sub,
        )
        strat.calculate_signals_window(_window(ts, {symbol: rows}), None)
        emitted.extend(_drain(events))
        price = close
        ts += _STEP_MS
    return emitted


def test_seasonal_micro_breakout_enters_on_aligned_slot_breakout() -> None:
    signals = _run_seasonal_micro_breakout(micro_agrees=True)
    longs = [s for s in signals if s.signal_type == "LONG"]
    assert longs, "expected LONG on favorable slot drift + breakout + micro confirmation"
    first = longs[0]
    assert first.metadata.get("slot_observations", 0) >= 4
    assert first.metadata.get("breakout_direction") == "LONG"
    assert first.metadata.get("micro_tick_agreement") is not None
    assert first.metadata.get("target_allocation") == 0.30
    assert first.metadata.get("max_symbol_exposure_pct") == 0.30
    assert first.metadata.get("max_order_value") == 5000.0


def test_seasonal_micro_breakout_vetoes_bad_micro_tape() -> None:
    signals = _run_seasonal_micro_breakout(micro_agrees=False)
    assert not [s for s in signals if s.signal_type == "LONG"]


def test_seasonal_micro_breakout_requires_slot_history() -> None:
    signals = _run_seasonal_micro_breakout(micro_agrees=True, min_slot_obs=10_000)
    assert not [s for s in signals if s.signal_type in {"LONG", "SHORT"}]


# --------------------------------------------------------------------------- #
# candidate wiring
# --------------------------------------------------------------------------- #
def _build_candidates(monkeypatch_perp: bool = True) -> list[dict[str, Any]]:
    original = candidate_library._has_perp_support_data
    try:
        if monkeypatch_perp:
            candidate_library._has_perp_support_data = lambda: True
        ctx = _CandidateBuildContext(
            normalized_timeframes=("30m", "1h", "4h", "1d"),
            normalized_symbols=("BTC/USDT", "ETH/USDT", "SOL/USDT"),
        )
        ctx.perp_support_data_available = True
        return [c.to_dict() for c in ctx.build()]
    finally:
        candidate_library._has_perp_support_data = original


def test_candidate_wiring_cadence_and_timeframes() -> None:
    cands = _build_candidates()
    for klass in MICRO_STRATEGIES:
        rows = [c for c in cands if c["strategy_class"] == klass]
        assert rows, f"no candidates wired for {klass}"
        for row in rows:
            assert row["metadata"]["decision_cadence_seconds"] == 1800
            assert row["strategy_timeframe"] in _TF
            assert len(row["symbols"]) == 1  # single-asset per-symbol

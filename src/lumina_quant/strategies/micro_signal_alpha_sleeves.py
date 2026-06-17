"""Micro-signal-informed alpha sleeves that LOOK intrabar but TRADE at >=30m.

Every sleeve in this module obeys the same plumbing mandate: the alpha *reads*
the micro / intrabar tape (the 1s OHLCV rows on ``event.bars_1s`` plus the
window-latched perp feature fields) yet *decides* only at a >=30m cadence.  The
canonical contract is therefore stamped on each class::

    decision_cadence_seconds = 1800
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

The cadence gate (``core/engine.py``) is a pure *emission* throttle, not a data
throttle: the engine still ingests every 1s row and evaluates stops at 1s
resolution, but the decision callback fires at most once per 30m bucket.  The
crucial consequence (the design's CRITICAL trap) is that a single gated firing
sees only the *last* window's ``bars_1s`` -- NOT the full 30m of 1s tape.  So
every sleeve here computes its CORE signal from accumulated >=30m DECISION BARS
(a ring buffer of one ``_window_snapshot`` per gated firing, lookbacks measured
in decision bars -- the ``_ReturnRiderBase`` / ``AdaptiveRegimeMomentumStrategy``
doctrine) and uses the current window's ``bars_1s`` / latest feature value only
as a FRESH micro reading / confirm at the decision instant.

Sleeves:

- :class:`IntradayFlowPressureRiderStrategy` (#1): per-symbol directional
  taker-flow *continuation* ride.  Distinct from the taker-flow *exhaustion
  fade* (which fades at 60s); this rides persistence at 1800s with the rider
  base's ATR trailing stop + pyramiding.  Long and short.
- :class:`VolOfVolRegimeTrendGateStrategy` (#2): per-symbol directional trend
  (``pv_trend_score``) whose SIZE is governed by a vol-of-vol / microstructure
  stress read -- Garman-Klass intrabar range vol versus close-to-close realized
  vol, plus a Kaufman-efficiency cleanliness gate.  OHLCV-only.  Long and short.
- :class:`VWAPCompressionReversionStrategy` (#5): per-symbol mean reversion
  gated by ``volcomp_vwap_pressure`` (only acts when its compression regime is
  active), anchored to the boundary window's 1s volume-weighted typical price for
  a precise deviation.  OHLCV-only.  Long and short.
- :class:`SeasonalMicroBreakoutRiderStrategy` (#6): per-symbol breakout
  continuation gated by the online intraday slot drift that just proved useful in
  backtest, then confirmed by the latest window's 1s tick agreement / VWAP edge.
  OHLCV-only.  Long and short.

All classes are data-local (no fetching, no artifact writes, no hidden
environment state), ``None``-safe (never raise on sparse / degenerate input),
keep bounded deques, and remain selectable through the plugin registry without
editing ``strategies.registry``.  They must still be validated on the
data-bearing machine.
"""

from __future__ import annotations

import statistics
import math
from collections import deque
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

from lumina_quant.core.plugin_registry import register
from lumina_quant.indicators.accelerated import garman_klass_volatility
from lumina_quant.indicators.oscillators import rate_of_change
from lumina_quant.indicators.advanced_alpha import (
    pv_trend_score,
    volcomp_vwap_pressure,
)
from lumina_quant.indicators.alpha_features import (
    log_return,
    order_flow_imbalance,
    realized_volatility,
    rolling_zscore,
)
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.indicators.momentum import kaufman_efficiency_ratio
from lumina_quant.strategies.external_alpha_sleeves import (
    _event_datetime_utc,
    _EPS,
    _Snapshot,
    _emit,
    _event_symbols,
    _market_snapshot,
    _row_dict,
    _safe_non_negative_int,
    _target_metadata,
    _window_snapshot,
)
from lumina_quant.strategies.return_rider_alpha_sleeves import (
    _RiderCommon,
    _RiderState,
    _ReturnRiderBase,
    _restore_deque,
)
from lumina_quant.strategy import Strategy
from lumina_quant.tuning import HyperParam, resolve_params_from_schema


# Minimum number of trailing GK/rv observations required before the vol-of-vol
# governor trusts a "relative-to-median" stress read; below this it stays neutral
# (multiplier 1.0) so the sleeve trades during warmup instead of going silent.
_GK_RV_MIN_HISTORY = 8


# --------------------------------------------------------------------------- #
# Shared micro-reading helpers (read raw 1s OHLCV rows from the boundary window)
# --------------------------------------------------------------------------- #
def _window_rows(event: Any, symbol: str) -> list[dict[str, Any]]:
    """Return the boundary window's 1s OHLCV rows as ``dict`` rows (ascending)."""
    bars_1s = dict(getattr(event, "bars_1s", {}) or {})
    rows = [_row_dict(row, getattr(event, "time", None)) for row in list(bars_1s.get(symbol) or [])]
    return [row for row in rows if row]


def _tick_agreement(rows: list[dict[str, Any]], *, direction: str) -> float | None:
    """Fraction of consecutive 1s close-to-close ticks agreeing with ``direction``.

    A fresh micro read from the *last* window only (the gated callback never sees
    the full 30m of 1s tape).  Returns ``None`` when there is too little tape to
    judge, so the caller can fall back to a best-effort gate rather than raise.
    """
    closes = [value for row in rows if (value := safe_float(row.get("close"))) is not None]
    if len(closes) < 3:
        return None
    ups = 0
    downs = 0
    moves = 0
    for prev, cur in pairwise(closes):
        delta = cur - prev
        if abs(delta) <= _EPS:
            continue
        moves += 1
        if delta > 0.0:
            ups += 1
        else:
            downs += 1
    if moves <= 0:
        return None
    agree = ups if direction == "LONG" else downs
    return float(agree) / float(moves)


def _window_vwap(rows: list[dict[str, Any]]) -> float | None:
    """Volume-weighted typical price across the boundary window's 1s rows.

    Uses the 1s tape as a precise micro anchor; falls back to a simple typical
    mean when volume is degenerate/zero so a flat-volume window still produces a
    usable anchor instead of ``None``.
    """
    typicals: list[float] = []
    weighted = 0.0
    total_volume = 0.0
    for row in rows:
        high = safe_float(row.get("high"))
        low = safe_float(row.get("low"))
        close = safe_float(row.get("close"))
        if close is None:
            continue
        typical = (
            (high if high is not None else close) + (low if low is not None else close) + close
        ) / 3.0
        typicals.append(typical)
        volume = safe_float(row.get("volume"))
        if volume is not None and volume > 0.0:
            weighted += typical * volume
            total_volume += volume
    if total_volume > _EPS:
        return float(weighted / total_volume)
    if typicals:
        return float(sum(typicals) / len(typicals))
    return None


_MicroSlotStat = tuple[float, float, int]


def _slot_stat_update(
    stats: dict[int, _MicroSlotStat],
    key: int,
    ret: float,
    decay: float,
) -> None:
    """One-bar-deferred EW mean/variance update for an intraday slot."""
    rec = stats.get(key)
    if rec is None:
        stats[key] = (float(ret), 0.0, 1)
        return
    mean, var, n = rec
    delta = float(ret) - mean
    mean += decay * delta
    var = (1.0 - decay) * (var + decay * delta * delta)
    stats[key] = (mean, var, n + 1)


def _slot_stat_signal(
    stats: dict[int, _MicroSlotStat],
    key: int | None,
    *,
    min_obs: int,
    t_threshold: float,
    decay: float,
) -> int:
    """Return significant slot-drift direction, capped by EW effective sample size."""
    if key is None:
        return 0
    rec = stats.get(key)
    if rec is None:
        return 0
    mean, var, n = rec
    if n < min_obs:
        return 0
    eff_n = min(float(n), 1.0 / decay if decay > _EPS else float(n))
    std = math.sqrt(var) if var > 0.0 else 0.0
    if std <= _EPS:
        return 1 if mean > 0.0 else (-1 if mean < 0.0 else 0)
    t_stat = abs(mean) / std * math.sqrt(max(eff_n, 1.0))
    if t_stat < t_threshold:
        return 0
    return 1 if mean > 0.0 else (-1 if mean < 0.0 else 0)


# --------------------------------------------------------------------------- #
# #1 IntradayFlowPressureRiderStrategy
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class _FlowRiderExtra:
    """Per-symbol taker-flow accumulator + the freshest micro confirm read.

    ``cum_flow`` holds the running cumulative sum of per-decision-bar flow
    imbalance; its trailing z-score is what gates entries -- sustained one-sided
    pressure makes the cumulative series trend strongly, which surfaces as a large
    z-score even when each bar's raw imbalance is similar (the per-bar series would
    otherwise be near-flat and degenerate).
    """

    flow_imbalance: deque[float]
    cum_flow: deque[float]
    cum_total: float = 0.0
    last_flow_z: float | None = None
    last_tick_agreement: float | None = None
    last_window_vwap: float | None = None
    last_close: float | None = None
    last_rows: tuple[dict[str, Any], ...] = ()


@register("strategy", "IntradayFlowPressureRiderStrategy", interface="event_driven")
class IntradayFlowPressureRiderStrategy(_ReturnRiderBase):
    """Ride sustained taker-flow PRESSURE as a >=30m directional continuation.

    Core (robust to the single-firing-window trap): on every gated 30m decision
    bar, compute the bar's taker order-flow imbalance from
    ``taker_buy_quote_volume`` / ``taker_sell_quote_volume`` (read via the
    ``_extract_feature`` cascade), append it to a per-symbol DECISION-BAR deque,
    and require the trailing-``flow_z_window`` z-score of cumulative flow to reach
    ``entry_z`` (long) / ``-entry_z`` (short).

    Declares NO ``required_features``: when the perp feature lookup is unconfigured
    (so ``taker_*`` quote/base volumes are absent), ``_resolve_flow`` degrades to an
    OHLCV up/down-volume proxy from the 1s tape rather than erroring out of the run.
    The candidate-library perp gate still controls when this sleeve is wired.

    Micro confirm (best-effort, last window only): require the latest window's 1s
    close-tick signs to agree with the flow direction on at least
    ``tick_agree_frac`` of moves (rejects single-print sweeps); and prefer a fill
    near the window VWAP (long below / short above).

    The position is then ridden with the base's ATR trailing stop and pyramided
    into continuation when flow z-score extends further.  Long and short.

    DISTINCT from :class:`TakerFlowExhaustionReversalStrategy`, which *fades*
    exhaustion at a 60s cadence -- this *rides* persistence at 1800s with the
    rider base's trailing-stop/pyramiding return machinery.
    """

    decision_cadence_seconds = 1800
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    strategy_name = "IntradayFlowPressureRiderStrategy"
    strategy_id = "intraday_flow_pressure_rider"

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "flow_z_window": HyperParam.integer("flow_z_window", default=8, low=3, high=4096),
            "entry_z": HyperParam.floating("entry_z", default=1.5, low=0.0, high=20.0),
            "pyramid_z": HyperParam.floating("pyramid_z", default=2.0, low=0.0, high=20.0),
            "tick_agree_frac": HyperParam.floating(
                "tick_agree_frac", default=0.55, low=0.0, high=1.0
            ),
            "require_vwap_edge": HyperParam.boolean(
                "require_vwap_edge", default=True, grid=[True, False]
            ),
            "min_dispersion": HyperParam.floating(
                "min_dispersion", default=1e-4, low=0.0, high=1.0
            ),
            "trail_atr_mult": HyperParam.floating(
                "trail_atr_mult", default=3.0, low=0.1, high=20.0
            ),
            "atr_period": HyperParam.integer("atr_period", default=14, low=2, high=1024),
            "max_adds": HyperParam.integer("max_adds", default=3, low=0, high=32),
            "add_step_atr": HyperParam.floating("add_step_atr", default=1.0, low=0.0, high=20.0),
            "vol_window": HyperParam.integer("vol_window", default=48, low=2, high=4096),
            "target_vol": HyperParam.floating("target_vol", default=0.02, low=0.0, high=1.0),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=180, low=1, high=200000),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
            "add_alloc_fraction": HyperParam.floating(
                "add_alloc_fraction", default=0.5, low=0.0, high=1.0
            ),
            "target_allocation": HyperParam.floating(
                "target_allocation", default=0.30, low=0.0, high=5.0, tunable=False
            ),
            "max_order_value": HyperParam.floating(
                "max_order_value", default=5000.0, low=0.0, high=1_000_000.0, tunable=False
            ),
            "min_price": HyperParam.floating("min_price", default=0.10, low=0.0, high=1_000_000.0),
        }

    def __init__(self, bars: Any, events: Any, **params: Any) -> None:
        super().__init__(bars, events, **params)
        self._extra = {
            symbol: _FlowRiderExtra(
                flow_imbalance=deque(maxlen=self._history_size),
                cum_flow=deque(maxlen=self._history_size),
            )
            for symbol in self.symbol_list
        }

    def _bind_params(self, resolved: dict[str, Any]) -> None:
        self.flow_z_window = max(3, int(resolved["flow_z_window"]))
        self.entry_z = max(0.0, float(resolved["entry_z"]))
        self.pyramid_z = max(0.0, float(resolved["pyramid_z"]))
        self.tick_agree_frac = max(0.0, min(1.0, float(resolved["tick_agree_frac"])))
        self.require_vwap_edge = bool(resolved["require_vwap_edge"])
        self.min_dispersion = max(0.0, float(resolved["min_dispersion"]))

    def _common_config(self, resolved: dict[str, Any]) -> _RiderCommon:
        return self._resolve_common(resolved, extra_window=self.flow_z_window + 2)

    # -- feature read (window-latched perp fields with OHLCV fallback) --------
    def _extract_feature(self, event: Any, symbol: str, field_name: str) -> float | None:
        direct = safe_float(getattr(event, field_name, None))
        if direct is not None:
            return direct
        getter = getattr(self.bars, "get_latest_feature_value", None)
        if callable(getter):
            try:
                parsed = safe_float(getter(symbol, field_name))
            except Exception:
                parsed = None
            if parsed is not None:
                return parsed
        getter = getattr(self.bars, "get_latest_bar_value", None)
        if callable(getter):
            try:
                return safe_float(getter(symbol, field_name))
            except Exception:
                return None
        return None

    def _resolve_flow(
        self, event: Any, symbol: str, rows: list[dict[str, Any]], close: float
    ) -> float | None:
        """Return this decision bar's taker order-flow imbalance, OHLCV fallback.

        Primary: window-latched taker quote volumes via the feature cascade.
        Fallback: split the window's 1s volume into up/down ticks (a coarse but
        always-available taker-flow proxy) so the sleeve degrades instead of
        going dark when perp fields are missing for a symbol/venue.
        """
        buy = self._extract_feature(event, symbol, "taker_buy_quote_volume")
        sell = self._extract_feature(event, symbol, "taker_sell_quote_volume")
        if buy is not None and sell is not None and (buy + sell) > 0.0:
            return order_flow_imbalance(max(0.0, float(buy)), max(0.0, float(sell)))
        buy_base = self._extract_feature(event, symbol, "taker_buy_base_volume")
        sell_base = self._extract_feature(event, symbol, "taker_sell_base_volume")
        if buy_base is not None and sell_base is not None and (buy_base + sell_base) > 0.0:
            return order_flow_imbalance(
                max(0.0, float(buy_base) * close), max(0.0, float(sell_base) * close)
            )
        # OHLCV fallback: up-volume vs down-volume from the 1s tape.
        up_volume = 0.0
        down_volume = 0.0
        prev_close: float | None = None
        for row in rows:
            row_close = safe_float(row.get("close"))
            row_volume = safe_float(row.get("volume"))
            if row_close is None:
                continue
            volume = max(0.0, row_volume if row_volume is not None else 0.0)
            if prev_close is not None and volume > 0.0:
                if row_close > prev_close:
                    up_volume += volume
                elif row_close < prev_close:
                    down_volume += volume
            prev_close = row_close
        if up_volume + down_volume > 0.0:
            return order_flow_imbalance(up_volume, down_volume)
        return None

    def calculate_signals_window(self, event: Any, aggregator: Any = None) -> None:
        _ = aggregator
        for symbol in _event_symbols(event, self.symbol_list):
            snapshot = _window_snapshot(event, symbol)
            if snapshot is None:
                continue
            rows = _window_rows(event, symbol)
            self._prepare_extra(event, symbol, snapshot, rows)
            self._process_symbol(symbol, snapshot)

    def calculate_signals(self, event: Any) -> None:
        if str(getattr(event, "type", "")).upper() == "MARKET_WINDOW":
            self.calculate_signals_window(event, None)
            return
        # MARKET ticks carry no window tape; ignore (this sleeve is window-only).

    def _prepare_extra(
        self, event: Any, symbol: str, snapshot: _Snapshot, rows: list[dict[str, Any]]
    ) -> None:
        """Fold this gated firing into the per-symbol decision-bar accumulator.

        Guards boundary dedup with ``time_key`` against the rider state's
        ``last_time_key`` so a repeated boundary firing never double-appends.
        """
        item = self._state.get(symbol)
        extra = self._extra.get(symbol)
        if item is None or extra is None:
            return
        key = time_key(snapshot.time)
        if key and key == item.last_time_key:
            return
        close = safe_float(snapshot.close)
        if close is None or close <= self.min_price:
            return
        flow = self._resolve_flow(event, symbol, rows, close)
        if flow is not None:
            extra.flow_imbalance.append(float(flow))
            extra.cum_total += float(flow)
            extra.cum_flow.append(extra.cum_total)
        extra.last_window_vwap = _window_vwap(rows)
        extra.last_close = close
        # Z-score the CUMULATIVE flow series: sustained one-sided pressure trends
        # the cumulative sum, which a trailing z-score surfaces as a strong reading
        # even when each bar's raw imbalance is similar.
        cum_values = list(extra.cum_flow)
        if len(cum_values) >= 3:
            # Guard the degenerate zero-sigma +-6 marker: require minimum
            # dispersion before trusting a z-score readout.
            hist = cum_values[-(self.flow_z_window + 1) :]
            dispersion = max(hist) - min(hist) if len(hist) >= 2 else 0.0
            if dispersion >= self.min_dispersion:
                extra.last_flow_z = rolling_zscore(cum_values, window=self.flow_z_window)
            else:
                extra.last_flow_z = None
        else:
            extra.last_flow_z = None
        # Tick agreement is computed lazily per candidate direction in the gate;
        # stash the boundary window's rows so the entry decision can read them.
        extra.last_tick_agreement = None
        extra.last_rows = tuple(rows)

    def _entry_decision(self, item: _RiderState) -> str:
        symbol = self._symbol_for_state(item)
        if symbol is None:
            return ""
        extra = self._extra.get(symbol)
        if extra is None or extra.last_flow_z is None:
            return ""
        flow_z = extra.last_flow_z
        if self.entry_z > 0.0 and abs(flow_z) < self.entry_z:
            return ""
        direction = "LONG" if flow_z > 0.0 else "SHORT"
        agreement = _tick_agreement(list(extra.last_rows), direction=direction)
        extra.last_tick_agreement = agreement
        if agreement is not None and agreement < self.tick_agree_frac:
            return ""
        if self.require_vwap_edge and extra.last_window_vwap is not None and extra.last_close:
            vwap = extra.last_window_vwap
            close = extra.last_close
            # Prefer a better fill: long only at/below VWAP, short at/above.
            if direction == "LONG" and close > vwap:
                return ""
            if direction == "SHORT" and close < vwap:
                return ""
        return direction

    def _should_pyramid(self, item: _RiderState) -> bool:
        symbol = self._symbol_for_state(item)
        if symbol is None:
            return False
        extra = self._extra.get(symbol)
        if extra is None or extra.last_flow_z is None:
            return False
        flow_z = extra.last_flow_z
        if item.mode == "LONG":
            return flow_z >= self.pyramid_z
        return flow_z <= -self.pyramid_z

    def _entry_metadata(self, item: _RiderState) -> dict[str, Any]:
        symbol = self._symbol_for_state(item)
        extra = self._extra.get(symbol) if symbol is not None else None
        return {
            "flow_zscore": float(extra.last_flow_z)
            if extra is not None and extra.last_flow_z is not None
            else None,
            "tick_agreement": float(extra.last_tick_agreement)
            if extra is not None and extra.last_tick_agreement is not None
            else None,
        }

    def _symbol_for_state(self, item: _RiderState) -> str | None:
        for symbol, state in self._state.items():
            if state is item:
                return symbol
        return None

    # -- state ---------------------------------------------------------------
    def get_state(self) -> dict[str, Any]:
        base = super().get_state()
        base["flow_state"] = {
            symbol: {
                "flow_imbalance": list(extra.flow_imbalance),
                "cum_flow": list(extra.cum_flow),
                "cum_total": extra.cum_total,
                "last_flow_z": extra.last_flow_z,
                "last_window_vwap": extra.last_window_vwap,
                "last_close": extra.last_close,
            }
            for symbol, extra in self._extra.items()
        }
        return base

    def set_state(self, state: dict[str, Any]) -> None:
        super().set_state(state)
        raw = state.get("flow_state") if isinstance(state, dict) else None
        if not isinstance(raw, dict):
            return
        for symbol, payload in raw.items():
            extra = self._extra.get(symbol)
            if extra is None or not isinstance(payload, dict):
                continue
            _restore_deque(extra.flow_imbalance, payload.get("flow_imbalance"))
            _restore_deque(extra.cum_flow, payload.get("cum_flow"))
            cum_total = safe_float(payload.get("cum_total"))
            extra.cum_total = cum_total if cum_total is not None else 0.0
            extra.last_flow_z = safe_float(payload.get("last_flow_z"))
            extra.last_window_vwap = safe_float(payload.get("last_window_vwap"))
            extra.last_close = safe_float(payload.get("last_close"))


# --------------------------------------------------------------------------- #
# #2 VolOfVolRegimeTrendGateStrategy
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class _VolOfVolExtra:
    """Per-symbol decision-bar OHLC history + the freshest regime governor read.

    ``gk_rv_history`` is a bounded trailing record of the raw GK/rv ratio per
    decision bar; the governor keys off the ratio RELATIVE to this symbol's own
    rolling median (stress = ratio elevated vs its norm) rather than an absolute
    level, because Garman-Klass range vol is structurally ~2-2.5x close-to-close
    realized vol on normal bars.
    """

    opens: deque[float]
    highs: deque[float]
    lows: deque[float]
    closes: deque[float]
    volumes: deque[float]
    gk_rv_history: deque[float] = field(default_factory=deque)
    last_size_multiplier: float = 1.0
    last_gk_rv_ratio: float | None = None
    last_gk_rv_relative: float | None = None
    last_ker: float | None = None


@register("strategy", "VolOfVolRegimeTrendGateStrategy", interface="event_driven")
class VolOfVolRegimeTrendGateStrategy(_ReturnRiderBase):
    """Directional >=30m trend whose SIZE is governed by a vol-of-vol stress read.

    Core direction: ``pv_trend_score`` (finally wiring the dead RG_PVTM primitive)
    on the per-symbol DECISION-BAR closes/volumes -- robust to the single-window
    trap because it reads accumulated 30m bars, not one window's 1s tape.

    Size governor (the novelty): compute the Garman-Klass intrabar range vol to
    close-to-close realized vol ratio (``GK/rv``), then key the governor off that
    ratio RELATIVE to this symbol's own trailing-median ratio -- NOT an absolute
    level.  GK range vol is structurally ~2-2.5x close-to-close realized vol on
    normal bars, so an absolute threshold would veto typical bars and the sleeve
    would silently never trade; keying on the relative elevation (stress = ratio
    high vs this symbol's norm) is what makes it calibration-correct.  When the
    relative ratio is elevated (hidden intrabar skids/gaps -> microstructure
    stress) the entry is DOWN-SIZED or VETOED; when the ratio sits at/below its
    norm AND ``kaufman_efficiency_ratio >= clean_threshold`` (a clean, efficient
    move) the entry is UP-SIZED, all within the base's existing <=2x allocation
    cap.  During warmup (insufficient history) the multiplier is neutral (1.0) so
    the sleeve trades.  Same directional core; capital flows only into clean trends.

    OHLCV-only (no perp fields), so the widest crypto universe.  Long and short.
    """

    decision_cadence_seconds = 1800
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    strategy_name = "VolOfVolRegimeTrendGateStrategy"
    strategy_id = "vol_of_vol_regime_trend_gate"

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "trend_z_window": HyperParam.integer("trend_z_window", default=48, low=16, high=4096),
            "trend_score_min": HyperParam.floating(
                "trend_score_min", default=0.10, low=0.0, high=20.0
            ),
            "gk_window": HyperParam.integer("gk_window", default=20, low=2, high=4096),
            "rv_window": HyperParam.integer("rv_window", default=20, low=2, high=4096),
            "ker_period": HyperParam.integer("ker_period", default=10, low=2, high=4096),
            "clean_threshold": HyperParam.floating(
                "clean_threshold", default=0.40, low=0.0, high=1.0
            ),
            "choppy_threshold": HyperParam.floating(
                "choppy_threshold", default=0.20, low=0.0, high=1.0
            ),
            "gk_rv_history_window": HyperParam.integer(
                "gk_rv_history_window", default=32, low=8, high=4096
            ),
            "gk_rv_veto_rel": HyperParam.floating(
                "gk_rv_veto_rel", default=1.6, low=1.0, high=10.0
            ),
            "gk_rv_clean_rel": HyperParam.floating(
                "gk_rv_clean_rel", default=1.0, low=0.1, high=5.0
            ),
            "downsize_floor": HyperParam.floating(
                "downsize_floor", default=0.40, low=0.0, high=1.0
            ),
            "upsize_cap": HyperParam.floating("upsize_cap", default=1.5, low=1.0, high=2.0),
            "trail_atr_mult": HyperParam.floating(
                "trail_atr_mult", default=3.0, low=0.1, high=20.0
            ),
            "atr_period": HyperParam.integer("atr_period", default=14, low=2, high=1024),
            "max_adds": HyperParam.integer("max_adds", default=2, low=0, high=32),
            "add_step_atr": HyperParam.floating("add_step_atr", default=1.0, low=0.0, high=20.0),
            "vol_window": HyperParam.integer("vol_window", default=48, low=2, high=4096),
            "target_vol": HyperParam.floating("target_vol", default=0.02, low=0.0, high=1.0),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=180, low=1, high=200000),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
            "add_alloc_fraction": HyperParam.floating(
                "add_alloc_fraction", default=0.5, low=0.0, high=1.0
            ),
            "target_allocation": HyperParam.floating(
                "target_allocation", default=0.30, low=0.0, high=5.0, tunable=False
            ),
            "max_order_value": HyperParam.floating(
                "max_order_value", default=5000.0, low=0.0, high=1_000_000.0, tunable=False
            ),
            "min_price": HyperParam.floating("min_price", default=0.10, low=0.0, high=1_000_000.0),
        }

    def __init__(self, bars: Any, events: Any, **params: Any) -> None:
        super().__init__(bars, events, **params)
        self._active_symbol: str | None = None
        self._extra = {
            symbol: _VolOfVolExtra(
                opens=deque(maxlen=self._history_size),
                highs=deque(maxlen=self._history_size),
                lows=deque(maxlen=self._history_size),
                closes=deque(maxlen=self._history_size),
                volumes=deque(maxlen=self._history_size),
                gk_rv_history=deque(maxlen=self.gk_rv_history_window),
            )
            for symbol in self.symbol_list
        }

    def _bind_params(self, resolved: dict[str, Any]) -> None:
        self.trend_z_window = max(16, int(resolved["trend_z_window"]))
        self.trend_score_min = max(0.0, float(resolved["trend_score_min"]))
        self.gk_window = max(2, int(resolved["gk_window"]))
        self.rv_window = max(2, int(resolved["rv_window"]))
        self.ker_period = max(2, int(resolved["ker_period"]))
        self.clean_threshold = max(0.0, min(1.0, float(resolved["clean_threshold"])))
        self.choppy_threshold = max(0.0, min(1.0, float(resolved["choppy_threshold"])))
        self.gk_rv_history_window = max(8, int(resolved["gk_rv_history_window"]))
        self.gk_rv_veto_rel = max(1.0, float(resolved["gk_rv_veto_rel"]))
        self.gk_rv_clean_rel = max(0.1, float(resolved["gk_rv_clean_rel"]))
        self.downsize_floor = max(0.0, min(1.0, float(resolved["downsize_floor"])))
        self.upsize_cap = max(1.0, min(2.0, float(resolved["upsize_cap"])))

    def _common_config(self, resolved: dict[str, Any]) -> _RiderCommon:
        return self._resolve_common(
            resolved,
            extra_window=max(self.trend_z_window, self.gk_window, self.rv_window, self.ker_period)
            + 2,
        )

    def calculate_signals_window(self, event: Any, aggregator: Any = None) -> None:
        _ = aggregator
        for symbol in _event_symbols(event, self.symbol_list):
            snapshot = _window_snapshot(event, symbol)
            if snapshot is None:
                continue
            self._prepare_extra(symbol, snapshot)
            self._process_symbol(symbol, snapshot)

    def calculate_signals(self, event: Any) -> None:
        if str(getattr(event, "type", "")).upper() == "MARKET_WINDOW":
            self.calculate_signals_window(event, None)

    def _process_symbol(self, symbol: str, snapshot: _Snapshot) -> None:
        # Thread the active symbol explicitly so ``_vol_scaled_allocation`` can
        # fetch its governor multiplier without re-deriving the symbol from
        # close-deque value-equality. Set BEFORE the base runs any _enter /
        # _maybe_pyramid sizing call.
        self._active_symbol = symbol
        super()._process_symbol(symbol, snapshot)

    def _prepare_extra(self, symbol: str, snapshot: _Snapshot) -> None:
        item = self._state.get(symbol)
        extra = self._extra.get(symbol)
        if item is None or extra is None:
            return
        key = time_key(snapshot.time)
        if key and key == item.last_time_key:
            return
        close = safe_float(snapshot.close)
        if close is None or close <= self.min_price:
            return
        extra.opens.append(float(snapshot.open if snapshot.open is not None else close))
        extra.highs.append(float(snapshot.high if snapshot.high is not None else close))
        extra.lows.append(float(snapshot.low if snapshot.low is not None else close))
        extra.closes.append(close)
        extra.volumes.append(max(0.0, float(snapshot.volume or 0.0)))
        self._refresh_governor(extra)

    def _refresh_governor(self, extra: _VolOfVolExtra) -> None:
        """Recompute the GK/rv + KER size governor from accumulated decision bars.

        The governor keys off the GK/rv ratio RELATIVE to this symbol's own
        trailing-median ratio (``relative = ratio / median``), not an absolute
        level: GK range vol runs ~2-2.5x close-to-close realized vol on normal
        bars, so an absolute threshold would veto typical bars and silently
        suppress all trading.  During warmup (fewer than ``_GK_RV_MIN_HISTORY``
        observations) the multiplier is neutral (1.0) so the sleeve still trades.
        """
        closes = list(extra.closes)
        gk = garman_klass_volatility(
            list(extra.opens),
            list(extra.highs),
            list(extra.lows),
            closes,
            window=self.gk_window,
            annualization=1.0,
        )
        rv = realized_volatility(closes, window=self.rv_window)
        ker = kaufman_efficiency_ratio(closes, period=self.ker_period)
        extra.last_ker = ker
        if gk is None or rv is None or rv <= _EPS:
            extra.last_gk_rv_ratio = None
            extra.last_gk_rv_relative = None
            extra.last_size_multiplier = 1.0
            return
        ratio = float(gk / rv)
        extra.last_gk_rv_ratio = ratio
        extra.gk_rv_history.append(ratio)
        if len(extra.gk_rv_history) < _GK_RV_MIN_HISTORY:
            # WARMUP: not enough history to judge "elevated vs norm" -> trade neutral.
            extra.last_gk_rv_relative = None
            extra.last_size_multiplier = 1.0
            return
        median_ratio = statistics.median(extra.gk_rv_history)
        relative = ratio / max(median_ratio, _EPS)
        extra.last_gk_rv_relative = relative
        multiplier = 1.0
        if relative >= self.gk_rv_veto_rel:
            multiplier = 0.0  # ratio elevated vs this symbol's norm -> veto
        elif relative <= self.gk_rv_clean_rel and ker is not None and ker >= self.clean_threshold:
            multiplier = self.upsize_cap  # clean efficient move at/below norm -> up-size
        elif relative > self.gk_rv_clean_rel:
            # Partial stress: scale down between clean and veto relative levels.
            span = max(_EPS, self.gk_rv_veto_rel - self.gk_rv_clean_rel)
            frac = (relative - self.gk_rv_clean_rel) / span
            multiplier = 1.0 - (1.0 - self.downsize_floor) * max(0.0, min(1.0, frac))
        extra.last_size_multiplier = max(0.0, multiplier)

    def _trend_direction(self, extra: _VolOfVolExtra) -> str:
        """Direction from a robust price trend, regime-gated by ``pv_trend_score``.

        ``pv_trend_score`` (the previously-dead RG_PVTM primitive) is wired as the
        regime gate: it must be available and clear ``trend_score_min`` in absolute
        terms (a meaningful regime-conditioned price-volume signal).  The DIRECTION
        is taken from the net log return over the trend window -- the RG_PVTM
        ``score`` is a z-scored-return composite that can invert sign on an
        established (percentage-decelerating) trend, so it gates the regime while
        net return sets the side.
        """
        closes = list(extra.closes)
        volumes = list(extra.volumes)
        if len(closes) < 64:
            return ""
        result = pv_trend_score(
            closes,
            volumes,
            high=list(extra.highs),
            low=list(extra.lows),
            z_window=self.trend_z_window,
        )
        if not result.get("available"):
            return ""
        if abs(float(result.get("score") or 0.0)) < self.trend_score_min:
            return ""
        net = log_return(closes, lookback=min(len(closes) - 1, self.trend_z_window))
        if net is None or abs(net) <= _EPS:
            return ""
        return "LONG" if net > 0.0 else "SHORT"

    def _entry_decision(self, item: _RiderState) -> str:
        symbol = self._symbol_for_state(item)
        if symbol is None:
            return ""
        extra = self._extra.get(symbol)
        if extra is None:
            return ""
        # Veto under hidden intrabar stress or a choppy regime.
        if extra.last_size_multiplier <= 0.0:
            return ""
        if extra.last_ker is not None and extra.last_ker < self.choppy_threshold:
            return ""
        return self._trend_direction(extra)

    def _should_pyramid(self, item: _RiderState) -> bool:
        symbol = self._symbol_for_state(item)
        if symbol is None:
            return False
        extra = self._extra.get(symbol)
        if extra is None or extra.last_size_multiplier <= 0.0:
            return False
        direction = self._trend_direction(extra)
        return direction == item.mode

    def _vol_scaled_allocation(self, closes: list[float]) -> float:
        """Apply the GK/rv + KER governor on top of the base vol-target sizing."""
        base = super()._vol_scaled_allocation(closes)
        multiplier = 1.0
        # Symbol is threaded explicitly via ``_active_symbol`` (set in
        # ``_process_symbol`` before any sizing call) rather than re-derived from
        # close-deque value-equality.
        symbol = self._active_symbol
        if symbol is not None:
            extra = self._extra.get(symbol)
            if extra is not None:
                multiplier = extra.last_size_multiplier
        scaled = base * multiplier
        # Never exceed the base's hard 2x cap regardless of governor up-size.
        return max(0.0, min(self.target_allocation * 2.0, scaled))

    def _entry_metadata(self, item: _RiderState) -> dict[str, Any]:
        symbol = self._symbol_for_state(item)
        extra = self._extra.get(symbol) if symbol is not None else None
        return {
            "size_multiplier": float(extra.last_size_multiplier) if extra is not None else 1.0,
            "gk_rv_ratio": float(extra.last_gk_rv_ratio)
            if extra is not None and extra.last_gk_rv_ratio is not None
            else None,
            "gk_rv_relative": float(extra.last_gk_rv_relative)
            if extra is not None and extra.last_gk_rv_relative is not None
            else None,
            "efficiency_ratio": float(extra.last_ker)
            if extra is not None and extra.last_ker is not None
            else None,
        }

    def _symbol_for_state(self, item: _RiderState) -> str | None:
        for symbol, state in self._state.items():
            if state is item:
                return symbol
        return None

    # -- state ---------------------------------------------------------------
    def get_state(self) -> dict[str, Any]:
        base = super().get_state()
        base["regime_state"] = {
            symbol: {
                "opens": list(extra.opens),
                "highs": list(extra.highs),
                "lows": list(extra.lows),
                "closes": list(extra.closes),
                "volumes": list(extra.volumes),
                "gk_rv_history": list(extra.gk_rv_history),
                "last_size_multiplier": extra.last_size_multiplier,
                "last_gk_rv_ratio": extra.last_gk_rv_ratio,
                "last_gk_rv_relative": extra.last_gk_rv_relative,
                "last_ker": extra.last_ker,
            }
            for symbol, extra in self._extra.items()
        }
        return base

    def set_state(self, state: dict[str, Any]) -> None:
        super().set_state(state)
        raw = state.get("regime_state") if isinstance(state, dict) else None
        if not isinstance(raw, dict):
            return
        for symbol, payload in raw.items():
            extra = self._extra.get(symbol)
            if extra is None or not isinstance(payload, dict):
                continue
            _restore_deque(extra.opens, payload.get("opens"))
            _restore_deque(extra.highs, payload.get("highs"))
            _restore_deque(extra.lows, payload.get("lows"))
            _restore_deque(extra.closes, payload.get("closes"))
            _restore_deque(extra.volumes, payload.get("volumes"))
            _restore_deque(extra.gk_rv_history, payload.get("gk_rv_history"))
            mult = safe_float(payload.get("last_size_multiplier"))
            extra.last_size_multiplier = mult if mult is not None else 1.0
            extra.last_gk_rv_ratio = safe_float(payload.get("last_gk_rv_ratio"))
            extra.last_gk_rv_relative = safe_float(payload.get("last_gk_rv_relative"))
            extra.last_ker = safe_float(payload.get("last_ker"))


# --------------------------------------------------------------------------- #
# #6 SeasonalMicroBreakoutRiderStrategy
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class _SeasonalMicroBreakoutExtra:
    """Per-symbol slot statistics + latest micro confirmation state."""

    slot_stats: dict[int, _MicroSlotStat] = field(default_factory=dict)
    decision_slot: int | None = None
    micro_vwap: float | None = None
    micro_agreement: float | None = None
    micro_rows: tuple[dict[str, Any], ...] = ()
    breakout_direction: str = ""
    breakout_level: float | None = None


@register("strategy", "SeasonalMicroBreakoutRiderStrategy", interface="event_driven")
class SeasonalMicroBreakoutRiderStrategy(_ReturnRiderBase):
    """Ride range breakouts only in historically favorable intraday slots.

    This sleeve deliberately combines the two strongest patterns from the latest
    local evaluation: sub-day slot drift (``IntradaySeasonalMomentumRider``) and
    directional range expansion (``VolatilityBreakoutRider``).  The robust core is
    computed on accumulated >=30m decision bars:

    1. maintain one-bar-deferred online mean/variance for each UTC time-of-day
       slot, so the current bar is never in its own statistic;
    2. require the current close to break the trailing decision-bar high/low;
    3. require the current slot's statistically significant drift to agree with
       the breakout direction and the medium-horizon ROC trend.

    When the boundary window carries a 1s tape, the sleeve uses it as a fresh
    micro confirmation: tick signs must agree with the breakout and, optionally,
    the close must finish on the correct side of the window VWAP.  If the 1s tape
    is absent or too sparse, the micro check is neutral rather than fabricated.
    Trades are still emitted only at ``decision_cadence_seconds == 1800``.
    """

    decision_cadence_seconds = 1800
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    strategy_name = "SeasonalMicroBreakoutRiderStrategy"
    strategy_id = "seasonal_micro_breakout_rider"

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "slot_minutes": HyperParam.integer(
                "slot_minutes", default=60, low=1, high=1440, tunable=False
            ),
            "seasonal_decay": HyperParam.floating(
                "seasonal_decay", default=0.08, low=0.001, high=1.0
            ),
            "min_slot_observations": HyperParam.integer(
                "min_slot_observations", default=6, low=1, high=100000
            ),
            "slot_t_threshold": HyperParam.floating(
                "slot_t_threshold", default=1.0, low=0.0, high=20.0
            ),
            "breakout_window": HyperParam.integer("breakout_window", default=24, low=2, high=20000),
            "breakout_buffer_atr_mult": HyperParam.floating(
                "breakout_buffer_atr_mult", default=0.05, low=0.0, high=20.0
            ),
            "trend_lookback": HyperParam.integer("trend_lookback", default=24, low=2, high=20000),
            "min_trend_roc": HyperParam.floating("min_trend_roc", default=0.0, low=0.0, high=1.0),
            "tick_agree_frac": HyperParam.floating(
                "tick_agree_frac", default=0.55, low=0.0, high=1.0
            ),
            "require_micro_vwap_edge": HyperParam.boolean(
                "require_micro_vwap_edge", default=True, grid=[True, False]
            ),
            "trail_atr_mult": HyperParam.floating(
                "trail_atr_mult", default=3.0, low=0.1, high=20.0
            ),
            "atr_period": HyperParam.integer("atr_period", default=14, low=2, high=1024),
            "max_adds": HyperParam.integer("max_adds", default=2, low=0, high=32),
            "add_step_atr": HyperParam.floating("add_step_atr", default=1.0, low=0.0, high=20.0),
            "vol_window": HyperParam.integer("vol_window", default=48, low=2, high=4096),
            "target_vol": HyperParam.floating("target_vol", default=0.02, low=0.0, high=1.0),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=180, low=1, high=200000),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
            "add_alloc_fraction": HyperParam.floating(
                "add_alloc_fraction", default=0.5, low=0.0, high=1.0
            ),
            "target_allocation": HyperParam.floating(
                "target_allocation", default=0.30, low=0.0, high=5.0, tunable=False
            ),
            "max_order_value": HyperParam.floating(
                "max_order_value", default=5000.0, low=0.0, high=1_000_000.0, tunable=False
            ),
            "min_price": HyperParam.floating("min_price", default=0.10, low=0.0, high=1_000_000.0),
        }

    def __init__(self, bars: Any, events: Any, **params: Any) -> None:
        super().__init__(bars, events, **params)
        self._extra = {symbol: _SeasonalMicroBreakoutExtra() for symbol in self.symbol_list}

    def _bind_params(self, resolved: dict[str, Any]) -> None:
        self.slot_minutes = max(1, min(1440, int(resolved["slot_minutes"])))
        self.seasonal_decay = max(0.001, min(1.0, float(resolved["seasonal_decay"])))
        self.min_slot_observations = max(1, int(resolved["min_slot_observations"]))
        self.slot_t_threshold = max(0.0, float(resolved["slot_t_threshold"]))
        self.breakout_window = max(2, int(resolved["breakout_window"]))
        self.breakout_buffer_atr_mult = max(0.0, float(resolved["breakout_buffer_atr_mult"]))
        self.trend_lookback = max(2, int(resolved["trend_lookback"]))
        self.min_trend_roc = max(0.0, float(resolved["min_trend_roc"]))
        self.tick_agree_frac = max(0.0, min(1.0, float(resolved["tick_agree_frac"])))
        self.require_micro_vwap_edge = bool(resolved["require_micro_vwap_edge"])

    def _common_config(self, resolved: dict[str, Any]) -> _RiderCommon:
        return self._resolve_common(
            resolved,
            extra_window=max(int(resolved["breakout_window"]), int(resolved["trend_lookback"])) + 2,
        )

    def calculate_signals_window(self, event: Any, aggregator: Any = None) -> None:
        _ = aggregator
        for symbol in _event_symbols(event, self.symbol_list):
            snapshot = _window_snapshot(event, symbol)
            if snapshot is None:
                continue
            rows = _window_rows(event, symbol)
            self._prepare_context(symbol, snapshot, rows)
            self._process_symbol(symbol, snapshot)

    def calculate_signals(self, event: Any) -> None:
        if str(getattr(event, "type", "")).upper() == "MARKET_WINDOW":
            self.calculate_signals_window(event, None)
            return
        if getattr(event, "type", None) != "MARKET":
            return
        symbol = getattr(event, "symbol", None)
        if symbol in self._state:
            snapshot = _market_snapshot(event)
            if snapshot is not None:
                # Backtest/event-driven paths may not expose MARKET_WINDOW bars_1s.
                # Treat the micro read as neutral while keeping the same >=30m
                # class-level decision cadence in the engine.
                self._prepare_context(str(symbol), snapshot, [])
                self._process_symbol(str(symbol), snapshot)

    def _slot_of(self, snapshot: _Snapshot) -> int | None:
        dt = _event_datetime_utc(snapshot.time)
        if dt is None:
            return None
        return (dt.hour * 60 + dt.minute) // self.slot_minutes

    def _prepare_context(
        self, symbol: str, snapshot: _Snapshot, rows: list[dict[str, Any]]
    ) -> None:
        item = self._state.get(symbol)
        extra = self._extra.get(symbol)
        if item is None or extra is None:
            return
        key = time_key(snapshot.time)
        if key and key == item.last_time_key:
            return
        close = safe_float(snapshot.close)
        if close is None or close <= self.min_price:
            return
        closes = list(item.closes)
        if extra.decision_slot is not None and len(closes) >= 2 and closes[-2] > _EPS:
            ret = closes[-1] / closes[-2] - 1.0
            if math.isfinite(ret):
                _slot_stat_update(
                    extra.slot_stats,
                    int(extra.decision_slot),
                    ret,
                    self.seasonal_decay,
                )
        extra.decision_slot = self._slot_of(snapshot)
        extra.micro_rows = tuple(rows)
        extra.micro_vwap = _window_vwap(rows)
        extra.micro_agreement = None
        extra.breakout_direction = ""
        extra.breakout_level = None

    def _symbol_for_state(self, item: _RiderState) -> str | None:
        for symbol, state in self._state.items():
            if state is item:
                return symbol
        return None

    def _trend_allows(self, closes: list[float], direction: str) -> bool:
        trend = rate_of_change(closes, period=self.trend_lookback)
        if trend is None or abs(trend) < self.min_trend_roc:
            return False
        return trend > 0.0 if direction == "LONG" else trend < 0.0

    def _breakout_direction(self, item: _RiderState) -> tuple[str, float | None]:
        closes = list(item.closes)
        highs = list(item.highs)
        lows = list(item.lows)
        if len(closes) <= self.breakout_window:
            return "", None
        close = closes[-1]
        previous_high = max(highs[-(self.breakout_window + 1) : -1])
        previous_low = min(lows[-(self.breakout_window + 1) : -1])
        atr = self._current_atr(item)
        buffer = (atr if atr is not None else 0.0) * self.breakout_buffer_atr_mult
        if close > previous_high + buffer:
            return "LONG", float(previous_high)
        if close < previous_low - buffer:
            return "SHORT", float(previous_low)
        return "", None

    def _aligned_direction(self, symbol: str, item: _RiderState) -> str:
        extra = self._extra.get(symbol)
        if extra is None:
            return ""
        direction, level = self._breakout_direction(item)
        extra.breakout_direction = direction
        extra.breakout_level = level
        if direction == "" or not self._trend_allows(list(item.closes), direction):
            return ""
        slot_sig = _slot_stat_signal(
            extra.slot_stats,
            extra.decision_slot,
            min_obs=self.min_slot_observations,
            t_threshold=self.slot_t_threshold,
            decay=self.seasonal_decay,
        )
        required = 1 if direction == "LONG" else -1
        if slot_sig != required:
            return ""
        agreement = _tick_agreement(list(extra.micro_rows), direction=direction)
        extra.micro_agreement = agreement
        if agreement is not None and agreement < self.tick_agree_frac:
            return ""
        if self.require_micro_vwap_edge and extra.micro_vwap is not None and item.closes:
            close = item.closes[-1]
            if direction == "LONG" and close < extra.micro_vwap:
                return ""
            if direction == "SHORT" and close > extra.micro_vwap:
                return ""
        return direction

    def _entry_decision(self, item: _RiderState) -> str:
        symbol = self._symbol_for_state(item)
        if symbol is None:
            return ""
        return self._aligned_direction(symbol, item)

    def _should_pyramid(self, item: _RiderState) -> bool:
        symbol = self._symbol_for_state(item)
        if symbol is None:
            return False
        return self._aligned_direction(symbol, item) == item.mode

    def _entry_metadata(self, item: _RiderState) -> dict[str, Any]:
        symbol = self._symbol_for_state(item)
        extra = self._extra.get(symbol) if symbol is not None else None
        rec = None
        if extra is not None and extra.decision_slot is not None:
            rec = extra.slot_stats.get(int(extra.decision_slot))
        return {
            "slot": int(extra.decision_slot)
            if extra is not None and extra.decision_slot is not None
            else None,
            "slot_mean_return": float(rec[0]) if rec is not None else None,
            "slot_observations": int(rec[2]) if rec is not None else 0,
            "breakout_direction": extra.breakout_direction if extra is not None else "",
            "breakout_level": float(extra.breakout_level)
            if extra is not None and extra.breakout_level is not None
            else None,
            "micro_tick_agreement": float(extra.micro_agreement)
            if extra is not None and extra.micro_agreement is not None
            else None,
            "micro_vwap": float(extra.micro_vwap)
            if extra is not None and extra.micro_vwap is not None
            else None,
        }

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state["seasonal_micro_breakout_state"] = {
            symbol: {
                "slot_stats": {
                    int(k): [float(v[0]), float(v[1]), int(v[2])]
                    for k, v in extra.slot_stats.items()
                },
                "decision_slot": extra.decision_slot,
            }
            for symbol, extra in self._extra.items()
        }
        return state

    def set_state(self, state: dict[str, Any]) -> None:
        super().set_state(state)
        raw = state.get("seasonal_micro_breakout_state") if isinstance(state, dict) else None
        if not isinstance(raw, dict):
            return
        for symbol, payload in raw.items():
            extra = self._extra.get(symbol)
            if extra is None or not isinstance(payload, dict):
                continue
            stats: dict[int, _MicroSlotStat] = {}
            for key, value in (payload.get("slot_stats") or {}).items():
                try:
                    seq = list(value)
                    stats[int(key)] = (float(seq[0]), float(seq[1]), int(seq[2]))
                except Exception:
                    continue
            extra.slot_stats = stats
            try:
                slot = payload.get("decision_slot")
                extra.decision_slot = int(slot) if slot is not None else None
            except Exception:
                extra.decision_slot = None


# --------------------------------------------------------------------------- #
# #5 VWAPCompressionReversionStrategy
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class _VWAPCompState:
    """Per-symbol decision-bar OHLCV history + position bookkeeping."""

    opens: deque[float]
    highs: deque[float]
    lows: deque[float]
    closes: deque[float]
    volumes: deque[float]
    mode: str = "OUT"
    entry_price: float | None = None
    bars_held: int = 0
    last_time_key: str = ""


def _new_vwap_state(size: int) -> _VWAPCompState:
    bound = max(2, int(size))
    return _VWAPCompState(
        opens=deque(maxlen=bound),
        highs=deque(maxlen=bound),
        lows=deque(maxlen=bound),
        closes=deque(maxlen=bound),
        volumes=deque(maxlen=bound),
    )


@register("strategy", "VWAPCompressionReversionStrategy", interface="event_driven")
class VWAPCompressionReversionStrategy(Strategy):
    """Vol-compression-gated VWAP-deviation mean reversion, decided at >=30m.

    Wires the previously-dead ``volcomp_vwap_pressure`` primitive: only act when
    its compression regime is ``active=True`` (Bollinger bandwidth in the bottom
    percentile AND volatility ratio low) -- a coiled, low-vol regime where
    VWAP-deviation reversion is high-hit-rate.

    Core (robust to the single-window trap): the compression gate + deviation
    z-score are computed on accumulated 30m DECISION BARS.  Micro read: the VWAP
    anchor is refined from the boundary window's 1s volume-weighted typical price
    so the deviation is precise at the decision instant.

    Entry: LONG when ``deviation_z <= -entry_z``, SHORT when ``>= +entry_z``.
    Exit: when ``deviation_z`` crosses zero, when ``active`` turns False (vol
    expands), or at ``max_hold_bars``.  Long and short.  OHLCV-only.

    Guards the ``rolling_zscore`` degenerate zero-sigma +-6 artifact by requiring
    a minimum deviation dispersion before trusting the z-score.
    """

    decision_cadence_seconds = 1800
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    strategy_name = "VWAPCompressionReversionStrategy"
    strategy_id = "vwap_compression_reversion"

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "vwap_window": HyperParam.integer("vwap_window", default=48, low=8, high=4096),
            "z_window": HyperParam.integer("z_window", default=96, low=16, high=4096),
            "bandwidth_window": HyperParam.integer(
                "bandwidth_window", default=48, low=8, high=4096
            ),
            "percentile_window": HyperParam.integer(
                "percentile_window", default=192, low=32, high=8192
            ),
            "compression_percentile": HyperParam.floating(
                "compression_percentile", default=0.30, low=0.0, high=1.0
            ),
            "compression_vol_ratio": HyperParam.floating(
                "compression_vol_ratio", default=0.85, low=0.0, high=20.0
            ),
            "entry_z": HyperParam.floating("entry_z", default=2.0, low=0.0, high=20.0),
            "exit_z": HyperParam.floating("exit_z", default=0.0, low=0.0, high=20.0),
            "min_dispersion": HyperParam.floating(
                "min_dispersion", default=1e-4, low=0.0, high=1.0
            ),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=48, low=1, high=200000),
            "stop_loss_pct": HyperParam.floating("stop_loss_pct", default=0.03, low=0.0, high=0.50),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
            "target_allocation": HyperParam.floating(
                "target_allocation", default=0.05, low=0.0, high=2.0, tunable=False
            ),
            "max_order_value": HyperParam.floating(
                "max_order_value", default=1000.0, low=0.0, high=1_000_000.0, tunable=False
            ),
            "min_price": HyperParam.floating("min_price", default=0.10, low=0.0, high=1_000_000.0),
        }

    def __init__(self, bars: Any, events: Any, **params: Any) -> None:
        self.bars = bars
        self.events = events
        self.symbol_list = list(getattr(self.bars, "symbol_list", []) or [])
        resolved = resolve_params_from_schema(self.get_param_schema(), params, keep_unknown=False)
        self.vwap_window = max(8, int(resolved["vwap_window"]))
        self.z_window = max(16, int(resolved["z_window"]))
        self.bandwidth_window = max(8, int(resolved["bandwidth_window"]))
        self.percentile_window = max(32, int(resolved["percentile_window"]))
        self.compression_percentile = max(0.0, min(1.0, float(resolved["compression_percentile"])))
        self.compression_vol_ratio = max(0.0, float(resolved["compression_vol_ratio"]))
        self.entry_z = max(0.0, float(resolved["entry_z"]))
        self.exit_z = max(0.0, float(resolved["exit_z"]))
        self.min_dispersion = max(0.0, float(resolved["min_dispersion"]))
        self.max_hold_bars = max(1, int(resolved["max_hold_bars"]))
        self.stop_loss_pct = max(0.0, float(resolved["stop_loss_pct"]))
        self.allow_short = bool(resolved["allow_short"])
        self.target_allocation = max(0.0, float(resolved["target_allocation"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        size = (
            max(
                self.vwap_window,
                self.z_window,
                self.bandwidth_window,
                self.percentile_window,
                self.max_hold_bars,
            )
            + 8
        )
        self._state = {symbol: _new_vwap_state(size) for symbol in self.symbol_list}

    def calculate_signals_window(self, event: Any, aggregator: Any = None) -> None:
        _ = aggregator
        for symbol in _event_symbols(event, self.symbol_list):
            snapshot = _window_snapshot(event, symbol)
            if snapshot is None:
                continue
            rows = _window_rows(event, symbol)
            self._process_symbol(symbol, snapshot, rows)

    def calculate_signals(self, event: Any) -> None:
        if str(getattr(event, "type", "")).upper() == "MARKET_WINDOW":
            self.calculate_signals_window(event, None)

    def _process_symbol(self, symbol: str, snapshot: _Snapshot, rows: list[dict[str, Any]]) -> None:
        item = self._state[symbol]
        key = time_key(snapshot.time)
        if key and key == item.last_time_key:
            return
        item.last_time_key = key
        close = safe_float(snapshot.close)
        if close is None or close <= self.min_price:
            return
        item.opens.append(float(snapshot.open if snapshot.open is not None else close))
        item.highs.append(float(snapshot.high if snapshot.high is not None else close))
        item.lows.append(float(snapshot.low if snapshot.low is not None else close))
        item.closes.append(close)
        item.volumes.append(max(0.0, float(snapshot.volume or 0.0)))

        pressure = volcomp_vwap_pressure(
            list(item.highs),
            list(item.lows),
            list(item.closes),
            list(item.volumes),
            vwap_window=self.vwap_window,
            z_window=self.z_window,
            bandwidth_window=self.bandwidth_window,
            percentile_window=self.percentile_window,
            compression_percentile=self.compression_percentile,
            compression_vol_ratio=self.compression_vol_ratio,
        )
        if not pressure.get("available"):
            return
        active = bool(pressure.get("active"))
        deviation_z = self._micro_deviation_z(item, rows, pressure)

        # Exit/ride management first so a flatten frees the symbol for re-entry.
        if item.mode in {"LONG", "SHORT"}:
            self._maybe_exit(symbol, item, snapshot, close, active, deviation_z)
        if item.mode != "OUT":
            return
        if not active or deviation_z is None:
            return
        if self.entry_z <= 0.0:
            return
        direction = ""
        if deviation_z <= -self.entry_z:
            direction = "LONG"
        elif deviation_z >= self.entry_z and self.allow_short:
            direction = "SHORT"
        if not direction:
            return
        self._enter(symbol, item, snapshot, close, direction, deviation_z, pressure)

    def _micro_deviation_z(
        self, item: _VWAPCompState, rows: list[dict[str, Any]], pressure: dict[str, Any]
    ) -> float | None:
        """Deviation z-score over decision bars, refined by the 1s micro VWAP.

        The robust CORE is the deviation z-score of the latest decision-bar close
        against its trailing rolling VWAP anchor (this is exactly what makes the
        sleeve robust to the single-firing-window trap -- it reads accumulated 30m
        bars).  When the boundary window carries genuine intra-window structure
        (multiple 1s rows whose volume-weighted typical price differs from the
        bar close), the LATEST bar's anchor is replaced by that precise micro VWAP
        for a sharper deviation read.  Guards the rolling_zscore zero-sigma +-6
        artifact by requiring a minimum deviation dispersion.
        """
        base_z = pressure.get("deviation_z")
        base_z = float(base_z) if base_z is not None else None
        closes = list(item.closes)
        if len(closes) <= self.z_window:
            return base_z
        highs = list(item.highs)
        lows = list(item.lows)
        volumes = list(item.volumes)
        typicals = [(highs[i] + lows[i] + closes[i]) / 3.0 for i in range(len(closes))]
        win = self.vwap_window
        # Trailing rolling-VWAP anchor EXCLUDING the current bar's own typical, so
        # the latest deviation reflects how far the close sits from the recent
        # value area rather than collapsing toward its own bar.
        deviations: list[float] = []
        for i in range(len(closes)):
            lo = max(0, i - win)
            seg_typ = typicals[lo:i] if i > lo else typicals[lo : i + 1]
            seg_vol = volumes[lo:i] if i > lo else volumes[lo : i + 1]
            tv = sum(t * v for t, v in zip(seg_typ, seg_vol, strict=True))
            sv = sum(seg_vol)
            anchor = (tv / sv) if sv > _EPS else (sum(seg_typ) / len(seg_typ))
            deviations.append(closes[i] / anchor - 1.0 if anchor > _EPS else 0.0)
        # Refine the latest deviation with the boundary window's micro VWAP when
        # the window has real intra-window structure (more than one 1s row).
        micro_vwap = _window_vwap(rows)
        if micro_vwap is not None and micro_vwap > _EPS and len(rows) > 1:
            deviations[-1] = closes[-1] / micro_vwap - 1.0
        tail = deviations[-(self.z_window + 1) :]
        dispersion = max(tail) - min(tail) if len(tail) >= 2 else 0.0
        if dispersion < self.min_dispersion:
            return None
        micro_z = rolling_zscore(deviations, window=self.z_window)
        return micro_z if micro_z is not None else base_z

    def _maybe_exit(
        self,
        symbol: str,
        item: _VWAPCompState,
        snapshot: _Snapshot,
        close: float,
        active: bool,
        deviation_z: float | None,
    ) -> None:
        item.bars_held += 1
        reason = ""
        stopped = (
            item.entry_price is not None
            and self.stop_loss_pct > 0.0
            and (
                (item.mode == "LONG" and close <= item.entry_price * (1.0 - self.stop_loss_pct))
                or (item.mode == "SHORT" and close >= item.entry_price * (1.0 + self.stop_loss_pct))
            )
        )
        if stopped:
            reason = "stop_loss"
        if not reason and not active:
            reason = "compression_expired"
        reverted = deviation_z is not None and (
            (item.mode == "LONG" and deviation_z >= -self.exit_z)
            or (item.mode == "SHORT" and deviation_z <= self.exit_z)
        )
        if not reason and reverted:
            reason = "reversion_complete"
        if not reason and item.bars_held >= self.max_hold_bars:
            reason = "max_hold"
        if not reason:
            return
        _emit(
            self.events,
            strategy_id=self.strategy_id,
            symbol=symbol,
            event_time=snapshot.time,
            signal_type="EXIT",
            price=close,
            metadata={"strategy": self.strategy_name, "reason": reason},
        )
        item.mode = "OUT"
        item.entry_price = None
        item.bars_held = 0

    def _enter(
        self,
        symbol: str,
        item: _VWAPCompState,
        snapshot: _Snapshot,
        close: float,
        direction: str,
        deviation_z: float,
        pressure: dict[str, Any],
    ) -> None:
        stop_loss = None
        if self.stop_loss_pct > 0.0:
            stop_loss = (
                close * (1.0 - self.stop_loss_pct)
                if direction == "LONG"
                else close * (1.0 + self.stop_loss_pct)
            )
        metadata = _target_metadata(
            strategy=self.strategy_name,
            target_allocation=self.target_allocation,
            max_order_value=self.max_order_value,
            target_mode=direction,
            action="entry",
            deviation_z=float(deviation_z),
            bandwidth_percentile=pressure.get("bandwidth_percentile"),
            volatility_ratio=pressure.get("volatility_ratio"),
        )
        _emit(
            self.events,
            strategy_id=self.strategy_id,
            symbol=symbol,
            event_time=snapshot.time,
            signal_type=direction,
            strength=max(0.25, min(3.0, abs(deviation_z) / max(self.entry_z, _EPS))),
            price=close,
            stop_loss=stop_loss,
            metadata=metadata,
        )
        item.mode = direction
        item.entry_price = close
        item.bars_held = 0

    # -- state ---------------------------------------------------------------
    def get_state(self) -> dict[str, Any]:
        return {
            "symbol_state": {
                symbol: {
                    "opens": list(item.opens),
                    "highs": list(item.highs),
                    "lows": list(item.lows),
                    "closes": list(item.closes),
                    "volumes": list(item.volumes),
                    "mode": item.mode,
                    "entry_price": item.entry_price,
                    "bars_held": int(item.bars_held),
                    "last_time_key": item.last_time_key,
                }
                for symbol, item in self._state.items()
            }
        }

    def set_state(self, state: dict[str, Any]) -> None:
        raw = state.get("symbol_state") if isinstance(state, dict) else None
        if not isinstance(raw, dict):
            return
        for symbol, payload in raw.items():
            if symbol not in self._state or not isinstance(payload, dict):
                continue
            item = self._state[symbol]
            _restore_deque(item.opens, payload.get("opens"))
            _restore_deque(item.highs, payload.get("highs"))
            _restore_deque(item.lows, payload.get("lows"))
            _restore_deque(item.closes, payload.get("closes"))
            _restore_deque(item.volumes, payload.get("volumes"))
            mode = str(payload.get("mode", "OUT")).upper()
            item.mode = mode if mode in {"OUT", "LONG", "SHORT"} else "OUT"
            item.entry_price = safe_float(payload.get("entry_price"))
            item.bars_held = _safe_non_negative_int(payload.get("bars_held"))
            item.last_time_key = str(payload.get("last_time_key", ""))


__all__ = [
    "IntradayFlowPressureRiderStrategy",
    "SeasonalMicroBreakoutRiderStrategy",
    "VWAPCompressionReversionStrategy",
    "VolOfVolRegimeTrendGateStrategy",
]

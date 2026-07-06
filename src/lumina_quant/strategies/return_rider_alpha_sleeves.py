"""Per-symbol directional return-rider alpha sleeves.

These sleeves are built to MAXIMIZE return and compound return on a per-symbol
basis, calibrated to the proven ``TopCapTimeSeriesMomentumStrategy`` return-maker
(meaningful directional exposure, tight-stop discipline, long/short rotation).
Where the incumbent caps gains with a fixed take-profit, these sleeves instead
RIDE WINNERS with an ATR/percent trailing stop and PYRAMID into trend
continuation so a single clean move can compound into an outsized return.

The classes are standalone, self-contained, param-driven event-driven
strategies (NOT wrappers over other registered strategies and NOT
portfolio-sets).  Each trades exactly ONE symbol per candidate, so each stays
``candidate_mix_type=="single"`` and outside the multi-asset admission gate;
the per-symbol structure mirrors :class:`ConfidenceGatedTrendStrategy`.

- :class:`AdaptiveTrendRiderStrategy`: enters on a KAMA/efficiency-confirmed
  trend (an inline Kaufman Adaptive Moving Average whose smoothing is driven by
  :func:`kaufman_efficiency_ratio`), then rides the move with an ATR trailing
  stop, pyramids into continuation up to ``max_adds``, and vol-targets size so
  clean strong trends carry more risk.  Long and short.
- :class:`VolatilityBreakoutRiderStrategy`: enters on a Donchian/N-bar high
  (long) or low (short) breakout confirmed by ATR expansion, then rides the
  breakout with an ATR trailing stop and pyramids on follow-through.  Captures
  explosive crypto range expansions.
- :class:`AccelerationRiderStrategy`: enters when momentum
  (:func:`rate_of_change`) is positive AND accelerating (rising ROC), shorts
  when negative and accelerating down, then rides with a trailing stop and exits
  on deceleration or a stop breach.  Captures parabolic/blow-off moves.

All three: an ATR/percent TRAILING stop is the primary return lever (winners
run; there is no tight fixed take-profit that caps gains); a meaningful
``target_allocation`` calibrated to the TopCap scale (``tunable=False`` by
default); vol-targeted sizing; a ``max_hold_bars`` cap; per-symbol state with
pyramiding add tracking; bounded deques; ``None``-safe graceful guards (never
raise).  They remain data-local (no fetching, no artifact writes, no hidden
environment state) and must still be validated on the data-bearing machine.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from lumina_quant.core.plugin_registry import register
from lumina_quant.indicators.alpha_features import (
    finite_floats,
    realized_volatility,
)
from lumina_quant.indicators.atr import average_true_range, true_range
from lumina_quant.indicators.bands import donchian_channel
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.indicators.momentum import kaufman_efficiency_ratio
from lumina_quant.indicators.oscillators import rate_of_change
from lumina_quant.strategies.external_alpha_sleeves import (
    _EPS,
    _Snapshot,
    _emit,
    _event_symbols,
    _market_snapshot,
    _safe_non_negative_int,
    _target_metadata,
    _window_snapshot,
)
from lumina_quant.strategy import Strategy
from lumina_quant.tuning import HyperParam, resolve_params_from_schema


def _restore_deque(target: deque[float], payload: Any) -> None:
    """Refill a bounded deque from a serialized payload, dropping non-finite values."""
    target.clear()
    for value in list(payload or [])[-int(target.maxlen or 0) :]:
        parsed = safe_float(value)
        if parsed is not None:
            target.append(parsed)


def _mode(raw: Any) -> str:
    """Coerce a serialized mode token to one of ``{OUT, LONG, SHORT}``."""
    parsed = str(raw or "OUT").upper()
    return parsed if parsed in {"OUT", "LONG", "SHORT"} else "OUT"


def _kama(values: list[float], *, period: int, fast: int, slow: int) -> float | None:
    """Return the latest Kaufman Adaptive Moving Average over the trailing window.

    The smoothing constant adapts between a ``fast`` and ``slow`` EMA span via the
    Kaufman efficiency ratio: trending (high-efficiency) windows track price
    closely, choppy (low-efficiency) windows smooth heavily.  Returns ``None`` when
    there is insufficient history or the result is non-finite.

    Args:
        values: Trailing close prices (oldest first).
        period: Efficiency-ratio lookback in bars.
        fast: Fast EMA span for the most-efficient smoothing.
        slow: Slow EMA span for the least-efficient smoothing.

    Returns:
        The latest KAMA level, or ``None`` when unavailable.
    """
    period_i = max(1, int(period))
    fast_i = max(1, int(fast))
    slow_i = max(fast_i + 1, int(slow))
    vals = finite_floats(values)
    if len(vals) <= period_i:
        return None
    fast_sc = 2.0 / (float(fast_i) + 1.0)
    slow_sc = 2.0 / (float(slow_i) + 1.0)
    kama = vals[0]
    for idx in range(1, len(vals)):
        # Only the trailing ``period_i + 1`` closes feed the efficiency ratio, so
        # slice the tail directly instead of building ``vals[: idx + 1]`` and then
        # re-slicing it (O(n) copy per bar → O(n^2) overall). ``max(0, idx -
        # period_i)`` reproduces the ``[-(period_i + 1):]`` tail exactly, including
        # the short-history case (idx < period_i) where both yield ``vals[0 : idx +
        # 1]``. Byte-identical; 2026-07-06 audit X1 perf fix.
        window = vals[max(0, idx - period_i) : idx + 1]
        er = kaufman_efficiency_ratio(window, period=period_i)
        er_value = 0.0 if er is None else max(0.0, min(1.0, float(er)))
        smoothing = (er_value * (fast_sc - slow_sc) + slow_sc) ** 2
        kama = kama + smoothing * (vals[idx] - kama)
    if not isinstance(kama, float):
        kama = float(kama)
    return kama if kama == kama else None  # NaN guard


# ---------------------------------------------------------------------------
# Incremental KAMA (opt-in ``incremental_kama`` flag; 2026-07-06 audit stage-2
# perf fix). ``_kama`` above recomputes the *entire* trailing window every
# call, so calling it per-bar on a ``list(item.closes)`` of up to
# ``history_size`` entries costs O(history_size) per bar (the #1 hot path the
# 2026-07-03 audit flagged). The accumulator below instead maintains a running
# KAMA value plus small deques bounded to the efficiency-ratio/slope windows
# (``kama_period``/``slope_lookback``), so each accepted bar costs O(period)
# -- a small, backtest-length-independent constant -- instead of O(n).
#
# PARITY ENVELOPE: ``_kama_inc_update`` mirrors ``_kama``'s loop body exactly
# (same smoothing formula, same summation order for the efficiency-ratio
# volatility term, same "insufficient window -> ER forced to 0" clamp for the
# first ``period - 1`` steps after any seed), so replaying a full close
# history bar-by-bar through it is bit-identical to a from-scratch ``_kama``
# recompute over the same prefix. ``item.closes`` is itself a *bounded* ring
# buffer (``history_size``), though, and the legacy path recomputes ``_kama``
# from ``list(item.closes)`` fresh every bar -- so once a symbol's bar count
# exceeds ``history_size`` and the ring buffer starts evicting its oldest
# close, the legacy computation re-seeds its recursion at a new, later
# starting point every subsequent bar, while this accumulator keeps running
# continuously from its own original seed. The two are proven bit-identical
# for as long as no eviction has occurred (the regime the parity test
# exercises); beyond that both are well-defined but are expected to diverge by
# a small, KAMA-smoothing-constant-dependent amount that does not vanish over
# time (it is bounded by how much a single ``history_size``-bar window fails
# to forget its own seed, not by elapsed bars since eviction started).
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _KamaIncState:
    """Per-symbol incremental KAMA accumulator; see the module note above."""

    value: float = 0.0
    count: int = 0
    last_close: float = 0.0
    diffs: deque[float] = field(default_factory=deque)
    recent_closes: deque[float] = field(default_factory=deque)
    recent_kama: deque[float] = field(default_factory=deque)


def _new_kama_inc_state(*, period: int, slope_lookback: int) -> _KamaIncState:
    """Build a fresh accumulator with deques bounded to the ER/slope windows."""
    period_i = max(1, int(period))
    slope_i = max(1, int(slope_lookback))
    return _KamaIncState(
        diffs=deque(maxlen=period_i),
        recent_closes=deque(maxlen=period_i + 1),
        recent_kama=deque(maxlen=slope_i + 1),
    )


def _kama_inc_seed(state: _KamaIncState, close: float) -> None:
    """Seed the accumulator at the first raw close it has ever observed."""
    state.value = close
    state.count = 1
    state.last_close = close
    state.diffs.clear()
    state.recent_closes.clear()
    state.recent_closes.append(close)
    state.recent_kama.clear()
    state.recent_kama.append(close)


def _kama_inc_update(
    state: _KamaIncState, close: float, *, period: int, fast_sc: float, slow_sc: float
) -> None:
    """Apply one O(period) step; mirrors ``_kama``'s loop body exactly.

    ``idx`` is ``state.count`` before this call, which lines up bar-for-bar
    with the ``idx`` in ``_kama``'s ``for idx in range(1, len(vals))`` loop
    (the efficiency ratio is forced to ``0`` for the first ``period - 1``
    updates after any seed, exactly as ``_kama`` forces it for the first
    ``period - 1`` positions of any window it recomputes).
    """
    idx = state.count
    new_diff = abs(close - state.last_close)
    state.diffs.append(new_diff)
    state.recent_closes.append(close)
    if idx < period:
        er_value = 0.0
    else:
        direction = abs(state.recent_closes[-1] - state.recent_closes[0])
        volatility = 0.0
        for diff in state.diffs:
            volatility += diff
        if volatility <= 1e-12:
            er_value = 0.0
        else:
            ratio = direction / volatility
            er_value = max(0.0, min(1.0, ratio)) if math.isfinite(ratio) else 0.0
    smoothing = (er_value * (fast_sc - slow_sc) + slow_sc) ** 2
    state.value = state.value + smoothing * (close - state.value)
    state.recent_kama.append(state.value)
    state.last_close = close
    state.count = idx + 1


def _bootstrap_kama_inc(
    closes: list[float], *, period: int, fast_sc: float, slow_sc: float, slope_lookback: int
) -> _KamaIncState:
    """Build a fresh accumulator by replaying ``closes`` from its head.

    Runs once per symbol: on the symbol's very first accepted bar
    (``len(closes) == 1``, the common/cheap case -- the replay loop below is
    empty) or once if ``incremental_kama`` is engaged on a strategy instance
    whose state was restored with existing history (an O(len(closes))
    one-time cost, not a per-bar one).
    """
    state = _new_kama_inc_state(period=period, slope_lookback=slope_lookback)
    if not closes:
        return state
    _kama_inc_seed(state, closes[0])
    for close in closes[1:]:
        _kama_inc_update(state, close, period=period, fast_sc=fast_sc, slow_sc=slow_sc)
    return state


def _atr_value(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    *,
    period: int,
) -> float | None:
    """Return the absolute ATR (not percent) over the trailing ``period`` bars."""
    period_i = max(1, int(period))
    n = min(len(highs), len(lows), len(closes))
    if n < period_i + 1:
        return None
    # Only the last ``period_i`` true ranges are consumed (simple mean), so slice
    # the tail instead of recomputing TR over the full history every bar
    # (2026-07-03 audit perf fix; numerically identical).
    take = period_i + 1
    highs_f = [float(value) for value in list(highs)[-take:]]
    lows_f = [float(value) for value in list(lows)[-take:]]
    closes_f = [float(value) for value in list(closes)[-take:]]
    tr_values = [true_range(highs_f[idx], lows_f[idx], closes_f[idx - 1]) for idx in range(1, take)]
    return average_true_range(tr_values, period_i)


@dataclass(slots=True)
class _RiderState:
    """Per-symbol rider state: OHLC history, position, trailing stop, pyramiding."""

    opens: deque[float]
    highs: deque[float]
    lows: deque[float]
    closes: deque[float]
    mode: str = "OUT"
    entry_price: float | None = None
    stop_price: float | None = None
    high_watermark: float | None = None
    low_watermark: float | None = None
    adds: int = 0
    last_add_price: float | None = None
    bars_held: int = 0
    last_time_key: str = ""
    prev_roc: float | None = None

    def reset_position(self) -> None:
        """Clear all live-position fields back to a flat ``OUT`` state."""
        self.mode = "OUT"
        self.entry_price = None
        self.stop_price = None
        self.high_watermark = None
        self.low_watermark = None
        self.adds = 0
        self.last_add_price = None
        self.bars_held = 0


def _new_state(size: int) -> _RiderState:
    """Build a fresh :class:`_RiderState` with bounded OHLC deques of length ``size``."""
    bound = max(2, int(size))
    return _RiderState(
        opens=deque(maxlen=bound),
        highs=deque(maxlen=bound),
        lows=deque(maxlen=bound),
        closes=deque(maxlen=bound),
    )


@dataclass(slots=True)
class _RiderCommon:
    """Shared resolved configuration for the return-rider sleeves."""

    trail_atr_mult: float
    atr_period: int
    max_adds: int
    add_step_atr: float
    vol_window: int
    target_vol: float
    target_allocation: float
    max_order_value: float
    max_hold_bars: int
    allow_short: bool
    min_price: float
    add_alloc_fraction: float
    history_size: int = field(default=2)


class _ReturnRiderBase(Strategy):
    """Common machinery for per-symbol directional return-rider sleeves.

    Subclasses implement :meth:`_entry_decision` (returns ``"LONG"``/``"SHORT"``/
    ``""``) and :meth:`_should_pyramid` (continuation gate).  This base owns the
    OHLC ring buffers, the ATR trailing-stop ride logic, pyramiding bookkeeping,
    vol-targeted sizing, state (de)serialization, and signal emission so the three
    sleeves stay DRY and behave identically on the ride/exit side.
    """

    decision_cadence_seconds = 1800
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    strategy_name: str = "_ReturnRiderBase"
    strategy_id: str = "return_rider_base"

    def __init__(self, bars: Any, events: Any, **params: Any) -> None:
        self.bars = bars
        self.events = events
        self.symbol_list = list(getattr(self.bars, "symbol_list", []) or [])
        resolved = resolve_params_from_schema(self.get_param_schema(), params, keep_unknown=False)
        self._bind_params(resolved)
        cfg = self._common_config(resolved)
        self.trail_atr_mult = cfg.trail_atr_mult
        self.atr_period = cfg.atr_period
        self.max_adds = cfg.max_adds
        self.add_step_atr = cfg.add_step_atr
        self.vol_window = cfg.vol_window
        self.target_vol = cfg.target_vol
        self.target_allocation = cfg.target_allocation
        self.max_order_value = cfg.max_order_value
        self.max_hold_bars = cfg.max_hold_bars
        self.allow_short = cfg.allow_short
        self.min_price = cfg.min_price
        self.add_alloc_fraction = cfg.add_alloc_fraction
        self._history_size = max(2, int(cfg.history_size))
        self._state = {symbol: _new_state(self._history_size) for symbol in self.symbol_list}

    # -- configuration hooks -------------------------------------------------
    @staticmethod
    def _resolve_common(resolved: dict[str, Any], *, extra_window: int) -> _RiderCommon:
        """Resolve the shared rider config and derive a bounded history size."""
        atr_period = max(2, int(resolved["atr_period"]))
        vol_window = max(2, int(resolved["vol_window"]))
        max_hold_bars = max(1, int(resolved["max_hold_bars"]))
        history = max(atr_period, vol_window, max_hold_bars, max(2, int(extra_window))) + 8
        return _RiderCommon(
            trail_atr_mult=max(0.1, float(resolved["trail_atr_mult"])),
            atr_period=atr_period,
            max_adds=max(0, int(resolved["max_adds"])),
            add_step_atr=max(0.0, float(resolved["add_step_atr"])),
            vol_window=vol_window,
            target_vol=max(0.0, float(resolved["target_vol"])),
            target_allocation=max(0.0, float(resolved["target_allocation"])),
            max_order_value=max(0.0, float(resolved["max_order_value"])),
            max_hold_bars=max_hold_bars,
            allow_short=bool(resolved["allow_short"]),
            min_price=max(0.0, float(resolved["min_price"])),
            add_alloc_fraction=max(0.0, min(1.0, float(resolved["add_alloc_fraction"]))),
            history_size=history,
        )

    def _bind_params(self, resolved: dict[str, Any]) -> None:
        """Bind subclass-specific resolved params; overridden per sleeve."""
        raise NotImplementedError

    def _common_config(self, resolved: dict[str, Any]) -> _RiderCommon:
        """Return the resolved shared rider config; overridden per sleeve."""
        raise NotImplementedError

    def _entry_decision(self, item: _RiderState) -> str:
        """Return ``"LONG"``/``"SHORT"``/``""`` for a flat symbol; per-sleeve."""
        raise NotImplementedError

    def _should_pyramid(self, item: _RiderState) -> bool:
        """Return whether to scale into an existing winner; per-sleeve."""
        raise NotImplementedError

    def _entry_metadata(self, item: _RiderState) -> dict[str, Any]:
        """Return extra metadata to attach to entry/add signals; per-sleeve."""
        return {}

    # -- sizing --------------------------------------------------------------
    def _vol_scaled_allocation(self, closes: list[float]) -> float:
        """Return the base entry allocation, scaled toward ``target_vol``.

        Clean low-vol trends are sized up toward the configured target volatility;
        noisy high-vol symbols are throttled.  The result is clamped so a single
        sleeve never exceeds twice its nominal ``target_allocation``.
        """
        alloc = self.target_allocation
        if self.target_vol > 0.0:
            vol = realized_volatility(closes, window=self.vol_window)
            if vol is not None and vol > _EPS:
                alloc = self.target_allocation * (self.target_vol / vol)
        return max(0.0, min(self.target_allocation * 2.0, alloc))

    # -- event entry points --------------------------------------------------
    def calculate_signals_window(self, event: Any, aggregator: Any = None) -> None:
        _ = aggregator
        for symbol in _event_symbols(event, self.symbol_list):
            snapshot = _window_snapshot(event, symbol)
            if snapshot is not None:
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
                self._process_symbol(str(symbol), snapshot)

    # -- core loop -----------------------------------------------------------
    def _append_snapshot(self, item: _RiderState, snapshot: _Snapshot, close: float) -> None:
        """Append the latest bar's OHLC to the per-symbol ring buffers."""
        item.opens.append(float(snapshot.open if snapshot.open is not None else close))
        item.highs.append(float(snapshot.high if snapshot.high is not None else close))
        item.lows.append(float(snapshot.low if snapshot.low is not None else close))
        item.closes.append(close)

    def _process_symbol(self, symbol: str, snapshot: _Snapshot) -> None:
        item = self._state[symbol]
        key = time_key(snapshot.time)
        if key and key == item.last_time_key:
            return
        item.last_time_key = key
        close = safe_float(snapshot.close)
        if close is None or close <= self.min_price:
            return
        self._append_snapshot(item, snapshot, close)
        # Exit/ride management first so a stop frees the symbol for re-entry.
        if item.mode in {"LONG", "SHORT"}:
            self._ride(symbol, item, snapshot, close)
        if item.mode in {"LONG", "SHORT"}:
            self._maybe_pyramid(symbol, item, snapshot, close)
            return
        direction = self._entry_decision(item)
        if direction in {"LONG", "SHORT"} and (self.allow_short or direction == "LONG"):
            self._enter(symbol, item, snapshot, close, direction)

    def _current_atr(self, item: _RiderState) -> float | None:
        """Return the latest absolute ATR for the symbol, or ``None``."""
        return _atr_value(
            list(item.highs),
            list(item.lows),
            list(item.closes),
            period=self.atr_period,
        )

    def _ride(self, symbol: str, item: _RiderState, snapshot: _Snapshot, close: float) -> None:
        """Trail the stop behind the favorable extreme; exit only on a breach/cap.

        This is the return lever: the stop only ever ratchets in the direction of
        profit, so winners are allowed to run for many bars while a pullback past
        the trailing band (or ``max_hold_bars``) is what closes the position.
        """
        item.bars_held += 1
        atr = self._current_atr(item)
        reason = ""
        if item.mode == "LONG":
            item.high_watermark = max(float(item.high_watermark or close), close)
            if atr is not None and item.high_watermark is not None:
                trail = item.high_watermark - self.trail_atr_mult * atr
                if item.stop_price is None or trail > item.stop_price:
                    item.stop_price = trail
            if item.stop_price is not None and close <= item.stop_price:
                reason = "trailing_stop"
        else:
            item.low_watermark = min(float(item.low_watermark or close), close)
            if atr is not None and item.low_watermark is not None:
                trail = item.low_watermark + self.trail_atr_mult * atr
                if item.stop_price is None or trail < item.stop_price:
                    item.stop_price = trail
            if item.stop_price is not None and close >= item.stop_price:
                reason = "trailing_stop"
        if not reason and item.bars_held >= self.max_hold_bars:
            reason = "max_hold"
        if reason:
            self._emit_exit(symbol, item, snapshot.time, close, reason)

    def _maybe_pyramid(
        self, symbol: str, item: _RiderState, snapshot: _Snapshot, close: float
    ) -> None:
        """Scale into a winner on confirmed continuation, up to ``max_adds`` adds."""
        if self.max_adds <= 0 or item.adds >= self.max_adds or item.entry_price is None:
            return
        atr = self._current_atr(item)
        if atr is None or atr <= _EPS:
            return
        ref = item.last_add_price if item.last_add_price is not None else item.entry_price
        step = self.add_step_atr * atr
        advanced = False
        if item.mode == "LONG":
            advanced = close >= ref + step
        else:
            advanced = close <= ref - step
        if not advanced or not self._should_pyramid(item):
            return
        item.adds += 1
        item.last_add_price = close
        add_alloc = self._vol_scaled_allocation(list(item.closes)) * self.add_alloc_fraction
        metadata = _target_metadata(
            strategy=self.strategy_name,
            target_allocation=add_alloc,
            max_order_value=self.max_order_value,
            target_mode=item.mode,
            action="pyramid_add",
            add_index=item.adds,
            **self._entry_metadata(item),
        )
        _emit(
            self.events,
            strategy_id=self.strategy_id,
            symbol=symbol,
            event_time=snapshot.time,
            signal_type=item.mode,
            strength=max(0.25, min(3.0, 1.0 + 0.5 * item.adds)),
            price=close,
            stop_loss=item.stop_price,
            trailing_percent=None,
            metadata=metadata,
        )

    def _enter(
        self, symbol: str, item: _RiderState, snapshot: _Snapshot, close: float, direction: str
    ) -> None:
        """Open a fresh position with an initial ATR-based trailing stop."""
        atr = self._current_atr(item)
        stop_price = None
        if atr is not None and atr > _EPS:
            stop_price = (
                close - self.trail_atr_mult * atr
                if direction == "LONG"
                else close + self.trail_atr_mult * atr
            )
        alloc = self._vol_scaled_allocation(list(item.closes))
        metadata = _target_metadata(
            strategy=self.strategy_name,
            target_allocation=alloc,
            max_order_value=self.max_order_value,
            target_mode=direction,
            action="entry",
            atr=float(atr) if atr is not None else None,
            **self._entry_metadata(item),
        )
        _emit(
            self.events,
            strategy_id=self.strategy_id,
            symbol=symbol,
            event_time=snapshot.time,
            signal_type=direction,
            strength=1.0,
            price=close,
            stop_loss=stop_price,
            trailing_percent=None,
            metadata=metadata,
        )
        item.mode = direction
        item.entry_price = close
        item.stop_price = stop_price
        item.high_watermark = close
        item.low_watermark = close
        item.adds = 0
        item.last_add_price = close
        item.bars_held = 0

    def _emit_exit(
        self, symbol: str, item: _RiderState, event_time: Any, close: float | None, reason: str
    ) -> None:
        """Emit a flattening EXIT and reset the per-symbol position state."""
        _emit(
            self.events,
            strategy_id=self.strategy_id,
            symbol=symbol,
            event_time=event_time,
            signal_type="EXIT",
            price=close,
            metadata={"strategy": self.strategy_name, "reason": reason},
        )
        item.reset_position()

    # -- state ---------------------------------------------------------------
    def get_state(self) -> dict[str, Any]:
        return {
            "symbol_state": {
                symbol: {
                    "opens": list(item.opens),
                    "highs": list(item.highs),
                    "lows": list(item.lows),
                    "closes": list(item.closes),
                    "mode": item.mode,
                    "entry_price": item.entry_price,
                    "stop_price": item.stop_price,
                    "high_watermark": item.high_watermark,
                    "low_watermark": item.low_watermark,
                    "adds": int(item.adds),
                    "last_add_price": item.last_add_price,
                    "bars_held": int(item.bars_held),
                    "last_time_key": item.last_time_key,
                    "prev_roc": item.prev_roc,
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
            item.mode = _mode(payload.get("mode"))
            item.entry_price = safe_float(payload.get("entry_price"))
            item.stop_price = safe_float(payload.get("stop_price"))
            item.high_watermark = safe_float(payload.get("high_watermark"))
            item.low_watermark = safe_float(payload.get("low_watermark"))
            item.adds = _safe_non_negative_int(payload.get("adds"))
            item.last_add_price = safe_float(payload.get("last_add_price"))
            item.bars_held = _safe_non_negative_int(payload.get("bars_held"))
            item.last_time_key = str(payload.get("last_time_key", ""))
            item.prev_roc = safe_float(payload.get("prev_roc"))


@register("strategy", "AdaptiveTrendRiderStrategy", interface="event_driven")
class AdaptiveTrendRiderStrategy(_ReturnRiderBase):
    """Ride KAMA/efficiency-confirmed adaptive trends with an ATR trailing stop.

    Entry: the inline Kaufman Adaptive Moving Average (:func:`_kama`) must slope up
    and price close above it with a Kaufman efficiency ratio above
    ``min_efficiency`` (long), or the mirror (short).  The position is then ridden
    with an ATR trailing stop and pyramided into continuation up to ``max_adds``;
    size is vol-targeted so clean strong trends carry more risk.  Long and short.
    Designed to capture sustained adaptive trends for high compound return.
    """

    strategy_name = "AdaptiveTrendRiderStrategy"
    strategy_id = "adaptive_trend_rider"

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "kama_period": HyperParam.integer("kama_period", default=10, low=2, high=4096),
            "kama_fast": HyperParam.integer("kama_fast", default=2, low=1, high=512),
            "kama_slow": HyperParam.integer("kama_slow", default=30, low=2, high=4096),
            "min_efficiency": HyperParam.floating(
                "min_efficiency", default=0.30, low=0.0, high=1.0
            ),
            "slope_lookback": HyperParam.integer("slope_lookback", default=3, low=1, high=512),
            "trail_atr_mult": HyperParam.floating(
                "trail_atr_mult", default=3.0, low=0.1, high=20.0
            ),
            "atr_period": HyperParam.integer("atr_period", default=14, low=2, high=1024),
            "max_adds": HyperParam.integer("max_adds", default=3, low=0, high=32),
            "add_step_atr": HyperParam.floating("add_step_atr", default=1.0, low=0.0, high=20.0),
            "vol_window": HyperParam.integer("vol_window", default=48, low=2, high=4096),
            "target_vol": HyperParam.floating("target_vol", default=0.02, low=0.0, high=1.0),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=240, low=1, high=200000),
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
            # Opt-in, config-gated (2026-07-06 audit stage-2 perf fix). OFF is
            # the exact legacy O(history_size)-per-bar recompute path
            # (byte-identical); ON switches to the O(period)-per-bar
            # incremental accumulator -- see the module note above
            # ``_KamaIncState``. Not tunable: it changes the compute path, not
            # the strategy's economics.
            "incremental_kama": HyperParam.boolean(
                "incremental_kama", default=False, tunable=False
            ),
        }

    def _bind_params(self, resolved: dict[str, Any]) -> None:
        self.kama_period = max(2, int(resolved["kama_period"]))
        self.kama_fast = max(1, int(resolved["kama_fast"]))
        self.kama_slow = max(self.kama_fast + 1, int(resolved["kama_slow"]))
        self.min_efficiency = max(0.0, min(1.0, float(resolved["min_efficiency"])))
        self.slope_lookback = max(1, int(resolved["slope_lookback"]))
        self.incremental_kama = bool(resolved["incremental_kama"])
        # Precomputed once (matches ``_kama``'s own internal normalization
        # exactly, since kama_fast/kama_slow are already clamped above), so the
        # incremental accumulator never redoes this per bar.
        self._kama_fast_sc = 2.0 / (float(self.kama_fast) + 1.0)
        self._kama_slow_sc = 2.0 / (float(self.kama_slow) + 1.0)

    def _common_config(self, resolved: dict[str, Any]) -> _RiderCommon:
        return self._resolve_common(
            resolved,
            extra_window=max(self.kama_period, self.kama_slow, self.slope_lookback) + 2,
        )

    def _kama_now_prev(self, closes: list[float]) -> tuple[float | None, float | None]:
        """Return ``(kama_now, kama_prev)`` for slope estimation, or ``(None, None)``."""
        if len(closes) <= self.kama_period + self.slope_lookback:
            return None, None
        now = _kama(closes, period=self.kama_period, fast=self.kama_fast, slow=self.kama_slow)
        prev = _kama(
            closes[: -self.slope_lookback],
            period=self.kama_period,
            fast=self.kama_fast,
            slow=self.kama_slow,
        )
        return now, prev

    def _kama_now_prev_cached(self, item: _RiderState) -> tuple[float | None, float | None]:
        """Per-bar memo of :meth:`_kama_now_prev` (2026-07-03 audit perf fix).

        The full-history KAMA recursion is O(n * period) and was recomputed up
        to three times per accepted bar (_entry_decision, _should_pyramid,
        _entry_metadata). The memo is keyed by the bar's time key, so the value
        is computed exactly once per bar and the numbers are identical.

        When ``incremental_kama`` is on, the cache-miss branch reads the
        O(period) accumulator (:meth:`_kama_now_prev_incremental`) instead of
        recomputing over ``list(item.closes)``; see the module note above
        ``_KamaIncState`` for the (flag-off-preserving) parity envelope.
        """
        cache = getattr(self, "_kama_bar_cache", None)
        if cache is None:
            cache = {}
            self._kama_bar_cache = cache
        key = item.last_time_key
        cached = cache.get(id(item))
        if cached is not None and cached[0] == key and key:
            return cached[1]
        if getattr(self, "incremental_kama", False):
            value = self._kama_now_prev_incremental(item)
        else:
            value = self._kama_now_prev(list(item.closes))
        cache[id(item)] = (key, value)
        return value

    def _kama_now_prev_incremental(self, item: _RiderState) -> tuple[float | None, float | None]:
        """Incremental-path equivalent of :meth:`_kama_now_prev` (flag ON only).

        Reads the accumulator maintained by :meth:`_advance_kama_inc` (fed
        once per accepted bar from :meth:`_append_snapshot`) instead of
        recomputing the KAMA recursion from scratch.
        """
        states = getattr(self, "_kama_inc_states", None)
        state = states.get(id(item)) if states else None
        if state is None or state.count <= self.kama_period + self.slope_lookback:
            return None, None
        if len(state.recent_kama) <= self.slope_lookback:
            return state.value, None
        return state.value, state.recent_kama[0]

    def _append_snapshot(self, item: _RiderState, snapshot: _Snapshot, close: float) -> None:
        super()._append_snapshot(item, snapshot, close)
        if getattr(self, "incremental_kama", False):
            self._advance_kama_inc(item, close)

    def _advance_kama_inc(self, item: _RiderState, close: float) -> None:
        """Feed one accepted bar into the O(period) incremental KAMA accumulator.

        Called from :meth:`_append_snapshot` so it runs exactly once per
        accepted bar, in calendar order, with no risk of skipping bars (unlike
        driving it lazily from :meth:`_kama_now_prev_cached`, which is not
        guaranteed to be invoked every bar -- e.g. while pyramiding is
        exhausted). The first call for a given ``item`` bootstraps by
        replaying its current ``closes`` (see :func:`_bootstrap_kama_inc`),
        which transparently covers both a brand-new symbol (a single-element
        replay) and a strategy instance resumed from serialized state with
        existing history (a one-time O(history) replay).
        """
        states = getattr(self, "_kama_inc_states", None)
        if states is None:
            states = {}
            self._kama_inc_states = states
        key = id(item)
        state = states.get(key)
        if state is None:
            states[key] = _bootstrap_kama_inc(
                list(item.closes),
                period=self.kama_period,
                fast_sc=self._kama_fast_sc,
                slow_sc=self._kama_slow_sc,
                slope_lookback=self.slope_lookback,
            )
            return
        _kama_inc_update(
            state,
            close,
            period=self.kama_period,
            fast_sc=self._kama_fast_sc,
            slow_sc=self._kama_slow_sc,
        )

    def _entry_decision(self, item: _RiderState) -> str:
        kama_now, kama_prev = self._kama_now_prev_cached(item)
        if kama_now is None or kama_prev is None:
            return ""
        if getattr(self, "incremental_kama", False):
            # Avoid materializing the (potentially large) ring buffer into a
            # fresh list just to read its trailing ``kama_period + 1`` entries
            # -- ``kaufman_efficiency_ratio`` slices a deque's tail directly.
            er = kaufman_efficiency_ratio(item.closes, period=self.kama_period)
            close = item.closes[-1]
        else:
            closes = list(item.closes)
            er = kaufman_efficiency_ratio(closes, period=self.kama_period)
            close = closes[-1]
        if er is None or er < self.min_efficiency:
            return ""
        if kama_now > kama_prev and close > kama_now:
            return "LONG"
        if kama_now < kama_prev and close < kama_now:
            return "SHORT"
        return ""

    def _should_pyramid(self, item: _RiderState) -> bool:
        kama_now, kama_prev = self._kama_now_prev_cached(item)
        if kama_now is None or kama_prev is None:
            return False
        if item.mode == "LONG":
            return kama_now > kama_prev
        return kama_now < kama_prev

    def _entry_metadata(self, item: _RiderState) -> dict[str, Any]:
        closes = item.closes if getattr(self, "incremental_kama", False) else list(item.closes)
        er = kaufman_efficiency_ratio(closes, period=self.kama_period)
        return {"efficiency_ratio": float(er) if er is not None else None}


@register("strategy", "VolatilityBreakoutRiderStrategy", interface="event_driven")
class VolatilityBreakoutRiderStrategy(_ReturnRiderBase):
    """Ride Donchian/N-bar breakouts confirmed by ATR expansion.

    Entry: price closes at or above the trailing ``donchian_window``-bar high
    (long) or at/below the low (short), AND the current ATR exceeds its own
    trailing average by ``atr_expansion_mult`` (an explicit volatility-expansion
    confirmation that filters dead-range pokes).  The breakout is then ridden with
    an ATR trailing stop and pyramided on follow-through.  Long and short.
    Designed to capture explosive crypto range expansions.
    """

    strategy_name = "VolatilityBreakoutRiderStrategy"
    strategy_id = "volatility_breakout_rider"

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "donchian_window": HyperParam.integer("donchian_window", default=20, low=2, high=4096),
            "atr_expansion_mult": HyperParam.floating(
                "atr_expansion_mult", default=1.1, low=0.1, high=20.0
            ),
            "atr_baseline_window": HyperParam.integer(
                "atr_baseline_window", default=48, low=2, high=4096
            ),
            "trail_atr_mult": HyperParam.floating(
                "trail_atr_mult", default=3.5, low=0.1, high=20.0
            ),
            "atr_period": HyperParam.integer("atr_period", default=14, low=2, high=1024),
            "max_adds": HyperParam.integer("max_adds", default=3, low=0, high=32),
            "add_step_atr": HyperParam.floating("add_step_atr", default=1.0, low=0.0, high=20.0),
            "vol_window": HyperParam.integer("vol_window", default=48, low=2, high=4096),
            "target_vol": HyperParam.floating("target_vol", default=0.02, low=0.0, high=1.0),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=240, low=1, high=200000),
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

    def _bind_params(self, resolved: dict[str, Any]) -> None:
        self.donchian_window = max(2, int(resolved["donchian_window"]))
        self.atr_expansion_mult = max(0.1, float(resolved["atr_expansion_mult"]))
        self.atr_baseline_window = max(2, int(resolved["atr_baseline_window"]))

    def _common_config(self, resolved: dict[str, Any]) -> _RiderCommon:
        return self._resolve_common(
            resolved,
            extra_window=max(self.donchian_window, self.atr_baseline_window) + 2,
        )

    def _atr_expanding(self, item: _RiderState) -> bool:
        """Return whether the current ATR exceeds its trailing-baseline ATR."""
        highs = list(item.highs)
        lows = list(item.lows)
        closes = list(item.closes)
        atr_now = _atr_value(highs, lows, closes, period=self.atr_period)
        atr_base = _atr_value(highs, lows, closes, period=self.atr_baseline_window)
        if atr_now is None or atr_base is None or atr_base <= _EPS:
            return False
        return atr_now >= self.atr_expansion_mult * atr_base

    def _entry_decision(self, item: _RiderState) -> str:
        highs = list(item.highs)
        lows = list(item.lows)
        if len(highs) <= self.donchian_window or len(lows) <= self.donchian_window:
            return ""
        # Exclude the live bar so the breakout level is the PRIOR N-bar extreme.
        upper, _middle, lower = donchian_channel(highs[:-1], lows[:-1], window=self.donchian_window)
        if upper is None or lower is None:
            return ""
        if not self._atr_expanding(item):
            return ""
        close = list(item.closes)[-1]
        if close >= upper:
            return "LONG"
        if close <= lower:
            return "SHORT"
        return ""

    def _should_pyramid(self, item: _RiderState) -> bool:
        # Follow-through is confirmed by continued favorable extension, which the
        # ATR add-step gate in the base already enforces; keep adding while the
        # ATR remains in an expansion regime so we only compound live breakouts.
        return self._atr_expanding(item)

    def _entry_metadata(self, item: _RiderState) -> dict[str, Any]:
        return {"donchian_window": int(self.donchian_window)}


@register("strategy", "AccelerationRiderStrategy", interface="event_driven")
class AccelerationRiderStrategy(_ReturnRiderBase):
    """Ride parabolic/blow-off moves where momentum is positive AND accelerating.

    Entry: the rate-of-change (:func:`rate_of_change`) over ``roc_period`` is
    positive and rising versus the prior bar (long), or negative and falling
    (short) — i.e. momentum is accelerating.  The move is ridden with an ATR
    trailing stop and pyramided while acceleration persists; the position exits on
    a trailing-stop breach, ``max_hold_bars``, or clear deceleration (ROC sign
    reversal beyond ``decel_tolerance``).  Designed to capture parabolic moves.
    """

    strategy_name = "AccelerationRiderStrategy"
    strategy_id = "acceleration_rider"

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "roc_period": HyperParam.integer("roc_period", default=8, low=1, high=4096),
            "min_roc": HyperParam.floating("min_roc", default=0.0, low=0.0, high=1.0),
            "decel_tolerance": HyperParam.floating(
                "decel_tolerance", default=0.0, low=0.0, high=1.0
            ),
            "trail_atr_mult": HyperParam.floating(
                "trail_atr_mult", default=2.5, low=0.1, high=20.0
            ),
            "atr_period": HyperParam.integer("atr_period", default=14, low=2, high=1024),
            "max_adds": HyperParam.integer("max_adds", default=4, low=0, high=32),
            "add_step_atr": HyperParam.floating("add_step_atr", default=0.75, low=0.0, high=20.0),
            "vol_window": HyperParam.integer("vol_window", default=48, low=2, high=4096),
            "target_vol": HyperParam.floating("target_vol", default=0.025, low=0.0, high=1.0),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=180, low=1, high=200000),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
            "add_alloc_fraction": HyperParam.floating(
                "add_alloc_fraction", default=0.6, low=0.0, high=1.0
            ),
            "target_allocation": HyperParam.floating(
                "target_allocation", default=0.30, low=0.0, high=5.0, tunable=False
            ),
            "max_order_value": HyperParam.floating(
                "max_order_value", default=5000.0, low=0.0, high=1_000_000.0, tunable=False
            ),
            "min_price": HyperParam.floating("min_price", default=0.10, low=0.0, high=1_000_000.0),
        }

    def _bind_params(self, resolved: dict[str, Any]) -> None:
        self.roc_period = max(1, int(resolved["roc_period"]))
        self.min_roc = max(0.0, float(resolved["min_roc"]))
        self.decel_tolerance = max(0.0, float(resolved["decel_tolerance"]))

    def _common_config(self, resolved: dict[str, Any]) -> _RiderCommon:
        return self._resolve_common(resolved, extra_window=self.roc_period + 2)

    def _entry_decision(self, item: _RiderState) -> str:
        closes = list(item.closes)
        roc = rate_of_change(closes, period=self.roc_period)
        prev_roc = item.prev_roc
        item.prev_roc = roc
        if roc is None or prev_roc is None:
            return ""
        if roc >= self.min_roc and roc > prev_roc and roc > 0.0:
            return "LONG"
        if roc <= -self.min_roc and roc < prev_roc and roc < 0.0:
            return "SHORT"
        return ""

    def _should_pyramid(self, item: _RiderState) -> bool:
        roc = rate_of_change(list(item.closes), period=self.roc_period)
        if roc is None:
            return False
        if item.mode == "LONG":
            return roc > -self.decel_tolerance and roc > 0.0
        return roc < self.decel_tolerance and roc < 0.0

    def _process_symbol(self, symbol: str, snapshot: _Snapshot) -> None:
        # Track ROC every accepted bar (including while in a position) so the
        # acceleration gate and prev_roc stay current across the whole ride.
        item = self._state.get(symbol)
        super()._process_symbol(symbol, snapshot)
        if item is not None and item.mode in {"LONG", "SHORT"}:
            item.prev_roc = rate_of_change(list(item.closes), period=self.roc_period)

    def _entry_metadata(self, item: _RiderState) -> dict[str, Any]:
        roc = rate_of_change(list(item.closes), period=self.roc_period)
        return {"rate_of_change": float(roc) if roc is not None else None}


__all__ = [
    "AccelerationRiderStrategy",
    "AdaptiveTrendRiderStrategy",
    "VolatilityBreakoutRiderStrategy",
]

"""Research-only RSI-divergence sleeve (regular + hidden) with a staged RSI exit.

Lineage
-------
Independent adaptation of the *publicly described* BTC-futures RSI-divergence
method of the Korean trader FlightF (handle "FlightF"), as catalogued in
``/home/hoky/dacapogo/docs/korean-trader-strategies.md`` (section "FlightF").
It is **not a reproduction**, not an endorsement, and carries **no performance
claim**: no audit-grade trade ledger exists for that trader, so nothing here
can be matched against real fills. Classification: ``research_only``.

The public rule, as audited
---------------------------
Everything in this list is what the public posts state; nothing else below is.

1. Instrument and timeframe: BTC futures read on **10-minute bars**
   (``decision_cadence_seconds = 600``).
2. Long setup: a **bullish RSI divergence while RSI is under 20** -- price
   makes a lower low, RSI makes a higher low, with the newer RSI low inside
   the sub-20 zone. The short setup is the exact mirror, **RSI above 80**.
3. Exit: **more than half** of the position is taken off once RSI reaches the
   **40~50 band**, and the **remainder above 60**. Shorts are symmetric
   (60~50 for the first stage, under 40 for the rest).
4. Invalidation: a **larger opposing-direction volume** print after the signal
   invalidates the divergence -- the setup is dropped rather than held.
5. Confirmation: the **30m / 1h / 4h / daily** charts are read together, and a
   trade is taken only when the higher timeframes agree with its direction.

A second post (dcinside *chartanalysis* 993821) shows the charting screen
rather than an order rule: **RSI length 11** with 25/75 band lines drawn, and
moving averages 7/20/50/100/200/400/800. The RSI length is taken from there;
the 25/75 lines are screen furniture and are *not* used as the divergence
zone -- rule 2 above (20/80) is. The two posts are separate public snapshots
and are not asserted to have been one live order rule set.

AUTHOR's choices (undisclosed by the sources, picked here)
----------------------------------------------------------
Everything not in the audited list is this module's own decision:

* RSI smoothing = Wilder (``IncrementalRsi``); pivot width = 3 bars, confirmed
  only when the right-hand bar closes; admissible pivot separation =
  ``[min_pivot_distance, max_pivot_distance]`` bars scanned over the last four
  confirmed pivots.
* Entry on the confirmation bar's close (the sources do not say whether entry
  was at the pivot or after confirmation).
* The staged exit of rule 3 is emitted as a real partial: on the first bar
  whose RSI reaches ``exit_rsi_first`` = 45.0 (the midpoint of the stated
  40~50 band) an ``EXIT`` carrying ``metadata["exit_fraction"]`` reduces the
  position by ``first_exit_fraction``, and the remainder leaves on a full
  ``EXIT`` at ``exit_rsi_second`` = 60.0. The source says "more than half";
  **0.6** is this module's reading of that phrase. Shorts mirror the levels
  (55.0 then 40.0). Risk exits -- pivot stop, opposing volume, time cap --
  always close everything, from either stage.
* Higher-timeframe confirmation (rule 5) is modelled with **one** higher
  timeframe, not four: the strategy's own bars are grouped ``htf_multiple`` at
  a time (default 6, i.e. 1h from 10m bars; 3/24/144 would be 30m/4h/daily)
  and the direction filter is "last completed HTF close above/below
  ``SMA(htf_ma_window)`` of the completed HTF closes". It is **off by
  default** (``require_htf_confirmation``); the sources never define how the
  four charts were combined.
* Volume invalidation (rule 4) is modelled twice, both times with thresholds
  the sources never gave: at setup time "the second pivot's volume must not
  exceed the first pivot's" (``require_volume_confirmation``, on by default,
  skipped when the feed has no volume), and in-trade as an exit when a bar
  *against* the position prints more than ``opposing_volume_multiple`` times
  the two pivot bars' average volume (off by default).
* Hidden divergence is a pure extension -- the sources describe only the
  regular case. It is gated by ``close > SMA(ma_fast) > SMA(ma_slow)`` (long)
  using *lagged* moving averages. The 7/100/200/400/800 averages are not
  modelled, only 20/50 direction.
* Position sizing, stop placement (pivot extreme) and the 12-hour time cap are
  this module's risk choices. A stop is triggered by a long bar's ``low <=
  stop`` or a short bar's ``high >= stop``. As a bar-close strategy it emits a
  market EXIT after observing that breach; the execution contract fills at the
  next available open, including a gap through the stop, rather than claiming
  an intrabar stop-price fill. The leverage/"challenge" money management of
  the public posts is explicitly out of scope.

Hypothesis
----------
On intraday crypto bars, an RSI divergence that is confirmed by a closed 3-bar
pivot, sits in the extreme RSI band (or agrees with the 20/50 trend for hidden
divergences), is not contradicted by a heavier opposing volume and -- when the
filter is enabled -- agrees with the higher timeframe, is followed by a
mean-reverting swing large enough to reach a neutral RSI level before the pivot
extreme is broken.

Cadence: intended for 10-minute bars (``decision_cadence_seconds = 600``), but
the rules are timeframe-agnostic and run on whatever bars are fed in.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
import math
from typing import Any

from lumina_quant.core.plugin_registry import register
from lumina_quant.indicators import simple_moving_average
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.indicators.rsi import IncrementalRsi
from lumina_quant.strategies.external_alpha_sleeves import (
    _Snapshot,
    _emit,
    _event_symbols,
    _market_snapshot,
    _target_metadata,
    _window_snapshot,
)
from lumina_quant.strategy import Strategy
from lumina_quant.tuning import HyperParam, resolve_params_from_schema

_STRATEGY_ID = "rsi_divergence_scale_out"


@dataclass(slots=True)
class _Pivot:
    """One confirmed 3-bar swing point."""

    bar_index: int
    price: float
    rsi: float
    volume: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "bar_index": int(self.bar_index),
            "price": float(self.price),
            "rsi": float(self.rsi),
            "volume": None if self.volume is None else float(self.volume),
        }


def _pivot_from_dict(payload: Any) -> _Pivot | None:
    if not isinstance(payload, dict):
        return None
    price = safe_float(payload.get("price"))
    rsi = safe_float(payload.get("rsi"))
    if price is None or rsi is None:
        return None
    try:
        bar_index = int(payload.get("bar_index", 0))
    except TypeError, ValueError:
        return None
    return _Pivot(bar_index, price, rsi, safe_float(payload.get("volume")))


@dataclass(slots=True)
class _State:
    closes: deque[float]
    highs: deque[float]
    lows: deque[float]
    volumes: deque[float | None]
    rsis: deque[float | None]
    htf_closes: deque[float]
    rsi_calc: IncrementalRsi
    pivot_lows: list[_Pivot] = field(default_factory=list)
    pivot_highs: list[_Pivot] = field(default_factory=list)
    bars_seen: int = 0
    htf_bucket: int | None = None
    htf_bucket_close: float | None = None
    mode: str = "OUT"
    entry_price: float | None = None
    stop_price: float | None = None
    pivot_volume_avg: float | None = None
    exit_stage: int = 0
    bars_held: int = 0
    last_time_key: str = ""


@register("strategy", "RsiDivergenceScaleOutStrategy", interface="event_driven")
class RsiDivergenceScaleOutStrategy(Strategy):
    """Trade confirmed RSI divergences and exit at the first neutral-RSI stage."""

    decision_cadence_seconds = 600
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "rsi_period": HyperParam.integer("rsi_period", default=11, low=2, high=200),
            "oversold": HyperParam.floating("oversold", default=20.0, low=1.0, high=49.0),
            "overbought": HyperParam.floating("overbought", default=80.0, low=51.0, high=99.0),
            "min_pivot_distance": HyperParam.integer(
                "min_pivot_distance", default=3, low=1, high=200
            ),
            "max_pivot_distance": HyperParam.integer(
                "max_pivot_distance", default=40, low=2, high=1000
            ),
            "use_hidden": HyperParam.boolean("use_hidden", default=True),
            "ma_fast": HyperParam.integer("ma_fast", default=20, low=2, high=500),
            "ma_slow": HyperParam.integer("ma_slow", default=50, low=3, high=1000),
            "require_volume_confirmation": HyperParam.boolean(
                "require_volume_confirmation", default=True
            ),
            "opposing_volume_multiple": HyperParam.floating(
                "opposing_volume_multiple", default=0.0, low=0.0, high=20.0
            ),
            "require_htf_confirmation": HyperParam.boolean(
                "require_htf_confirmation", default=False
            ),
            "htf_multiple": HyperParam.integer(
                "htf_multiple", default=6, low=1, high=1000, tunable=False
            ),
            "htf_ma_window": HyperParam.integer("htf_ma_window", default=20, low=2, high=500),
            "exit_rsi_first": HyperParam.floating(
                "exit_rsi_first", default=45.0, low=5.0, high=95.0
            ),
            "exit_rsi_second": HyperParam.floating(
                "exit_rsi_second", default=60.0, low=5.0, high=95.0
            ),
            "first_exit_fraction": HyperParam.floating(
                "first_exit_fraction", default=0.6, low=0.0, high=1.0, tunable=False
            ),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=72, low=1, high=5000),
            "allow_short": HyperParam.boolean("allow_short", default=True),
            "target_allocation": HyperParam.floating(
                "target_allocation", default=0.10, low=0.0, high=1.0, tunable=False
            ),
            "max_order_value": HyperParam.floating(
                "max_order_value", default=500.0, low=0.0, high=1_000_000.0, tunable=False
            ),
        }

    def __init__(self, bars: Any, events: Any, **params: Any) -> None:
        self.bars = bars
        self.events = events
        self.symbol_list = list(getattr(bars, "symbol_list", []) or [])
        resolved = resolve_params_from_schema(self.get_param_schema(), params, keep_unknown=False)
        self.rsi_period = max(2, int(resolved["rsi_period"]))
        self.oversold = min(49.0, max(1.0, float(resolved["oversold"])))
        self.overbought = min(99.0, max(51.0, float(resolved["overbought"])))
        self.min_pivot_distance = max(1, int(resolved["min_pivot_distance"]))
        self.max_pivot_distance = max(self.min_pivot_distance, int(resolved["max_pivot_distance"]))
        self.use_hidden = bool(resolved["use_hidden"])
        self.ma_fast = max(2, int(resolved["ma_fast"]))
        self.ma_slow = max(self.ma_fast + 1, int(resolved["ma_slow"]))
        self.require_volume_confirmation = bool(resolved["require_volume_confirmation"])
        self.opposing_volume_multiple = max(0.0, float(resolved["opposing_volume_multiple"]))
        self.require_htf_confirmation = bool(resolved["require_htf_confirmation"])
        self.htf_multiple = max(1, int(resolved["htf_multiple"]))
        self.htf_ma_window = max(2, int(resolved["htf_ma_window"]))
        self.exit_rsi_first = min(95.0, max(5.0, float(resolved["exit_rsi_first"])))
        self.exit_rsi_second = min(95.0, max(5.0, float(resolved["exit_rsi_second"])))
        # A reversed ladder can cross the final threshold before a partial
        # exit is representable. Refuse new entries instead of reordering
        # frozen parameters or leaving an unsafe open position.
        self._valid_exit_thresholds = self.exit_rsi_first < self.exit_rsi_second
        # "More than half" in the source; 0.6 is this module's reading of it.
        self.first_exit_fraction = min(1.0, max(0.0, float(resolved["first_exit_fraction"])))
        self.max_hold_bars = max(1, int(resolved["max_hold_bars"]))
        self.allow_short = bool(resolved["allow_short"])
        self.target_allocation = max(0.0, float(resolved["target_allocation"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        size = max(
            self.rsi_period * 4 + self.max_pivot_distance + 10,
            self.ma_slow + 5,
        )
        self._state = {symbol: self._new_state(size) for symbol in self.symbol_list}

    def _new_state(self, size: int) -> _State:
        return _State(
            closes=deque(maxlen=size),
            highs=deque(maxlen=size),
            lows=deque(maxlen=size),
            volumes=deque(maxlen=size),
            rsis=deque(maxlen=size),
            htf_closes=deque(maxlen=self.htf_ma_window),
            rsi_calc=IncrementalRsi(self.rsi_period),
        )

    # ------------------------------------------------------------------
    # state round-trip
    # ------------------------------------------------------------------
    def get_state(self) -> dict[str, Any]:
        return {
            "symbol_state": {
                symbol: {
                    "closes": list(item.closes),
                    "highs": list(item.highs),
                    "lows": list(item.lows),
                    "volumes": list(item.volumes),
                    "rsis": list(item.rsis),
                    "htf_closes": list(item.htf_closes),
                    "htf_bucket": item.htf_bucket,
                    "htf_bucket_close": item.htf_bucket_close,
                    "rsi_calc": item.rsi_calc.to_state(),
                    "pivot_lows": [pivot.as_dict() for pivot in item.pivot_lows],
                    "pivot_highs": [pivot.as_dict() for pivot in item.pivot_highs],
                    "bars_seen": int(item.bars_seen),
                    "mode": item.mode,
                    "entry_price": item.entry_price,
                    "stop_price": item.stop_price,
                    "pivot_volume_avg": item.pivot_volume_avg,
                    "exit_stage": int(item.exit_stage),
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
            for name in ("closes", "highs", "lows", "htf_closes"):
                target: deque[Any] = getattr(item, name)
                target.clear()
                for value in list(payload.get(name) or [])[-int(target.maxlen or 0) :]:
                    parsed = safe_float(value)
                    if parsed is not None:
                        target.append(parsed)
            for name in ("volumes", "rsis"):
                # These two may legitimately hold ``None`` (no volume feed, or
                # RSI still warming up); keep the slot so they stay aligned
                # with the price deques.
                optional: deque[Any] = getattr(item, name)
                optional.clear()
                for value in list(payload.get(name) or [])[-int(optional.maxlen or 0) :]:
                    optional.append(safe_float(value))
            item.rsi_calc = IncrementalRsi(self.rsi_period)
            item.rsi_calc.load_state(payload.get("rsi_calc") or {})
            for name in ("pivot_lows", "pivot_highs"):
                pivots = [
                    pivot
                    for raw_pivot in list(payload.get(name) or [])
                    if (pivot := _pivot_from_dict(raw_pivot)) is not None
                ]
                setattr(item, name, pivots)
            try:
                item.bars_seen = max(0, int(payload.get("bars_seen", 0)))
            except TypeError, ValueError:
                item.bars_seen = 0
            cutoff = item.bars_seen - 2 - self.max_pivot_distance
            item.pivot_lows[:] = [pivot for pivot in item.pivot_lows if pivot.bar_index >= cutoff]
            item.pivot_highs[:] = [pivot for pivot in item.pivot_highs if pivot.bar_index >= cutoff]
            try:
                item.htf_bucket = int(payload.get("htf_bucket"))
            except TypeError, ValueError:
                item.htf_bucket = None
            item.htf_bucket_close = safe_float(payload.get("htf_bucket_close"))
            mode = str(payload.get("mode", "OUT")).upper()
            item.mode = mode if mode in ("LONG", "SHORT") else "OUT"
            item.entry_price = safe_float(payload.get("entry_price"))
            item.stop_price = safe_float(payload.get("stop_price"))
            item.pivot_volume_avg = safe_float(payload.get("pivot_volume_avg"))
            try:
                item.exit_stage = 1 if int(payload.get("exit_stage", 0)) == 1 else 0
            except TypeError, ValueError:
                item.exit_stage = 0
            try:
                item.bars_held = max(0, int(payload.get("bars_held", 0)))
            except TypeError, ValueError:
                item.bars_held = 0
            item.last_time_key = str(payload.get("last_time_key", ""))

    # ------------------------------------------------------------------
    # event plumbing
    # ------------------------------------------------------------------
    def calculate_signals_window(self, event: Any, aggregator: Any = None) -> None:
        _ = aggregator
        for symbol in _event_symbols(event, self.symbol_list):
            snapshot = _window_snapshot(event, symbol)
            if snapshot is not None:
                self._process(symbol, snapshot)

    def calculate_signals(self, event: Any) -> None:
        event_type = str(getattr(event, "type", "")).upper()
        if event_type == "MARKET_WINDOW":
            self.calculate_signals_window(event)
            return
        if event_type != "MARKET":
            return
        symbol = str(getattr(event, "symbol", ""))
        if symbol in self._state and (snapshot := _market_snapshot(event)) is not None:
            self._process(symbol, snapshot)

    # ------------------------------------------------------------------
    # core
    # ------------------------------------------------------------------
    def _process(self, symbol: str, snapshot: _Snapshot) -> None:
        item = self._state[symbol]
        key = time_key(snapshot.time)
        if key and key == item.last_time_key:
            return
        close = safe_float(snapshot.close)
        high = safe_float(snapshot.high)
        low = safe_float(snapshot.low)
        if close is None or high is None or low is None:
            return
        item.last_time_key = key
        item.closes.append(close)
        item.highs.append(high)
        item.lows.append(low)
        item.volumes.append(safe_float(snapshot.volume))
        rsi = item.rsi_calc.update(close)
        item.rsis.append(rsi)
        item.bars_seen += 1
        self._record_htf_close(item, snapshot.time, close)

        new_low, new_high = self._record_pivots(item)

        if item.mode != "OUT":
            self._manage_position(symbol, item, snapshot, close, rsi)
            return
        if self.target_allocation <= 0.0 or not self._valid_exit_thresholds:
            # ``_target_metadata`` drops the key when the allocation is not
            # positive, so the portfolio would size this entry off its own
            # config default.  Refuse rather than emit an unsized signal.
            # Pivots are still recorded above, so the book stays warm.
            return
        if new_low is not None and self._enter_long(symbol, item, snapshot, close, new_low):
            return
        if self.allow_short and new_high is not None:
            self._enter_short(symbol, item, snapshot, close, new_high)

    def _record_pivots(self, item: _State) -> tuple[_Pivot | None, _Pivot | None]:
        """Confirm a 3-bar pivot on the *previous* bar now that its right bar closed."""
        if len(item.lows) < 3 or len(item.highs) < 3 or len(item.rsis) < 2:
            return None, None
        pivot_rsi = item.rsis[-2]
        if pivot_rsi is None:
            return None, None
        pivot_index = item.bars_seen - 2
        pivot_volume = item.volumes[-2]
        new_low: _Pivot | None = None
        new_high: _Pivot | None = None
        if item.lows[-2] < item.lows[-3] and item.lows[-2] < item.lows[-1]:
            new_low = _Pivot(pivot_index, item.lows[-2], pivot_rsi, pivot_volume)
            item.pivot_lows.append(new_low)
        if item.highs[-2] > item.highs[-3] and item.highs[-2] > item.highs[-1]:
            new_high = _Pivot(pivot_index, item.highs[-2], pivot_rsi, pivot_volume)
            item.pivot_highs.append(new_high)
        cutoff = item.bars_seen - 2 - self.max_pivot_distance
        item.pivot_lows[:] = [pivot for pivot in item.pivot_lows if pivot.bar_index >= cutoff]
        item.pivot_highs[:] = [pivot for pivot in item.pivot_highs if pivot.bar_index >= cutoff]
        return new_low, new_high

    def _record_htf_close(self, item: _State, raw_time: Any, close: float) -> None:
        """Publish only completed closes from UTC-aligned HTF buckets.

        Missing base bars leave missing buckets instead of shifting all later
        groups. A stop or entry decision can therefore use only a completed
        higher-timeframe close, never a partial bucket.
        """
        bucket = self._htf_bucket(raw_time, item.bars_seen - 1)
        if item.htf_bucket is None:
            item.htf_bucket = bucket
        elif bucket > item.htf_bucket:
            if item.htf_bucket_close is not None:
                item.htf_closes.append(item.htf_bucket_close)
            item.htf_bucket = bucket
        elif bucket < item.htf_bucket:
            return  # ignore out-of-order bars; state remains causal
        item.htf_bucket_close = close

    def _htf_bucket(self, raw_time: Any, fallback_index: int) -> int:
        """Return a UTC bucket; bar ordinals are cadence-spaced logical time."""
        if isinstance(raw_time, (int, float)) and math.isfinite(float(raw_time)):
            value = float(raw_time)
            if abs(value) < 100_000_000:
                return math.floor(value / self.htf_multiple)
            if abs(value) > 100_000_000_000:
                value /= 1000.0
            return math.floor(value / (self.decision_cadence_seconds * self.htf_multiple))
        parsed: datetime | None = None
        if isinstance(raw_time, datetime):
            parsed = (
                raw_time.astimezone(UTC)
                if raw_time.tzinfo is not None
                else raw_time.replace(tzinfo=UTC)
            )
        elif raw_time is not None:
            try:
                parsed = datetime.fromisoformat(str(raw_time).replace("Z", "+00:00"))
                parsed = (
                    parsed.astimezone(UTC)
                    if parsed.tzinfo is not None
                    else parsed.replace(tzinfo=UTC)
                )
            except ValueError:
                pass
        if parsed is None:
            return fallback_index // self.htf_multiple
        return math.floor(parsed.timestamp() / (self.decision_cadence_seconds * self.htf_multiple))

    def _volume_ok(self, earlier: _Pivot, newest: _Pivot) -> bool:
        """Public rule: a heavier opposing print can invalidate the divergence."""
        if not self.require_volume_confirmation:
            return True
        if earlier.volume is None or newest.volume is None:
            return True  # volume feed unavailable -> rule cannot be evaluated
        return newest.volume <= earlier.volume

    def _htf_ok(self, item: _State, side: str) -> bool:
        """Public rule: act only when the higher timeframe agrees with the trade.

        One higher timeframe stands in for the 30m/1h/4h/daily stack; see the
        module docstring. Insufficient completed HTF history blocks the entry.
        """
        if not self.require_htf_confirmation:
            return True
        average = simple_moving_average(item.htf_closes, self.htf_ma_window)
        if average is None:
            return False
        last = item.htf_closes[-1]
        return last > average if side == "LONG" else last < average

    def _trend_up(self, item: _State) -> bool:
        history = list(item.closes)[:-1]
        fast = simple_moving_average(history, self.ma_fast)
        slow = simple_moving_average(history, self.ma_slow)
        if fast is None or slow is None:
            return False
        return item.closes[-1] > fast > slow

    def _trend_down(self, item: _State) -> bool:
        history = list(item.closes)[:-1]
        fast = simple_moving_average(history, self.ma_fast)
        slow = simple_moving_average(history, self.ma_slow)
        if fast is None or slow is None:
            return False
        return item.closes[-1] < fast < slow

    def _bullish_setup(self, item: _State, newest: _Pivot) -> tuple[str, _Pivot] | None:
        for earlier in reversed(item.pivot_lows[:-1]):
            distance = newest.bar_index - earlier.bar_index
            if distance < self.min_pivot_distance:
                continue
            if distance > self.max_pivot_distance:
                break
            if not self._volume_ok(earlier, newest):
                continue
            if (
                newest.price < earlier.price
                and newest.rsi > earlier.rsi
                and newest.rsi <= self.oversold
            ):
                return "regular", earlier
            if (
                self.use_hidden
                and newest.price > earlier.price
                and newest.rsi < earlier.rsi
                and self._trend_up(item)
            ):
                return "hidden", earlier
        return None

    def _bearish_setup(self, item: _State, newest: _Pivot) -> tuple[str, _Pivot] | None:
        for earlier in reversed(item.pivot_highs[:-1]):
            distance = newest.bar_index - earlier.bar_index
            if distance < self.min_pivot_distance:
                continue
            if distance > self.max_pivot_distance:
                break
            if not self._volume_ok(earlier, newest):
                continue
            if (
                newest.price > earlier.price
                and newest.rsi < earlier.rsi
                and newest.rsi >= self.overbought
            ):
                return "regular", earlier
            if (
                self.use_hidden
                and newest.price < earlier.price
                and newest.rsi > earlier.rsi
                and self._trend_down(item)
            ):
                return "hidden", earlier
        return None

    def _enter_long(
        self, symbol: str, item: _State, snapshot: _Snapshot, close: float, newest: _Pivot
    ) -> bool:
        if not self._htf_ok(item, "LONG"):
            return False
        setup = self._bullish_setup(item, newest)
        if setup is None:
            return False
        divergence_type, earlier = setup
        self._open(
            symbol,
            item,
            snapshot,
            close,
            side="LONG",
            divergence_type=divergence_type,
            earlier=earlier,
            newest=newest,
        )
        return True

    def _enter_short(
        self, symbol: str, item: _State, snapshot: _Snapshot, close: float, newest: _Pivot
    ) -> bool:
        if not self._htf_ok(item, "SHORT"):
            return False
        setup = self._bearish_setup(item, newest)
        if setup is None:
            return False
        divergence_type, earlier = setup
        self._open(
            symbol,
            item,
            snapshot,
            close,
            side="SHORT",
            divergence_type=divergence_type,
            earlier=earlier,
            newest=newest,
        )
        return True

    def _open(
        self,
        symbol: str,
        item: _State,
        snapshot: _Snapshot,
        close: float,
        *,
        side: str,
        divergence_type: str,
        earlier: _Pivot,
        newest: _Pivot,
    ) -> None:
        metadata = _target_metadata(
            strategy=self.__class__.__name__,
            target_allocation=self.target_allocation,
            max_order_value=self.max_order_value,
            divergence_type=divergence_type,
            side=side,
            rsi_new=newest.rsi,
            rsi_prev=earlier.rsi,
            pivot_price_new=newest.price,
            pivot_price_prev=earlier.price,
            pivot_distance=newest.bar_index - earlier.bar_index,
            stop_price=newest.price,
        )
        _emit(
            self.events,
            strategy_id=_STRATEGY_ID,
            symbol=symbol,
            event_time=snapshot.time,
            signal_type=side,
            strength=self.target_allocation or 1.0,
            price=close,
            metadata=metadata,
        )
        item.mode = side
        item.entry_price = close
        item.stop_price = newest.price
        pivot_volumes = [volume for volume in (earlier.volume, newest.volume) if volume is not None]
        item.pivot_volume_avg = (
            sum(pivot_volumes) / len(pivot_volumes) if len(pivot_volumes) == 2 else None
        )
        item.bars_held = 0
        item.exit_stage = 0

    def _exit_levels(self, side: str) -> tuple[float, float]:
        """Staged RSI levels for ``side``: (first stage, final stage)."""
        if side == "LONG":
            return self.exit_rsi_first, self.exit_rsi_second
        return 100.0 - self.exit_rsi_first, 100.0 - self.exit_rsi_second

    @staticmethod
    def _reached(rsi: float, level: float, side: str) -> bool:
        return rsi >= level if side == "LONG" else rsi <= level

    def _opposing_volume(self, item: _State, snapshot: _Snapshot, close: float) -> bool:
        """Public rule: a larger opposing-direction volume invalidates the setup.

        The multiple is this module's choice (off by default); the sources give
        no window or threshold. Unavailable volume leaves the rule unevaluated.
        """
        if self.opposing_volume_multiple <= 0.0 or item.pivot_volume_avg is None:
            return False
        volume = safe_float(snapshot.volume)
        open_price = safe_float(snapshot.open)
        if volume is None or open_price is None:
            return False
        against = close < open_price if item.mode == "LONG" else close > open_price
        return against and volume > self.opposing_volume_multiple * item.pivot_volume_avg

    def _emit_exit(
        self,
        symbol: str,
        item: _State,
        snapshot: _Snapshot,
        close: float,
        rsi: float | None,
        *,
        reason: str,
        exit_fraction: float | None = None,
    ) -> None:
        metadata: dict[str, Any] = {
            "strategy": self.__class__.__name__,
            "reason": reason,
            "side": item.mode,
            "rsi": rsi,
            "entry_price": item.entry_price,
            "stop_price": item.stop_price,
            "bars_held": item.bars_held,
        }
        if exit_fraction is not None:
            # A FULL exit deliberately carries NO ``exit_fraction`` key, so a
            # consumer that does not implement partial exits still flattens.
            metadata["exit_fraction"] = float(exit_fraction)
        _emit(
            self.events,
            strategy_id=_STRATEGY_ID,
            symbol=symbol,
            event_time=snapshot.time,
            signal_type="EXIT",
            price=close,
            metadata=metadata,
        )

    def _close(
        self,
        symbol: str,
        item: _State,
        snapshot: _Snapshot,
        close: float,
        rsi: float | None,
        *,
        reason: str,
    ) -> None:
        self._emit_exit(symbol, item, snapshot, close, rsi, reason=reason)
        item.mode = "OUT"
        item.entry_price = None
        item.stop_price = None
        item.pivot_volume_avg = None
        item.bars_held = 0
        item.exit_stage = 0

    def _staged_rsi_exit(
        self, symbol: str, item: _State, snapshot: _Snapshot, close: float, rsi: float
    ) -> bool:
        """Public rule: >50% off in the RSI 40~50 band, the rest above 60.

        Returns ``True`` once this bar's staged decision has been emitted.
        """
        first, second = self._exit_levels(item.mode)
        if item.exit_stage == 0:
            # The repository's EXIT contract treats ``exit_fraction`` as the
            # fraction of the *current* position. A bar that reaches the final
            # target therefore closes all of it once, never emits a partial
            # and waits to discover whether that partial filled.
            if self._reached(rsi, second, item.mode):
                self._close(symbol, item, snapshot, close, rsi, reason="stage2_rsi")
                return True
            if not self._reached(rsi, first, item.mode):
                return False
            if not 0.0 < self.first_exit_fraction < 1.0:
                # Zero is not a valid execution fraction and one is a full
                # exit. In either case do not advance stage without a
                # representable partial fill.
                self._close(symbol, item, snapshot, close, rsi, reason="stage1_rsi")
                return True
            self._emit_exit(
                symbol,
                item,
                snapshot,
                close,
                rsi,
                reason="stage1_rsi",
                exit_fraction=self.first_exit_fraction,
            )
            item.exit_stage = 1  # the remainder stays open
            return True
        if not self._reached(rsi, second, item.mode):
            return False
        self._close(symbol, item, snapshot, close, rsi, reason="stage2_rsi")
        return True

    def _manage_position(
        self, symbol: str, item: _State, snapshot: _Snapshot, close: float, rsi: float | None
    ) -> None:
        item.bars_held += 1
        stopped = item.stop_price is not None and (
            safe_float(snapshot.low) <= item.stop_price
            if item.mode == "LONG"
            else safe_float(snapshot.high) >= item.stop_price
        )
        if stopped:
            self._close(symbol, item, snapshot, close, rsi, reason="pivot_stop")
            return
        if rsi is not None and self._staged_rsi_exit(symbol, item, snapshot, close, rsi):
            return
        if self._opposing_volume(item, snapshot, close):
            self._close(symbol, item, snapshot, close, rsi, reason="opposing_volume")
            return
        if item.bars_held >= self.max_hold_bars:
            self._close(symbol, item, snapshot, close, rsi, reason="max_hold")

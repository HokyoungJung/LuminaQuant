"""Turtle-style unit pyramiding: ATR-unit sizing, +0.5N adds, hard 2N last-fill stop.

Research-only sleeve.

Lineage
-------
Independent adaptation of the publicly described rules of the classic Turtle
trading system (Richard Dennis / William Eckhardt; the "Original Turtle Trading
Rules" have been in open circulation for decades), in the shape it is popularised
in Korean retail-quant education by the handle Multan Chanbap -- whose slogan is
"pyramid up instead of averaging down" (bul-ta-gi, not mul-ta-gi), taught with
ATR "unit" money management on the 20/10 Donchian skeleton plus the 55/20
long-term variant.  This is not a reproduction of anyone's code, not an
endorsement, and makes no performance claim; no proprietary material was used.

Hypothesis
----------
The edge of a channel breakout is claimed to live in its risk shape rather than
in its timing: every unit carries the same currency risk (``unit_risk_pct`` of
equity per 1N adverse move), size is only added after the market has already
paid ``add_step_atr`` * N, and the hard stop is re-anchored to the LAST fill so
the whole stack always risks about ``stop_atr_multiple`` * N of the most recent
unit.  That last-fill anchor is what caps the giveback of a late add, and is the
part this sleeve exists to measure.

What the public source states vs. what is this author's choice
--------------------------------------------------------------
Publicly stated by the source rules: Donchian breakout entry with a shorter
opposite-channel exit (20/10 and 55/20), N = 20-period ATR, one "unit" = a
fixed risk fraction per 1N, add a unit every 1/2 N of favourable excursion up to
4 units, a hard 2N stop measured from the most recent fill, and the System-1
filter that skips a breakout when the previous trade was a winner.

This author's choices: the 55/20 long-term pair as the default (rather than
20/10); expressing a unit as a NOTIONAL fraction of equity
(``unit_risk_pct`` * price / N) because this engine sizes orders from
``target_allocation`` and not from contracts/points; the
``max_unit_allocation`` = 0.25 notional cap that keeps a low-N regime from
demanding absurd leverage; close-based (not intrabar) evaluation of breakouts
and stops; a single full EXIT instead of unit-by-unit scale-outs; the System-1
filter defaulting OFF (``require_prior_loser_skip``); ``max_hold_bars`` = 0
(disabled) and ``max_order_value`` = 500.

Public ``물탄찬밥`` preset
----------------------------
Public 물탄찬밥 rule (as documented in the newsystock academy previews and
mirrored in the sibling Codex lane): 20-day close new-high entry, 10-day close
new-low exit, -3.5% stop, 120-day MA gate; expressed here with
``channel_source="close"``, ``entry_lookback=20``, ``exit_lookback=10``,
``stop_loss_pct=0.035``, ``trend_ma_window=120``, ``max_units=1``,
``use_n_stop=False``, ``allow_short=False``.  Those four extra parameters all
default to the legacy/OFF setting, so nothing above changes unless the preset is
asked for.  This is again an independent adaptation of publicly described rules
-- not a reproduction of anyone's code, not an endorsement, and no performance
claim: no audited ledger of that source is public, and the unit sizing, the
notional-cap and the bar-close evaluation remain this author's engine choices.

Relationship to ``DonchianAtrTrendStrategy``
--------------------------------------------
That sleeve is a single-entry breakout with a high-watermark ATR trail and a
fixed ``target_allocation``.  This one is deliberately different: risk-scaled
UNIT sizing, up to ``max_units`` pyramided entries, and a stop that ratchets
with each new fill rather than with the favourable extreme.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from lumina_quant.core.plugin_registry import register
from lumina_quant.indicators import average_true_range, simple_moving_average, true_range
from lumina_quant.indicators.common import safe_float, time_key
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

_EPS = 1e-12
_STRATEGY_ID = "turtle_unit_pyramiding"
_MODES = ("LONG", "SHORT")
_CHANNEL_SOURCES = ("hl", "close")


@dataclass(slots=True)
class _TurtleState:
    """Per-symbol rolling history plus the live unit stack."""

    highs: deque[float]
    lows: deque[float]
    closes: deque[float]
    tr_values: deque[float]
    mode: str = "OUT"
    units: int = 0
    last_fill_price: float | None = None
    stop_price: float | None = None
    entry_price: float | None = None
    bars_held: int = 0
    last_trade_won: bool | None = None
    last_time_key: str = ""


@register("strategy", "TurtleUnitPyramidingStrategy", interface="event_driven")
class TurtleUnitPyramidingStrategy(Strategy):
    """Breakout entry sized in ATR units, pyramided every +0.5N, stopped at 2N."""

    decision_cadence_seconds = 86400
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "entry_lookback": HyperParam.integer("entry_lookback", default=55, low=2, high=10080),
            "exit_lookback": HyperParam.integer("exit_lookback", default=20, low=2, high=10080),
            "channel_source": HyperParam.categorical(
                "channel_source", default="hl", choices=("hl", "close")
            ),
            "atr_window": HyperParam.integer("atr_window", default=20, low=2, high=4096),
            "unit_risk_pct": HyperParam.floating(
                "unit_risk_pct", default=0.01, low=0.001, high=0.10
            ),
            "max_units": HyperParam.integer("max_units", default=4, low=1, high=20),
            "add_step_atr": HyperParam.floating("add_step_atr", default=0.5, low=0.05, high=5.0),
            "stop_atr_multiple": HyperParam.floating(
                "stop_atr_multiple", default=2.0, low=0.25, high=10.0
            ),
            "use_n_stop": HyperParam.boolean("use_n_stop", default=True, grid=[True, False]),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
            "trend_ma_window": HyperParam.integer("trend_ma_window", default=0, low=0, high=10080),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=0, low=0, high=200000),
            "require_prior_loser_skip": HyperParam.boolean(
                "require_prior_loser_skip", default=False, grid=[True, False]
            ),
            "stop_loss_pct": HyperParam.floating(
                "stop_loss_pct", default=0.0, low=0.0, high=0.99, tunable=False
            ),
            "max_unit_allocation": HyperParam.floating(
                "max_unit_allocation", default=0.25, low=0.0, high=2.0, tunable=False
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
        self.entry_lookback = max(2, int(resolved["entry_lookback"]))
        self.exit_lookback = max(2, int(resolved["exit_lookback"]))
        source = str(resolved["channel_source"]).lower()
        self.channel_source = source if source in _CHANNEL_SOURCES else "hl"
        self.atr_window = max(2, int(resolved["atr_window"]))
        self.unit_risk_pct = max(0.0, float(resolved["unit_risk_pct"]))
        self.max_units = max(1, int(resolved["max_units"]))
        self.add_step_atr = max(0.0, float(resolved["add_step_atr"]))
        self.stop_atr_multiple = max(0.0, float(resolved["stop_atr_multiple"]))
        self.use_n_stop = bool(resolved["use_n_stop"])
        self.allow_short = bool(resolved["allow_short"])
        self.trend_ma_window = max(0, int(resolved["trend_ma_window"]))
        self.stop_loss_pct = min(0.99, max(0.0, float(resolved["stop_loss_pct"])))
        self.max_hold_bars = max(0, int(resolved["max_hold_bars"]))
        self.require_prior_loser_skip = bool(resolved["require_prior_loser_skip"])
        self.max_unit_allocation = max(0.0, float(resolved["max_unit_allocation"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        size = (
            max(self.entry_lookback, self.exit_lookback, self.atr_window, self.trend_ma_window) + 3
        )
        self._state = {
            symbol: _TurtleState(
                highs=deque(maxlen=size),
                lows=deque(maxlen=size),
                closes=deque(maxlen=size),
                tr_values=deque(maxlen=size),
            )
            for symbol in self.symbol_list
        }

    # ------------------------------------------------------------------ state

    def get_state(self) -> dict[str, Any]:
        return {
            "symbol_state": {
                symbol: {
                    "highs": list(item.highs),
                    "lows": list(item.lows),
                    "closes": list(item.closes),
                    "tr_values": list(item.tr_values),
                    "mode": item.mode,
                    "units": int(item.units),
                    "last_fill_price": item.last_fill_price,
                    "stop_price": item.stop_price,
                    "entry_price": item.entry_price,
                    "bars_held": int(item.bars_held),
                    "last_trade_won": item.last_trade_won,
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
            for name in ("highs", "lows", "closes", "tr_values"):
                target = getattr(item, name)
                target.clear()
                for value in list(payload.get(name) or [])[-int(target.maxlen or 0) :]:
                    parsed = safe_float(value)
                    if parsed is not None:
                        target.append(parsed)
            mode = str(payload.get("mode", "OUT")).upper()
            item.mode = mode if mode in _MODES else "OUT"
            item.units = self._clamped_units(payload.get("units"))
            item.last_fill_price = safe_float(payload.get("last_fill_price"))
            item.stop_price = safe_float(payload.get("stop_price"))
            item.entry_price = safe_float(payload.get("entry_price"))
            try:
                item.bars_held = max(0, int(payload.get("bars_held", 0)))
            except TypeError, ValueError:
                item.bars_held = 0
            won = payload.get("last_trade_won")
            item.last_trade_won = None if won is None else bool(won)
            item.last_time_key = str(payload.get("last_time_key", ""))
            if item.mode == "OUT":
                item.units = 0

    def _clamped_units(self, value: Any) -> int:
        try:
            return max(0, min(self.max_units, int(value)))
        except TypeError, ValueError:
            return 0

    # ----------------------------------------------------------------- events

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

    # ------------------------------------------------------------------ rules

    def _channel(self, item: _TurtleState, window: int) -> tuple[float | None, float | None]:
        """Return the ``(upper, lower)`` breakout channel of the stored history.

        ``channel_source="hl"`` measures the classic Donchian extremes of the
        highs/lows; ``"close"`` measures the close-only new-high / new-low
        channel.  The current bar is appended only after this is read, so either
        way every level is measured strictly on prior bars.
        """
        if self.channel_source == "close":
            if len(item.closes) < window:
                return None, None
            tail = list(item.closes)[-window:]
            return max(tail), min(tail)
        if len(item.highs) < window or len(item.lows) < window:
            return None, None
        return max(list(item.highs)[-window:]), min(list(item.lows)[-window:])

    def _unit_allocation(self, close: float, n_atr: float) -> float:
        """Notional fraction of equity whose 1N move costs ``unit_risk_pct``."""
        if close <= 0.0 or n_atr <= _EPS or self.unit_risk_pct <= 0.0:
            return 0.0
        return min(self.max_unit_allocation, self.unit_risk_pct * close / n_atr)

    def _stop_for(self, mode: str, fill_price: float, n_atr: float) -> float | None:
        if not self.use_n_stop:
            return None
        offset = self.stop_atr_multiple * n_atr
        return fill_price - offset if mode == "LONG" else fill_price + offset

    def _pct_stop_price(self, item: _TurtleState) -> float | None:
        """Fixed-percentage stop measured off the FIRST fill.

        ``entry_price`` is set once by ``_maybe_enter`` and is never moved by an
        add, so it already is the first-fill anchor the public rule wants -- and
        it already round-trips through ``get_state``/``set_state``, so the
        percentage stop needs no extra state of its own.
        """
        if self.stop_loss_pct <= 0.0 or item.entry_price is None:
            return None
        if item.mode == "LONG":
            return item.entry_price * (1.0 - self.stop_loss_pct)
        if item.mode == "SHORT":
            return item.entry_price * (1.0 + self.stop_loss_pct)
        return None

    def _effective_stop(self, item: _TurtleState) -> float | None:
        """The stop actually in force: the tighter of the N-stop and the %-stop."""
        pct_stop = self._pct_stop_price(item)
        if item.stop_price is None:
            return pct_stop
        if pct_stop is None:
            return item.stop_price
        return (
            max(item.stop_price, pct_stop)
            if item.mode == "LONG"
            else min(item.stop_price, pct_stop)
        )

    @staticmethod
    def _stop_hit(mode: str, close: float, level: float) -> bool:
        return close <= level if mode == "LONG" else close >= level

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

        # Lagged reads BEFORE the current bar joins the history.
        n_atr = average_true_range(item.tr_values, self.atr_window)
        entry_upper, entry_lower = self._channel(item, self.entry_lookback)
        exit_upper, exit_lower = self._channel(item, self.exit_lookback)
        trend_ma = (
            simple_moving_average(item.closes, self.trend_ma_window)
            if self.trend_ma_window > 0
            else None
        )
        prev_close = item.closes[-1] if item.closes else None

        item.highs.append(high)
        item.lows.append(low)
        item.closes.append(close)
        item.tr_values.append(true_range(high, low, prev_close))

        if item.mode in _MODES:
            self._maybe_exit(symbol, item, snapshot, close, exit_upper, exit_lower)
        if item.mode in _MODES:
            self._maybe_add(symbol, item, snapshot, close, n_atr)
            return
        self._maybe_enter(symbol, item, snapshot, close, n_atr, entry_upper, entry_lower, trend_ma)

    def _maybe_exit(
        self,
        symbol: str,
        item: _TurtleState,
        snapshot: _Snapshot,
        close: float,
        exit_upper: float | None,
        exit_lower: float | None,
    ) -> None:
        """Close the whole stack on a stop, the exit channel or age."""
        item.bars_held += 1
        pct_stop = self._pct_stop_price(item)
        hits: list[tuple[float, str]] = []
        if item.stop_price is not None and self._stop_hit(item.mode, close, item.stop_price):
            hits.append((item.stop_price, "unit_stop"))
        if pct_stop is not None and self._stop_hit(item.mode, close, pct_stop):
            hits.append((pct_stop, "pct_stop"))
        reason = ""
        if hits:
            # Both stops are evaluated: the level sitting CLOSER to the market
            # would have been touched first, so it names the exit.  ``max``/
            # ``min`` return the first extreme, so a tie keeps the N-stop.
            pick = max if item.mode == "LONG" else min
            reason = pick(hits, key=lambda hit: hit[0])[1]
        elif item.mode == "LONG":
            if exit_lower is not None and close < exit_lower:
                reason = "exit_channel"
        elif exit_upper is not None and close > exit_upper:
            reason = "exit_channel"
        if not reason and self.max_hold_bars > 0 and item.bars_held >= self.max_hold_bars:
            reason = "max_hold"
        if not reason:
            return
        entry_price = item.entry_price
        if entry_price is not None:
            item.last_trade_won = (
                close > entry_price if item.mode == "LONG" else close < entry_price
            )
        metadata: dict[str, Any] = {
            "strategy": self.__class__.__name__,
            "reason": reason,
            "unit": int(item.units),
            "stop": self._effective_stop(item),
            "entry_price": entry_price,
        }
        if pct_stop is not None:
            metadata["pct_stop"] = pct_stop
        # ponytail: the portfolio has no partial-exit semantics, so the entire
        # unit stack leaves on one EXIT instead of peeling units off one by one.
        _emit(
            self.events,
            strategy_id=_STRATEGY_ID,
            symbol=symbol,
            event_time=snapshot.time,
            signal_type="EXIT",
            price=close,
            metadata=metadata,
        )
        item.mode = "OUT"
        item.units = 0
        item.last_fill_price = None
        item.stop_price = None
        item.entry_price = None
        item.bars_held = 0

    def _maybe_add(
        self,
        symbol: str,
        item: _TurtleState,
        snapshot: _Snapshot,
        close: float,
        n_atr: float | None,
    ) -> None:
        """Add one more unit once price has paid another ``add_step_atr`` * N."""
        if item.units >= self.max_units or item.last_fill_price is None:
            return
        if n_atr is None or n_atr <= _EPS:
            return
        step = self.add_step_atr * n_atr
        if item.mode == "LONG":
            if close < item.last_fill_price + step:
                return
        elif close > item.last_fill_price - step:
            return
        unit_alloc = self._unit_allocation(close, n_atr)
        if unit_alloc <= 0.0:
            return
        item.units += 1
        item.last_fill_price = close
        item.stop_price = self._stop_for(item.mode, close, n_atr)
        self._emit_fill(symbol, item, snapshot, close, n_atr, unit_alloc, "pyramid_add")

    def _maybe_enter(
        self,
        symbol: str,
        item: _TurtleState,
        snapshot: _Snapshot,
        close: float,
        n_atr: float | None,
        entry_upper: float | None,
        entry_lower: float | None,
        trend_ma: float | None,
    ) -> None:
        """Open unit 1 on a close beyond the lagged entry channel."""
        if n_atr is None or n_atr <= _EPS or entry_upper is None or entry_lower is None:
            return
        if close > entry_upper:
            direction = "LONG"
        elif self.allow_short and close < entry_lower:
            direction = "SHORT"
        else:
            return
        if self.trend_ma_window > 0:
            # Regime gate on the lagged MA.  Checked before the System-1 filter
            # so a breakout refused by the regime does not burn the skip token,
            # and it never forces an exit -- it only vetoes new entries.
            if trend_ma is None:
                return
            if direction == "LONG" and close <= trend_ma:
                return
            if direction == "SHORT" and close >= trend_ma:
                return
        if self.require_prior_loser_skip and item.last_trade_won:
            # System-1 filter: one breakout is forfeited after a winning trade.
            item.last_trade_won = None
            return
        unit_alloc = self._unit_allocation(close, n_atr)
        if unit_alloc <= 0.0:
            return
        item.mode = direction
        item.units = 1
        item.entry_price = close
        item.last_fill_price = close
        item.bars_held = 0
        item.last_trade_won = None
        item.stop_price = self._stop_for(direction, close, n_atr)
        self._emit_fill(symbol, item, snapshot, close, n_atr, unit_alloc, "unit_entry")

    def _emit_fill(
        self,
        symbol: str,
        item: _TurtleState,
        snapshot: _Snapshot,
        close: float,
        n_atr: float,
        unit_alloc: float,
        reason: str,
    ) -> None:
        effective_stop = self._effective_stop(item)
        extra: dict[str, Any] = {
            "unit": int(item.units),
            "n_atr": float(n_atr),
            "stop": effective_stop,
            "reason": reason,
            "target_mode": item.mode,
        }
        pct_stop = self._pct_stop_price(item)
        if pct_stop is not None:
            extra["pct_stop"] = pct_stop
        metadata = _target_metadata(
            strategy=self.__class__.__name__,
            target_allocation=unit_alloc,
            max_order_value=self.max_order_value,
            **extra,
        )
        _emit(
            self.events,
            strategy_id=_STRATEGY_ID,
            symbol=symbol,
            event_time=snapshot.time,
            signal_type=item.mode,
            strength=unit_alloc,
            price=close,
            stop_loss=effective_stop,
            metadata=metadata,
        )

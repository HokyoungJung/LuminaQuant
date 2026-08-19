"""Session-high breakout scalp - a bar-based proxy of publicly observed intraday
breakout-scalping behaviour.  research_only.

Lineage (public sources only)
-----------------------------
This is an INDEPENDENT ADAPTATION of the *publicly observed behaviour* of the
2021 Kiwoom contest-winning intraday bot '돌파고' (Dolpago), as documented in the
public write-up at ``/home/hoky/dacapogo`` (``README.md`` -> "돌파고 실제
매매내역 분석"; ``docs/research.md`` -> section 4, the author's public Q&A
reproduction).  It is **NOT a reproduction, endorsement or performance claim**,
and it is not a port of that bot's private formula - which the same sources
state is *undisclosed and non-identifiable*: the ledger fixes the behaviour, not
the state equation, and infinitely many rules reproduce the same ledger.

What the PUBLIC SOURCE actually states (facts, not choices made here):

* Features were computed from the *same day only*, from live tape (trades) plus
  order book, with the trade side weighted more heavily; minute bars were
  explicitly described as a human-viewing layer, not the program's input.
* Third-party observation (2021-07-22): entry at the moment the intraday prior
  high is broken, take profit about +1..2%, stop under 1%, flat within ~5 min.
* Measured ledger: median hold 1 minute, p90 3 minutes, 100% same-day flat,
  win rate 42.8% with a payoff ratio that carries it, ~86.3% of entries in the
  KST 09:00-12:00 morning window, ~120 positions/day, ~41 names/day.
* Universe: roughly the 100-200 names with the largest *same-day* volume, ranked
  live; the exact ranking formula was never published.
* Economics: gross +112M KRW collapsed to +27M KRW after fees/tax/slippage, and
  the author conceded a hard capacity ceiling (~15M KRW seed).  The same
  write-up shows the *generic* published breakout recipe losing money over the
  identical window (-15.2%), i.e. the edge lived in the undisclosed state
  formula and in book-level execution, not in "breakout" as such.

What is the AUTHOR's choice here (this file, not the public source)
------------------------------------------------------------------
Everything mechanical: this class is a **bar-based (1s/1m) proxy for Binance
USD-M perps**, so it has no tape and no order book at all.  The order-flow surge
is proxied by *bar volume* (mean of the last ``surge_bars`` bar volumes vs. the
session's mean bar volume so far); the "intraday prior high" is the running
session high excluding the current bar; the morning concentration is proxied by
a UTC-session minute window (``entry_start_minute``..``entry_end_minute``, the
crypto market has no 09:00-12:00 KST session, so the window is a *stand-in* for
"trade the early, liquid part of the session", not a translation of it); the
universe filter ranks by the *previous* session's turnover (the source ranked
same-day turnover live, which a bar feed cannot do causally).  Thresholds
(``surge_multiple``, ``breakout_buffer_pct``, ``min_session_bars``,
``max_entries_per_session``, ``max_hold_bars``) are this author's picks.  Only
the tight exit geometry is calibrated to public numbers: ``take_profit_pct``
0.015 sits inside the observed +1..2%, ``stop_loss_pct`` 0.007 under the
observed <1%, and ``max_hold_bars`` 300 at 1s cadence equals the observed ~5
minute ceiling.

READ THIS BEFORE USING ANY BACKTEST NUMBER
------------------------------------------
The public ledger analysis is unambiguous that this behaviour is *cost
dominated*: ~76% of the gross P&L in the source ledger was eaten by
fees/tax/slippage, and a naive minute-bar transcription of the same skeleton
returned -36.7% on crypto.  A run of this strategy is therefore **meaningless
unless fees, slippage and queue/fill realism are modelled**; with zero-cost
fills it will manufacture a fantasy equity curve out of ~1-minute round trips.
Capacity is likewise bounded - the source's own author hit a ceiling.  Tier:
research_only.  No claim is made that this reproduces the private formula, the
name selection, the fills, or the returns.
Execution-timing note: signals are emitted on bar close and the engine fills
market orders at the NEXT bar open.  The session time-cut / day-end flat is
emitted on the first bar of the new session, so its fill lands one bar after
the session boundary; treat that one-bar lag as part of the proxy (a
last-bar-of-session pre-emptive exit would need the bar interval, which the
event contract does not carry).

Double-managed bracket caveat: the entry signal carries ``stop_loss`` and
``take_profit`` AND the sleeve re-checks the very same levels close-to-close in
:meth:`_manage_position`.  That is the incumbent pattern in this book, but the
two layers do not talk to each other: if the engine's bracket fills intrabar
the sleeve is not told, so it keeps managing a position that is already flat
(``bars_held`` keeps counting, and it will emit its own EXIT once a bar CLOSES
through the level or the time stop trips).  The stale EXIT is harmless for a
portfolio that ignores exits on a flat book, but any bar-level P&L attribution
that trusts the sleeve's own bookkeeping will disagree with the fills; on a 1s
cadence with a 0.7% stop that gap is the normal case, not the exception.  Read
every backtest of this sleeve as bracket-fills-OR-close-rules, never both.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from lumina_quant.core.plugin_registry import register
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.strategies.external_alpha_sleeves import (
    _Snapshot,
    _emit,
    _event_datetime_utc,
    _event_symbols,
    _market_snapshot,
    _session_key,
    _target_metadata,
    _window_snapshot,
)
from lumina_quant.strategy import Strategy
from lumina_quant.tuning import HyperParam, resolve_params_from_schema

_STRATEGY_ID = "session_high_breakout_scalp"


def _minutes_since_session_start(raw_time: Any, *, start_minute_utc: int) -> float | None:
    """Minutes elapsed since the UTC session open that ``raw_time`` belongs to.

    Mirrors :func:`_session_key` exactly (same anchor, same wrap rule) so the
    time-of-day window and the session bucket can never disagree.
    """
    moment = _event_datetime_utc(raw_time)
    if moment is None:
        return None
    offset = max(0, min(1439, int(start_minute_utc)))
    session_start = moment.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(
        minutes=offset
    )
    if moment < session_start:
        session_start -= timedelta(days=1)
    return (moment - session_start).total_seconds() / 60.0


@dataclass(slots=True)
class _State:
    """Same-day-only aggregates for one symbol (nothing survives the session)."""

    volumes: deque[float]
    session_key: str = ""
    session_high: float | None = None
    session_low: float | None = None
    bars_seen: int = 0
    volume_sum: float = 0.0
    volume_bars: int = 0
    turnover: float = 0.0
    prev_turnover: float | None = None
    entries_this_session: int = 0
    mode: str = "OUT"
    entry_price: float | None = None
    bars_held: int = 0
    reentry_high_barrier: float | None = None
    reentry_low_barrier: float | None = None
    last_time_key: str = ""


@register("strategy", "SessionHighBreakoutScalpStrategy", interface="event_driven")
class SessionHighBreakoutScalpStrategy(Strategy):
    """Scalp the break of the intraday session high on a bar-volume surge."""

    decision_cadence_seconds = 1
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "session_start_minute_utc": HyperParam.integer(
                "session_start_minute_utc", default=0, low=0, high=1439, tunable=False
            ),
            "entry_start_minute": HyperParam.integer(
                "entry_start_minute", default=0, low=0, high=1439
            ),
            "entry_end_minute": HyperParam.integer(
                "entry_end_minute", default=240, low=0, high=1439
            ),
            "min_session_bars": HyperParam.integer(
                "min_session_bars", default=60, low=1, high=100_000
            ),
            "surge_bars": HyperParam.integer("surge_bars", default=30, low=1, high=10_000),
            "surge_multiple": HyperParam.floating(
                "surge_multiple", default=2.0, low=1.0, high=50.0
            ),
            "breakout_buffer_pct": HyperParam.floating(
                "breakout_buffer_pct", default=0.0005, low=0.0, high=0.05
            ),
            "take_profit_pct": HyperParam.floating(
                "take_profit_pct", default=0.015, low=0.0005, high=0.20
            ),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=300, low=1, high=100_000),
            "max_entries_per_session": HyperParam.integer(
                "max_entries_per_session", default=3, low=1, high=1000
            ),
            "max_symbols_by_turnover": HyperParam.integer(
                "max_symbols_by_turnover", default=0, low=0, high=1000
            ),
            "require_new_high_after_exit": HyperParam.boolean(
                "require_new_high_after_exit", default=True
            ),
            "allow_short": HyperParam.boolean("allow_short", default=False),
            "stop_loss_pct": HyperParam.floating(
                "stop_loss_pct", default=0.007, low=0.0005, high=0.20, tunable=False
            ),
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
        self.session_start_minute_utc = max(0, min(1439, int(resolved["session_start_minute_utc"])))
        self.entry_start_minute = max(0, int(resolved["entry_start_minute"]))
        self.entry_end_minute = max(self.entry_start_minute, int(resolved["entry_end_minute"]))
        self.min_session_bars = max(1, int(resolved["min_session_bars"]))
        self.surge_bars = max(1, int(resolved["surge_bars"]))
        self.surge_multiple = max(0.0, float(resolved["surge_multiple"]))
        self.breakout_buffer_pct = max(0.0, float(resolved["breakout_buffer_pct"]))
        self.take_profit_pct = max(0.0, float(resolved["take_profit_pct"]))
        self.stop_loss_pct = max(0.0, float(resolved["stop_loss_pct"]))
        self.max_hold_bars = max(1, int(resolved["max_hold_bars"]))
        self.max_entries_per_session = max(1, int(resolved["max_entries_per_session"]))
        self.max_symbols_by_turnover = max(0, int(resolved["max_symbols_by_turnover"]))
        self.require_new_high_after_exit = bool(resolved["require_new_high_after_exit"])
        self.allow_short = bool(resolved["allow_short"])
        self.target_allocation = max(0.0, float(resolved["target_allocation"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self._state = {
            symbol: _State(volumes=deque(maxlen=self.surge_bars)) for symbol in self.symbol_list
        }
        self._allowed_symbols: set[str] | None = None

    # ------------------------------------------------------------------ state

    def get_state(self) -> dict[str, Any]:
        return {
            "allowed_symbols": (
                None if self._allowed_symbols is None else sorted(self._allowed_symbols)
            ),
            "symbol_state": {
                symbol: {
                    "volumes": list(item.volumes),
                    "session_key": item.session_key,
                    "session_high": item.session_high,
                    "session_low": item.session_low,
                    "bars_seen": item.bars_seen,
                    "volume_sum": item.volume_sum,
                    "volume_bars": item.volume_bars,
                    "turnover": item.turnover,
                    "prev_turnover": item.prev_turnover,
                    "entries_this_session": item.entries_this_session,
                    "mode": item.mode,
                    "entry_price": item.entry_price,
                    "bars_held": item.bars_held,
                    "reentry_high_barrier": item.reentry_high_barrier,
                    "reentry_low_barrier": item.reentry_low_barrier,
                    "last_time_key": item.last_time_key,
                }
                for symbol, item in self._state.items()
            },
        }

    def set_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        allowed = state.get("allowed_symbols")
        self._allowed_symbols = (
            None if allowed is None else {str(symbol) for symbol in list(allowed)}
        )
        raw = state.get("symbol_state")
        if not isinstance(raw, dict):
            return
        for symbol, payload in raw.items():
            if symbol not in self._state or not isinstance(payload, dict):
                continue
            item = self._state[symbol]
            item.volumes.clear()
            for value in list(payload.get("volumes") or [])[-self.surge_bars :]:
                parsed = safe_float(value)
                if parsed is not None:
                    item.volumes.append(parsed)
            item.session_key = str(payload.get("session_key", ""))
            item.session_high = safe_float(payload.get("session_high"))
            item.session_low = safe_float(payload.get("session_low"))
            item.bars_seen = _as_count(payload.get("bars_seen"))
            item.volume_sum = float(safe_float(payload.get("volume_sum")) or 0.0)
            item.volume_bars = _as_count(payload.get("volume_bars"))
            item.turnover = float(safe_float(payload.get("turnover")) or 0.0)
            item.prev_turnover = safe_float(payload.get("prev_turnover"))
            item.entries_this_session = _as_count(payload.get("entries_this_session"))
            mode = str(payload.get("mode", "OUT")).upper()
            item.mode = mode if mode in ("LONG", "SHORT") else "OUT"
            item.entry_price = safe_float(payload.get("entry_price"))
            item.bars_held = _as_count(payload.get("bars_held"))
            item.reentry_high_barrier = safe_float(payload.get("reentry_high_barrier"))
            item.reentry_low_barrier = safe_float(payload.get("reentry_low_barrier"))
            item.last_time_key = str(payload.get("last_time_key", ""))

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

    # ------------------------------------------------------------------- core

    def _process(self, symbol: str, snapshot: _Snapshot) -> None:
        item = self._state[symbol]
        key = time_key(snapshot.time)
        if key and key == item.last_time_key:
            return
        close = safe_float(snapshot.close)
        if close is None or close <= 0.0:
            return
        high = safe_float(snapshot.high)
        low = safe_float(snapshot.low)
        if high is None:
            high = close
        if low is None:
            low = close
        session = _session_key(snapshot.time, start_minute_utc=self.session_start_minute_utc)
        if not session:
            return
        item.last_time_key = key

        if session != item.session_key:
            self._roll_session(symbol, item, snapshot, session, close)

        # Snapshot everything the entry rule is allowed to see BEFORE this bar is
        # folded in: the break must clear the session high made by *completed*
        # bars, and the surge baseline must not contain the surging bar itself.
        prior_high = item.session_high
        prior_low = item.session_low
        prior_bars = item.bars_seen
        prior_volume_mean = (
            item.volume_sum / float(item.volume_bars) if item.volume_bars > 0 else None
        )

        volume = safe_float(snapshot.volume)
        if volume is not None and volume >= 0.0:
            item.volumes.append(volume)
            item.volume_sum += volume
            item.volume_bars += 1
            item.turnover += close * volume
        else:
            volume = None
        item.bars_seen = prior_bars + 1
        item.session_high = high if item.session_high is None else max(item.session_high, high)
        item.session_low = low if item.session_low is None else min(item.session_low, low)

        if self._manage_position(symbol, item, snapshot, close):
            # ponytail: no same-bar re-entry after an exit - one decision per bar
            # keeps the round trip auditable and cannot double-count the fill.
            return
        if item.mode != "OUT":
            return
        self._maybe_enter(
            symbol,
            item,
            snapshot,
            close=close,
            volume=volume,
            prior_high=prior_high,
            prior_low=prior_low,
            prior_bars=prior_bars,
            prior_volume_mean=prior_volume_mean,
        )

    def _roll_session(
        self, symbol: str, item: _State, snapshot: _Snapshot, session: str, close: float
    ) -> None:
        """Flatten, archive turnover, reset the day, re-rank the universe."""
        if item.mode != "OUT":
            # ponytail: the source bot was 100% same-day flat; with bar data the
            # earliest observable flatten price is the first bar of the NEW
            # session, so a genuine close-of-session fill is not modelled.
            self._exit(symbol, item, snapshot, close, "session_flat")
        item.prev_turnover = item.turnover if item.bars_seen > 0 else None
        item.volumes.clear()
        item.session_key = session
        item.session_high = None
        item.session_low = None
        item.bars_seen = 0
        item.volume_sum = 0.0
        item.volume_bars = 0
        item.turnover = 0.0
        item.entries_this_session = 0
        item.reentry_high_barrier = None
        item.reentry_low_barrier = None
        # ponytail: the rank is refreshed on EVERY symbol's rollover rather than
        # once per wall-clock boundary.  Symbols that have not rolled yet still
        # contribute their last completed session's turnover (strictly past data,
        # so no look-ahead); the set self-corrects as the rest of the book rolls.
        self._refresh_universe()

    def _refresh_universe(self) -> None:
        if self.max_symbols_by_turnover <= 0:
            self._allowed_symbols = None
            return
        scored = [
            (symbol, float(item.prev_turnover))
            for symbol, item in self._state.items()
            if item.prev_turnover is not None and item.prev_turnover > 0.0
        ]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        self._allowed_symbols = {symbol for symbol, _ in scored[: self.max_symbols_by_turnover]}

    def _manage_position(
        self, symbol: str, item: _State, snapshot: _Snapshot, close: float
    ) -> bool:
        if item.mode == "OUT" or item.entry_price is None:
            return False
        item.bars_held += 1
        entry = float(item.entry_price)
        if item.mode == "LONG":
            if close >= entry * (1.0 + self.take_profit_pct):
                reason = "take_profit"
            elif close <= entry * (1.0 - self.stop_loss_pct):
                reason = "stop_loss"
            elif item.bars_held >= self.max_hold_bars:
                reason = "time_stop"
            else:
                return False
        else:
            if close <= entry * (1.0 - self.take_profit_pct):
                reason = "take_profit"
            elif close >= entry * (1.0 + self.stop_loss_pct):
                reason = "stop_loss"
            elif item.bars_held >= self.max_hold_bars:
                reason = "time_stop"
            else:
                return False
        self._exit(symbol, item, snapshot, close, reason)
        return True

    def _exit(
        self, symbol: str, item: _State, snapshot: _Snapshot, close: float, reason: str
    ) -> None:
        # ponytail: EXIT closes the WHOLE position - the book has no partial
        # exits, so the observed "scale out into strength" is collapsed into a
        # single all-or-nothing fill at the first target that trips.
        _emit(
            self.events,
            strategy_id=_STRATEGY_ID,
            symbol=symbol,
            event_time=snapshot.time,
            signal_type="EXIT",
            price=close,
            metadata={
                "strategy": self.__class__.__name__,
                "reason": reason,
                "side": item.mode,
                "entry_price": item.entry_price,
                "bars_held": item.bars_held,
            },
        )
        if self.require_new_high_after_exit:
            # The barrier is the session extreme observed up to this exit, NOT
            # the entry price: the entry bar's own high (and everything the
            # trade ran through) already sits at or above the entry price, so a
            # barrier pinned there can never block a re-entry after a winner -
            # ``prior_high <= barrier`` would already be false on the next bar.
            # Anchoring on the session extreme means re-arming needs the
            # session to print a genuinely new high (low) beyond the level this
            # round trip was just closed out of.
            if item.mode == "LONG" and item.session_high is not None:
                item.reentry_high_barrier = float(item.session_high)
            elif item.mode == "SHORT" and item.session_low is not None:
                item.reentry_low_barrier = float(item.session_low)
        item.mode = "OUT"
        item.entry_price = None
        item.bars_held = 0

    def _maybe_enter(
        self,
        symbol: str,
        item: _State,
        snapshot: _Snapshot,
        *,
        close: float,
        volume: float | None,
        prior_high: float | None,
        prior_low: float | None,
        prior_bars: int,
        prior_volume_mean: float | None,
    ) -> None:
        if self.target_allocation <= 0.0:
            # ``_target_metadata`` drops the key when the allocation is not
            # positive, and the portfolio then falls back to its own config
            # default - i.e. the sleeve would silently place an order it never
            # sized.  Refuse the entry instead of emitting an unsized signal.
            return
        if item.entries_this_session >= self.max_entries_per_session:
            return
        if prior_bars < self.min_session_bars:
            return
        if self.max_symbols_by_turnover > 0 and (
            self._allowed_symbols is None or symbol not in self._allowed_symbols
        ):
            return
        minutes = _minutes_since_session_start(
            snapshot.time, start_minute_utc=self.session_start_minute_utc
        )
        if minutes is None or minutes < self.entry_start_minute or minutes > self.entry_end_minute:
            return
        if volume is None or prior_volume_mean is None or prior_volume_mean <= 0.0:
            return
        # ponytail: the divisor is the FIXED window length, not len(volumes) - a
        # partly filled window then understates the surge, which can only ever
        # suppress an entry, never manufacture one.
        surge_mean = sum(item.volumes) / float(self.surge_bars)
        if surge_mean < self.surge_multiple * prior_volume_mean:
            return
        surge_ratio = surge_mean / prior_volume_mean

        side = ""
        level: float | None = None
        if prior_high is not None:
            long_level = prior_high * (1.0 + self.breakout_buffer_pct)
            blocked = (
                self.require_new_high_after_exit
                and item.reentry_high_barrier is not None
                and prior_high <= item.reentry_high_barrier
            )
            if close > long_level and not blocked:
                side, level = "LONG", long_level
        if not side and self.allow_short and prior_low is not None:
            short_level = prior_low * (1.0 - self.breakout_buffer_pct)
            blocked = (
                self.require_new_high_after_exit
                and item.reentry_low_barrier is not None
                and prior_low >= item.reentry_low_barrier
            )
            if close < short_level and not blocked:
                side, level = "SHORT", short_level
        if not side or level is None:
            return

        if side == "LONG":
            stop_loss = close * (1.0 - self.stop_loss_pct)
            take_profit = close * (1.0 + self.take_profit_pct)
        else:
            stop_loss = close * (1.0 + self.stop_loss_pct)
            take_profit = close * (1.0 - self.take_profit_pct)
        item.entries_this_session += 1
        metadata = _target_metadata(
            strategy=self.__class__.__name__,
            target_allocation=self.target_allocation,
            max_order_value=self.max_order_value,
            side=side,
            session_high=prior_high,
            session_low=prior_low,
            breakout_level=float(level),
            surge_ratio=float(surge_ratio),
            minutes_since_open=float(minutes),
            entries_this_session=int(item.entries_this_session),
        )
        _emit(
            self.events,
            strategy_id=_STRATEGY_ID,
            symbol=symbol,
            event_time=snapshot.time,
            signal_type=side,
            strength=self.target_allocation or 1.0,
            price=close,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata=metadata,
        )
        item.mode = side
        item.entry_price = close
        item.bars_held = 0


def _as_count(value: Any) -> int:
    try:
        return max(0, int(value))
    except TypeError, ValueError:
        return 0

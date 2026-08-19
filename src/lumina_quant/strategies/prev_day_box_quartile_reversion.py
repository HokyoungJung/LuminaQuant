"""Research-only previous-day box quartile reversion (AOA-inspired proxy).

Lineage and attribution
-----------------------
This module is an **independent adaptation** of the *publicly described* trading
approach of the Korean crypto trader known as ``워뇨띠``/``AOA``.  It is **not a
reproduction**, not an endorsement, and carries **no performance claim**: no
audited trade-by-trade ledger of that trader is public, so nothing here can be
compared against real fills.

Public sources (what the trader actually said):

* BitMEX official interview, 2025-06-11 (``https://www.bitmex.com/blog/whale-trader-talks-aoa``):
  preference for BTC/ETH and other large-cap names, chart-based low leverage,
  portfolio-wide gross exposure held around 1.5-2x, maximum account loss kept
  under ~30% of capital, entry technique varies by regime while risk management
  stays fixed.
* Preserved third-party Q&A repost (DCinside ``chartanalysis`` 959039): direction
  is read from candles and trend, volume is only a *confidence* aid, "boxes" have
  a cheap zone and an expensive zone -- buy cheap, sell expensive.

Those statements are qualitative.  The machine rules below were codified as a
deterministic comparison proxy in
``/home/hoky/dacapogo/docs/korean-trader-strategies.md`` (section "워뇨띠/AOA 프록시"):
15-minute bars, the box is the previous UTC calendar day's high/low with 25/50/75%
levels, the first rebound out of the lower/upper quartile is taken, take profit at
the box mid, stop at the opposite box end, and anything still open is flattened at
the UTC day end.

Author's choices (NOT stated by the public source, chosen by the proxy author
after looking at an evaluation window -- so they are ``posthoc_exploratory`` and
any positive result they produce validates nothing about AOA):

* ``wick >= body`` as the quantitative reading of "the candle rejected the level".
* one signal per symbol per session.
* the confirmation volume threshold = previous session's median 15m bar volume
  (the trader said volume is a confidence aid, never a formula).
* box length = one UTC calendar day, quartiles at 25/50/75%.
* per-symbol ``target_allocation`` and the hard ``max_order_value`` cap.

Deliberately NOT implemented: cross-exchange volume corroboration, multi-timeframe
conflict resolution, and the account-level 30% loss cap (the last one is a
portfolio/risk-engine concern, not a per-strategy signal rule).

Hypothesis
----------
Inside a range regime, the previous day's high/low box frames the day's "cheap"
and "expensive" zones; the first 15m bar that pierces a quartile and closes back
inside it, with a rejection wick at least as long as its body and above-normal
volume, marks a short-lived liquidity flush that reverts toward the box mid.

Classification: ``research_only``.
Execution-timing note: signals are emitted on bar close and the engine fills
market orders at the NEXT bar open.  The session time-cut / day-end flat is
emitted on the first bar of the new session, so its fill lands one bar after
the session boundary; treat that one-bar lag as part of the proxy (a
last-bar-of-session pre-emptive exit would need the bar interval, which the
event contract does not carry).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Any

from lumina_quant.core.plugin_registry import register
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.strategies.external_alpha_sleeves import (
    _Snapshot,
    _emit,
    _event_symbols,
    _market_snapshot,
    _session_key,
    _target_metadata,
    _window_snapshot,
)
from lumina_quant.strategy import Strategy
from lumina_quant.tuning import HyperParam, resolve_params_from_schema

_STRATEGY_ID = "prev_day_box_quartile_reversion"


@dataclass(slots=True)
class _State:
    """Per-symbol running session stats, the frozen box, and position bookkeeping."""

    session: str = ""
    session_high: float | None = None
    session_low: float | None = None
    session_volumes: list[float] = field(default_factory=list)
    box_high: float | None = None
    box_low: float | None = None
    prev_median_volume: float | None = None
    mode: str = "OUT"
    entry_price: float | None = None
    entry_stop: float | None = None
    entry_target: float | None = None
    bars_held: int = 0
    signaled_session: str = ""
    last_time_key: str = ""


@register("strategy", "PrevDayBoxQuartileReversionStrategy", interface="event_driven")
class PrevDayBoxQuartileReversionStrategy(Strategy):
    """Fade the first wick-rejection out of the previous day's box quartiles."""

    decision_cadence_seconds = 900
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "session_start_minute_utc": HyperParam.integer(
                "session_start_minute_utc", default=0, low=0, high=1439, tunable=False
            ),
            "min_session_bars": HyperParam.integer("min_session_bars", default=8, low=1, high=1440),
            "require_volume": HyperParam.boolean("require_volume", default=True),
            "allow_short": HyperParam.boolean("allow_short", default=True),
            "one_signal_per_session": HyperParam.boolean("one_signal_per_session", default=True),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=0, low=0, high=1440),
            "target_allocation": HyperParam.floating(
                "target_allocation", default=0.25, low=0.0, high=1.0, tunable=False
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
        self.min_session_bars = max(1, int(resolved["min_session_bars"]))
        self.require_volume = bool(resolved["require_volume"])
        self.allow_short = bool(resolved["allow_short"])
        self.one_signal_per_session = bool(resolved["one_signal_per_session"])
        self.max_hold_bars = max(0, int(resolved["max_hold_bars"]))
        self.target_allocation = max(0.0, float(resolved["target_allocation"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self._state = {symbol: _State() for symbol in self.symbol_list}

    def get_state(self) -> dict[str, Any]:
        return {
            "symbol_state": {
                symbol: {
                    "session": item.session,
                    "session_high": item.session_high,
                    "session_low": item.session_low,
                    "session_volumes": list(item.session_volumes),
                    "box_high": item.box_high,
                    "box_low": item.box_low,
                    "prev_median_volume": item.prev_median_volume,
                    "mode": item.mode,
                    "entry_price": item.entry_price,
                    "entry_stop": item.entry_stop,
                    "entry_target": item.entry_target,
                    "bars_held": item.bars_held,
                    "signaled_session": item.signaled_session,
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
            item.session = str(payload.get("session", ""))
            item.session_high = safe_float(payload.get("session_high"))
            item.session_low = safe_float(payload.get("session_low"))
            volumes = [
                parsed
                for value in list(payload.get("session_volumes") or [])
                if (parsed := safe_float(value)) is not None
            ]
            item.session_volumes = volumes
            item.box_high = safe_float(payload.get("box_high"))
            item.box_low = safe_float(payload.get("box_low"))
            item.prev_median_volume = safe_float(payload.get("prev_median_volume"))
            mode = str(payload.get("mode", "OUT")).upper()
            item.mode = mode if mode in ("LONG", "SHORT") else "OUT"
            item.entry_price = safe_float(payload.get("entry_price"))
            item.entry_stop = safe_float(payload.get("entry_stop"))
            item.entry_target = safe_float(payload.get("entry_target"))
            try:
                item.bars_held = max(0, int(payload.get("bars_held", 0)))
            except TypeError, ValueError:
                item.bars_held = 0
            item.signaled_session = str(payload.get("signaled_session", ""))
            item.last_time_key = str(payload.get("last_time_key", ""))

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

    def _box_levels(self, item: _State) -> tuple[float, float, float, float, float] | None:
        """Return ``(low, q25, mid, q75, high)`` for the frozen box, or ``None``."""
        low, high = item.box_low, item.box_high
        if low is None or high is None or high <= low:
            return None
        span = high - low
        return (low, low + 0.25 * span, low + 0.5 * span, low + 0.75 * span, high)

    def _process(self, symbol: str, snapshot: _Snapshot) -> None:
        item = self._state[symbol]
        key = time_key(snapshot.time)
        if key and key == item.last_time_key:
            return
        close = safe_float(snapshot.close)
        high = safe_float(snapshot.high)
        low = safe_float(snapshot.low)
        if close is None or high is None or low is None or high < low:
            return
        item.last_time_key = key
        open_ = safe_float(snapshot.open)
        if open_ is None:
            open_ = close
        volume = safe_float(snapshot.volume)
        session = _session_key(snapshot.time, start_minute_utc=self.session_start_minute_utc)
        if not session:
            return

        if session != item.session:
            self._rollover(symbol, item, snapshot, close, session)

        item.session_high = high if item.session_high is None else max(item.session_high, high)
        item.session_low = low if item.session_low is None else min(item.session_low, low)
        if volume is not None and volume >= 0.0:
            item.session_volumes.append(volume)

        if item.mode != "OUT":
            self._manage(symbol, item, snapshot, close)
            return
        self._maybe_enter(symbol, item, snapshot, open_, high, low, close, volume, session)

    def _rollover(
        self, symbol: str, item: _State, snapshot: _Snapshot, close: float, session: str
    ) -> None:
        """Flatten anything still open, freeze the finished session as the new box."""
        if item.mode != "OUT":
            _emit(
                self.events,
                strategy_id=_STRATEGY_ID,
                symbol=symbol,
                event_time=snapshot.time,
                signal_type="EXIT",
                price=close,
                metadata={"strategy": self.__class__.__name__, "reason": "session_flat"},
            )
            self._flatten(item)
        prev_high, prev_low = item.session_high, item.session_low
        if prev_high is not None and prev_low is not None and prev_high > prev_low:
            item.box_high, item.box_low = prev_high, prev_low
        else:
            item.box_high = item.box_low = None
        if len(item.session_volumes) >= self.min_session_bars:
            item.prev_median_volume = float(median(item.session_volumes))
        else:
            item.prev_median_volume = None
        item.session = session
        item.session_high = item.session_low = None
        item.session_volumes = []

    def _flatten(self, item: _State) -> None:
        item.mode = "OUT"
        item.entry_price = item.entry_stop = item.entry_target = None
        item.bars_held = 0

    def _manage(self, symbol: str, item: _State, snapshot: _Snapshot, close: float) -> None:
        """Emit close-based take-profit / stop-loss / max-hold exits."""
        item.bars_held += 1
        target, stop = item.entry_target, item.entry_stop
        reason = ""
        if item.mode == "LONG":
            if target is not None and close >= target:
                reason = "take_profit"
            elif stop is not None and close <= stop:
                reason = "stop_loss"
        elif target is not None and close <= target:
            reason = "take_profit"
        elif stop is not None and close >= stop:
            reason = "stop_loss"
        if not reason and self.max_hold_bars > 0 and item.bars_held >= self.max_hold_bars:
            reason = "max_hold"
        if not reason:
            return
        # The documented proxy takes profit in full at the box mid, so a plain
        # whole-position EXIT is the faithful rule here (no scale-out needed).
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
                "bars_held": item.bars_held,
            },
        )
        self._flatten(item)

    def _maybe_enter(
        self,
        symbol: str,
        item: _State,
        snapshot: _Snapshot,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: float | None,
        session: str,
    ) -> None:
        if self.target_allocation <= 0.0:
            # ``_target_metadata`` drops the key when the allocation is not
            # positive, so the portfolio would size this entry off its own
            # config default.  Refuse rather than emit an unsized signal.
            return
        if self.one_signal_per_session and item.signaled_session == session:
            return
        levels = self._box_levels(item)
        if levels is None:
            return
        box_low, q25, mid, q75, box_high = levels
        if not self._volume_confirms(item, volume):
            return

        side = ""
        if low <= q25 and close > q25 and close > open_ and (open_ - low) >= (close - open_):
            side = "LONG"
        elif (
            self.allow_short
            and high >= q75
            and close < q75
            and close < open_
            and (high - open_) >= (open_ - close)
        ):
            side = "SHORT"
        if not side:
            return

        stop = box_low if side == "LONG" else box_high
        metadata = _target_metadata(
            strategy=self.__class__.__name__,
            target_allocation=self.target_allocation,
            max_order_value=self.max_order_value,
            reason="lower_quartile_rebound" if side == "LONG" else "upper_quartile_rejection",
            side=side,
            session=session,
            box_high=box_high,
            box_low=box_low,
            box_q25=q25,
            box_mid=mid,
            box_q75=q75,
            stop_price=stop,
            target_price=mid,
            prev_median_volume=item.prev_median_volume,
            bar_volume=volume,
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
        item.entry_stop = stop
        item.entry_target = mid
        item.bars_held = 0
        item.signaled_session = session

    def _volume_confirms(self, item: _State, volume: float | None) -> bool:
        if not self.require_volume:
            return True
        if volume is None:
            # ponytail: volume is optional in the bar contract; without it the
            # confidence check cannot be evaluated, so the setup is skipped.
            return False
        reference = item.prev_median_volume
        return reference is None or volume > reference

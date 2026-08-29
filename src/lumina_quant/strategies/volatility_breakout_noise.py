"""Session volatility breakout with a noise filter, MA-score timing and vol control.

Research-only.  This is an INDEPENDENT ADAPTATION of the publicly described
"volatility breakout + noise filter + moving-average-score market timing +
volatility control" family popularised by the Korean blogger *systrader79*
(itself in the Larry Williams open-anchored range-breakout lineage, and widely
reproduced in Korean crypto-bot writeups).  It is NOT a reproduction of any
author's live system, NOT an endorsement, and carries NO performance claim; the
original rules were never published with executable parameters, so everything
below is a hypothesis encoded in code.

What the public sources state (structure only):

* Buy when price trades through ``session_open + K * previous_session_range``
  during the session; flatten at the session boundary ("time cut").
* ``K`` may be fixed (Williams used ~0.5) or set from the trailing average bar
  *noise ratio* ``1 - |close-open| / (high-low)``: noisier markets need a wider
  trigger, quieter/trendier markets a tighter one.
* Scale exposure by a "market-timing score" = the fraction of a set of trailing
  moving averages sitting below the last close.
* Scale exposure again by a crude volatility control
  ``target_vol / (prev_range / prev_close)``.

What is the AUTHOR's choice here (not stated by the public source):

* Session = UTC calendar day offset by ``session_start_minute_utc`` (0 => the
  09:00 KST crypto-day convention lands on 00:00 UTC).
* ``k_min``/``k_max`` clip band ``[0.3, 0.7]`` for the noise-derived ``K``.
* ``min_ma_score`` entry gate (0.25), ``target_session_vol`` (0.02 per session),
  ``max_vol_weight`` cap, ``target_allocation`` / ``max_position_allocation``
  sizing, the optional short side (default OFF), the optional intra-session
  ``stop_loss_pct`` (default OFF), and the cross-sectional
  ``max_symbols_by_noise`` universe filter (default OFF).
* One entry per symbol per session, exit-in-full at the session boundary.

Hypothesis: in instruments whose sessions travel directionally (low noise
ratio), an open-anchored range breakout captures the continuation of that
session's drift, while the noise filter, the MA score and the range-based vol
target keep exposure out of chop and off high-volatility sessions.
Execution-timing note: signals are emitted on bar close and the engine fills
market orders at the NEXT bar open.  The session time-cut / day-end flat is
emitted on the first bar of the new session, so its fill lands one bar after
the session boundary; treat that one-bar lag as part of the proxy (a
last-bar-of-session pre-emptive exit would need the bar interval, which the
event contract does not carry).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from lumina_quant.core.plugin_registry import register
from lumina_quant.indicators import (
    average_noise_ratio,
    moving_average_score,
    range_volatility_target_weight,
    volatility_breakout_levels,
)
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

_STRATEGY_ID = "volatility_breakout_noise"
_DEFAULT_MA_WINDOWS: tuple[int, ...] = (3, 5, 10, 20)


def _parse_windows(raw: Any) -> tuple[int, ...]:
    """Parse a ``"3,5,10,20"`` style string into a sorted unique window tuple."""
    parsed: list[int] = []
    for chunk in str(raw).replace(";", ",").split(","):
        text = chunk.strip()
        if not text:
            continue
        try:
            value = int(float(text))
        except TypeError, ValueError:
            continue
        if value >= 1:
            parsed.append(value)
    return tuple(sorted(set(parsed))) or _DEFAULT_MA_WINDOWS


@dataclass(slots=True)
class _State:
    """Completed-session OHLC history plus the in-progress session for one symbol."""

    opens: deque[float]
    highs: deque[float]
    lows: deque[float]
    closes: deque[float]
    session_key: str = ""
    session_open: float | None = None
    session_high: float | None = None
    session_low: float | None = None
    session_close: float | None = None
    mode: str = "OUT"
    entry_price: float | None = None
    entered_session_key: str = ""
    last_time_key: str = ""


@register("strategy", "NoiseFilteredVolatilityBreakoutStrategy", interface="event_driven")
class NoiseFilteredVolatilityBreakoutStrategy(Strategy):
    """Open-anchored session breakout, sized by noise, MA score and range vol."""

    decision_cadence_seconds = 60
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "k_mode": HyperParam.categorical("k_mode", default="noise", choices=("fixed", "noise")),
            "k": HyperParam.floating("k", default=0.5, low=0.05, high=2.0),
            "noise_period": HyperParam.integer("noise_period", default=20, low=2, high=500),
            "k_min": HyperParam.floating("k_min", default=0.3, low=0.05, high=2.0),
            "k_max": HyperParam.floating("k_max", default=0.7, low=0.05, high=2.0),
            "use_ma_score": HyperParam.boolean("use_ma_score", default=True),
            "ma_score_windows": HyperParam.string(
                "ma_score_windows", default="3,5,10,20", tunable=False
            ),
            "min_ma_score": HyperParam.floating("min_ma_score", default=0.25, low=0.0, high=1.0),
            "use_vol_target": HyperParam.boolean("use_vol_target", default=True),
            "target_session_vol": HyperParam.floating(
                "target_session_vol", default=0.02, low=0.001, high=0.50
            ),
            "max_vol_weight": HyperParam.floating("max_vol_weight", default=1.0, low=0.1, high=5.0),
            "allow_short": HyperParam.boolean("allow_short", default=False),
            "max_symbols_by_noise": HyperParam.integer(
                "max_symbols_by_noise", default=0, low=0, high=100
            ),
            "max_position_allocation": HyperParam.floating(
                "max_position_allocation", default=0.30, low=0.0, high=1.0
            ),
            "stop_loss_pct": HyperParam.floating(
                "stop_loss_pct", default=0.0, low=0.0, high=1.0, tunable=False
            ),
            "target_allocation": HyperParam.floating(
                "target_allocation", default=0.10, low=0.0, high=1.0, tunable=False
            ),
            "max_order_value": HyperParam.floating(
                "max_order_value", default=500.0, low=0.0, high=1_000_000.0, tunable=False
            ),
            "session_start_minute_utc": HyperParam.integer(
                "session_start_minute_utc", default=0, low=0, high=1439, tunable=False
            ),
        }

    def __init__(self, bars: Any, events: Any, **params: Any) -> None:
        self.bars = bars
        self.events = events
        self.symbol_list = list(getattr(bars, "symbol_list", []) or [])
        resolved = resolve_params_from_schema(self.get_param_schema(), params, keep_unknown=False)
        self.k_mode = str(resolved["k_mode"])
        self.k = max(0.0, float(resolved["k"]))
        self.noise_period = max(2, int(resolved["noise_period"]))
        self.k_min = max(0.0, float(resolved["k_min"]))
        self.k_max = max(self.k_min, float(resolved["k_max"]))
        self.use_ma_score = bool(resolved["use_ma_score"])
        self.ma_score_windows = _parse_windows(resolved["ma_score_windows"])
        self.min_ma_score = min(1.0, max(0.0, float(resolved["min_ma_score"])))
        self.use_vol_target = bool(resolved["use_vol_target"])
        self.target_session_vol = max(0.0, float(resolved["target_session_vol"]))
        self.max_vol_weight = max(0.0, float(resolved["max_vol_weight"]))
        self.allow_short = bool(resolved["allow_short"])
        self.max_symbols_by_noise = max(0, int(resolved["max_symbols_by_noise"]))
        self.max_position_allocation = max(0.0, float(resolved["max_position_allocation"]))
        self.stop_loss_pct = max(0.0, float(resolved["stop_loss_pct"]))
        self.target_allocation = max(0.0, float(resolved["target_allocation"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.session_start_minute_utc = max(0, min(1439, int(resolved["session_start_minute_utc"])))
        size = self.noise_period + max(self.ma_score_windows) + 5
        self._state = {
            symbol: _State(
                opens=deque(maxlen=size),
                highs=deque(maxlen=size),
                lows=deque(maxlen=size),
                closes=deque(maxlen=size),
            )
            for symbol in self.symbol_list
        }
        self._allowed_symbols: set[str] | None = None
        self._rank_session_key = ""

    # ------------------------------------------------------------------ state

    def get_state(self) -> dict[str, Any]:
        return {
            "rank_session_key": self._rank_session_key,
            "allowed_symbols": (
                None if self._allowed_symbols is None else sorted(self._allowed_symbols)
            ),
            "symbol_state": {
                symbol: {
                    "opens": list(item.opens),
                    "highs": list(item.highs),
                    "lows": list(item.lows),
                    "closes": list(item.closes),
                    "session_key": item.session_key,
                    "session_open": item.session_open,
                    "session_high": item.session_high,
                    "session_low": item.session_low,
                    "session_close": item.session_close,
                    "mode": item.mode,
                    "entry_price": item.entry_price,
                    "entered_session_key": item.entered_session_key,
                    "last_time_key": item.last_time_key,
                }
                for symbol, item in self._state.items()
            },
        }

    def set_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        self._rank_session_key = str(state.get("rank_session_key", ""))
        allowed = state.get("allowed_symbols")
        self._allowed_symbols = (
            {str(symbol) for symbol in allowed} if isinstance(allowed, (list, tuple, set)) else None
        )
        raw = state.get("symbol_state")
        if not isinstance(raw, dict):
            return
        for symbol, payload in raw.items():
            if symbol not in self._state or not isinstance(payload, dict):
                continue
            item = self._state[symbol]
            for name in ("opens", "highs", "lows", "closes"):
                target = getattr(item, name)
                target.clear()
                for value in list(payload.get(name) or [])[-int(target.maxlen or 0) :]:
                    parsed = safe_float(value)
                    if parsed is not None:
                        target.append(parsed)
            item.session_key = str(payload.get("session_key", ""))
            item.session_open = safe_float(payload.get("session_open"))
            item.session_high = safe_float(payload.get("session_high"))
            item.session_low = safe_float(payload.get("session_low"))
            item.session_close = safe_float(payload.get("session_close"))
            mode = str(payload.get("mode", "OUT")).upper()
            item.mode = mode if mode in ("LONG", "SHORT") else "OUT"
            item.entry_price = safe_float(payload.get("entry_price"))
            item.entered_session_key = str(payload.get("entered_session_key", ""))
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

    def calculate_signals_batch(self, event: Any) -> None:
        prepared = []
        for bar in sorted(getattr(event, "bars", ()), key=lambda item: str(item.symbol)):
            symbol = str(getattr(bar, "symbol", ""))
            if (
                symbol in self._state
                and (snapshot := _market_snapshot(bar)) is not None
                and (context := self._prepare(symbol, snapshot, refresh_ranking=False))
            ):
                prepared.append(context)
        if prepared:
            self._refresh_noise_ranking(prepared[0][3])
        for symbol, snapshot, item, session, high, low, close in prepared:
            self._evaluate(symbol, snapshot, item, session, high, low, close)

    # ------------------------------------------------------------------- core

    def _process(self, symbol: str, snapshot: _Snapshot) -> None:
        if context := self._prepare(symbol, snapshot, refresh_ranking=True):
            self._evaluate(*context)

    def _prepare(
        self, symbol: str, snapshot: _Snapshot, *, refresh_ranking: bool
    ) -> tuple[str, _Snapshot, _State, str, float, float, float] | None:
        item = self._state[symbol]
        key = time_key(snapshot.time)
        if key and key == item.last_time_key:
            return
        close = safe_float(snapshot.close)
        high = safe_float(snapshot.high)
        low = safe_float(snapshot.low)
        if close is None or high is None or low is None:
            return
        bar_open = safe_float(snapshot.open)
        if bar_open is None:
            bar_open = close
        session = _session_key(snapshot.time, start_minute_utc=self.session_start_minute_utc)
        if not session:
            return
        item.last_time_key = key

        if session != item.session_key:
            self._roll_session(
                symbol, item, snapshot, session, bar_open, high, low, close, refresh_ranking
            )
        else:
            item.session_high = high if item.session_high is None else max(item.session_high, high)
            item.session_low = low if item.session_low is None else min(item.session_low, low)
            item.session_close = close
        return symbol, snapshot, item, session, high, low, close

    def _evaluate(
        self,
        symbol: str,
        snapshot: _Snapshot,
        item: _State,
        session: str,
        high: float,
        low: float,
        close: float,
    ) -> None:
        if item.mode != "OUT" and self._stop_hit(item, close):
            self._exit(symbol, snapshot, close, "stop_loss")
        self._maybe_enter(symbol, item, snapshot, session, high, low, close)

    def _roll_session(
        self,
        symbol: str,
        item: _State,
        snapshot: _Snapshot,
        session: str,
        bar_open: float,
        high: float,
        low: float,
        close: float,
        refresh_ranking: bool = True,
    ) -> None:
        """Archive the finished session, time-cut any open position, open the new one."""
        if (
            item.session_key
            and item.session_open is not None
            and item.session_high is not None
            and item.session_low is not None
            and item.session_close is not None
        ):
            item.opens.append(float(item.session_open))
            item.highs.append(float(item.session_high))
            item.lows.append(float(item.session_low))
            item.closes.append(float(item.session_close))
        if item.mode != "OUT":
            # The rule is a hard time cut: flatten before the new session is judged.
            self._exit(symbol, snapshot, close, "session_time_cut")
        item.session_key = session
        item.session_open = bar_open
        item.session_high = high
        item.session_low = low
        item.session_close = close
        if refresh_ranking:
            self._refresh_noise_ranking(session)

    def _maybe_enter(
        self,
        symbol: str,
        item: _State,
        snapshot: _Snapshot,
        session: str,
        high: float,
        low: float,
        close: float,
    ) -> None:
        # ponytail: at most ONE entry per symbol per session - no pyramiding and no
        # re-entry after a stop, which keeps the session accounting a single round trip.
        if item.mode != "OUT" or item.entered_session_key == session or not item.closes:
            return
        if self.max_symbols_by_noise > 0 and (
            self._allowed_symbols is None or symbol not in self._allowed_symbols
        ):
            return
        session_open = item.session_open
        if session_open is None:
            return
        prev_high = item.highs[-1]
        prev_low = item.lows[-1]
        prev_close = item.closes[-1]
        k = self._resolve_k(item)
        if k is None:
            return
        upper, lower = volatility_breakout_levels(session_open, prev_high, prev_low, k=k)
        if upper is None or lower is None:
            return
        # Intrabar cross is intentional: the live rule is a resting stop order at the
        # trigger, so the level is taken the moment the bar's range touches it.
        # ponytail: on pure daily bars each bar IS its own session, so the trigger
        # degenerates to "today's high >= today's open + K * yesterday's range" and the
        # realistic fill is the NEXT open; the backtest fill here is optimistic by that
        # much on daily data and is only faithful on intraday bars.
        # ponytail: when a single bar spans both triggers the long side wins rather
        # than modelling which side printed first (tick data would be required).
        if high >= upper:
            direction = "LONG"
        elif self.allow_short and low <= lower:
            direction = "SHORT"
        else:
            return

        ma_score = 1.0
        if self.use_ma_score:
            scored = moving_average_score(list(item.closes), windows=self.ma_score_windows)
            if scored is None or scored < self.min_ma_score:
                return
            ma_score = float(scored)
        vol_weight = 1.0
        if self.use_vol_target:
            weighted = range_volatility_target_weight(
                prev_high,
                prev_low,
                prev_close,
                target_vol=self.target_session_vol,
                cap=self.max_vol_weight,
            )
            if weighted is None:
                return
            vol_weight = float(weighted)
        allocation = min(
            self.max_position_allocation, self.target_allocation * ma_score * vol_weight
        )
        if allocation <= 0.0:
            return

        if direction == "LONG":
            price = max(float(upper), close)
            stop_loss = price * (1.0 - self.stop_loss_pct) if self.stop_loss_pct > 0.0 else None
        else:
            price = min(float(lower), close)
            stop_loss = price * (1.0 + self.stop_loss_pct) if self.stop_loss_pct > 0.0 else None
        metadata = _target_metadata(
            strategy=self.__class__.__name__,
            target_allocation=allocation,
            max_order_value=self.max_order_value,
            k=float(k),
            noise=self._noise_estimate(item),
            ma_score=ma_score,
            vol_weight=vol_weight,
            session=session,
            upper=float(upper),
            lower=float(lower),
            stop_loss=stop_loss,
        )
        _emit(
            self.events,
            strategy_id=_STRATEGY_ID,
            symbol=symbol,
            event_time=snapshot.time,
            signal_type=direction,
            strength=allocation,
            price=price,
            metadata=metadata,
        )
        item.mode = direction
        item.entry_price = price
        item.entered_session_key = session

    def _exit(self, symbol: str, snapshot: _Snapshot, close: float, reason: str) -> None:
        item = self._state[symbol]
        # ponytail: EXIT closes the WHOLE position - the portfolio has no partial
        # exits, so the session time cut and the stop are both all-or-nothing.
        _emit(
            self.events,
            strategy_id=_STRATEGY_ID,
            symbol=symbol,
            event_time=snapshot.time,
            signal_type="EXIT",
            price=close,
            metadata={"strategy": self.__class__.__name__, "reason": reason},
        )
        item.mode = "OUT"
        item.entry_price = None

    # -------------------------------------------------------------- internals

    def _stop_hit(self, item: _State, close: float) -> bool:
        entry = item.entry_price
        if self.stop_loss_pct <= 0.0 or entry is None or entry <= 0.0:
            return False
        if item.mode == "LONG":
            return close <= entry * (1.0 - self.stop_loss_pct)
        return close >= entry * (1.0 + self.stop_loss_pct)

    def _noise_estimate(self, item: _State) -> float | None:
        return average_noise_ratio(
            list(item.opens),
            list(item.highs),
            list(item.lows),
            list(item.closes),
            period=self.noise_period,
        )

    def _resolve_k(self, item: _State) -> float | None:
        if self.k_mode == "fixed":
            return self.k if self.k > 0.0 else None
        noise = self._noise_estimate(item)
        if noise is None:
            return None
        return min(self.k_max, max(self.k_min, float(noise)))

    def _refresh_noise_ranking(self, session: str) -> None:
        if self.max_symbols_by_noise <= 0:
            self._allowed_symbols = None
            self._rank_session_key = session
            return
        scored = [
            (noise, symbol)
            for symbol, item in self._state.items()
            if (noise := self._noise_estimate(item)) is not None
        ]
        scored.sort()
        self._allowed_symbols = {symbol for _, symbol in scored[: self.max_symbols_by_noise]}
        self._rank_session_key = session

"""Profit-first MARKET_WINDOW strategy families for moonshot research.

These sleeves are intentionally simple, event-driven, and live-equivalent
compatible: each ``MARKET_WINDOW`` decision tick contributes exactly one bar per
symbol and no ``TimeframeAggregator`` is required.  The classes are meant to
expand the alpha search space away from cashlike/no-trade modes while keeping
risk metadata explicit for the portfolio sizing layer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from itertools import pairwise
from statistics import mean, stdev
from typing import Any

from lumina_quant.core.events import SignalEvent
from lumina_quant.indicators.common import safe_float, safe_int, time_key
from lumina_quant.strategy import Strategy
from lumina_quant.tuning import HyperParam


def _returns(values: list[float]) -> list[float]:
    out: list[float] = []
    for before, after in pairwise(values):
        if before > 0.0 and after > 0.0:
            out.append((after / before) - 1.0)
    return out


def _zscore(value: float, sample: list[float]) -> float:
    if len(sample) < 3:
        return 0.0
    sigma = stdev(sample)
    if sigma <= 0.0:
        return 0.0
    return (float(value) - mean(sample)) / sigma


class _MoonshotWindowBase(Strategy, ABC):
    """Shared one-bar-per-decision MARKET_WINDOW implementation."""

    decision_cadence_seconds = 60
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False
    strategy_id = "profit_moonshot_base"

    def __init__(
        self,
        bars: Any,
        events: Any,
        *,
        lookback_bars: int = 48,
        fast_lookback_bars: int = 12,
        slow_lookback_bars: int = 144,
        rebalance_bars: int = 6,
        entry_threshold: float = 0.012,
        exit_threshold: float = 0.002,
        max_longs: int = 2,
        max_shorts: int = 2,
        gross_exposure: float = 0.08,
        max_order_value: float = 1000.0,
        stop_loss_pct: float = 0.045,
        take_profit_pct: float = 0.12,
        trailing_exit_pct: float = 0.055,
        max_hold_bars: int = 240,
        min_price: float = 0.10,
        allow_shorts: bool = True,
        **_: Any,
    ) -> None:
        self.bars = bars
        self.events = events
        self.symbol_list = list(getattr(self.bars, "symbol_list", []) or [])
        if not self.symbol_list:
            raise ValueError(f"{self.__class__.__name__} requires at least one symbol")

        self.lookback_bars = max(1, int(lookback_bars))
        self.fast_lookback_bars = max(1, int(fast_lookback_bars))
        self.slow_lookback_bars = max(self.lookback_bars, int(slow_lookback_bars))
        self.rebalance_bars = max(1, int(rebalance_bars))
        self.entry_threshold = max(0.0, float(entry_threshold))
        self.exit_threshold = max(0.0, float(exit_threshold))
        self.max_longs = max(0, int(max_longs))
        self.max_shorts = max(0, int(max_shorts)) if bool(allow_shorts) else 0
        self.gross_exposure = max(0.0, float(gross_exposure))
        self.max_order_value = max(0.0, float(max_order_value))
        self.stop_loss_pct = max(0.0, float(stop_loss_pct))
        self.take_profit_pct = max(0.0, float(take_profit_pct))
        self.trailing_exit_pct = max(0.0, float(trailing_exit_pct))
        self.max_hold_bars = max(0, int(max_hold_bars))
        self.min_price = max(0.0, float(min_price))

        history_len = max(self.fast_lookback_bars, self.lookback_bars, self.slow_lookback_bars) + 8
        self._close: dict[str, deque[float]] = {
            symbol: deque(maxlen=history_len) for symbol in self.symbol_list
        }
        self._high: dict[str, deque[float]] = {
            symbol: deque(maxlen=history_len) for symbol in self.symbol_list
        }
        self._low: dict[str, deque[float]] = {
            symbol: deque(maxlen=history_len) for symbol in self.symbol_list
        }
        self._volume: dict[str, deque[float]] = {
            symbol: deque(maxlen=history_len) for symbol in self.symbol_list
        }
        self._last_symbol_time_key = dict.fromkeys(self.symbol_list, "")
        self._last_eval_time_key = ""
        self._position_state = dict.fromkeys(self.symbol_list, "OUT")
        self._entry_price = dict.fromkeys(self.symbol_list, 0.0)
        self._high_watermark = dict.fromkeys(self.symbol_list, 0.0)
        self._low_watermark = dict.fromkeys(self.symbol_list, 0.0)
        self._bars_held = dict.fromkeys(self.symbol_list, 0)
        self._tick = 0

    @classmethod
    def _base_schema(cls) -> dict[str, HyperParam]:
        return {
            "lookback_bars": HyperParam.integer(
                "lookback_bars", default=48, low=2, high=10080, tunable=False
            ),
            "fast_lookback_bars": HyperParam.integer(
                "fast_lookback_bars", default=12, low=1, high=1440, tunable=False
            ),
            "slow_lookback_bars": HyperParam.integer(
                "slow_lookback_bars", default=144, low=2, high=10080, tunable=False
            ),
            "rebalance_bars": HyperParam.integer(
                "rebalance_bars", default=6, low=1, high=1440, tunable=False
            ),
            "entry_threshold": HyperParam.floating(
                "entry_threshold", default=0.012, low=0.0, high=1.0, tunable=False
            ),
            "exit_threshold": HyperParam.floating(
                "exit_threshold", default=0.002, low=0.0, high=1.0, tunable=False
            ),
            "max_longs": HyperParam.integer("max_longs", default=2, low=0, high=32, tunable=False),
            "max_shorts": HyperParam.integer(
                "max_shorts", default=2, low=0, high=32, tunable=False
            ),
            "gross_exposure": HyperParam.floating(
                "gross_exposure", default=0.08, low=0.0, high=2.0, tunable=False
            ),
            "max_order_value": HyperParam.floating(
                "max_order_value", default=1000.0, low=0.0, high=1_000_000.0, tunable=False
            ),
            "stop_loss_pct": HyperParam.floating(
                "stop_loss_pct", default=0.045, low=0.0, high=1.0, tunable=False
            ),
            "take_profit_pct": HyperParam.floating(
                "take_profit_pct", default=0.12, low=0.0, high=2.0, tunable=False
            ),
            "trailing_exit_pct": HyperParam.floating(
                "trailing_exit_pct", default=0.055, low=0.0, high=1.0, tunable=False
            ),
            "max_hold_bars": HyperParam.integer(
                "max_hold_bars", default=240, low=0, high=10080, tunable=False
            ),
            "min_price": HyperParam.floating(
                "min_price", default=0.10, low=0.0, high=1_000_000.0, tunable=False
            ),
            "allow_shorts": HyperParam.boolean("allow_shorts", default=True, tunable=False),
        }

    def get_state(self) -> dict[str, Any]:
        return {
            "last_symbol_time_key": dict(self._last_symbol_time_key),
            "last_eval_time_key": self._last_eval_time_key,
            "position_state": dict(self._position_state),
            "entry_price": dict(self._entry_price),
            "high_watermark": dict(self._high_watermark),
            "low_watermark": dict(self._low_watermark),
            "bars_held": dict(self._bars_held),
            "tick": int(self._tick),
            "close": {symbol: list(values) for symbol, values in self._close.items()},
            "high": {symbol: list(values) for symbol, values in self._high.items()},
            "low": {symbol: list(values) for symbol, values in self._low.items()},
            "volume": {symbol: list(values) for symbol, values in self._volume.items()},
        }

    def set_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        self._last_eval_time_key = str(state.get("last_eval_time_key", ""))
        self._tick = max(0, safe_int(state.get("tick", 0), 0))
        for name, target in (
            ("last_symbol_time_key", self._last_symbol_time_key),
            ("position_state", self._position_state),
            ("entry_price", self._entry_price),
            ("high_watermark", self._high_watermark),
            ("low_watermark", self._low_watermark),
            ("bars_held", self._bars_held),
        ):
            raw = state.get(name)
            if not isinstance(raw, dict):
                continue
            for symbol, value in raw.items():
                if symbol not in target:
                    continue
                if name == "position_state":
                    if value in {"OUT", "LONG", "SHORT"}:
                        target[symbol] = str(value)
                elif name in {"last_symbol_time_key"}:
                    target[symbol] = str(value)
                elif name == "bars_held":
                    target[symbol] = max(0, safe_int(value, 0))
                else:
                    target[symbol] = float(safe_float(value) or 0.0)
        for name, target in (
            ("close", self._close),
            ("high", self._high),
            ("low", self._low),
            ("volume", self._volume),
        ):
            raw = state.get(name)
            if not isinstance(raw, dict):
                continue
            for symbol, values in raw.items():
                if symbol not in target or not isinstance(values, list):
                    continue
                target[symbol].clear()
                keep = int(target[symbol].maxlen or len(values))
                for value in values[-keep:]:
                    parsed = safe_float(value)
                    if parsed is not None and parsed >= 0.0:
                        target[symbol].append(float(parsed))

    def _row_ohlcv(self, row: Any) -> tuple[float, float, float, float] | None:
        if isinstance(row, dict):
            close = safe_float(row.get("close"))
            high = safe_float(row.get("high"))
            low = safe_float(row.get("low"))
            volume = safe_float(row.get("volume"))
        elif isinstance(row, (tuple, list)) and len(row) >= 6:
            high = safe_float(row[2])
            low = safe_float(row[3])
            close = safe_float(row[4])
            volume = safe_float(row[5])
        else:
            return None
        if close is None or close <= 0.0:
            return None
        high = float(high if high is not None and high > 0.0 else close)
        low = float(low if low is not None and low > 0.0 else close)
        volume = float(volume if volume is not None and volume >= 0.0 else 0.0)
        return high, low, float(close), volume

    def _append_bar(self, symbol: str, event_time_key: str, row: Any) -> bool:
        if symbol not in self._close or self._last_symbol_time_key.get(symbol) == event_time_key:
            return False
        parsed = self._row_ohlcv(row)
        if parsed is None:
            return False
        high, low, close, volume = parsed
        if close < self.min_price:
            return False
        self._last_symbol_time_key[symbol] = event_time_key
        self._high[symbol].append(high)
        self._low[symbol].append(low)
        self._close[symbol].append(close)
        self._volume[symbol].append(volume)
        return True

    def _momentum(self, symbol: str, lookback: int) -> float | None:
        values = self._close.get(symbol)
        if values is None or len(values) <= lookback:
            return None
        latest = float(values[-1])
        base = float(values[-1 - lookback])
        if latest <= 0.0 or base <= 0.0:
            return None
        return latest / base - 1.0

    def _latest_price(self, symbol: str) -> float | None:
        values = self._close.get(symbol)
        if values:
            return float(values[-1])
        parsed = safe_float(self.bars.get_latest_bar_value(symbol, "close"))
        if parsed is None or parsed <= 0.0:
            return None
        return float(parsed)

    def _allocation(self, n_targets: int) -> float:
        if n_targets <= 0:
            return 0.0
        return max(0.0, float(self.gross_exposure) / float(n_targets))

    def _emit(
        self,
        *,
        symbol: str,
        event_time: Any,
        signal_type: str,
        price: float,
        score: float = 0.0,
        target_allocation: float = 0.0,
        reason: str = "",
    ) -> None:
        stop_loss = None
        take_profit = None
        trailing_percent = None
        if signal_type == "LONG":
            if self.stop_loss_pct > 0.0:
                stop_loss = price * (1.0 - self.stop_loss_pct)
            if self.take_profit_pct > 0.0:
                take_profit = price * (1.0 + self.take_profit_pct)
            if self.trailing_exit_pct > 0.0:
                trailing_percent = self.trailing_exit_pct
        elif signal_type == "SHORT":
            if self.stop_loss_pct > 0.0:
                stop_loss = price * (1.0 + self.stop_loss_pct)
            if self.take_profit_pct > 0.0:
                take_profit = price * (1.0 - self.take_profit_pct)
            if self.trailing_exit_pct > 0.0:
                trailing_percent = self.trailing_exit_pct
        metadata = {
            "strategy": self.__class__.__name__,
            "score": float(score),
            "reason": reason,
            "target_allocation": float(target_allocation) if target_allocation > 0.0 else 0.0,
            "target_allocation_scale": float(target_allocation) if target_allocation > 0.0 else 0.0,
            "max_symbol_exposure_pct": float(target_allocation) if target_allocation > 0.0 else 0.0,
            "max_order_value": float(self.max_order_value),
            "gross_exposure": float(self.gross_exposure),
        }
        self.events.put(
            SignalEvent(
                strategy_id=self.strategy_id,
                symbol=symbol,
                datetime=event_time,
                signal_type=signal_type,
                strength=float(target_allocation if target_allocation > 0.0 else 1.0),
                price=price,
                position_side=signal_type if signal_type in {"LONG", "SHORT"} else None,
                stop_loss=stop_loss,
                take_profit=take_profit,
                trailing_percent=trailing_percent,
                metadata=metadata,
            )
        )

    def _enter(self, symbol: str, side: str, price: float) -> None:
        self._position_state[symbol] = side
        self._entry_price[symbol] = price
        self._high_watermark[symbol] = price
        self._low_watermark[symbol] = price
        self._bars_held[symbol] = 0

    def _exit(self, symbol: str) -> None:
        self._position_state[symbol] = "OUT"
        self._entry_price[symbol] = 0.0
        self._high_watermark[symbol] = 0.0
        self._low_watermark[symbol] = 0.0
        self._bars_held[symbol] = 0

    def _risk_exit_reason(self, symbol: str, price: float) -> str:
        state = self._position_state.get(symbol, "OUT")
        if state == "OUT":
            return ""
        self._bars_held[symbol] = int(self._bars_held.get(symbol, 0) or 0) + 1
        entry = float(self._entry_price.get(symbol, 0.0) or 0.0)
        if entry <= 0.0:
            return "missing_entry"
        self._high_watermark[symbol] = max(
            float(self._high_watermark.get(symbol, price) or price), price
        )
        self._low_watermark[symbol] = min(
            float(self._low_watermark.get(symbol, price) or price), price
        )
        if state == "LONG":
            if self.stop_loss_pct > 0.0 and price <= entry * (1.0 - self.stop_loss_pct):
                return "stop_loss"
            if self.take_profit_pct > 0.0 and price >= entry * (1.0 + self.take_profit_pct):
                return "take_profit"
            if self.trailing_exit_pct > 0.0 and price <= self._high_watermark[symbol] * (
                1.0 - self.trailing_exit_pct
            ):
                return "trailing_exit"
        if state == "SHORT":
            if self.stop_loss_pct > 0.0 and price >= entry * (1.0 + self.stop_loss_pct):
                return "stop_loss"
            if self.take_profit_pct > 0.0 and price <= entry * (1.0 - self.take_profit_pct):
                return "take_profit"
            if self.trailing_exit_pct > 0.0 and price >= self._low_watermark[symbol] * (
                1.0 + self.trailing_exit_pct
            ):
                return "trailing_exit"
        if (
            self.max_hold_bars > 0
            and int(self._bars_held.get(symbol, 0) or 0) >= self.max_hold_bars
        ):
            return "max_hold"
        return ""

    @abstractmethod
    def _targets(self) -> tuple[dict[str, str], dict[str, float]]:
        """Return symbol -> LONG/SHORT target and per-symbol score."""

    def _process_decision_bar(self, event_time: Any, event_time_key: str) -> None:
        if event_time_key == self._last_eval_time_key:
            return
        self._last_eval_time_key = event_time_key
        self._tick += 1
        should_rebalance = self._tick % self.rebalance_bars == 0
        targets: dict[str, str] = {}
        scores: dict[str, float] = {}
        if should_rebalance:
            targets, scores = self._targets()

        for symbol in self.symbol_list:
            state = self._position_state.get(symbol, "OUT")
            if state == "OUT":
                continue
            price = self._latest_price(symbol)
            if price is None:
                continue
            reason = self._risk_exit_reason(symbol, price)
            target_state = targets.get(symbol, "OUT") if should_rebalance else state
            if (
                should_rebalance
                and target_state != state
                and abs(scores.get(symbol, 0.0)) <= self.exit_threshold
            ):
                reason = reason or "signal_decay"
            if reason:
                self._emit(
                    symbol=symbol,
                    event_time=event_time,
                    signal_type="EXIT",
                    price=price,
                    score=scores.get(symbol, 0.0),
                    reason=reason,
                )
                self._exit(symbol)

        if not should_rebalance:
            return

        allocations = self._allocation(len(targets))
        for symbol in self.symbol_list:
            target_state = targets.get(symbol, "OUT")
            state = self._position_state.get(symbol, "OUT")
            if target_state == state:
                continue
            price = self._latest_price(symbol)
            if price is None:
                continue
            if state != "OUT":
                self._emit(
                    symbol=symbol,
                    event_time=event_time,
                    signal_type="EXIT",
                    price=price,
                    score=scores.get(symbol, 0.0),
                    reason="rebalance",
                )
                self._exit(symbol)
            if target_state in {"LONG", "SHORT"}:
                self._emit(
                    symbol=symbol,
                    event_time=event_time,
                    signal_type=target_state,
                    price=price,
                    score=scores.get(symbol, 0.0),
                    target_allocation=allocations,
                    reason="entry",
                )
                self._enter(symbol, target_state, price)

    def calculate_signals_window(self, event: Any, aggregator: Any) -> None:
        _ = aggregator
        if str(getattr(event, "type", "")).upper() != "MARKET_WINDOW":
            self.calculate_signals(event)
            return
        event_time = getattr(event, "time", None)
        event_time_key = time_key(event_time)
        if not event_time_key:
            return
        for symbol, rows in dict(getattr(event, "bars_1s", {}) or {}).items():
            if symbol in self._close and rows:
                self._append_bar(symbol, event_time_key, list(rows)[-1])
        self._process_decision_bar(event_time, event_time_key)

    def calculate_signals(self, event: Any) -> None:
        if getattr(event, "type", None) != "MARKET":
            return
        symbol = str(getattr(event, "symbol", "") or "")
        if symbol not in self._close:
            return
        event_time = getattr(event, "time", None) or self.bars.get_latest_bar_datetime(symbol)
        event_time_key = time_key(event_time)
        if not event_time_key:
            return
        row = (
            event_time,
            getattr(event, "open", 0.0),
            getattr(event, "high", 0.0),
            getattr(event, "low", 0.0),
            getattr(event, "close", 0.0),
            getattr(event, "volume", 0.0),
        )
        if self._append_bar(symbol, event_time_key, row):
            self._process_decision_bar(event_time, event_time_key)


class ProfitMoonshotTrendStrategy(_MoonshotWindowBase):
    """Cross-sectional multi-horizon trend acceleration sleeve."""

    strategy_id = "profit_moonshot_trend"

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        schema = cls._base_schema()
        schema.update(
            {
                "breadth_threshold": HyperParam.floating(
                    "breadth_threshold", default=0.0, low=-1.0, high=1.0, tunable=False
                ),
            }
        )
        return schema

    def __init__(self, *args: Any, breadth_threshold: float = 0.0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.breadth_threshold = float(breadth_threshold)

    def _targets(self) -> tuple[dict[str, str], dict[str, float]]:
        raw_scores: dict[str, float] = {}
        for symbol in self.symbol_list:
            fast = self._momentum(symbol, self.fast_lookback_bars)
            medium = self._momentum(symbol, self.lookback_bars)
            slow = self._momentum(symbol, self.slow_lookback_bars)
            if fast is None or medium is None or slow is None:
                continue
            acceleration = fast - medium
            raw_scores[symbol] = 0.40 * fast + 0.40 * medium + 0.20 * slow + 0.25 * acceleration
        if not raw_scores:
            return {}, {}
        breadth = mean(raw_scores.values())
        centered = {symbol: score - 0.20 * breadth for symbol, score in raw_scores.items()}
        targets: dict[str, str] = {}
        if breadth >= self.breadth_threshold and self.max_longs > 0:
            for symbol, score in sorted(centered.items(), key=lambda item: item[1], reverse=True):
                if score < self.entry_threshold or len(targets) >= self.max_longs:
                    break
                targets[symbol] = "LONG"
        if self.max_shorts > 0:
            short_count = 0
            for symbol, score in sorted(centered.items(), key=lambda item: item[1]):
                if score > -self.entry_threshold or short_count >= self.max_shorts:
                    break
                if symbol not in targets:
                    targets[symbol] = "SHORT"
                    short_count += 1
        return targets, centered


class ProfitMoonshotBreakoutStrategy(_MoonshotWindowBase):
    """Donchian breakout sleeve gated by volatility expansion."""

    strategy_id = "profit_moonshot_breakout"

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        schema = cls._base_schema()
        schema.update(
            {
                "breakout_buffer": HyperParam.floating(
                    "breakout_buffer", default=0.002, low=0.0, high=1.0, tunable=False
                ),
                "squeeze_ratio_max": HyperParam.floating(
                    "squeeze_ratio_max", default=1.35, low=0.0, high=10.0, tunable=False
                ),
                "volume_z_min": HyperParam.floating(
                    "volume_z_min", default=0.0, low=-10.0, high=10.0, tunable=False
                ),
            }
        )
        return schema

    def __init__(
        self,
        *args: Any,
        breakout_buffer: float = 0.002,
        squeeze_ratio_max: float = 1.35,
        volume_z_min: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.breakout_buffer = max(0.0, float(breakout_buffer))
        self.squeeze_ratio_max = max(0.0, float(squeeze_ratio_max))
        self.volume_z_min = float(volume_z_min)

    def _targets(self) -> tuple[dict[str, str], dict[str, float]]:
        scores: dict[str, float] = {}
        for symbol in self.symbol_list:
            close = self._close[symbol]
            highs = self._high[symbol]
            lows = self._low[symbol]
            volumes = self._volume[symbol]
            if len(close) <= self.lookback_bars or len(close) <= self.slow_lookback_bars:
                continue
            prev_high = max(list(highs)[-self.lookback_bars - 1 : -1])
            prev_low = min(list(lows)[-self.lookback_bars - 1 : -1])
            latest = float(close[-1])
            if prev_high <= 0.0 or prev_low <= 0.0:
                continue
            ret_fast = _returns(list(close)[-(self.fast_lookback_bars + 1) :])
            ret_slow = _returns(list(close)[-(self.slow_lookback_bars + 1) :])
            fast_vol = stdev(ret_fast) if len(ret_fast) > 1 else 0.0
            slow_vol = stdev(ret_slow) if len(ret_slow) > 1 else fast_vol
            squeeze_ratio = fast_vol / slow_vol if slow_vol > 0.0 else 1.0
            volume_sample = list(volumes)[-self.lookback_bars - 1 : -1]
            volume_z = _zscore(float(volumes[-1]), [float(item) for item in volume_sample])
            expansion_ok = squeeze_ratio <= self.squeeze_ratio_max or volume_z >= self.volume_z_min
            if not expansion_ok:
                continue
            up = latest / prev_high - 1.0
            down = latest / prev_low - 1.0
            if up >= self.breakout_buffer:
                scores[symbol] = up + max(0.0, volume_z) * 0.001
            elif down <= -self.breakout_buffer:
                scores[symbol] = down - max(0.0, volume_z) * 0.001
        targets: dict[str, str] = {}
        for symbol, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)[
            : self.max_longs
        ]:
            if score >= self.entry_threshold:
                targets[symbol] = "LONG"
        short_count = 0
        for symbol, score in sorted(scores.items(), key=lambda item: item[1]):
            if (
                score <= -self.entry_threshold
                and symbol not in targets
                and short_count < self.max_shorts
            ):
                targets[symbol] = "SHORT"
                short_count += 1
        return targets, scores


class ProfitMoonshotReversionStrategy(_MoonshotWindowBase):
    """Volume/range shock fade sleeve for overextended moves."""

    strategy_id = "profit_moonshot_reversion"

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        schema = cls._base_schema()
        schema.update(
            {
                "return_z_min": HyperParam.floating(
                    "return_z_min", default=1.4, low=0.0, high=10.0, tunable=False
                ),
                "volume_z_min": HyperParam.floating(
                    "volume_z_min", default=0.5, low=-10.0, high=10.0, tunable=False
                ),
                "range_z_min": HyperParam.floating(
                    "range_z_min", default=0.5, low=-10.0, high=10.0, tunable=False
                ),
            }
        )
        return schema

    def __init__(
        self,
        *args: Any,
        return_z_min: float = 1.4,
        volume_z_min: float = 0.5,
        range_z_min: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.return_z_min = max(0.0, float(return_z_min))
        self.volume_z_min = float(volume_z_min)
        self.range_z_min = float(range_z_min)

    def _targets(self) -> tuple[dict[str, str], dict[str, float]]:
        scores: dict[str, float] = {}
        for symbol in self.symbol_list:
            close = self._close[symbol]
            highs = self._high[symbol]
            lows = self._low[symbol]
            volumes = self._volume[symbol]
            if len(close) <= self.lookback_bars + 1:
                continue
            closes = list(close)[-(self.lookback_bars + 1) :]
            returns = _returns(closes)
            if len(returns) < 3:
                continue
            latest_return = returns[-1]
            return_z = _zscore(latest_return, returns[:-1])
            ranges = [
                (float(high_value) - float(low_value)) / float(close_value)
                for high_value, low_value, close_value in zip(
                    list(highs)[-self.lookback_bars - 1 : -1],
                    list(lows)[-self.lookback_bars - 1 : -1],
                    list(close)[-self.lookback_bars - 1 : -1],
                    strict=False,
                )
                if float(close_value) > 0.0
            ]
            latest_range = (float(highs[-1]) - float(lows[-1])) / float(close[-1])
            range_z = _zscore(latest_range, ranges)
            volume_z = _zscore(
                float(volumes[-1]), [float(v) for v in list(volumes)[-self.lookback_bars - 1 : -1]]
            )
            if abs(return_z) < self.return_z_min:
                continue
            if volume_z < self.volume_z_min and range_z < self.range_z_min:
                continue
            # Fade extreme positive moves with shorts and extreme negative moves with longs.
            scores[symbol] = -return_z * max(1.0, 1.0 + 0.05 * max(volume_z, range_z, 0.0))
        targets: dict[str, str] = {}
        for symbol, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)[
            : self.max_longs
        ]:
            if score >= self.entry_threshold:
                targets[symbol] = "LONG"
        short_count = 0
        for symbol, score in sorted(scores.items(), key=lambda item: item[1]):
            if (
                score <= -self.entry_threshold
                and symbol not in targets
                and short_count < self.max_shorts
            ):
                targets[symbol] = "SHORT"
                short_count += 1
        return targets, scores


__all__ = [
    "ProfitMoonshotBreakoutStrategy",
    "ProfitMoonshotReversionStrategy",
    "ProfitMoonshotTrendStrategy",
]

"""Additional theory-grounded alpha sleeves for forward testing.

The classes in this module are intentionally data-local: they do not fetch data,
write artifacts, or depend on hidden environment state.  Each strategy registers
as an event-driven drop-in candidate and is designed around an established alpha
family that should still be validated on the data-bearing research machine:

- Donchian/ATR trend following: captures convex trend continuation after new highs/lows.
- Volatility squeeze breakout: waits for compression before range expansion.
- Taker-flow imbalance continuation: follows aggressive buy/sell pressure when price confirms.
- Pair-spread mean reversion: trades temporary divergence between related instruments.
- Cross-sectional short-term reversal: provides liquidity after recent dislocations.
"""

from __future__ import annotations

import math
from itertools import pairwise
from collections import deque
from dataclasses import dataclass
from typing import Any

from lumina_quant.core.plugin_registry import register
from lumina_quant.indicators.alpha_features import (
    log_return,
    order_flow_imbalance,
    realized_volatility,
    simple_return,
    trend_efficiency,
    volatility_ratio,
    volume_zscore,
)
from lumina_quant.indicators.atr import average_true_range, true_range
from lumina_quant.indicators.common import safe_float, time_key
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


@dataclass(slots=True)
class _ChannelState:
    highs: deque[float]
    lows: deque[float]
    closes: deque[float]
    volumes: deque[float]
    tr_values: deque[float]
    ranges: deque[float]
    mode: str = "OUT"
    entry_price: float | None = None
    entry_atr: float | None = None
    high_watermark: float | None = None
    low_watermark: float | None = None
    bars_held: int = 0
    ticks_seen: int = 0
    last_time_key: str = ""


@dataclass(slots=True)
class _FlowState:
    closes: deque[float]
    volumes: deque[float]
    buy_quote: deque[float]
    sell_quote: deque[float]
    imbalances: deque[float]
    mode: str = "OUT"
    entry_price: float | None = None
    bars_held: int = 0
    last_time_key: str = ""


@dataclass(slots=True)
class _PairState:
    symbol_a: str
    symbol_b: str
    closes_a: deque[float]
    closes_b: deque[float]
    spreads: deque[float]
    mode: str = "OUT"
    beta: float = 1.0
    entry_z: float | None = None
    bars_held: int = 0
    last_eval_key: str = ""


@dataclass(slots=True)
class _CrossSectionalState:
    closes: deque[float]
    volumes: deque[float]
    mode: str = "OUT"
    entry_price: float | None = None
    bars_held: int = 0
    last_time_key: str = ""


def _append_market_state(item: _ChannelState, snapshot: _Snapshot) -> bool:
    close = safe_float(snapshot.close)
    if close is None or close <= 0.0:
        return False
    high = safe_float(snapshot.high)
    low = safe_float(snapshot.low)
    if high is None:
        high = close
    if low is None:
        low = close
    prev_close = item.closes[-1] if item.closes else None
    item.highs.append(float(high))
    item.lows.append(float(low))
    item.closes.append(float(close))
    volume = max(0.0, float(snapshot.volume or 0.0))
    item.volumes.append(volume)
    item.tr_values.append(true_range(float(high), float(low), prev_close))
    item.ranges.append(max(0.0, float(high) - float(low)) / max(abs(float(close)), _EPS))
    item.ticks_seen += 1
    return True


def _restore_deque(target: deque[float], payload: Any) -> None:
    target.clear()
    for value in list(payload or [])[-int(target.maxlen or 0) :]:
        parsed = safe_float(value)
        if parsed is not None:
            target.append(parsed)


def _mode(raw: Any) -> str:
    parsed = str(raw or "OUT").upper()
    return parsed if parsed in {"OUT", "LONG", "SHORT", "LONG_SPREAD", "SHORT_SPREAD"} else "OUT"


def _mean_tail(values: deque[float], window: int) -> float | None:
    win = max(1, int(window))
    if len(values) < win:
        return None
    tail = list(values)[-win:]
    if not tail:
        return None
    return float(sum(tail) / len(tail))


def _feature_value(bars: Any, event: Any, symbol: str, field: str) -> float | None:
    direct = safe_float(getattr(event, field, None))
    if direct is not None:
        return direct
    getter = getattr(bars, "get_latest_feature_value", None)
    if callable(getter):
        try:
            parsed = safe_float(getter(symbol, field))
        except Exception:
            parsed = None
        if parsed is not None:
            return parsed
    getter = getattr(bars, "get_latest_bar_value", None)
    if callable(getter):
        try:
            return safe_float(getter(symbol, field))
        except Exception:
            return None
    return None


def _log_return_series(values: list[float]) -> list[float]:
    out: list[float] = []
    for prev, cur in pairwise(values):
        if prev > 0.0 and cur > 0.0:
            out.append(math.log(cur / prev))
    return out


def _rolling_beta(
    a: deque[float], b: deque[float], window: int, beta_floor: float, beta_cap: float
) -> float | None:
    n = min(len(a), len(b), max(2, int(window)) + 1)
    if n < 4:
        return None
    returns_a = _log_return_series(list(a)[-n:])
    returns_b = _log_return_series(list(b)[-n:])
    n_ret = min(len(returns_a), len(returns_b))
    if n_ret < 3:
        return None
    returns_a = returns_a[-n_ret:]
    returns_b = returns_b[-n_ret:]
    mean_a = sum(returns_a) / float(n_ret)
    mean_b = sum(returns_b) / float(n_ret)
    cov = sum((ra - mean_a) * (rb - mean_b) for ra, rb in zip(returns_a, returns_b, strict=True))
    var_b = sum((rb - mean_b) ** 2 for rb in returns_b)
    if var_b <= _EPS:
        return None
    beta = cov / var_b
    if not math.isfinite(beta):
        return None
    return max(float(beta_floor), min(float(beta_cap), float(beta)))


def _spread_zscore(
    pair: _PairState, spread_window: int, beta: float
) -> tuple[float | None, float | None]:
    n = min(len(pair.closes_a), len(pair.closes_b), max(3, int(spread_window)))
    if n < max(3, int(spread_window)):
        return None, None
    spreads: list[float] = []
    for price_a, price_b in zip(list(pair.closes_a)[-n:], list(pair.closes_b)[-n:], strict=True):
        if price_a <= 0.0 or price_b <= 0.0:
            continue
        spreads.append(math.log(price_a) - beta * math.log(price_b))
    if len(spreads) < max(3, int(spread_window)):
        return None, None
    latest = spreads[-1]
    mean_value = sum(spreads) / float(len(spreads))
    variance = sum((value - mean_value) ** 2 for value in spreads) / float(len(spreads) - 1)
    sigma = math.sqrt(max(0.0, variance))
    if sigma <= _EPS:
        return None, latest
    pair.spreads.append(float(latest))
    return float((latest - mean_value) / sigma), float(latest)


class _StatePackingMixin:
    @staticmethod
    def _pack_channel_state(item: _ChannelState) -> dict[str, Any]:
        return {
            "highs": list(item.highs),
            "lows": list(item.lows),
            "closes": list(item.closes),
            "volumes": list(item.volumes),
            "tr_values": list(item.tr_values),
            "ranges": list(item.ranges),
            "mode": item.mode,
            "entry_price": item.entry_price,
            "entry_atr": item.entry_atr,
            "high_watermark": item.high_watermark,
            "low_watermark": item.low_watermark,
            "bars_held": int(item.bars_held),
            "ticks_seen": int(item.ticks_seen),
            "last_time_key": item.last_time_key,
        }

    @staticmethod
    def _restore_channel_state(item: _ChannelState, payload: dict[str, Any]) -> None:
        for attr in ("highs", "lows", "closes", "volumes", "tr_values", "ranges"):
            _restore_deque(getattr(item, attr), payload.get(attr))
        item.mode = _mode(payload.get("mode"))
        item.entry_price = safe_float(payload.get("entry_price"))
        item.entry_atr = safe_float(payload.get("entry_atr"))
        item.high_watermark = safe_float(payload.get("high_watermark"))
        item.low_watermark = safe_float(payload.get("low_watermark"))
        item.bars_held = _safe_non_negative_int(payload.get("bars_held"))
        item.ticks_seen = _safe_non_negative_int(payload.get("ticks_seen"))
        item.last_time_key = str(payload.get("last_time_key", ""))


@register("strategy", "DonchianAtrTrendStrategy", interface="event_driven")
class DonchianAtrTrendStrategy(_StatePackingMixin, Strategy):
    """Classic channel breakout trend-following sleeve with ATR risk controls."""

    decision_cadence_seconds = 60
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "breakout_lookback": HyperParam.integer(
                "breakout_lookback", default=120, low=4, high=10080
            ),
            "exit_lookback": HyperParam.integer("exit_lookback", default=60, low=2, high=10080),
            "atr_window": HyperParam.integer("atr_window", default=48, low=2, high=4096),
            "entry_atr_buffer": HyperParam.floating(
                "entry_atr_buffer", default=0.10, low=0.0, high=5.0
            ),
            "stop_atr_multiple": HyperParam.floating(
                "stop_atr_multiple", default=2.5, low=0.0, high=20.0
            ),
            "trail_atr_multiple": HyperParam.floating(
                "trail_atr_multiple", default=3.0, low=0.0, high=20.0
            ),
            "min_trend_efficiency": HyperParam.floating(
                "min_trend_efficiency", default=0.18, low=0.0, high=1.0
            ),
            "max_vol_ratio": HyperParam.floating("max_vol_ratio", default=4.0, low=0.0, high=50.0),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=2880, low=1, high=200000),
            "target_allocation": HyperParam.floating(
                "target_allocation", default=0.015, low=0.0, high=2.0, tunable=False
            ),
            "max_order_value": HyperParam.floating(
                "max_order_value", default=300.0, low=0.0, high=1_000_000.0, tunable=False
            ),
            "min_price": HyperParam.floating("min_price", default=0.10, low=0.0, high=1_000_000.0),
        }

    def __init__(self, bars: Any, events: Any, **params: Any) -> None:
        self.bars = bars
        self.events = events
        self.symbol_list = list(getattr(self.bars, "symbol_list", []) or [])
        resolved = resolve_params_from_schema(self.get_param_schema(), params, keep_unknown=False)
        self.breakout_lookback = max(2, int(resolved["breakout_lookback"]))
        self.exit_lookback = max(2, int(resolved["exit_lookback"]))
        self.atr_window = max(2, int(resolved["atr_window"]))
        self.entry_atr_buffer = max(0.0, float(resolved["entry_atr_buffer"]))
        self.stop_atr_multiple = max(0.0, float(resolved["stop_atr_multiple"]))
        self.trail_atr_multiple = max(0.0, float(resolved["trail_atr_multiple"]))
        self.min_trend_efficiency = max(0.0, min(1.0, float(resolved["min_trend_efficiency"])))
        self.max_vol_ratio = max(0.0, float(resolved["max_vol_ratio"]))
        self.allow_short = bool(resolved["allow_short"])
        self.max_hold_bars = max(1, int(resolved["max_hold_bars"]))
        self.target_allocation = max(0.0, float(resolved["target_allocation"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        size = (
            max(self.breakout_lookback, self.exit_lookback, self.atr_window, self.max_hold_bars) + 8
        )
        self._state = {
            symbol: _ChannelState(
                deque(maxlen=size),
                deque(maxlen=size),
                deque(maxlen=size),
                deque(maxlen=size),
                deque(maxlen=size),
                deque(maxlen=size),
            )
            for symbol in self.symbol_list
        }

    def get_state(self) -> dict[str, Any]:
        return {
            "symbol_state": {
                symbol: self._pack_channel_state(item) for symbol, item in self._state.items()
            }
        }

    def set_state(self, state: dict[str, Any]) -> None:
        raw = state.get("symbol_state") if isinstance(state, dict) else None
        if not isinstance(raw, dict):
            return
        for symbol, payload in raw.items():
            if symbol in self._state and isinstance(payload, dict):
                self._restore_channel_state(self._state[symbol], payload)

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

    def _maybe_exit(
        self,
        symbol: str,
        item: _ChannelState,
        snapshot: _Snapshot,
        prior_mid: float | None,
        atr: float | None,
    ) -> None:
        close = safe_float(snapshot.close)
        if item.mode not in {"LONG", "SHORT"} or item.entry_price is None or close is None:
            return
        item.bars_held += 1
        reason = ""
        atr_value = max(float(atr or item.entry_atr or 0.0), _EPS)
        if item.mode == "LONG":
            item.high_watermark = max(float(item.high_watermark or close), close)
            stop = item.entry_price - self.stop_atr_multiple * atr_value
            trail = item.high_watermark - self.trail_atr_multiple * atr_value
            if self.stop_atr_multiple > 0.0 and close <= stop:
                reason = "atr_stop"
            elif self.trail_atr_multiple > 0.0 and close <= trail:
                reason = "atr_trail"
            elif prior_mid is not None and close < prior_mid:
                reason = "channel_mid_exit"
        else:
            item.low_watermark = min(float(item.low_watermark or close), close)
            stop = item.entry_price + self.stop_atr_multiple * atr_value
            trail = item.low_watermark + self.trail_atr_multiple * atr_value
            if self.stop_atr_multiple > 0.0 and close >= stop:
                reason = "atr_stop"
            elif self.trail_atr_multiple > 0.0 and close >= trail:
                reason = "atr_trail"
            elif prior_mid is not None and close > prior_mid:
                reason = "channel_mid_exit"
        if not reason and item.bars_held >= self.max_hold_bars:
            reason = "max_hold"
        if not reason:
            return
        _emit(
            self.events,
            strategy_id="donchian_atr_trend",
            symbol=symbol,
            event_time=snapshot.time,
            signal_type="EXIT",
            price=close,
            metadata={"strategy": "DonchianAtrTrendStrategy", "reason": reason},
        )
        item.mode = "OUT"
        item.entry_price = None
        item.entry_atr = None
        item.high_watermark = None
        item.low_watermark = None
        item.bars_held = 0

    def _process_symbol(self, symbol: str, snapshot: _Snapshot) -> None:
        item = self._state[symbol]
        key = time_key(snapshot.time)
        if key and key == item.last_time_key:
            return
        item.last_time_key = key
        close = safe_float(snapshot.close)
        if close is None or close <= self.min_price:
            return
        ready = len(item.highs) >= self.breakout_lookback and len(item.tr_values) >= self.atr_window
        prior_upper = max(list(item.highs)[-self.breakout_lookback :]) if ready else None
        prior_lower = min(list(item.lows)[-self.breakout_lookback :]) if ready else None
        exit_ready = len(item.highs) >= self.exit_lookback
        prior_mid = None
        if exit_ready:
            upper = max(list(item.highs)[-self.exit_lookback :])
            lower = min(list(item.lows)[-self.exit_lookback :])
            prior_mid = (upper + lower) / 2.0
        atr = (
            average_true_range(item.tr_values, self.atr_window)
            if len(item.tr_values) >= self.atr_window
            else None
        )
        self._maybe_exit(symbol, item, snapshot, prior_mid, atr)
        if (
            item.mode == "OUT"
            and ready
            and atr is not None
            and prior_upper is not None
            and prior_lower is not None
        ):
            eff = trend_efficiency(
                item.closes, window=min(self.breakout_lookback, max(3, self.atr_window))
            )
            vr = volatility_ratio(
                item.closes,
                fast_window=max(2, self.atr_window // 2),
                slow_window=max(self.atr_window + 1, self.breakout_lookback),
            )
            if (
                eff is not None
                and eff >= self.min_trend_efficiency
                and (self.max_vol_ratio <= 0.0 or vr is None or vr <= self.max_vol_ratio)
            ):
                buffer = self.entry_atr_buffer * atr
                signal_type = ""
                if close > prior_upper + buffer:
                    signal_type = "LONG"
                elif self.allow_short and close < prior_lower - buffer:
                    signal_type = "SHORT"
                if signal_type:
                    stop_loss = None
                    trailing_percent = None
                    if self.stop_atr_multiple > 0.0:
                        stop_loss = (
                            close - self.stop_atr_multiple * atr
                            if signal_type == "LONG"
                            else close + self.stop_atr_multiple * atr
                        )
                    if self.trail_atr_multiple > 0.0 and close > 0.0:
                        trailing_percent = self.trail_atr_multiple * atr / close
                    metadata = _target_metadata(
                        strategy="DonchianAtrTrendStrategy",
                        target_allocation=self.target_allocation,
                        max_order_value=self.max_order_value,
                        prior_channel_high=prior_upper,
                        prior_channel_low=prior_lower,
                        atr=atr,
                        trend_efficiency=eff,
                        vol_ratio=vr,
                        target_mode=signal_type,
                    )
                    _emit(
                        self.events,
                        strategy_id="donchian_atr_trend",
                        symbol=symbol,
                        event_time=snapshot.time,
                        signal_type=signal_type,
                        strength=max(
                            0.25,
                            min(
                                3.0,
                                abs(close - (prior_upper if signal_type == "LONG" else prior_lower))
                                / max(atr, _EPS),
                            ),
                        ),
                        price=close,
                        stop_loss=stop_loss,
                        trailing_percent=trailing_percent,
                        metadata=metadata,
                    )
                    item.mode = signal_type
                    item.entry_price = close
                    item.entry_atr = atr
                    item.high_watermark = close
                    item.low_watermark = close
                    item.bars_held = 0
        _append_market_state(item, snapshot)


@register("strategy", "VolatilitySqueezeBreakoutStrategy", interface="event_driven")
class VolatilitySqueezeBreakoutStrategy(_StatePackingMixin, Strategy):
    """Trade range expansion after realized range compression and volume confirmation."""

    decision_cadence_seconds = 60
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "compression_window": HyperParam.integer(
                "compression_window", default=48, low=4, high=4096
            ),
            "baseline_window": HyperParam.integer(
                "baseline_window", default=240, low=16, high=20000
            ),
            "breakout_lookback": HyperParam.integer(
                "breakout_lookback", default=48, low=4, high=4096
            ),
            "range_ratio_max": HyperParam.floating(
                "range_ratio_max", default=0.65, low=0.05, high=2.0
            ),
            "min_volume_z": HyperParam.floating("min_volume_z", default=0.5, low=-5.0, high=20.0),
            "momentum_lookback": HyperParam.integer(
                "momentum_lookback", default=12, low=1, high=4096
            ),
            "momentum_threshold": HyperParam.floating(
                "momentum_threshold", default=0.001, low=0.0, high=1.0
            ),
            "atr_window": HyperParam.integer("atr_window", default=48, low=2, high=4096),
            "stop_atr_multiple": HyperParam.floating(
                "stop_atr_multiple", default=2.0, low=0.0, high=20.0
            ),
            "trail_atr_multiple": HyperParam.floating(
                "trail_atr_multiple", default=2.5, low=0.0, high=20.0
            ),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=720, low=1, high=100000),
            "target_allocation": HyperParam.floating(
                "target_allocation", default=0.012, low=0.0, high=2.0, tunable=False
            ),
            "max_order_value": HyperParam.floating(
                "max_order_value", default=250.0, low=0.0, high=1_000_000.0, tunable=False
            ),
            "min_price": HyperParam.floating("min_price", default=0.10, low=0.0, high=1_000_000.0),
        }

    def __init__(self, bars: Any, events: Any, **params: Any) -> None:
        self.bars = bars
        self.events = events
        self.symbol_list = list(getattr(self.bars, "symbol_list", []) or [])
        resolved = resolve_params_from_schema(self.get_param_schema(), params, keep_unknown=False)
        self.compression_window = max(3, int(resolved["compression_window"]))
        self.baseline_window = max(self.compression_window + 1, int(resolved["baseline_window"]))
        self.breakout_lookback = max(2, int(resolved["breakout_lookback"]))
        self.range_ratio_max = max(0.01, float(resolved["range_ratio_max"]))
        self.min_volume_z = float(resolved["min_volume_z"])
        self.momentum_lookback = max(1, int(resolved["momentum_lookback"]))
        self.momentum_threshold = max(0.0, float(resolved["momentum_threshold"]))
        self.atr_window = max(2, int(resolved["atr_window"]))
        self.stop_atr_multiple = max(0.0, float(resolved["stop_atr_multiple"]))
        self.trail_atr_multiple = max(0.0, float(resolved["trail_atr_multiple"]))
        self.allow_short = bool(resolved["allow_short"])
        self.max_hold_bars = max(1, int(resolved["max_hold_bars"]))
        self.target_allocation = max(0.0, float(resolved["target_allocation"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        size = (
            max(self.baseline_window, self.breakout_lookback, self.atr_window, self.max_hold_bars)
            + 8
        )
        self._state = {
            symbol: _ChannelState(
                deque(maxlen=size),
                deque(maxlen=size),
                deque(maxlen=size),
                deque(maxlen=size),
                deque(maxlen=size),
                deque(maxlen=size),
            )
            for symbol in self.symbol_list
        }

    def get_state(self) -> dict[str, Any]:
        return {
            "symbol_state": {
                symbol: self._pack_channel_state(item) for symbol, item in self._state.items()
            }
        }

    def set_state(self, state: dict[str, Any]) -> None:
        raw = state.get("symbol_state") if isinstance(state, dict) else None
        if not isinstance(raw, dict):
            return
        for symbol, payload in raw.items():
            if symbol in self._state and isinstance(payload, dict):
                self._restore_channel_state(self._state[symbol], payload)

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

    def _exit_if_needed(
        self, symbol: str, item: _ChannelState, snapshot: _Snapshot, atr: float | None
    ) -> None:
        close = safe_float(snapshot.close)
        if item.mode not in {"LONG", "SHORT"} or item.entry_price is None or close is None:
            return
        item.bars_held += 1
        atr_value = max(float(atr or item.entry_atr or 0.0), _EPS)
        reason = ""
        if item.mode == "LONG":
            item.high_watermark = max(float(item.high_watermark or close), close)
            if (
                self.stop_atr_multiple > 0.0
                and close <= item.entry_price - self.stop_atr_multiple * atr_value
            ):
                reason = "atr_stop"
            elif (
                self.trail_atr_multiple > 0.0
                and close <= item.high_watermark - self.trail_atr_multiple * atr_value
            ):
                reason = "atr_trail"
        else:
            item.low_watermark = min(float(item.low_watermark or close), close)
            if (
                self.stop_atr_multiple > 0.0
                and close >= item.entry_price + self.stop_atr_multiple * atr_value
            ):
                reason = "atr_stop"
            elif (
                self.trail_atr_multiple > 0.0
                and close >= item.low_watermark + self.trail_atr_multiple * atr_value
            ):
                reason = "atr_trail"
        if not reason and item.bars_held >= self.max_hold_bars:
            reason = "max_hold"
        if not reason:
            return
        _emit(
            self.events,
            strategy_id="volatility_squeeze_breakout",
            symbol=symbol,
            event_time=snapshot.time,
            signal_type="EXIT",
            price=close,
            metadata={"strategy": "VolatilitySqueezeBreakoutStrategy", "reason": reason},
        )
        item.mode = "OUT"
        item.entry_price = None
        item.entry_atr = None
        item.high_watermark = None
        item.low_watermark = None
        item.bars_held = 0

    def _process_symbol(self, symbol: str, snapshot: _Snapshot) -> None:
        item = self._state[symbol]
        key = time_key(snapshot.time)
        if key and key == item.last_time_key:
            return
        item.last_time_key = key
        close = safe_float(snapshot.close)
        if close is None or close <= self.min_price:
            return
        ready = (
            len(item.ranges) >= self.baseline_window
            and len(item.highs) >= self.breakout_lookback
            and len(item.tr_values) >= self.atr_window
        )
        prior_upper = max(list(item.highs)[-self.breakout_lookback :]) if ready else None
        prior_lower = min(list(item.lows)[-self.breakout_lookback :]) if ready else None
        short_range = _mean_tail(item.ranges, self.compression_window)
        base_range = _mean_tail(item.ranges, self.baseline_window)
        atr = (
            average_true_range(item.tr_values, self.atr_window)
            if len(item.tr_values) >= self.atr_window
            else None
        )
        self._exit_if_needed(symbol, item, snapshot, atr)
        _append_market_state(item, snapshot)
        if (
            item.mode != "OUT"
            or not ready
            or prior_upper is None
            or prior_lower is None
            or atr is None
        ):
            return
        if short_range is None or base_range is None or base_range <= _EPS:
            return
        if short_range / base_range > self.range_ratio_max:
            return
        vol_z = volume_zscore(
            item.volumes, window=min(self.baseline_window, max(8, self.compression_window))
        )
        if vol_z is not None and vol_z < self.min_volume_z:
            return
        mom = log_return(item.closes, lookback=self.momentum_lookback)
        signal_type = ""
        if close > prior_upper and mom is not None and mom >= self.momentum_threshold:
            signal_type = "LONG"
        elif (
            self.allow_short
            and close < prior_lower
            and mom is not None
            and mom <= -self.momentum_threshold
        ):
            signal_type = "SHORT"
        if not signal_type:
            return
        stop_loss = None
        trailing_percent = None
        if self.stop_atr_multiple > 0.0:
            stop_loss = (
                close - self.stop_atr_multiple * atr
                if signal_type == "LONG"
                else close + self.stop_atr_multiple * atr
            )
        if self.trail_atr_multiple > 0.0:
            trailing_percent = self.trail_atr_multiple * atr / max(close, _EPS)
        metadata = _target_metadata(
            strategy="VolatilitySqueezeBreakoutStrategy",
            target_allocation=self.target_allocation,
            max_order_value=self.max_order_value,
            compression_ratio=short_range / base_range,
            volume_z=vol_z,
            momentum=mom,
            atr=atr,
            target_mode=signal_type,
        )
        _emit(
            self.events,
            strategy_id="volatility_squeeze_breakout",
            symbol=symbol,
            event_time=snapshot.time,
            signal_type=signal_type,
            strength=max(0.25, min(3.0, abs(mom) / max(self.momentum_threshold, _EPS))),
            price=close,
            stop_loss=stop_loss,
            trailing_percent=trailing_percent,
            metadata=metadata,
        )
        item.mode = signal_type
        item.entry_price = close
        item.entry_atr = atr
        item.high_watermark = close
        item.low_watermark = close
        item.bars_held = 0


@register("strategy", "TakerFlowImbalanceContinuationStrategy", interface="event_driven")
class TakerFlowImbalanceContinuationStrategy(Strategy):
    """Follow short-horizon aggressive taker-flow imbalance when price confirms."""

    decision_cadence_seconds = 60
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False
    required_features = ("taker_buy_quote_volume", "taker_sell_quote_volume")

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "flow_window": HyperParam.integer("flow_window", default=12, low=1, high=1440),
            "price_confirm_bars": HyperParam.integer(
                "price_confirm_bars", default=6, low=1, high=1440
            ),
            "imbalance_entry": HyperParam.floating(
                "imbalance_entry", default=0.22, low=0.0, high=1.0
            ),
            "imbalance_exit": HyperParam.floating(
                "imbalance_exit", default=0.04, low=0.0, high=1.0
            ),
            "min_price_move": HyperParam.floating(
                "min_price_move", default=0.001, low=0.0, high=1.0
            ),
            "min_volume_z": HyperParam.floating("min_volume_z", default=-0.5, low=-5.0, high=20.0),
            "volume_window": HyperParam.integer("volume_window", default=96, low=8, high=4096),
            "stop_loss_pct": HyperParam.floating(
                "stop_loss_pct", default=0.018, low=0.0, high=0.50
            ),
            "take_profit_pct": HyperParam.floating(
                "take_profit_pct", default=0.045, low=0.0, high=1.0
            ),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=96, low=1, high=100000),
            "target_allocation": HyperParam.floating(
                "target_allocation", default=0.010, low=0.0, high=2.0, tunable=False
            ),
            "max_order_value": HyperParam.floating(
                "max_order_value", default=250.0, low=0.0, high=1_000_000.0, tunable=False
            ),
            "min_price": HyperParam.floating("min_price", default=0.10, low=0.0, high=1_000_000.0),
        }

    def __init__(self, bars: Any, events: Any, **params: Any) -> None:
        self.bars = bars
        self.events = events
        self.symbol_list = list(getattr(self.bars, "symbol_list", []) or [])
        resolved = resolve_params_from_schema(self.get_param_schema(), params, keep_unknown=False)
        self.flow_window = max(1, int(resolved["flow_window"]))
        self.price_confirm_bars = max(1, int(resolved["price_confirm_bars"]))
        self.imbalance_entry = max(0.0, min(1.0, float(resolved["imbalance_entry"])))
        self.imbalance_exit = max(0.0, min(1.0, float(resolved["imbalance_exit"])))
        self.min_price_move = max(0.0, float(resolved["min_price_move"]))
        self.min_volume_z = float(resolved["min_volume_z"])
        self.volume_window = max(3, int(resolved["volume_window"]))
        self.stop_loss_pct = max(0.0, float(resolved["stop_loss_pct"]))
        self.take_profit_pct = max(0.0, float(resolved["take_profit_pct"]))
        self.allow_short = bool(resolved["allow_short"])
        self.max_hold_bars = max(1, int(resolved["max_hold_bars"]))
        self.target_allocation = max(0.0, float(resolved["target_allocation"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        size = (
            max(self.flow_window, self.price_confirm_bars, self.volume_window, self.max_hold_bars)
            + 8
        )
        self._state = {
            symbol: _FlowState(
                deque(maxlen=size),
                deque(maxlen=size),
                deque(maxlen=size),
                deque(maxlen=size),
                deque(maxlen=size),
            )
            for symbol in self.symbol_list
        }

    def get_state(self) -> dict[str, Any]:
        return {
            "symbol_state": {
                symbol: {
                    "closes": list(item.closes),
                    "volumes": list(item.volumes),
                    "buy_quote": list(item.buy_quote),
                    "sell_quote": list(item.sell_quote),
                    "imbalances": list(item.imbalances),
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
            for attr in ("closes", "volumes", "buy_quote", "sell_quote", "imbalances"):
                _restore_deque(getattr(item, attr), payload.get(attr))
            item.mode = _mode(payload.get("mode"))
            item.entry_price = safe_float(payload.get("entry_price"))
            item.bars_held = _safe_non_negative_int(payload.get("bars_held"))
            item.last_time_key = str(payload.get("last_time_key", ""))

    def calculate_signals_window(self, event: Any, aggregator: Any = None) -> None:
        _ = aggregator
        for symbol in _event_symbols(event, self.symbol_list):
            snapshot = _window_snapshot(event, symbol)
            if snapshot is not None:
                self._process_symbol(event, symbol, snapshot)

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
                self._process_symbol(event, str(symbol), snapshot)

    def _exit_if_needed(
        self, symbol: str, item: _FlowState, snapshot: _Snapshot, imbalance: float | None
    ) -> None:
        close = safe_float(snapshot.close)
        if item.mode not in {"LONG", "SHORT"} or item.entry_price is None or close is None:
            return
        item.bars_held += 1
        reason = ""
        if item.mode == "LONG":
            if self.stop_loss_pct > 0.0 and close <= item.entry_price * (1.0 - self.stop_loss_pct):
                reason = "stop_loss"
            elif self.take_profit_pct > 0.0 and close >= item.entry_price * (
                1.0 + self.take_profit_pct
            ):
                reason = "take_profit"
            elif imbalance is not None and imbalance <= self.imbalance_exit:
                reason = "flow_faded"
        else:
            if self.stop_loss_pct > 0.0 and close >= item.entry_price * (1.0 + self.stop_loss_pct):
                reason = "stop_loss"
            elif self.take_profit_pct > 0.0 and close <= item.entry_price * (
                1.0 - self.take_profit_pct
            ):
                reason = "take_profit"
            elif imbalance is not None and imbalance >= -self.imbalance_exit:
                reason = "flow_faded"
        if not reason and item.bars_held >= self.max_hold_bars:
            reason = "max_hold"
        if not reason:
            return
        _emit(
            self.events,
            strategy_id="taker_flow_imbalance_continuation",
            symbol=symbol,
            event_time=snapshot.time,
            signal_type="EXIT",
            price=close,
            metadata={
                "strategy": "TakerFlowImbalanceContinuationStrategy",
                "reason": reason,
                "imbalance": imbalance,
            },
        )
        item.mode = "OUT"
        item.entry_price = None
        item.bars_held = 0

    def _process_symbol(self, event: Any, symbol: str, snapshot: _Snapshot) -> None:
        item = self._state[symbol]
        key = time_key(snapshot.time)
        if key and key == item.last_time_key:
            return
        item.last_time_key = key
        close = safe_float(snapshot.close)
        if close is None or close <= self.min_price:
            return
        buy = _feature_value(self.bars, event, symbol, "taker_buy_quote_volume")
        sell = _feature_value(self.bars, event, symbol, "taker_sell_quote_volume")
        if buy is None:
            buy = _feature_value(self.bars, event, symbol, "taker_buy_base_volume")
            if buy is not None:
                buy *= close
        if sell is None:
            sell = _feature_value(self.bars, event, symbol, "taker_sell_base_volume")
            if sell is not None:
                sell *= close
        if buy is None or sell is None:
            return
        item.closes.append(close)
        item.volumes.append(max(0.0, float(snapshot.volume or 0.0)))
        item.buy_quote.append(max(0.0, float(buy)))
        item.sell_quote.append(max(0.0, float(sell)))
        imbalance = order_flow_imbalance(item.buy_quote, item.sell_quote, window=self.flow_window)
        if imbalance is not None:
            item.imbalances.append(imbalance)
        self._exit_if_needed(symbol, item, snapshot, imbalance)
        if item.mode != "OUT" or imbalance is None:
            return
        ret = log_return(item.closes, lookback=self.price_confirm_bars)
        if ret is None:
            return
        vol_z = volume_zscore(item.volumes, window=self.volume_window)
        if vol_z is not None and vol_z < self.min_volume_z:
            return
        signal_type = ""
        if imbalance >= self.imbalance_entry and ret >= self.min_price_move:
            signal_type = "LONG"
        elif (
            self.allow_short and imbalance <= -self.imbalance_entry and ret <= -self.min_price_move
        ):
            signal_type = "SHORT"
        if not signal_type:
            return
        stop_loss = None
        take_profit = None
        if self.stop_loss_pct > 0.0:
            stop_loss = close * (
                1.0 - self.stop_loss_pct if signal_type == "LONG" else 1.0 + self.stop_loss_pct
            )
        if self.take_profit_pct > 0.0:
            take_profit = close * (
                1.0 + self.take_profit_pct if signal_type == "LONG" else 1.0 - self.take_profit_pct
            )
        metadata = _target_metadata(
            strategy="TakerFlowImbalanceContinuationStrategy",
            target_allocation=self.target_allocation,
            max_order_value=self.max_order_value,
            imbalance=imbalance,
            price_confirmation=ret,
            volume_z=vol_z,
            target_mode=signal_type,
        )
        _emit(
            self.events,
            strategy_id="taker_flow_imbalance_continuation",
            symbol=symbol,
            event_time=snapshot.time,
            signal_type=signal_type,
            strength=max(0.25, min(3.0, abs(imbalance) / max(self.imbalance_entry, _EPS))),
            price=close,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata=metadata,
        )
        item.mode = signal_type
        item.entry_price = close
        item.bars_held = 0


@register("strategy", "PairsSpreadMeanReversionStrategy", interface="event_driven")
class PairsSpreadMeanReversionStrategy(Strategy):
    """Beta-adjusted pair-spread mean reversion for related instruments."""

    decision_cadence_seconds = 60
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "pairs": HyperParam.string("pairs", default="", tunable=False),
            "spread_window": HyperParam.integer("spread_window", default=240, low=20, high=20000),
            "hedge_window": HyperParam.integer("hedge_window", default=240, low=20, high=20000),
            "entry_z": HyperParam.floating("entry_z", default=2.0, low=0.25, high=10.0),
            "exit_z": HyperParam.floating("exit_z", default=0.35, low=0.0, high=5.0),
            "stop_z": HyperParam.floating("stop_z", default=4.0, low=0.5, high=20.0),
            "beta_floor": HyperParam.floating(
                "beta_floor", default=0.25, low=0.01, high=10.0, tunable=False
            ),
            "beta_cap": HyperParam.floating(
                "beta_cap", default=4.0, low=0.01, high=20.0, tunable=False
            ),
            "rebalance_bars": HyperParam.integer(
                "rebalance_bars", default=1, low=1, high=10080, tunable=False
            ),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=1440, low=1, high=200000),
            "target_allocation": HyperParam.floating(
                "target_allocation", default=0.012, low=0.0, high=2.0, tunable=False
            ),
            "max_order_value": HyperParam.floating(
                "max_order_value", default=300.0, low=0.0, high=1_000_000.0, tunable=False
            ),
            "min_price": HyperParam.floating("min_price", default=0.10, low=0.0, high=1_000_000.0),
        }

    def __init__(self, bars: Any, events: Any, **params: Any) -> None:
        self.bars = bars
        self.events = events
        self.symbol_list = list(getattr(self.bars, "symbol_list", []) or [])
        resolved = resolve_params_from_schema(self.get_param_schema(), params, keep_unknown=False)
        self.spread_window = max(3, int(resolved["spread_window"]))
        self.hedge_window = max(3, int(resolved["hedge_window"]))
        self.entry_z = max(0.0, float(resolved["entry_z"]))
        self.exit_z = max(0.0, float(resolved["exit_z"]))
        self.stop_z = max(self.entry_z, float(resolved["stop_z"]))
        self.beta_floor = max(0.01, float(resolved["beta_floor"]))
        self.beta_cap = max(self.beta_floor, float(resolved["beta_cap"]))
        self.rebalance_bars = max(1, int(resolved["rebalance_bars"]))
        self.max_hold_bars = max(1, int(resolved["max_hold_bars"]))
        self.target_allocation = max(0.0, float(resolved["target_allocation"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        self._latest_close: dict[str, float] = {}
        self._last_seen: dict[str, str] = {}
        self._last_eval_time_key = ""
        self._tick = 0
        size = max(self.spread_window, self.hedge_window, self.max_hold_bars) + 8
        pairs = self._parse_pairs(str(resolved["pairs"] or ""))
        if not pairs and len(self.symbol_list) >= 2:
            pairs = [(self.symbol_list[0], self.symbol_list[1])]
        self._pairs = {
            f"{a}|{b}": _PairState(a, b, deque(maxlen=size), deque(maxlen=size), deque(maxlen=size))
            for a, b in pairs
            if a in self.symbol_list and b in self.symbol_list and a != b
        }

    @staticmethod
    def _parse_pairs(raw: str) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        for chunk in raw.split(","):
            text = chunk.strip()
            if not text or ":" not in text:
                continue
            a, b = [part.strip() for part in text.split(":", 1)]
            if a and b and a != b:
                pairs.append((a, b))
        return pairs

    def get_state(self) -> dict[str, Any]:
        return {
            "last_eval_time_key": self._last_eval_time_key,
            "tick": int(self._tick),
            "latest_close": dict(self._latest_close),
            "last_seen": dict(self._last_seen),
            "pairs": {
                key: {
                    "closes_a": list(pair.closes_a),
                    "closes_b": list(pair.closes_b),
                    "spreads": list(pair.spreads),
                    "mode": pair.mode,
                    "beta": pair.beta,
                    "entry_z": pair.entry_z,
                    "bars_held": int(pair.bars_held),
                    "last_eval_key": pair.last_eval_key,
                }
                for key, pair in self._pairs.items()
            },
        }

    def set_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        self._last_eval_time_key = str(state.get("last_eval_time_key", ""))
        self._tick = _safe_non_negative_int(state.get("tick"))
        self._latest_close = {
            str(k): float(v)
            for k, v in dict(state.get("latest_close") or {}).items()
            if safe_float(v) is not None
        }
        self._last_seen = {str(k): str(v) for k, v in dict(state.get("last_seen") or {}).items()}
        raw_pairs = state.get("pairs")
        if not isinstance(raw_pairs, dict):
            return
        for key, payload in raw_pairs.items():
            if key not in self._pairs or not isinstance(payload, dict):
                continue
            pair = self._pairs[key]
            _restore_deque(pair.closes_a, payload.get("closes_a"))
            _restore_deque(pair.closes_b, payload.get("closes_b"))
            _restore_deque(pair.spreads, payload.get("spreads"))
            pair.mode = _mode(payload.get("mode"))
            pair.beta = float(safe_float(payload.get("beta")) or 1.0)
            pair.entry_z = safe_float(payload.get("entry_z"))
            pair.bars_held = _safe_non_negative_int(payload.get("bars_held"))
            pair.last_eval_key = str(payload.get("last_eval_key", ""))

    def calculate_signals_window(self, event: Any, aggregator: Any = None) -> None:
        _ = aggregator
        updated = False
        event_key = time_key(getattr(event, "time", None))
        for symbol in _event_symbols(event, self.symbol_list):
            snapshot = _window_snapshot(event, symbol)
            if snapshot is not None and self._update_symbol(symbol, snapshot):
                updated = True
        if updated and event_key and event_key != self._last_eval_time_key:
            self._last_eval_time_key = event_key
            self._tick += 1
            self._evaluate_pairs(getattr(event, "time", None), event_key)

    def calculate_signals(self, event: Any) -> None:
        if str(getattr(event, "type", "")).upper() == "MARKET_WINDOW":
            self.calculate_signals_window(event, None)
            return
        if getattr(event, "type", None) != "MARKET":
            return
        symbol = getattr(event, "symbol", None)
        if symbol not in self.symbol_list:
            return
        snapshot = _market_snapshot(event)
        if snapshot is None:
            return
        if self._update_symbol(str(symbol), snapshot):
            key = time_key(snapshot.time)
            if key and key != self._last_eval_time_key:
                self._last_eval_time_key = key
                self._tick += 1
            self._evaluate_pairs(snapshot.time, key)

    def _update_symbol(self, symbol: str, snapshot: _Snapshot) -> bool:
        close = safe_float(snapshot.close)
        if close is None or close <= self.min_price:
            return False
        key = time_key(snapshot.time)
        if key and key == self._last_seen.get(symbol):
            return False
        self._last_seen[symbol] = key
        self._latest_close[symbol] = close
        for pair in self._pairs.values():
            if pair.symbol_a == symbol:
                pair.closes_a.append(close)
            if pair.symbol_b == symbol:
                pair.closes_b.append(close)
        return True

    def _emit_pair_leg(
        self,
        pair: _PairState,
        event_time: Any,
        symbol: str,
        signal_type: str,
        close: float | None,
        zscore: float,
        reason: str,
        leg: str,
        beta: float,
    ) -> None:
        metadata = _target_metadata(
            strategy="PairsSpreadMeanReversionStrategy",
            target_allocation=self.target_allocation / 2.0,
            max_order_value=self.max_order_value,
            pair=f"{pair.symbol_a}:{pair.symbol_b}",
            zscore=zscore,
            beta=beta,
            reason=reason,
            leg=leg,
        )
        _emit(
            self.events,
            strategy_id="pairs_spread_mean_reversion",
            symbol=symbol,
            event_time=event_time,
            signal_type=signal_type,
            strength=max(0.25, min(3.0, abs(zscore) / max(self.entry_z, _EPS))),
            price=close,
            metadata=metadata,
        )

    def _exit_pair(self, pair: _PairState, event_time: Any, zscore: float, reason: str) -> None:
        self._emit_pair_leg(
            pair,
            event_time,
            pair.symbol_a,
            "EXIT",
            self._latest_close.get(pair.symbol_a),
            zscore,
            reason,
            "a",
            pair.beta,
        )
        self._emit_pair_leg(
            pair,
            event_time,
            pair.symbol_b,
            "EXIT",
            self._latest_close.get(pair.symbol_b),
            zscore,
            reason,
            "b",
            pair.beta,
        )
        pair.mode = "OUT"
        pair.entry_z = None
        pair.bars_held = 0

    def _enter_pair(self, pair: _PairState, event_time: Any, mode: str, zscore: float) -> None:
        if mode == "LONG_SPREAD":
            a_signal, b_signal = "LONG", "SHORT"
        else:
            a_signal, b_signal = "SHORT", "LONG"
        self._emit_pair_leg(
            pair,
            event_time,
            pair.symbol_a,
            a_signal,
            self._latest_close.get(pair.symbol_a),
            zscore,
            "spread_entry",
            "a",
            pair.beta,
        )
        self._emit_pair_leg(
            pair,
            event_time,
            pair.symbol_b,
            b_signal,
            self._latest_close.get(pair.symbol_b),
            zscore,
            "spread_entry",
            "b",
            pair.beta,
        )
        pair.mode = mode
        pair.entry_z = zscore
        pair.bars_held = 0

    def _evaluate_pairs(self, event_time: Any, event_key: str = "") -> None:
        if self._tick % self.rebalance_bars:
            return
        for pair in self._pairs.values():
            if event_key:
                if pair.last_eval_key == event_key:
                    continue
                if self._last_seen.get(pair.symbol_a) != event_key:
                    continue
                if self._last_seen.get(pair.symbol_b) != event_key:
                    continue
                pair.last_eval_key = event_key
            beta = _rolling_beta(
                pair.closes_a, pair.closes_b, self.hedge_window, self.beta_floor, self.beta_cap
            )
            if beta is None:
                beta = pair.beta
            pair.beta = beta
            zscore, _ = _spread_zscore(pair, self.spread_window, beta)
            if zscore is None:
                continue
            if pair.mode != "OUT":
                pair.bars_held += 1
                if abs(zscore) <= self.exit_z:
                    self._exit_pair(pair, event_time, zscore, "mean_reverted")
                elif abs(zscore) >= self.stop_z:
                    self._exit_pair(pair, event_time, zscore, "spread_stop")
                elif pair.bars_held >= self.max_hold_bars:
                    self._exit_pair(pair, event_time, zscore, "max_hold")
                continue
            if zscore >= self.entry_z:
                self._enter_pair(pair, event_time, "SHORT_SPREAD", zscore)
            elif zscore <= -self.entry_z:
                self._enter_pair(pair, event_time, "LONG_SPREAD", zscore)


@register("strategy", "CrossSectionalShortTermReversalStrategy", interface="event_driven")
class CrossSectionalShortTermReversalStrategy(Strategy):
    """Liquidity-provision style cross-sectional short-term reversal sleeve."""

    decision_cadence_seconds = 60
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "lookback_bars": HyperParam.integer("lookback_bars", default=24, low=2, high=4096),
            "vol_window": HyperParam.integer("vol_window", default=96, low=4, high=4096),
            "rebalance_bars": HyperParam.integer("rebalance_bars", default=12, low=1, high=10080),
            "signal_threshold": HyperParam.floating(
                "signal_threshold", default=0.45, low=0.0, high=20.0
            ),
            "max_longs": HyperParam.integer("max_longs", default=3, low=0, high=128),
            "max_shorts": HyperParam.integer("max_shorts", default=3, low=0, high=128),
            "max_trend_efficiency": HyperParam.floating(
                "max_trend_efficiency", default=0.55, low=0.0, high=1.0
            ),
            "min_volume_z": HyperParam.floating("min_volume_z", default=-1.0, low=-5.0, high=20.0),
            "volume_window": HyperParam.integer("volume_window", default=96, low=8, high=4096),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
            "stop_loss_pct": HyperParam.floating(
                "stop_loss_pct", default=0.035, low=0.0, high=0.50
            ),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=96, low=1, high=100000),
            "target_gross_exposure": HyperParam.floating(
                "target_gross_exposure", default=0.36, low=0.0, high=5.0, tunable=False
            ),
            "max_order_value": HyperParam.floating(
                "max_order_value", default=300.0, low=0.0, high=1_000_000.0, tunable=False
            ),
            "min_price": HyperParam.floating("min_price", default=0.10, low=0.0, high=1_000_000.0),
        }

    def __init__(self, bars: Any, events: Any, **params: Any) -> None:
        self.bars = bars
        self.events = events
        self.symbol_list = list(getattr(self.bars, "symbol_list", []) or [])
        resolved = resolve_params_from_schema(self.get_param_schema(), params, keep_unknown=False)
        self.lookback_bars = max(1, int(resolved["lookback_bars"]))
        self.vol_window = max(2, int(resolved["vol_window"]))
        self.rebalance_bars = max(1, int(resolved["rebalance_bars"]))
        self.signal_threshold = max(0.0, float(resolved["signal_threshold"]))
        self.max_longs = max(0, int(resolved["max_longs"]))
        self.max_shorts = max(0, int(resolved["max_shorts"]))
        self.max_trend_efficiency = max(0.0, min(1.0, float(resolved["max_trend_efficiency"])))
        self.min_volume_z = float(resolved["min_volume_z"])
        self.volume_window = max(3, int(resolved["volume_window"]))
        self.allow_short = bool(resolved["allow_short"])
        self.stop_loss_pct = max(0.0, float(resolved["stop_loss_pct"]))
        self.max_hold_bars = max(1, int(resolved["max_hold_bars"]))
        self.target_gross_exposure = max(0.0, float(resolved["target_gross_exposure"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        size = max(self.lookback_bars, self.vol_window, self.volume_window, self.max_hold_bars) + 8
        self._state = {
            symbol: _CrossSectionalState(deque(maxlen=size), deque(maxlen=size))
            for symbol in self.symbol_list
        }
        self._last_eval_time_key = ""
        self._tick = 0

    def get_state(self) -> dict[str, Any]:
        return {
            "last_eval_time_key": self._last_eval_time_key,
            "tick": int(self._tick),
            "symbol_state": {
                symbol: {
                    "closes": list(item.closes),
                    "volumes": list(item.volumes),
                    "mode": item.mode,
                    "entry_price": item.entry_price,
                    "bars_held": int(item.bars_held),
                    "last_time_key": item.last_time_key,
                }
                for symbol, item in self._state.items()
            },
        }

    def set_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        self._last_eval_time_key = str(state.get("last_eval_time_key", ""))
        self._tick = _safe_non_negative_int(state.get("tick"))
        raw = state.get("symbol_state")
        if not isinstance(raw, dict):
            return
        for symbol, payload in raw.items():
            if symbol not in self._state or not isinstance(payload, dict):
                continue
            item = self._state[symbol]
            _restore_deque(item.closes, payload.get("closes"))
            _restore_deque(item.volumes, payload.get("volumes"))
            item.mode = _mode(payload.get("mode"))
            item.entry_price = safe_float(payload.get("entry_price"))
            item.bars_held = _safe_non_negative_int(payload.get("bars_held"))
            item.last_time_key = str(payload.get("last_time_key", ""))

    def calculate_signals_window(self, event: Any, aggregator: Any = None) -> None:
        _ = aggregator
        updated = False
        event_key = time_key(getattr(event, "time", None))
        for symbol in _event_symbols(event, self.symbol_list):
            snapshot = _window_snapshot(event, symbol)
            if snapshot is not None and self._update_symbol(symbol, snapshot):
                updated = True
        if updated and event_key and event_key != self._last_eval_time_key:
            self._last_eval_time_key = event_key
            self._tick += 1
            self._rebalance(getattr(event, "time", None))

    def calculate_signals(self, event: Any) -> None:
        if str(getattr(event, "type", "")).upper() == "MARKET_WINDOW":
            self.calculate_signals_window(event, None)
            return
        if getattr(event, "type", None) != "MARKET":
            return
        symbol = getattr(event, "symbol", None)
        if symbol not in self._state:
            return
        snapshot = _market_snapshot(event)
        if snapshot is not None and self._update_symbol(str(symbol), snapshot):
            key = time_key(snapshot.time)
            if key and key != self._last_eval_time_key:
                self._last_eval_time_key = key
                self._tick += 1
                self._rebalance(snapshot.time)

    def _update_symbol(self, symbol: str, snapshot: _Snapshot) -> bool:
        close = safe_float(snapshot.close)
        if close is None or close <= self.min_price:
            return False
        item = self._state[symbol]
        key = time_key(snapshot.time)
        if key and key == item.last_time_key:
            return False
        item.last_time_key = key
        item.closes.append(close)
        item.volumes.append(max(0.0, float(snapshot.volume or 0.0)))
        return True

    def _score_symbol(self, symbol: str) -> tuple[float, float, float] | None:
        item = self._state[symbol]
        ret = simple_return(item.closes, lookback=self.lookback_bars)
        vol = realized_volatility(item.closes, window=self.vol_window)
        eff = trend_efficiency(
            item.closes,
            window=max(self.lookback_bars, min(self.vol_window, self.lookback_bars * 2)),
        )
        if ret is None or vol is None or vol <= _EPS:
            return None
        if eff is not None and eff > self.max_trend_efficiency:
            return None
        vol_z = volume_zscore(item.volumes, window=self.volume_window)
        if vol_z is not None and vol_z < self.min_volume_z:
            return None
        score = -ret / max(vol, _EPS)
        return float(score), float(ret), float(vol)

    def _emit_exit(
        self, symbol: str, item: _CrossSectionalState, event_time: Any, reason: str
    ) -> None:
        price = item.closes[-1] if item.closes else None
        _emit(
            self.events,
            strategy_id="cross_sectional_short_term_reversal",
            symbol=symbol,
            event_time=event_time,
            signal_type="EXIT",
            price=price,
            metadata={"strategy": "CrossSectionalShortTermReversalStrategy", "reason": reason},
        )
        item.mode = "OUT"
        item.entry_price = None
        item.bars_held = 0

    def _rebalance(self, event_time: Any) -> None:
        rows: list[tuple[float, str, float, float]] = []
        for symbol in self.symbol_list:
            scored = self._score_symbol(symbol)
            if scored is not None:
                score, ret, vol = scored
                rows.append((score, symbol, ret, vol))
        if self._tick % self.rebalance_bars:
            self._age_positions(event_time)
            return
        long_rows = [row for row in sorted(rows, reverse=True) if row[0] >= self.signal_threshold][
            : self.max_longs
        ]
        short_rows = []
        if self.allow_short:
            short_rows = [row for row in sorted(rows) if row[0] <= -self.signal_threshold][
                : self.max_shorts
            ]
        targets: dict[str, tuple[str, float, float, float]] = {}
        selected_count = len(long_rows) + len(short_rows)
        base_alloc = self.target_gross_exposure / float(selected_count) if selected_count else 0.0
        for score, symbol, ret, vol in long_rows:
            targets[symbol] = ("LONG", score, ret, vol)
        for score, symbol, ret, vol in short_rows:
            targets[symbol] = ("SHORT", score, ret, vol)
        for symbol, item in self._state.items():
            target = targets.get(symbol)
            price = item.closes[-1] if item.closes else None
            if target is None:
                if item.mode != "OUT":
                    self._emit_exit(symbol, item, event_time, "rebalance_removed")
                continue
            target_mode, score, ret, vol = target
            if item.mode == target_mode:
                item.bars_held += 1
                continue
            if item.mode != "OUT":
                self._emit_exit(symbol, item, event_time, "side_flip")
            stop_loss = None
            if price is not None and self.stop_loss_pct > 0.0:
                stop_loss = price * (
                    1.0 - self.stop_loss_pct if target_mode == "LONG" else 1.0 + self.stop_loss_pct
                )
            metadata = _target_metadata(
                strategy="CrossSectionalShortTermReversalStrategy",
                target_allocation=base_alloc,
                max_order_value=self.max_order_value,
                score=score,
                lookback_return=ret,
                realized_vol=vol,
                target_mode=target_mode,
            )
            _emit(
                self.events,
                strategy_id="cross_sectional_short_term_reversal",
                symbol=symbol,
                event_time=event_time,
                signal_type=target_mode,
                strength=max(0.25, min(3.0, abs(score) / max(self.signal_threshold, _EPS))),
                price=price,
                stop_loss=stop_loss,
                metadata=metadata,
            )
            item.mode = target_mode
            item.entry_price = price
            item.bars_held = 0

    def _age_positions(self, event_time: Any) -> None:
        for symbol, item in self._state.items():
            if item.mode == "OUT":
                continue
            item.bars_held += 1
            price = item.closes[-1] if item.closes else None
            if price is not None and item.entry_price is not None and self.stop_loss_pct > 0.0:
                if item.mode == "LONG" and price <= item.entry_price * (1.0 - self.stop_loss_pct):
                    self._emit_exit(symbol, item, event_time, "stop_loss")
                    continue
                if item.mode == "SHORT" and price >= item.entry_price * (1.0 + self.stop_loss_pct):
                    self._emit_exit(symbol, item, event_time, "stop_loss")
                    continue
            if item.bars_held >= self.max_hold_bars:
                self._emit_exit(symbol, item, event_time, "max_hold")


__all__ = [
    "CrossSectionalShortTermReversalStrategy",
    "DonchianAtrTrendStrategy",
    "PairsSpreadMeanReversionStrategy",
    "TakerFlowImbalanceContinuationStrategy",
    "VolatilitySqueezeBreakoutStrategy",
]

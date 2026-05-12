"""Crypto/FX Alpha Zoo state strategy.

v0 trades crypto only and uses FX as a risk-regime filter.  It is deliberately
live opt-in and calibration-gated: without a positive lower-confidence edge for
(symbol, side) or a default side bucket, entries fail closed.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from statistics import mean, stdev
from typing import Any

from lumina_quant.alpha_zoo.crypto_fx_factors import infer_market, normalize_symbol
from lumina_quant.core.events import MarketBatchEvent, MarketWindowEvent, SignalEvent
from lumina_quant.indicators.common import safe_float, safe_int, time_key
from lumina_quant.strategy import Strategy
from lumina_quant.tuning import HyperParam, resolve_params_from_schema

CRYPTO_DEFAULTS = ("BTC", "ETH", "SOL", "BNB", "TRX")
FX_DEFAULTS = ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF")
MIN_ZSCORE_OBS = 2 + 1
WINDOW_TUPLE_MIN_FIELDS = 4 + 2
WINDOW_TUPLE_OPEN_IDX = 1
WINDOW_TUPLE_HIGH_IDX = 2
WINDOW_TUPLE_LOW_IDX = 2 + 1
WINDOW_TUPLE_CLOSE_IDX = 4
WINDOW_TUPLE_VOLUME_IDX = 4 + 1


@dataclass(slots=True)
class _SymbolState:
    opens: deque[float]
    highs: deque[float]
    lows: deque[float]
    closes: deque[float]
    volumes: deque[float]
    mode: str = "OUT"
    entry_price: float = 0.0
    bars_held: int = 0
    last_time_key: str = ""


def _ret(values: deque[float], window: int) -> float | None:
    w = max(1, int(window))
    if len(values) <= w:
        return None
    prev = float(list(values)[-1 - w])
    curr = float(values[-1])
    if prev <= 0.0 or curr <= 0.0:
        return None
    return (curr / prev) - 1.0


def _returns(values: deque[float], window: int) -> list[float]:
    seq = list(values)
    w = max(1, int(window))
    out: list[float] = []
    for idx in range(w, len(seq)):
        before = float(seq[idx - w])
        after = float(seq[idx])
        if before > 0.0 and after > 0.0:
            out.append((after / before) - 1.0)
    return out


def _zscore(value: float, sample: list[float]) -> float:
    clean = [float(item) for item in sample if math.isfinite(float(item))]
    if len(clean) < MIN_ZSCORE_OBS:
        return 0.0
    sigma = stdev(clean)
    if sigma <= 1e-12:
        return 0.0
    return (float(value) - mean(clean)) / sigma


def _trend_efficiency(values: deque[float], window: int) -> float:
    seq = list(values)
    w = max(2, int(window))
    if len(seq) <= w:
        return 0.0
    tail = seq[-w - 1 :]
    net = abs(float(tail[-1]) - float(tail[0]))
    path = sum(abs(float(tail[idx]) - float(tail[idx - 1])) for idx in range(1, len(tail)))
    return 0.0 if path <= 1e-12 else net / path


def _volume_z(volumes: deque[float], window: int) -> float:
    if not volumes:
        return 0.0
    seq = list(volumes)[-max(MIN_ZSCORE_OBS, int(window)) :]
    return _zscore(float(seq[-1]), seq[:-1]) if len(seq) >= 4 else 0.0


class CryptoFxAlphaZooStateStrategy(Strategy):
    """Trade crypto residual/factor scores with FX regime and edge gates."""

    strategy_id = "crypto_fx_alpha_zoo_state"
    decision_cadence_seconds = 64 - 4
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False
    required_inputs = ("OHLCV",)
    required_features = (
        "crypto_residual_momentum",
        "crypto_residual_reversal",
        "volume_vwap_pressure",
        "breakout_failure",
        "fx_usd_jpy_risk_regime",
        "calibrated_edge_lower_bound",
    )
    strategy_validity = {
        "pass": True,
        "calendar_primary": False,
        "primary_signal_type": "formulaic_state_factor",
        "causal_state_only": True,
        "lookahead_safe": True,
        "locked_oos_role": "gate_report_only",
        "rejection_reasons": [],
    }

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "fast_lookback_bars": HyperParam.integer("fast_lookback_bars", default=4, low=1, high=2880, tunable=False),
            "slow_lookback_bars": HyperParam.integer("slow_lookback_bars", default=24, low=2, high=10080, tunable=False),
            "history_window": HyperParam.integer("history_window", default=96, low=16, high=20000, tunable=False),
            "entry_threshold": HyperParam.floating("entry_threshold", default=0.75, low=0.0, high=10.0, tunable=False),
            "exit_threshold": HyperParam.floating("exit_threshold", default=0.20, low=0.0, high=10.0, tunable=False),
            "stop_loss_pct": HyperParam.floating("stop_loss_pct", default=0.035, low=0.0, high=1.0, tunable=False),
            "take_profit_pct": HyperParam.floating("take_profit_pct", default=0.075, low=0.0, high=2.0, tunable=False),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=96, low=1, high=10080, tunable=False),
            "max_longs": HyperParam.integer("max_longs", default=2, low=0, high=16, tunable=False),
            "max_shorts": HyperParam.integer("max_shorts", default=2, low=0, high=16, tunable=False),
            "allow_shorts": HyperParam.boolean("allow_shorts", default=True, tunable=False),
            "use_fx_filter": HyperParam.boolean("use_fx_filter", default=True, tunable=False),
            "require_calibrated_edge": HyperParam.boolean("require_calibrated_edge", default=True, tunable=False),
            "min_calibrated_edge_bps": HyperParam.floating("min_calibrated_edge_bps", default=0.0, low=-10000.0, high=10000.0, tunable=False),
            "risk_off_threshold": HyperParam.floating("risk_off_threshold", default=0.015, low=0.0, high=1.0, tunable=False),
            "risk_on_threshold": HyperParam.floating("risk_on_threshold", default=-0.010, low=-1.0, high=0.0, tunable=False),
            "residual_momentum_weight": HyperParam.floating("residual_momentum_weight", default=0.35, low=0.0, high=1.0, tunable=False),
            "residual_reversal_weight": HyperParam.floating("residual_reversal_weight", default=0.25, low=0.0, high=1.0, tunable=False),
            "vwap_pressure_weight": HyperParam.floating("vwap_pressure_weight", default=0.15, low=0.0, high=1.0, tunable=False),
            "breakout_failure_weight": HyperParam.floating("breakout_failure_weight", default=0.15, low=0.0, high=1.0, tunable=False),
            "trend_efficiency_weight": HyperParam.floating("trend_efficiency_weight", default=0.10, low=0.0, high=1.0, tunable=False),
            "risk_off_long_multiplier": HyperParam.floating("risk_off_long_multiplier", default=0.25, low=0.0, high=2.0, tunable=False),
            "risk_off_short_multiplier": HyperParam.floating("risk_off_short_multiplier", default=1.15, low=0.0, high=2.0, tunable=False),
            "risk_on_long_multiplier": HyperParam.floating("risk_on_long_multiplier", default=1.10, low=0.0, high=2.0, tunable=False),
            "risk_on_short_multiplier": HyperParam.floating("risk_on_short_multiplier", default=0.50, low=0.0, high=2.0, tunable=False),
            "min_signal_strength": HyperParam.floating("min_signal_strength", default=0.10, low=0.0, high=1.0, tunable=False),
        }

    def __init__(
        self,
        bars: Any,
        events: Any,
        *,
        calibrated_edges: Mapping[str, float] | None = None,
        crypto_symbols: tuple[str, ...] | None = None,
        fx_symbols: tuple[str, ...] | None = None,
        **params: Any,
    ) -> None:
        self.bars = bars
        self.events = events
        self.symbol_list = list(getattr(bars, "symbol_list", []) or [])
        if not self.symbol_list:
            raise ValueError("CryptoFxAlphaZooStateStrategy requires at least one symbol")
        resolved = resolve_params_from_schema(self.get_param_schema(), params, keep_unknown=True)
        self.fast_lookback_bars = max(1, int(resolved["fast_lookback_bars"]))
        self.slow_lookback_bars = max(self.fast_lookback_bars + 1, int(resolved["slow_lookback_bars"]))
        self.history_window = max(self.slow_lookback_bars + 8, int(resolved["history_window"]))
        self.entry_threshold = max(0.0, float(resolved["entry_threshold"]))
        self.exit_threshold = max(0.0, float(resolved["exit_threshold"]))
        self.stop_loss_pct = max(0.0, float(resolved["stop_loss_pct"]))
        self.take_profit_pct = max(0.0, float(resolved["take_profit_pct"]))
        self.max_hold_bars = max(1, int(resolved["max_hold_bars"]))
        self.max_longs = max(0, int(resolved["max_longs"]))
        self.max_shorts = max(0, int(resolved["max_shorts"])) if bool(resolved["allow_shorts"]) else 0
        self.allow_shorts = bool(resolved["allow_shorts"])
        self.use_fx_filter = bool(resolved["use_fx_filter"])
        self.require_calibrated_edge = bool(resolved["require_calibrated_edge"])
        self.min_calibrated_edge_bps = float(resolved["min_calibrated_edge_bps"])
        self.risk_off_threshold = float(resolved["risk_off_threshold"])
        self.risk_on_threshold = float(resolved["risk_on_threshold"])
        self.residual_momentum_weight = float(resolved["residual_momentum_weight"])
        self.residual_reversal_weight = float(resolved["residual_reversal_weight"])
        self.vwap_pressure_weight = float(resolved["vwap_pressure_weight"])
        self.breakout_failure_weight = float(resolved["breakout_failure_weight"])
        self.trend_efficiency_weight = float(resolved["trend_efficiency_weight"])
        self.risk_off_long_multiplier = float(resolved["risk_off_long_multiplier"])
        self.risk_off_short_multiplier = float(resolved["risk_off_short_multiplier"])
        self.risk_on_long_multiplier = float(resolved["risk_on_long_multiplier"])
        self.risk_on_short_multiplier = float(resolved["risk_on_short_multiplier"])
        self.min_signal_strength = float(resolved["min_signal_strength"])
        self.calibrated_edges = {str(key): float(value) for key, value in dict(calibrated_edges or {}).items()}

        requested_crypto = {normalize_symbol(item) for item in (crypto_symbols or CRYPTO_DEFAULTS)}
        requested_fx = {normalize_symbol(item) for item in (fx_symbols or FX_DEFAULTS)}
        self.crypto_symbols = [symbol for symbol in self.symbol_list if normalize_symbol(symbol) in requested_crypto or infer_market(symbol) == "crypto"]
        self.fx_symbols = [symbol for symbol in self.symbol_list if normalize_symbol(symbol) in requested_fx or infer_market(symbol) == "fx"]
        if not self.crypto_symbols:
            raise ValueError("CryptoFxAlphaZooStateStrategy requires at least one crypto symbol")
        maxlen = self.history_window + self.slow_lookback_bars + 8
        self._state = {
            symbol: _SymbolState(
                opens=deque(maxlen=maxlen),
                highs=deque(maxlen=maxlen),
                lows=deque(maxlen=maxlen),
                closes=deque(maxlen=maxlen),
                volumes=deque(maxlen=maxlen),
            )
            for symbol in self.symbol_list
        }

    def get_state(self) -> dict[str, Any]:
        return {
            "symbol_state": {
                symbol: {
                    "opens": list(state.opens),
                    "highs": list(state.highs),
                    "lows": list(state.lows),
                    "closes": list(state.closes),
                    "volumes": list(state.volumes),
                    "mode": state.mode,
                    "entry_price": state.entry_price,
                    "bars_held": state.bars_held,
                    "last_time_key": state.last_time_key,
                }
                for symbol, state in self._state.items()
            }
        }

    def set_state(self, state: dict[str, Any]) -> None:
        raw_states = state.get("symbol_state") if isinstance(state, dict) else None
        if not isinstance(raw_states, dict):
            return
        for symbol, raw in raw_states.items():
            if symbol not in self._state or not isinstance(raw, dict):
                continue
            item = self._state[symbol]
            for attr in ("opens", "highs", "lows", "closes", "volumes"):
                target = getattr(item, attr)
                target.clear()
                for value in list(raw.get(attr) or [])[-target.maxlen :]:
                    parsed = safe_float(value)
                    if parsed is not None:
                        target.append(float(parsed))
            mode = str(raw.get("mode", "OUT")).upper()
            item.mode = mode if mode in {"OUT", "LONG", "SHORT"} else "OUT"
            item.entry_price = float(safe_float(raw.get("entry_price")) or 0.0)
            item.bars_held = max(0, safe_int(raw.get("bars_held"), 0))
            item.last_time_key = str(raw.get("last_time_key", ""))

    def _append_bar(self, event_time: Any, symbol: str, open_: Any, high: Any, low: Any, close: Any, volume: Any) -> bool:
        if symbol not in self._state:
            return False
        close_f = safe_float(close)
        if close_f is None or close_f <= 0.0:
            return False
        item = self._state[symbol]
        key = time_key(event_time)
        if key and key == item.last_time_key:
            return False
        item.last_time_key = key
        open_f = safe_float(open_) or close_f
        high_f = safe_float(high) or close_f
        low_f = safe_float(low) or close_f
        volume_f = safe_float(volume) or 0.0
        item.opens.append(max(0.0, open_f))
        item.highs.append(max(close_f, high_f))
        item.lows.append(min(close_f, low_f))
        item.closes.append(close_f)
        item.volumes.append(max(0.0, volume_f))
        if item.mode != "OUT":
            item.bars_held += 1
        return True

    def _resolve_market_event(self, event: Any) -> tuple[Any, str, float, float, float, float, float] | None:
        symbol = str(getattr(event, "symbol", ""))
        if symbol not in self._state:
            return None
        event_time = getattr(event, "time", getattr(event, "datetime", None))
        open_f = safe_float(getattr(event, "open", None))
        high_f = safe_float(getattr(event, "high", None))
        low_f = safe_float(getattr(event, "low", None))
        close_f = safe_float(getattr(event, "close", None))
        volume_f = safe_float(getattr(event, "volume", None))
        if close_f is not None:
            return event_time, symbol, open_f or close_f, high_f or close_f, low_f or close_f, close_f, volume_f or 0.0
        return None

    def _fx_risk_score(self) -> float:
        weights = {"EURUSD": -1.0, "GBPUSD": -1.0, "AUDUSD": -1.0, "USDJPY": 1.0, "USDCHF": 1.0}
        score = 0.0
        count = 0
        for symbol in self.fx_symbols:
            name = normalize_symbol(symbol)
            if name not in weights:
                continue
            value = _ret(self._state[symbol].closes, self.fast_lookback_bars)
            if value is None:
                continue
            score += weights[name] * value
            count += 1
        jpy = next((symbol for symbol in self.fx_symbols if normalize_symbol(symbol) == "USDJPY"), None)
        if jpy is not None:
            jpy_ret = _ret(self._state[jpy].closes, self.fast_lookback_bars)
            if jpy_ret is not None:
                score += -jpy_ret
                count += 1
        return 0.0 if count == 0 else score

    def _fx_risk_state(self) -> str:
        if not self.use_fx_filter:
            return "neutral"
        score = self._fx_risk_score()
        if score >= self.risk_off_threshold:
            return "risk_off"
        if score <= self.risk_on_threshold:
            return "risk_on"
        return "neutral"

    def _calibrated_edge_bps(self, symbol: str, side: str) -> float | None:
        side_token = side.upper()
        keys = (
            f"{symbol}:{side_token}",
            f"{normalize_symbol(symbol)}:{side_token}",
            f"default:{side_token}",
            "default",
        )
        for key in keys:
            if key in self.calibrated_edges:
                return float(self.calibrated_edges[key])
        return None

    def _edge_gate(self, symbol: str, side: str) -> tuple[bool, float | None]:
        edge = self._calibrated_edge_bps(symbol, side)
        if not self.require_calibrated_edge:
            return True, edge
        return edge is not None and edge > self.min_calibrated_edge_bps, edge

    def _score_symbol(self, symbol: str) -> float | None:
        item = self._state[symbol]
        btc_symbol = next((candidate for candidate in self.crypto_symbols if normalize_symbol(candidate) == "BTC"), None)
        if btc_symbol is None or len(item.closes) <= self.slow_lookback_bars:
            return None
        btc = self._state[btc_symbol]
        fast_ret = _ret(item.closes, self.fast_lookback_bars)
        slow_ret = _ret(item.closes, self.slow_lookback_bars)
        btc_fast = _ret(btc.closes, self.fast_lookback_bars)
        btc_slow = _ret(btc.closes, self.slow_lookback_bars)
        if fast_ret is None or slow_ret is None or btc_fast is None or btc_slow is None:
            return None
        residual_fast = fast_ret - btc_fast
        residual_slow = slow_ret - btc_slow
        residual_hist = [ret - (btc_ret or 0.0) for ret, btc_ret in zip(_returns(item.closes, self.fast_lookback_bars), _returns(btc.closes, self.fast_lookback_bars), strict=False)]
        residual_mom = _zscore(residual_fast, residual_hist[-self.history_window :])
        residual_reversal = -_zscore(residual_slow, residual_hist[-self.history_window :]) if residual_fast * residual_slow < 0.0 else 0.0
        vwap_components = (float(item.highs[-1]), float(item.lows[-1]), float(item.closes[-1]))
        vwap_proxy = sum(vwap_components) / len(vwap_components)
        vwap_pressure = _volume_z(item.volumes, self.slow_lookback_bars) * (1.0 if item.closes[-1] >= vwap_proxy else -1.0)
        prior_high = max(list(item.highs)[-self.slow_lookback_bars - 1 : -1]) if len(item.highs) > self.slow_lookback_bars else item.highs[-1]
        prior_low = min(list(item.lows)[-self.slow_lookback_bars - 1 : -1]) if len(item.lows) > self.slow_lookback_bars else item.lows[-1]
        breakout_failure = 0.0
        if item.highs[-1] > prior_high and item.closes[-1] < prior_high:
            breakout_failure = -1.0
        elif item.lows[-1] < prior_low and item.closes[-1] > prior_low:
            breakout_failure = 1.0
        trend_eff = _trend_efficiency(item.closes, self.fast_lookback_bars)
        return (
            self.residual_momentum_weight * residual_mom
            + self.residual_reversal_weight * residual_reversal
            + self.vwap_pressure_weight * vwap_pressure
            + self.breakout_failure_weight * breakout_failure
            + self.trend_efficiency_weight * trend_eff * (1.0 if residual_fast >= 0.0 else -1.0)
        )

    def _adjust_for_fx(self, score: float, risk_state: str) -> float:
        if risk_state == "risk_off":
            return score * (self.risk_off_long_multiplier if score > 0.0 else self.risk_off_short_multiplier)
        if risk_state == "risk_on":
            return score * (self.risk_on_long_multiplier if score > 0.0 else self.risk_on_short_multiplier)
        return score

    def _emit(self, symbol: str, event_time: Any, signal_type: str, *, score: float, edge_bps: float | None, risk_state: str) -> None:
        close = float(self._state[symbol].closes[-1])
        side = "LONG" if signal_type == "LONG" else "SHORT" if signal_type == "SHORT" else None
        stop = None
        take = None
        if signal_type == "LONG":
            stop = close * (1.0 - self.stop_loss_pct)
            take = close * (1.0 + self.take_profit_pct)
        elif signal_type == "SHORT":
            stop = close * (1.0 + self.stop_loss_pct)
            take = close * (1.0 - self.take_profit_pct)
        self.events.put(
            SignalEvent(
                strategy_id=self.strategy_id,
                symbol=symbol,
                datetime=event_time,
                signal_type=signal_type,
                strength=min(1.0, max(self.min_signal_strength, abs(score) / max(self.entry_threshold, 1e-9))),
                price=close,
                stop_loss=stop,
                take_profit=take,
                position_side=side,
                metadata={
                    "strategy": self.__class__.__name__,
                    "factor_score": float(score),
                    "fx_risk_state": risk_state,
                    "calibrated_lower_bound_edge_bps": edge_bps,
                    "calendar_primary": False,
                    "uses_locked_oos_for_selection": False,
                },
            )
        )

    def _evaluate(self, event_time: Any) -> None:
        risk_state = self._fx_risk_state()
        raw_scores = {symbol: self._score_symbol(symbol) for symbol in self.crypto_symbols}
        scores = {symbol: self._adjust_for_fx(score, risk_state) for symbol, score in raw_scores.items() if score is not None}
        if not scores:
            return
        longs = [symbol for symbol, score in sorted(scores.items(), key=lambda item: item[1], reverse=True) if score > self.entry_threshold][: self.max_longs]
        shorts = [symbol for symbol, score in sorted(scores.items(), key=lambda item: item[1]) if score < -self.entry_threshold][: self.max_shorts]
        for symbol, score in scores.items():
            item = self._state[symbol]
            close = float(item.closes[-1])
            if item.mode == "LONG":
                stop_hit = close <= item.entry_price * (1.0 - self.stop_loss_pct)
                take_hit = close >= item.entry_price * (1.0 + self.take_profit_pct)
                stale = item.bars_held >= self.max_hold_bars
                if stop_hit or take_hit or stale or score <= self.exit_threshold or risk_state == "risk_off":
                    self._emit(symbol, event_time, "EXIT", score=score, edge_bps=self._calibrated_edge_bps(symbol, "LONG"), risk_state=risk_state)
                    item.mode = "OUT"
                    item.entry_price = 0.0
                    item.bars_held = 0
                continue
            if item.mode == "SHORT":
                stop_hit = close >= item.entry_price * (1.0 + self.stop_loss_pct)
                take_hit = close <= item.entry_price * (1.0 - self.take_profit_pct)
                stale = item.bars_held >= self.max_hold_bars
                if stop_hit or take_hit or stale or score >= -self.exit_threshold or risk_state == "risk_on":
                    self._emit(symbol, event_time, "EXIT", score=score, edge_bps=self._calibrated_edge_bps(symbol, "SHORT"), risk_state=risk_state)
                    item.mode = "OUT"
                    item.entry_price = 0.0
                    item.bars_held = 0
                continue
            if symbol in longs:
                allowed, edge = self._edge_gate(symbol, "LONG")
                if allowed:
                    self._emit(symbol, event_time, "LONG", score=score, edge_bps=edge, risk_state=risk_state)
                    item.mode = "LONG"
                    item.entry_price = close
                    item.bars_held = 0
            elif self.allow_shorts and symbol in shorts:
                allowed, edge = self._edge_gate(symbol, "SHORT")
                if allowed:
                    self._emit(symbol, event_time, "SHORT", score=score, edge_bps=edge, risk_state=risk_state)
                    item.mode = "SHORT"
                    item.entry_price = close
                    item.bars_held = 0

    def calculate_signals(self, event: Any) -> None:
        if isinstance(event, MarketBatchEvent) or getattr(event, "type", None) == "MARKET_BATCH":
            event_time = getattr(event, "time", None)
            for bar in getattr(event, "bars", ()):
                parsed = self._resolve_market_event(bar)
                if parsed is None:
                    continue
                _, symbol, open_f, high_f, low_f, close_f, volume_f = parsed
                self._append_bar(event_time, symbol, open_f, high_f, low_f, close_f, volume_f)
            self._evaluate(event_time)
            return
        parsed = self._resolve_market_event(event)
        if parsed is None:
            return
        event_time, symbol, open_f, high_f, low_f, close_f, volume_f = parsed
        if self._append_bar(event_time, symbol, open_f, high_f, low_f, close_f, volume_f):
            self._evaluate(event_time)

    def calculate_signals_window(self, event: Any, aggregator: Any) -> None:
        _ = aggregator
        if not isinstance(event, MarketWindowEvent) and getattr(event, "type", None) != "MARKET_WINDOW":
            self.calculate_signals(event)
            return
        event_time = getattr(event, "time", None)
        for symbol, rows in dict(getattr(event, "bars_1s", {}) or {}).items():
            if symbol not in self._state or not rows:
                continue
            row = rows[-1]
            if isinstance(row, Mapping):
                self._append_bar(
                    event_time,
                    symbol,
                    row.get("open"),
                    row.get("high"),
                    row.get("low"),
                    row.get("close"),
                    row.get("volume"),
                )
            elif isinstance(row, (tuple, list)) and len(row) >= WINDOW_TUPLE_MIN_FIELDS:
                self._append_bar(
                    event_time,
                    symbol,
                    row[WINDOW_TUPLE_OPEN_IDX],
                    row[WINDOW_TUPLE_HIGH_IDX],
                    row[WINDOW_TUPLE_LOW_IDX],
                    row[WINDOW_TUPLE_CLOSE_IDX],
                    row[WINDOW_TUPLE_VOLUME_IDX],
                )
        self._evaluate(event_time)

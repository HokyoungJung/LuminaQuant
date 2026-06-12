"""More adaptive crypto alpha sleeves for forward testing.

These strategies are additional candidate hypotheses, not validated results.  They
use only local event/bar inputs and avoid data collection, I/O, or hidden
configuration buses.  The implemented alpha families are intentionally diverse:
benchmark lead-lag continuation, beta-residual momentum, low-volatility momentum,
near-high anchoring momentum, and false-breakout reversal.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from lumina_quant.core.plugin_registry import register
from lumina_quant.indicators.alpha_features import (
    log_return,
    realized_volatility,
    trend_efficiency,
    volume_zscore,
)
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
from lumina_quant.strategies.robust_alpha_sleeves import (
    _ChannelState,
    _CrossSectionalState,
    _StatePackingMixin,
    _append_market_state,
    _mode,
    _restore_deque,
    _rolling_beta,
)
from lumina_quant.strategy import Strategy
from lumina_quant.tuning import HyperParam, resolve_params_from_schema


@dataclass(slots=True)
class _LeadLagState:
    closes: deque[float]
    volumes: deque[float]
    mode: str = "OUT"
    entry_price: float | None = None
    bars_held: int = 0
    last_time_key: str = ""


class _CrossSectionPackingMixin:
    @staticmethod
    def _pack_state(item: _CrossSectionalState) -> dict[str, Any]:
        return {
            "closes": list(item.closes),
            "volumes": list(item.volumes),
            "mode": item.mode,
            "entry_price": item.entry_price,
            "bars_held": int(item.bars_held),
            "last_time_key": item.last_time_key,
        }

    @staticmethod
    def _restore_state(item: _CrossSectionalState, payload: dict[str, Any]) -> None:
        _restore_deque(item.closes, payload.get("closes"))
        _restore_deque(item.volumes, payload.get("volumes"))
        item.mode = _mode(payload.get("mode"))
        item.entry_price = safe_float(payload.get("entry_price"))
        item.bars_held = _safe_non_negative_int(payload.get("bars_held"))
        item.last_time_key = str(payload.get("last_time_key", ""))


def _default_benchmark(symbols: list[str], preferred: str) -> str:
    if preferred in symbols:
        return preferred
    for candidate in ("BTC/USDT", "BTCUSDT", "ETH/USDT", "ETHUSDT"):
        if candidate in symbols:
            return candidate
    return symbols[0] if symbols else preferred


def _state_size(*values: int) -> int:
    return max(8, max(int(value) for value in values) + 8)


def _ranked_targets(
    rows: list[tuple[float, str, dict[str, Any]]],
    *,
    threshold: float,
    max_longs: int,
    max_shorts: int,
    allow_short: bool,
) -> dict[str, tuple[str, float, dict[str, Any]]]:
    targets: dict[str, tuple[str, float, dict[str, Any]]] = {}
    long_rows = [
        row for row in sorted(rows, key=lambda item: item[0], reverse=True) if row[0] >= threshold
    ]
    short_rows: list[tuple[float, str, dict[str, Any]]] = []
    if allow_short:
        short_rows = [row for row in sorted(rows, key=lambda item: item[0]) if row[0] <= -threshold]
    for score, symbol, meta in long_rows[: max(0, int(max_longs))]:
        targets[symbol] = ("LONG", score, meta)
    for score, symbol, meta in short_rows[: max(0, int(max_shorts))]:
        targets[symbol] = ("SHORT", score, meta)
    return targets


def _emit_rebalance_targets(
    events: Any,
    state: dict[str, _CrossSectionalState],
    targets: dict[str, tuple[str, float, dict[str, Any]]],
    *,
    event_time: Any,
    strategy_id: str,
    strategy_name: str,
    target_gross_exposure: float,
    max_order_value: float,
    stop_loss_pct: float,
    max_hold_bars: int,
    threshold: float,
) -> None:
    selected = len(targets)
    base_alloc = float(target_gross_exposure) / float(selected) if selected else 0.0
    for symbol, item in state.items():
        target = targets.get(symbol)
        price = item.closes[-1] if item.closes else None
        if target is None:
            if item.mode != "OUT":
                _emit(
                    events,
                    strategy_id=strategy_id,
                    symbol=symbol,
                    event_time=event_time,
                    signal_type="EXIT",
                    price=price,
                    metadata={"strategy": strategy_name, "reason": "rebalance_removed"},
                )
                item.mode = "OUT"
                item.entry_price = None
                item.bars_held = 0
            continue
        target_mode, score, meta = target
        if item.mode == target_mode:
            item.bars_held += 1
            continue
        if item.mode != "OUT":
            _emit(
                events,
                strategy_id=strategy_id,
                symbol=symbol,
                event_time=event_time,
                signal_type="EXIT",
                price=price,
                metadata={"strategy": strategy_name, "reason": "side_flip"},
            )
        stop_loss = None
        if price is not None and stop_loss_pct > 0.0:
            stop_loss = price * (
                1.0 - stop_loss_pct if target_mode == "LONG" else 1.0 + stop_loss_pct
            )
        metadata = _target_metadata(
            strategy=strategy_name,
            target_allocation=max(0.0, base_alloc),
            max_order_value=max_order_value,
            score=score,
            target_mode=target_mode,
            **meta,
        )
        _emit(
            events,
            strategy_id=strategy_id,
            symbol=symbol,
            event_time=event_time,
            signal_type=target_mode,
            strength=max(0.25, min(3.0, abs(score) / max(threshold, _EPS))),
            price=price,
            stop_loss=stop_loss,
            metadata=metadata,
        )
        item.mode = target_mode
        item.entry_price = price
        item.bars_held = 0
    for symbol, item in state.items():
        if item.mode == "OUT" or symbol in targets:
            continue
        price = item.closes[-1] if item.closes else None
        item.bars_held += 1
        stop = False
        if price is not None and item.entry_price is not None and stop_loss_pct > 0.0:
            if item.mode == "LONG" and price <= item.entry_price * (1.0 - stop_loss_pct):
                stop = True
            if item.mode == "SHORT" and price >= item.entry_price * (1.0 + stop_loss_pct):
                stop = True
        if stop or item.bars_held >= max_hold_bars:
            _emit(
                events,
                strategy_id=strategy_id,
                symbol=symbol,
                event_time=event_time,
                signal_type="EXIT",
                price=price,
                metadata={"strategy": strategy_name, "reason": "stop_loss" if stop else "max_hold"},
            )
            item.mode = "OUT"
            item.entry_price = None
            item.bars_held = 0


def _age_cross_positions(
    events: Any,
    state: dict[str, _CrossSectionalState],
    *,
    event_time: Any,
    strategy_id: str,
    strategy_name: str,
    stop_loss_pct: float,
    max_hold_bars: int,
) -> None:
    for symbol, item in state.items():
        if item.mode == "OUT":
            continue
        price = item.closes[-1] if item.closes else None
        item.bars_held += 1
        stop = False
        if price is not None and item.entry_price is not None and stop_loss_pct > 0.0:
            if item.mode == "LONG" and price <= item.entry_price * (1.0 - stop_loss_pct):
                stop = True
            if item.mode == "SHORT" and price >= item.entry_price * (1.0 + stop_loss_pct):
                stop = True
        if not stop and item.bars_held < max_hold_bars:
            continue
        _emit(
            events,
            strategy_id=strategy_id,
            symbol=symbol,
            event_time=event_time,
            signal_type="EXIT",
            price=price,
            metadata={"strategy": strategy_name, "reason": "stop_loss" if stop else "max_hold"},
        )
        item.mode = "OUT"
        item.entry_price = None
        item.bars_held = 0


@register("strategy", "BenchmarkLeadLagContinuationStrategy", interface="event_driven")
class BenchmarkLeadLagContinuationStrategy(Strategy):
    """Trade follower symbols when a benchmark impulse leads and the follower has not chased."""

    decision_cadence_seconds = 60
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "benchmark_symbol": HyperParam.string(
                "benchmark_symbol", default="BTC/USDT", tunable=False
            ),
            "leader_lookback_bars": HyperParam.integer(
                "leader_lookback_bars", default=6, low=1, high=4096
            ),
            "follower_lookback_bars": HyperParam.integer(
                "follower_lookback_bars", default=3, low=1, high=4096
            ),
            "leader_threshold": HyperParam.floating(
                "leader_threshold", default=0.006, low=0.0, high=1.0
            ),
            "max_follower_move": HyperParam.floating(
                "max_follower_move", default=0.010, low=0.0, high=1.0
            ),
            "min_volume_z": HyperParam.floating("min_volume_z", default=-0.5, low=-5.0, high=20.0),
            "volume_window": HyperParam.integer("volume_window", default=96, low=8, high=4096),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
            "stop_loss_pct": HyperParam.floating(
                "stop_loss_pct", default=0.025, low=0.0, high=0.50
            ),
            "take_profit_pct": HyperParam.floating(
                "take_profit_pct", default=0.055, low=0.0, high=1.0
            ),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=120, low=1, high=100000),
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
        self.benchmark_symbol = _default_benchmark(
            self.symbol_list, str(resolved["benchmark_symbol"])
        )
        self.leader_lookback_bars = max(1, int(resolved["leader_lookback_bars"]))
        self.follower_lookback_bars = max(1, int(resolved["follower_lookback_bars"]))
        self.leader_threshold = max(0.0, float(resolved["leader_threshold"]))
        self.max_follower_move = max(0.0, float(resolved["max_follower_move"]))
        self.min_volume_z = float(resolved["min_volume_z"])
        self.volume_window = max(3, int(resolved["volume_window"]))
        self.allow_short = bool(resolved["allow_short"])
        self.stop_loss_pct = max(0.0, float(resolved["stop_loss_pct"]))
        self.take_profit_pct = max(0.0, float(resolved["take_profit_pct"]))
        self.max_hold_bars = max(1, int(resolved["max_hold_bars"]))
        self.target_allocation = max(0.0, float(resolved["target_allocation"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        size = _state_size(
            self.leader_lookback_bars,
            self.follower_lookback_bars,
            self.volume_window,
            self.max_hold_bars,
        )
        self._state = {
            symbol: _LeadLagState(deque(maxlen=size), deque(maxlen=size))
            for symbol in self.symbol_list
        }
        self._last_eval_time_key = ""

    def get_state(self) -> dict[str, Any]:
        return {
            "last_eval_time_key": self._last_eval_time_key,
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
            self._evaluate(getattr(event, "time", None))

    def calculate_signals(self, event: Any) -> None:
        if str(getattr(event, "type", "")).upper() == "MARKET_WINDOW":
            self.calculate_signals_window(event, None)
            return
        if getattr(event, "type", None) != "MARKET":
            return
        symbol = getattr(event, "symbol", None)
        if symbol in self._state:
            snapshot = _market_snapshot(event)
            if snapshot is not None and self._update_symbol(str(symbol), snapshot):
                key = time_key(snapshot.time)
                if key and key != self._last_eval_time_key:
                    self._last_eval_time_key = key
                    self._evaluate(snapshot.time)

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

    def _exit_if_needed(
        self, symbol: str, item: _LeadLagState, event_time: Any, leader_ret: float | None
    ) -> None:
        if item.mode == "OUT":
            return
        price = item.closes[-1] if item.closes else None
        item.bars_held += 1
        reason = ""
        if price is not None and item.entry_price is not None:
            if item.mode == "LONG":
                if self.stop_loss_pct > 0.0 and price <= item.entry_price * (
                    1.0 - self.stop_loss_pct
                ):
                    reason = "stop_loss"
                elif self.take_profit_pct > 0.0 and price >= item.entry_price * (
                    1.0 + self.take_profit_pct
                ):
                    reason = "take_profit"
                elif leader_ret is not None and leader_ret < 0.0:
                    reason = "leader_reversed"
            elif item.mode == "SHORT":
                if self.stop_loss_pct > 0.0 and price >= item.entry_price * (
                    1.0 + self.stop_loss_pct
                ):
                    reason = "stop_loss"
                elif self.take_profit_pct > 0.0 and price <= item.entry_price * (
                    1.0 - self.take_profit_pct
                ):
                    reason = "take_profit"
                elif leader_ret is not None and leader_ret > 0.0:
                    reason = "leader_reversed"
        if not reason and item.bars_held >= self.max_hold_bars:
            reason = "max_hold"
        if not reason:
            return
        _emit(
            self.events,
            strategy_id="benchmark_lead_lag_continuation",
            symbol=symbol,
            event_time=event_time,
            signal_type="EXIT",
            price=price,
            metadata={
                "strategy": "BenchmarkLeadLagContinuationStrategy",
                "reason": reason,
                "leader_return": leader_ret,
            },
        )
        item.mode = "OUT"
        item.entry_price = None
        item.bars_held = 0

    def _evaluate(self, event_time: Any) -> None:
        benchmark = self._state.get(self.benchmark_symbol)
        if benchmark is None:
            return
        leader_ret = log_return(benchmark.closes, lookback=self.leader_lookback_bars)
        if leader_ret is None:
            return
        for symbol, item in self._state.items():
            if symbol == self.benchmark_symbol:
                continue
            self._exit_if_needed(symbol, item, event_time, leader_ret)
            if item.mode != "OUT":
                continue
            follower_ret = log_return(item.closes, lookback=self.follower_lookback_bars)
            if follower_ret is None:
                continue
            vol_z = volume_zscore(item.volumes, window=self.volume_window)
            if vol_z is not None and vol_z < self.min_volume_z:
                continue
            signal_type = ""
            if leader_ret >= self.leader_threshold and follower_ret <= self.max_follower_move:
                signal_type = "LONG"
            elif (
                self.allow_short
                and leader_ret <= -self.leader_threshold
                and follower_ret >= -self.max_follower_move
            ):
                signal_type = "SHORT"
            if not signal_type:
                continue
            price = item.closes[-1] if item.closes else None
            stop_loss = None
            take_profit = None
            if price is not None and self.stop_loss_pct > 0.0:
                stop_loss = price * (
                    1.0 - self.stop_loss_pct if signal_type == "LONG" else 1.0 + self.stop_loss_pct
                )
            if price is not None and self.take_profit_pct > 0.0:
                take_profit = price * (
                    1.0 + self.take_profit_pct
                    if signal_type == "LONG"
                    else 1.0 - self.take_profit_pct
                )
            metadata = _target_metadata(
                strategy="BenchmarkLeadLagContinuationStrategy",
                target_allocation=self.target_allocation,
                max_order_value=self.max_order_value,
                benchmark_symbol=self.benchmark_symbol,
                leader_return=leader_ret,
                follower_return=follower_ret,
                volume_z=vol_z,
                target_mode=signal_type,
            )
            _emit(
                self.events,
                strategy_id="benchmark_lead_lag_continuation",
                symbol=symbol,
                event_time=event_time,
                signal_type=signal_type,
                strength=max(0.25, min(3.0, abs(leader_ret) / max(self.leader_threshold, _EPS))),
                price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata=metadata,
            )
            item.mode = signal_type
            item.entry_price = price
            item.bars_held = 0


@register("strategy", "ResidualMomentumRotationStrategy", interface="event_driven")
class ResidualMomentumRotationStrategy(_CrossSectionPackingMixin, Strategy):
    """Rotate on beta-adjusted residual momentum instead of raw momentum."""

    decision_cadence_seconds = 60
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "benchmark_symbol": HyperParam.string(
                "benchmark_symbol", default="BTC/USDT", tunable=False
            ),
            "momentum_lookback_bars": HyperParam.integer(
                "momentum_lookback_bars", default=168, low=4, high=10080
            ),
            "beta_window": HyperParam.integer("beta_window", default=240, low=20, high=20000),
            "vol_window": HyperParam.integer("vol_window", default=96, low=4, high=4096),
            "rebalance_bars": HyperParam.integer("rebalance_bars", default=24, low=1, high=10080),
            "signal_threshold": HyperParam.floating(
                "signal_threshold", default=0.30, low=0.0, high=20.0
            ),
            "max_longs": HyperParam.integer("max_longs", default=3, low=0, high=128),
            "max_shorts": HyperParam.integer("max_shorts", default=3, low=0, high=128),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
            "stop_loss_pct": HyperParam.floating(
                "stop_loss_pct", default=0.050, low=0.0, high=0.50
            ),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=1440, low=1, high=200000),
            "target_gross_exposure": HyperParam.floating(
                "target_gross_exposure", default=0.42, low=0.0, high=5.0, tunable=False
            ),
            "max_order_value": HyperParam.floating(
                "max_order_value", default=500.0, low=0.0, high=1_000_000.0, tunable=False
            ),
            "min_price": HyperParam.floating("min_price", default=0.10, low=0.0, high=1_000_000.0),
        }

    def __init__(self, bars: Any, events: Any, **params: Any) -> None:
        self.bars = bars
        self.events = events
        self.symbol_list = list(getattr(self.bars, "symbol_list", []) or [])
        resolved = resolve_params_from_schema(self.get_param_schema(), params, keep_unknown=False)
        self.benchmark_symbol = _default_benchmark(
            self.symbol_list, str(resolved["benchmark_symbol"])
        )
        self.momentum_lookback_bars = max(1, int(resolved["momentum_lookback_bars"]))
        self.beta_window = max(3, int(resolved["beta_window"]))
        self.vol_window = max(2, int(resolved["vol_window"]))
        self.rebalance_bars = max(1, int(resolved["rebalance_bars"]))
        self.signal_threshold = max(0.0, float(resolved["signal_threshold"]))
        self.max_longs = max(0, int(resolved["max_longs"]))
        self.max_shorts = max(0, int(resolved["max_shorts"]))
        self.allow_short = bool(resolved["allow_short"])
        self.stop_loss_pct = max(0.0, float(resolved["stop_loss_pct"]))
        self.max_hold_bars = max(1, int(resolved["max_hold_bars"]))
        self.target_gross_exposure = max(0.0, float(resolved["target_gross_exposure"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        size = _state_size(
            self.momentum_lookback_bars, self.beta_window, self.vol_window, self.max_hold_bars
        )
        self._state = {
            symbol: _CrossSectionalState(deque(maxlen=size), deque(maxlen=size))
            for symbol in self.symbol_list
        }
        self._last_eval_time_key = ""
        self._tick = 0

    def get_state(self) -> dict[str, Any]:
        return {
            "last_eval_time_key": self._last_eval_time_key,
            "tick": self._tick,
            "symbol_state": {
                symbol: self._pack_state(item) for symbol, item in self._state.items()
            },
        }

    def set_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        self._last_eval_time_key = str(state.get("last_eval_time_key", ""))
        self._tick = _safe_non_negative_int(state.get("tick"))
        raw = state.get("symbol_state")
        if isinstance(raw, dict):
            for symbol, payload in raw.items():
                if symbol in self._state and isinstance(payload, dict):
                    self._restore_state(self._state[symbol], payload)

    def calculate_signals_window(self, event: Any, aggregator: Any = None) -> None:
        _ = aggregator
        event_key = time_key(getattr(event, "time", None))
        updated = False
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
        if symbol in self._state:
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

    def _score_rows(self) -> list[tuple[float, str, dict[str, Any]]]:
        benchmark = self._state.get(self.benchmark_symbol)
        if benchmark is None or len(benchmark.closes) <= self.momentum_lookback_bars:
            return []
        bench_ret = log_return(benchmark.closes, lookback=self.momentum_lookback_bars)
        if bench_ret is None:
            return []
        rows: list[tuple[float, str, dict[str, Any]]] = []
        for symbol, item in self._state.items():
            if symbol == self.benchmark_symbol:
                continue
            raw_ret = log_return(item.closes, lookback=self.momentum_lookback_bars)
            vol = realized_volatility(item.closes, window=self.vol_window)
            beta = _rolling_beta(item.closes, benchmark.closes, self.beta_window, 0.0, 5.0)
            if raw_ret is None or vol is None or vol <= _EPS or beta is None:
                continue
            residual = raw_ret - beta * bench_ret
            score = residual / max(vol, _EPS)
            rows.append(
                (
                    float(score),
                    symbol,
                    {
                        "raw_return": raw_ret,
                        "benchmark_return": bench_ret,
                        "beta": beta,
                        "residual_return": residual,
                        "realized_vol": vol,
                    },
                )
            )
        return rows

    def _rebalance(self, event_time: Any) -> None:
        if self._tick % self.rebalance_bars:
            _age_cross_positions(
                self.events,
                self._state,
                event_time=event_time,
                strategy_id="residual_momentum_rotation",
                strategy_name="ResidualMomentumRotationStrategy",
                stop_loss_pct=self.stop_loss_pct,
                max_hold_bars=self.max_hold_bars,
            )
            return
        targets = _ranked_targets(
            self._score_rows(),
            threshold=self.signal_threshold,
            max_longs=self.max_longs,
            max_shorts=self.max_shorts,
            allow_short=self.allow_short,
        )
        _emit_rebalance_targets(
            self.events,
            self._state,
            targets,
            event_time=event_time,
            strategy_id="residual_momentum_rotation",
            strategy_name="ResidualMomentumRotationStrategy",
            target_gross_exposure=self.target_gross_exposure,
            max_order_value=self.max_order_value,
            stop_loss_pct=self.stop_loss_pct,
            max_hold_bars=self.max_hold_bars,
            threshold=self.signal_threshold,
        )


@register("strategy", "LowVolatilityMomentumStrategy", interface="event_driven")
class LowVolatilityMomentumStrategy(_CrossSectionPackingMixin, Strategy):
    """Select positive-momentum symbols with relatively low realized volatility."""

    decision_cadence_seconds = 60
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "momentum_lookback_bars": HyperParam.integer(
                "momentum_lookback_bars", default=168, low=4, high=10080
            ),
            "vol_window": HyperParam.integer("vol_window", default=96, low=4, high=4096),
            "rebalance_bars": HyperParam.integer("rebalance_bars", default=24, low=1, high=10080),
            "min_momentum": HyperParam.floating("min_momentum", default=0.004, low=-1.0, high=1.0),
            "max_vol_rank": HyperParam.floating("max_vol_rank", default=0.50, low=0.05, high=1.0),
            "max_longs": HyperParam.integer("max_longs", default=4, low=0, high=128),
            "stop_loss_pct": HyperParam.floating(
                "stop_loss_pct", default=0.045, low=0.0, high=0.50
            ),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=1440, low=1, high=200000),
            "target_gross_exposure": HyperParam.floating(
                "target_gross_exposure", default=0.40, low=0.0, high=5.0, tunable=False
            ),
            "max_order_value": HyperParam.floating(
                "max_order_value", default=500.0, low=0.0, high=1_000_000.0, tunable=False
            ),
            "min_price": HyperParam.floating("min_price", default=0.10, low=0.0, high=1_000_000.0),
        }

    def __init__(self, bars: Any, events: Any, **params: Any) -> None:
        self.bars = bars
        self.events = events
        self.symbol_list = list(getattr(self.bars, "symbol_list", []) or [])
        resolved = resolve_params_from_schema(self.get_param_schema(), params, keep_unknown=False)
        self.momentum_lookback_bars = max(1, int(resolved["momentum_lookback_bars"]))
        self.vol_window = max(2, int(resolved["vol_window"]))
        self.rebalance_bars = max(1, int(resolved["rebalance_bars"]))
        self.min_momentum = float(resolved["min_momentum"])
        self.max_vol_rank = max(0.01, min(1.0, float(resolved["max_vol_rank"])))
        self.max_longs = max(0, int(resolved["max_longs"]))
        self.stop_loss_pct = max(0.0, float(resolved["stop_loss_pct"]))
        self.max_hold_bars = max(1, int(resolved["max_hold_bars"]))
        self.target_gross_exposure = max(0.0, float(resolved["target_gross_exposure"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        size = _state_size(self.momentum_lookback_bars, self.vol_window, self.max_hold_bars)
        self._state = {
            symbol: _CrossSectionalState(deque(maxlen=size), deque(maxlen=size))
            for symbol in self.symbol_list
        }
        self._last_eval_time_key = ""
        self._tick = 0

    def get_state(self) -> dict[str, Any]:
        return {
            "last_eval_time_key": self._last_eval_time_key,
            "tick": self._tick,
            "symbol_state": {
                symbol: self._pack_state(item) for symbol, item in self._state.items()
            },
        }

    def set_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        self._last_eval_time_key = str(state.get("last_eval_time_key", ""))
        self._tick = _safe_non_negative_int(state.get("tick"))
        raw = state.get("symbol_state")
        if isinstance(raw, dict):
            for symbol, payload in raw.items():
                if symbol in self._state and isinstance(payload, dict):
                    self._restore_state(self._state[symbol], payload)

    def calculate_signals_window(self, event: Any, aggregator: Any = None) -> None:
        _ = aggregator
        event_key = time_key(getattr(event, "time", None))
        updated = False
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
        if symbol in self._state:
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

    def _rebalance(self, event_time: Any) -> None:
        if self._tick % self.rebalance_bars:
            _age_cross_positions(
                self.events,
                self._state,
                event_time=event_time,
                strategy_id="low_volatility_momentum",
                strategy_name="LowVolatilityMomentumStrategy",
                stop_loss_pct=self.stop_loss_pct,
                max_hold_bars=self.max_hold_bars,
            )
            return
        rows: list[tuple[float, str, dict[str, Any]]] = []
        vol_rows: list[tuple[float, str, float]] = []
        for symbol, item in self._state.items():
            mom = log_return(item.closes, lookback=self.momentum_lookback_bars)
            vol = realized_volatility(item.closes, window=self.vol_window)
            if mom is None or vol is None or vol <= _EPS:
                continue
            vol_rows.append((vol, symbol, mom))
        if not vol_rows:
            return
        vol_rows.sort(key=lambda row: row[0])
        cutoff = max(1, int(len(vol_rows) * self.max_vol_rank))
        for vol, symbol, mom in vol_rows[:cutoff]:
            if mom < self.min_momentum:
                continue
            score = mom / max(vol, _EPS)
            rows.append(
                (
                    float(score),
                    symbol,
                    {"momentum": mom, "realized_vol": vol, "vol_rank_bucket": "low"},
                )
            )
        targets = _ranked_targets(
            rows,
            threshold=max(abs(self.min_momentum), _EPS),
            max_longs=self.max_longs,
            max_shorts=0,
            allow_short=False,
        )
        _emit_rebalance_targets(
            self.events,
            self._state,
            targets,
            event_time=event_time,
            strategy_id="low_volatility_momentum",
            strategy_name="LowVolatilityMomentumStrategy",
            target_gross_exposure=self.target_gross_exposure,
            max_order_value=self.max_order_value,
            stop_loss_pct=self.stop_loss_pct,
            max_hold_bars=self.max_hold_bars,
            threshold=max(abs(self.min_momentum), _EPS),
        )


@register("strategy", "NearHighMomentumStrategy", interface="event_driven")
class NearHighMomentumStrategy(_StatePackingMixin, Strategy):
    """Anchoring-style momentum: favor symbols near trailing highs with positive returns."""

    decision_cadence_seconds = 60
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "high_lookback_bars": HyperParam.integer(
                "high_lookback_bars", default=720, low=20, high=50000
            ),
            "momentum_lookback_bars": HyperParam.integer(
                "momentum_lookback_bars", default=168, low=4, high=10080
            ),
            "rebalance_bars": HyperParam.integer("rebalance_bars", default=24, low=1, high=10080),
            "near_high_pct": HyperParam.floating("near_high_pct", default=0.06, low=0.0, high=1.0),
            "min_momentum": HyperParam.floating("min_momentum", default=0.006, low=-1.0, high=1.0),
            "max_longs": HyperParam.integer("max_longs", default=4, low=0, high=128),
            "stop_loss_pct": HyperParam.floating(
                "stop_loss_pct", default=0.055, low=0.0, high=0.50
            ),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=2880, low=1, high=200000),
            "target_gross_exposure": HyperParam.floating(
                "target_gross_exposure", default=0.40, low=0.0, high=5.0, tunable=False
            ),
            "max_order_value": HyperParam.floating(
                "max_order_value", default=500.0, low=0.0, high=1_000_000.0, tunable=False
            ),
            "min_price": HyperParam.floating("min_price", default=0.10, low=0.0, high=1_000_000.0),
        }

    def __init__(self, bars: Any, events: Any, **params: Any) -> None:
        self.bars = bars
        self.events = events
        self.symbol_list = list(getattr(self.bars, "symbol_list", []) or [])
        resolved = resolve_params_from_schema(self.get_param_schema(), params, keep_unknown=False)
        self.high_lookback_bars = max(3, int(resolved["high_lookback_bars"]))
        self.momentum_lookback_bars = max(1, int(resolved["momentum_lookback_bars"]))
        self.rebalance_bars = max(1, int(resolved["rebalance_bars"]))
        self.near_high_pct = max(0.0, float(resolved["near_high_pct"]))
        self.min_momentum = float(resolved["min_momentum"])
        self.max_longs = max(0, int(resolved["max_longs"]))
        self.stop_loss_pct = max(0.0, float(resolved["stop_loss_pct"]))
        self.max_hold_bars = max(1, int(resolved["max_hold_bars"]))
        self.target_gross_exposure = max(0.0, float(resolved["target_gross_exposure"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        size = _state_size(self.high_lookback_bars, self.momentum_lookback_bars, self.max_hold_bars)
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
        self._last_eval_time_key = ""
        self._tick = 0

    def get_state(self) -> dict[str, Any]:
        return {
            "last_eval_time_key": self._last_eval_time_key,
            "tick": self._tick,
            "symbol_state": {
                symbol: self._pack_channel_state(item) for symbol, item in self._state.items()
            },
        }

    def set_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        self._last_eval_time_key = str(state.get("last_eval_time_key", ""))
        self._tick = _safe_non_negative_int(state.get("tick"))
        raw = state.get("symbol_state")
        if isinstance(raw, dict):
            for symbol, payload in raw.items():
                if symbol in self._state and isinstance(payload, dict):
                    self._restore_channel_state(self._state[symbol], payload)

    def calculate_signals_window(self, event: Any, aggregator: Any = None) -> None:
        _ = aggregator
        event_key = time_key(getattr(event, "time", None))
        updated = False
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
        if symbol in self._state:
            snapshot = _market_snapshot(event)
            if snapshot is not None and self._update_symbol(str(symbol), snapshot):
                key = time_key(snapshot.time)
                if key and key != self._last_eval_time_key:
                    self._last_eval_time_key = key
                    self._tick += 1
                    self._rebalance(snapshot.time)

    def _update_symbol(self, symbol: str, snapshot: _Snapshot) -> bool:
        item = self._state[symbol]
        key = time_key(snapshot.time)
        if key and key == item.last_time_key:
            return False
        item.last_time_key = key
        if snapshot.close is None or snapshot.close <= self.min_price:
            return False
        return _append_market_state(item, snapshot)

    def _rebalance(self, event_time: Any) -> None:
        if self._tick % self.rebalance_bars:
            return
        rows: list[tuple[float, str, dict[str, Any]]] = []
        for symbol, item in self._state.items():
            if (
                len(item.highs) < self.high_lookback_bars
                or len(item.closes) <= self.momentum_lookback_bars
            ):
                continue
            close = item.closes[-1]
            trailing_high = max(list(item.highs)[-self.high_lookback_bars :])
            if trailing_high <= 0.0:
                continue
            distance = close / trailing_high - 1.0
            momentum = log_return(item.closes, lookback=self.momentum_lookback_bars)
            if momentum is None or momentum < self.min_momentum or distance < -self.near_high_pct:
                continue
            score = momentum + max(-self.near_high_pct, distance)
            rows.append(
                (
                    float(score),
                    symbol,
                    {
                        "momentum": momentum,
                        "distance_to_high": distance,
                        "trailing_high": trailing_high,
                    },
                )
            )
        targets = _ranked_targets(
            rows,
            threshold=max(self.min_momentum * 0.5, _EPS),
            max_longs=self.max_longs,
            max_shorts=0,
            allow_short=False,
        )
        proxy_state = {
            symbol: _CrossSectionalState(
                item.closes,
                item.volumes,
                item.mode,
                item.entry_price,
                item.bars_held,
                item.last_time_key,
            )
            for symbol, item in self._state.items()
        }
        _emit_rebalance_targets(
            self.events,
            proxy_state,
            targets,
            event_time=event_time,
            strategy_id="near_high_momentum",
            strategy_name="NearHighMomentumStrategy",
            target_gross_exposure=self.target_gross_exposure,
            max_order_value=self.max_order_value,
            stop_loss_pct=self.stop_loss_pct,
            max_hold_bars=self.max_hold_bars,
            threshold=max(self.min_momentum * 0.5, _EPS),
        )
        for symbol, proxy in proxy_state.items():
            item = self._state[symbol]
            item.mode = proxy.mode
            item.entry_price = proxy.entry_price
            item.bars_held = proxy.bars_held


@register("strategy", "FalseBreakoutReversalStrategy", interface="event_driven")
class FalseBreakoutReversalStrategy(_StatePackingMixin, Strategy):
    """Fade failed channel breaks that close back inside the prior range."""

    decision_cadence_seconds = 60
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "channel_lookback": HyperParam.integer(
                "channel_lookback", default=96, low=4, high=10080
            ),
            "break_buffer_pct": HyperParam.floating(
                "break_buffer_pct", default=0.002, low=0.0, high=0.20
            ),
            "min_volume_z": HyperParam.floating("min_volume_z", default=0.8, low=-5.0, high=20.0),
            "volume_window": HyperParam.integer("volume_window", default=96, low=8, high=4096),
            "max_trend_efficiency": HyperParam.floating(
                "max_trend_efficiency", default=0.65, low=0.0, high=1.0
            ),
            "stop_loss_pct": HyperParam.floating(
                "stop_loss_pct", default=0.020, low=0.0, high=0.50
            ),
            "take_profit_pct": HyperParam.floating(
                "take_profit_pct", default=0.040, low=0.0, high=1.0
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
        self.channel_lookback = max(2, int(resolved["channel_lookback"]))
        self.break_buffer_pct = max(0.0, float(resolved["break_buffer_pct"]))
        self.min_volume_z = float(resolved["min_volume_z"])
        self.volume_window = max(3, int(resolved["volume_window"]))
        self.max_trend_efficiency = max(0.0, min(1.0, float(resolved["max_trend_efficiency"])))
        self.stop_loss_pct = max(0.0, float(resolved["stop_loss_pct"]))
        self.take_profit_pct = max(0.0, float(resolved["take_profit_pct"]))
        self.allow_short = bool(resolved["allow_short"])
        self.max_hold_bars = max(1, int(resolved["max_hold_bars"]))
        self.target_allocation = max(0.0, float(resolved["target_allocation"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        size = _state_size(self.channel_lookback, self.volume_window, self.max_hold_bars)
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
        if isinstance(raw, dict):
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

    def _exit_if_needed(self, symbol: str, item: _ChannelState, snapshot: _Snapshot) -> None:
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
        else:
            if self.stop_loss_pct > 0.0 and close >= item.entry_price * (1.0 + self.stop_loss_pct):
                reason = "stop_loss"
            elif self.take_profit_pct > 0.0 and close <= item.entry_price * (
                1.0 - self.take_profit_pct
            ):
                reason = "take_profit"
        if not reason and item.bars_held >= self.max_hold_bars:
            reason = "max_hold"
        if not reason:
            return
        _emit(
            self.events,
            strategy_id="false_breakout_reversal",
            symbol=symbol,
            event_time=snapshot.time,
            signal_type="EXIT",
            price=close,
            metadata={"strategy": "FalseBreakoutReversalStrategy", "reason": reason},
        )
        item.mode = "OUT"
        item.entry_price = None
        item.bars_held = 0

    def _process_symbol(self, symbol: str, snapshot: _Snapshot) -> None:
        item = self._state[symbol]
        key = time_key(snapshot.time)
        if key and key == item.last_time_key:
            return
        item.last_time_key = key
        close = safe_float(snapshot.close)
        high = safe_float(snapshot.high)
        low = safe_float(snapshot.low)
        if close is None or close <= self.min_price:
            return
        ready = len(item.highs) >= self.channel_lookback
        prior_upper = max(list(item.highs)[-self.channel_lookback :]) if ready else None
        prior_lower = min(list(item.lows)[-self.channel_lookback :]) if ready else None
        self._exit_if_needed(symbol, item, snapshot)
        if item.mode == "OUT" and ready and prior_upper is not None and prior_lower is not None:
            vol_z = volume_zscore(item.volumes, window=self.volume_window)
            eff = trend_efficiency(
                item.closes, window=min(self.channel_lookback, max(3, self.volume_window))
            )
            if (vol_z is None or vol_z >= self.min_volume_z) and (
                eff is None or eff <= self.max_trend_efficiency
            ):
                signal_type = ""
                if (
                    high is not None
                    and high > prior_upper * (1.0 + self.break_buffer_pct)
                    and close < prior_upper
                    and self.allow_short
                ):
                    signal_type = "SHORT"
                elif (
                    low is not None
                    and low < prior_lower * (1.0 - self.break_buffer_pct)
                    and close > prior_lower
                ):
                    signal_type = "LONG"
                if signal_type:
                    stop_loss = (
                        close
                        * (
                            1.0 - self.stop_loss_pct
                            if signal_type == "LONG"
                            else 1.0 + self.stop_loss_pct
                        )
                        if self.stop_loss_pct > 0.0
                        else None
                    )
                    take_profit = (
                        close
                        * (
                            1.0 + self.take_profit_pct
                            if signal_type == "LONG"
                            else 1.0 - self.take_profit_pct
                        )
                        if self.take_profit_pct > 0.0
                        else None
                    )
                    metadata = _target_metadata(
                        strategy="FalseBreakoutReversalStrategy",
                        target_allocation=self.target_allocation,
                        max_order_value=self.max_order_value,
                        prior_channel_high=prior_upper,
                        prior_channel_low=prior_lower,
                        volume_z=vol_z,
                        trend_efficiency=eff,
                        target_mode=signal_type,
                    )
                    _emit(
                        self.events,
                        strategy_id="false_breakout_reversal",
                        symbol=symbol,
                        event_time=snapshot.time,
                        signal_type=signal_type,
                        strength=1.0,
                        price=close,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        metadata=metadata,
                    )
                    item.mode = signal_type
                    item.entry_price = close
                    item.bars_held = 0
        _append_market_state(item, snapshot)


__all__ = [
    "BenchmarkLeadLagContinuationStrategy",
    "FalseBreakoutReversalStrategy",
    "LowVolatilityMomentumStrategy",
    "NearHighMomentumStrategy",
    "ResidualMomentumRotationStrategy",
]

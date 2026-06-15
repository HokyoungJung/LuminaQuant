"""Dormant dual-momentum index rotation alpha sleeve (S4).

``DualMomentumIndexRotationStrategy`` combines an absolute-momentum filter (a
risk-on/off gate on the benchmark index, e.g. SPY) with a relative-momentum rank
across a set of index ETFs (SPY, QQQ, SOXL, EWY, EWJ, EWT).  When the benchmark's
trailing return is non-positive the sleeve goes flat (cash); otherwise it holds
the top-3 indices ranked by a blended 1/3/6/12-bar return, each confirmed above
its 200-bar SMA.

The sleeve is coded but DORMANT: it only emits when at least ``min_symbols``
(default 4) index perps carry sufficient history, and its candidate wiring
intersects with the materialized universe.  It auto-discovers through the plugin
registry; validate scoring on the data-bearing machine.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from lumina_quant.core.plugin_registry import register
from lumina_quant.indicators.alpha_features import simple_return
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.strategies.adaptive_crypto_alpha_sleeves import (
    _age_cross_positions,
    _emit_rebalance_targets,
    _ranked_targets,
)
from lumina_quant.strategies.external_alpha_sleeves import (
    _EPS,
    _Snapshot,
    _event_symbols,
    _market_snapshot,
    _safe_non_negative_int,
    _window_snapshot,
)
from lumina_quant.strategies.robust_alpha_sleeves import (
    _CrossSectionalState,
    _mode,
    _restore_deque,
)
from lumina_quant.strategy import Strategy
from lumina_quant.tuning import HyperParam, resolve_params_from_schema

_DEFAULT_BENCHMARKS = ("SPYUSDT", "SPY/USDT", "QQQUSDT", "QQQ/USDT", "BTC/USDT", "BTCUSDT")


def _default_benchmark(symbols: list[str], preferred: str) -> str:
    if preferred in symbols:
        return preferred
    for candidate in _DEFAULT_BENCHMARKS:
        if candidate in symbols:
            return candidate
    return symbols[0] if symbols else preferred


def _state_size(*values: int) -> int:
    return max(8, max(int(value) for value in values) + 8)


@register("strategy", "DualMomentumIndexRotationStrategy", interface="event_driven")
class DualMomentumIndexRotationStrategy(Strategy):
    """Absolute risk-on gate plus relative blended-momentum index rotation."""

    decision_cadence_seconds = 86400
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "benchmark_symbol": HyperParam.string(
                "benchmark_symbol", default="SPYUSDT", tunable=False
            ),
            "absolute_lookback_bars": HyperParam.integer(
                "absolute_lookback_bars", default=12, low=1, high=10080
            ),
            "blend_lookbacks": HyperParam.string(
                "blend_lookbacks", default="1,3,6,12", tunable=False
            ),
            "sma_bars": HyperParam.integer("sma_bars", default=200, low=8, high=20000),
            "rebalance_bars": HyperParam.integer("rebalance_bars", default=21, low=1, high=10080),
            "max_holdings": HyperParam.integer("max_holdings", default=3, low=1, high=64),
            "min_symbols": HyperParam.integer("min_symbols", default=4, low=2, high=512),
            "stop_loss_pct": HyperParam.floating(
                "stop_loss_pct", default=0.12, low=0.0, high=0.50
            ),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=252, low=1, high=200000),
            "target_gross_exposure": HyperParam.floating(
                "target_gross_exposure", default=0.45, low=0.0, high=5.0, tunable=False
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
        self.absolute_lookback_bars = max(1, int(resolved["absolute_lookback_bars"]))
        self.blend_lookbacks = self._parse_lookbacks(str(resolved["blend_lookbacks"] or ""))
        self.sma_bars = max(2, int(resolved["sma_bars"]))
        self.rebalance_bars = max(1, int(resolved["rebalance_bars"]))
        self.max_holdings = max(1, int(resolved["max_holdings"]))
        self.min_symbols = max(2, int(resolved["min_symbols"]))
        self.stop_loss_pct = max(0.0, float(resolved["stop_loss_pct"]))
        self.max_hold_bars = max(1, int(resolved["max_hold_bars"]))
        self.target_gross_exposure = max(0.0, float(resolved["target_gross_exposure"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        size = _state_size(
            max(self.blend_lookbacks, default=12),
            self.absolute_lookback_bars,
            self.sma_bars,
            self.max_hold_bars,
        )
        self._state = {
            symbol: _CrossSectionalState(deque(maxlen=size), deque(maxlen=size))
            for symbol in self.symbol_list
        }
        self._last_eval_time_key = ""
        self._tick = 0

    @staticmethod
    def _parse_lookbacks(raw: str) -> list[int]:
        out: list[int] = []
        for chunk in raw.split(","):
            text = chunk.strip()
            if not text:
                continue
            try:
                value = int(float(text))
            except Exception:
                continue
            if value >= 1:
                out.append(value)
        return out or [1, 3, 6, 12]

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
        if isinstance(raw, dict):
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

    def _go_flat(self, event_time: Any) -> None:
        _age_cross_positions(
            self.events,
            self._state,
            event_time=event_time,
            strategy_id="dual_momentum_index_rotation",
            strategy_name="DualMomentumIndexRotationStrategy",
            stop_loss_pct=0.0,
            max_hold_bars=1,
        )

    def _rebalance(self, event_time: Any) -> None:
        if self._tick % self.rebalance_bars:
            _age_cross_positions(
                self.events,
                self._state,
                event_time=event_time,
                strategy_id="dual_momentum_index_rotation",
                strategy_name="DualMomentumIndexRotationStrategy",
                stop_loss_pct=self.stop_loss_pct,
                max_hold_bars=self.max_hold_bars,
            )
            return
        # Absolute-momentum filter: benchmark return must be positive (risk-on).
        benchmark = self._state.get(self.benchmark_symbol)
        if benchmark is None or len(benchmark.closes) <= self.absolute_lookback_bars:
            return
        bench_ret = simple_return(benchmark.closes, lookback=self.absolute_lookback_bars)
        if bench_ret is None or bench_ret <= 0.0:
            # Risk-off -> flatten everything (move to cash).
            self._go_flat(event_time)
            return
        max_lookback = max(self.blend_lookbacks, default=12)
        rows: list[tuple[float, str, dict[str, Any]]] = []
        for symbol, item in self._state.items():
            if len(item.closes) <= max_lookback or len(item.closes) < self.sma_bars:
                continue
            blended: list[float] = []
            for lb in self.blend_lookbacks:
                ret = simple_return(item.closes, lookback=lb)
                if ret is not None:
                    blended.append(ret)
            if not blended:
                continue
            score = sum(blended) / float(len(blended))
            close = item.closes[-1]
            sma = sum(list(item.closes)[-self.sma_bars :]) / float(self.sma_bars)
            # 200-bar SMA confirmation: only hold indices above trend.
            if close < sma or score <= 0.0:
                continue
            rows.append(
                (
                    float(score),
                    symbol,
                    {"blended_momentum": score, "sma": sma, "benchmark_return": bench_ret},
                )
            )
        if len(rows) < min(self.min_symbols, max(2, self.max_holdings)):
            # Not enough qualifying indices: flatten rather than over-concentrate.
            self._go_flat(event_time)
            return
        targets = _ranked_targets(
            rows,
            threshold=0.0,
            max_longs=self.max_holdings,
            max_shorts=0,
            allow_short=False,
        )
        _emit_rebalance_targets(
            self.events,
            self._state,
            targets,
            event_time=event_time,
            strategy_id="dual_momentum_index_rotation",
            strategy_name="DualMomentumIndexRotationStrategy",
            target_gross_exposure=self.target_gross_exposure,
            max_order_value=self.max_order_value,
            stop_loss_pct=self.stop_loss_pct,
            max_hold_bars=self.max_hold_bars,
            threshold=_EPS,
        )


__all__ = ["DualMomentumIndexRotationStrategy"]

"""Order-book imbalance / spread mean-reversion microstructure sleeve.

This module ships a single Pass-1 live-universe sleeve,
:class:`OrderBookImbalanceReversionStrategy` (S12).  It fades extreme order-book
depth-imbalance and spread z-scores on liquid perpetual symbols.  The sleeve is
intentionally data-local: it reads the microstructure features through its own
``_extract_feature`` copy (the canonical 3-arg fallback chain), keeps bounded
deques, and self-skips on missing features / insufficient history.  It carries a
rebalance band (``entry_z`` to act, ``exit_z``/``hold_bars`` to close) and a
per-symbol cooldown so it does not flip every bar, and the candidate slice
defaults it to a coarse timeframe (5m/15m, not 1s) to keep turnover bounded.

The class registers through the plugin registry, so it auto-discovers without
editing ``strategies.registry``.  Validate scoring on the data-bearing machine.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from lumina_quant.core.plugin_registry import register
from lumina_quant.indicators.alpha_features import rolling_zscore
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.indicators.rolling_stats import rolling_quantile
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
class _BookState:
    closes: deque[float]
    imbalances: deque[float]
    spreads: deque[float]
    mode: str = "OUT"
    entry_price: float | None = None
    bars_held: int = 0
    cooldown: int = 0
    last_time_key: str = ""


def _restore_deque(target: deque[float], payload: Any) -> None:
    target.clear()
    for value in list(payload or [])[-int(target.maxlen or 0) :]:
        parsed = safe_float(value)
        if parsed is not None:
            target.append(parsed)


def _mode(raw: Any) -> str:
    parsed = str(raw or "OUT").upper()
    return parsed if parsed in {"OUT", "LONG", "SHORT"} else "OUT"


@register("strategy", "OrderBookImbalanceReversionStrategy", interface="event_driven")
class OrderBookImbalanceReversionStrategy(Strategy):
    """Fade extreme order-book imbalance/spread z-scores with a short hold."""

    decision_cadence_seconds = 300
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False
    required_features = ("book_depth_imbalance_1pct", "bbo_spread_bps")

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "imbalance_z_window": HyperParam.integer(
                "imbalance_z_window", default=120, low=8, high=4096
            ),
            "entry_z": HyperParam.floating("entry_z", default=2.0, low=0.25, high=10.0),
            "exit_z": HyperParam.floating("exit_z", default=0.40, low=0.0, high=5.0),
            "spread_z_window": HyperParam.integer(
                "spread_z_window", default=120, low=8, high=4096
            ),
            "max_spread_z": HyperParam.floating("max_spread_z", default=3.0, low=0.0, high=20.0),
            "spread_quantile_window": HyperParam.integer(
                "spread_quantile_window", default=240, low=8, high=20000
            ),
            "max_spread_quantile": HyperParam.floating(
                "max_spread_quantile", default=0.90, low=0.05, high=1.0
            ),
            "hold_bars": HyperParam.integer("hold_bars", default=12, low=1, high=10080),
            "stop_pct": HyperParam.floating("stop_pct", default=0.010, low=0.0, high=0.50),
            "cooldown_bars": HyperParam.integer("cooldown_bars", default=6, low=0, high=10080),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
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
        self.imbalance_z_window = max(3, int(resolved["imbalance_z_window"]))
        self.entry_z = max(0.0, float(resolved["entry_z"]))
        self.exit_z = max(0.0, float(resolved["exit_z"]))
        self.spread_z_window = max(3, int(resolved["spread_z_window"]))
        self.max_spread_z = max(0.0, float(resolved["max_spread_z"]))
        self.spread_quantile_window = max(2, int(resolved["spread_quantile_window"]))
        self.max_spread_quantile = max(0.01, min(1.0, float(resolved["max_spread_quantile"])))
        self.hold_bars = max(1, int(resolved["hold_bars"]))
        self.stop_pct = max(0.0, float(resolved["stop_pct"]))
        self.cooldown_bars = max(0, int(resolved["cooldown_bars"]))
        self.allow_short = bool(resolved["allow_short"])
        self.target_allocation = max(0.0, float(resolved["target_allocation"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        size = (
            max(
                self.imbalance_z_window,
                self.spread_z_window,
                self.spread_quantile_window,
                self.hold_bars,
            )
            + 8
        )
        self._state = {
            symbol: _BookState(
                deque(maxlen=size),
                deque(maxlen=size),
                deque(maxlen=size),
            )
            for symbol in self.symbol_list
        }

    def _extract_feature(self, event: Any, symbol: str, field: str) -> float | None:
        direct = safe_float(getattr(event, field, None))
        if direct is not None:
            return direct
        getter = getattr(self.bars, "get_latest_feature_value", None)
        if callable(getter):
            try:
                value = getter(symbol, field)
            except Exception:
                value = None
            parsed = safe_float(value)
            if parsed is not None:
                return parsed
        getter = getattr(self.bars, "get_latest_bar_value", None)
        if callable(getter):
            try:
                value = getter(symbol, field)
            except Exception:
                value = None
            return safe_float(value)
        return None

    def get_state(self) -> dict[str, Any]:
        return {
            "symbol_state": {
                symbol: {
                    "closes": list(item.closes),
                    "imbalances": list(item.imbalances),
                    "spreads": list(item.spreads),
                    "mode": item.mode,
                    "entry_price": item.entry_price,
                    "bars_held": int(item.bars_held),
                    "cooldown": int(item.cooldown),
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
            _restore_deque(item.closes, payload.get("closes"))
            _restore_deque(item.imbalances, payload.get("imbalances"))
            _restore_deque(item.spreads, payload.get("spreads"))
            item.mode = _mode(payload.get("mode"))
            item.entry_price = safe_float(payload.get("entry_price"))
            item.bars_held = _safe_non_negative_int(payload.get("bars_held"))
            item.cooldown = _safe_non_negative_int(payload.get("cooldown"))
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

    def _exit_if_needed(self, symbol: str, item: _BookState, snapshot: _Snapshot) -> None:
        close = safe_float(snapshot.close)
        if item.mode not in {"LONG", "SHORT"} or item.entry_price is None or close is None:
            return
        item.bars_held += 1
        reason = ""
        if item.mode == "LONG":
            if self.stop_pct > 0.0 and close <= item.entry_price * (1.0 - self.stop_pct):
                reason = "stop_loss"
        else:
            if self.stop_pct > 0.0 and close >= item.entry_price * (1.0 + self.stop_pct):
                reason = "stop_loss"
        if not reason and item.bars_held >= self.hold_bars:
            reason = "max_hold"
        if not reason:
            imb_z = rolling_zscore(item.imbalances, window=self.imbalance_z_window)
            if imb_z is not None and abs(imb_z) <= self.exit_z:
                reason = "imbalance_reverted"
        if not reason:
            return
        _emit(
            self.events,
            strategy_id="orderbook_imbalance_reversion",
            symbol=symbol,
            event_time=snapshot.time,
            signal_type="EXIT",
            price=close,
            metadata={"strategy": "OrderBookImbalanceReversionStrategy", "reason": reason},
        )
        item.mode = "OUT"
        item.entry_price = None
        item.bars_held = 0
        item.cooldown = self.cooldown_bars

    def _process_symbol(self, event: Any, symbol: str, snapshot: _Snapshot) -> None:
        item = self._state[symbol]
        key = time_key(snapshot.time)
        if key and key == item.last_time_key:
            return
        item.last_time_key = key
        close = safe_float(snapshot.close)
        if close is None or close <= self.min_price:
            return
        imbalance = self._extract_feature(event, symbol, "book_depth_imbalance_1pct")
        spread = self._extract_feature(event, symbol, "bbo_spread_bps")
        if imbalance is None or spread is None:
            return
        item.closes.append(close)
        item.imbalances.append(float(imbalance))
        item.spreads.append(max(0.0, float(spread)))
        if item.cooldown > 0:
            item.cooldown -= 1
        self._exit_if_needed(symbol, item, snapshot)
        if item.mode != "OUT" or item.cooldown > 0:
            return
        imb_z = rolling_zscore(item.imbalances, window=self.imbalance_z_window)
        if imb_z is None or abs(imb_z) < self.entry_z:
            return
        spread_z = rolling_zscore(item.spreads, window=self.spread_z_window)
        if self.max_spread_z > 0.0 and spread_z is not None and abs(spread_z) > self.max_spread_z:
            return
        spread_cap = rolling_quantile(
            item.spreads, window=self.spread_quantile_window, q=self.max_spread_quantile
        )
        if spread_cap is not None and item.spreads[-1] > spread_cap:
            return
        # Fade the imbalance: heavy bid pressure (positive z) -> short, vice versa.
        signal_type = ""
        if imb_z >= self.entry_z and self.allow_short:
            signal_type = "SHORT"
        elif imb_z <= -self.entry_z:
            signal_type = "LONG"
        if not signal_type:
            return
        stop_loss = None
        if self.stop_pct > 0.0:
            stop_loss = close * (
                1.0 - self.stop_pct if signal_type == "LONG" else 1.0 + self.stop_pct
            )
        metadata = _target_metadata(
            strategy="OrderBookImbalanceReversionStrategy",
            target_allocation=self.target_allocation,
            max_order_value=self.max_order_value,
            imbalance_z=imb_z,
            spread_z=spread_z,
            spread_bps=item.spreads[-1],
            target_mode=signal_type,
        )
        _emit(
            self.events,
            strategy_id="orderbook_imbalance_reversion",
            symbol=symbol,
            event_time=snapshot.time,
            signal_type=signal_type,
            strength=max(0.25, min(3.0, abs(imb_z) / max(self.entry_z, _EPS))),
            price=close,
            stop_loss=stop_loss,
            metadata=metadata,
        )
        item.mode = signal_type
        item.entry_price = close
        item.bars_held = 0


__all__ = ["OrderBookImbalanceReversionStrategy"]

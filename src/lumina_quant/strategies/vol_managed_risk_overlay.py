"""Volatility-managed risk overlay (a reusable WRAPPER meta-strategy).

``VolManagedRiskOverlayStrategy`` instantiates a CHILD return-seeking strategy
and rescales the notional of every signal the child emits.  It introduces ZERO
new return-seeking signal -- it can only down-weight (and never lever beyond) the
child's own intent -- so it cannot worsen the child's overfitting.

Two well-documented overlays are stacked multiplicatively:

- VOL TARGETING (Moreira-Muir "Volatility-Managed Portfolios"): the child's
  notional is scaled by ``target_vol_per_bar / max(realized_vol, eps)`` of a
  configurable benchmark, clamped to ``[min_scale, max_scale]`` with
  ``max_scale <= 1.0`` so the overlay never adds leverage.
- A CRASH GATE (Daniel-Moskowitz "Momentum Crashes" conjunction): the benchmark
  must be SIMULTANEOUSLY in a deep trailing drawdown (``drawdown_from_peak`` at or
  below ``crash_drawdown_threshold``) AND showing an elevated realized-vol
  z-score (``>= crash_vol_z_threshold``).  Only when BOTH fire together does the
  gate collapse the scale to ``crash_scale``; either condition alone leaves the
  gate at ``1.0``.

``overlay_scale = clamp(target_vol_per_bar / max(realized_vol, eps), min_scale,
max_scale) * crash_gate``.  Non-EXIT child signals have their ``target_allocation``,
``max_order_value`` and ``strength`` multiplied by ``overlay_scale``; EXIT signals
are forwarded UNSCALED so de-risking is never blocked.  A ``rebalance_band``
suppresses re-emit of a non-EXIT signal whose multiplier has not moved beyond the
band since the prior emit for that symbol.

The overlay is data-local: it tracks only the benchmark's closed-bar closes in a
bounded deque and computes ``realized_volatility`` / ``drawdown_from_peak`` on
closed bars (no-lookahead -- the scale at bar ``t`` never reads ``t+1``).  It
reuses the child-execution machinery of ``ArtifactPortfolioModeStrategy`` and
``calculate_signals`` never raises.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from lumina_quant.core.events import SignalEvent
from lumina_quant.core.plugin_registry import register
from lumina_quant.indicators.alpha_features import (
    drawdown_from_peak,
    realized_volatility,
)
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.strategies.adaptive_crypto_alpha_sleeves import _state_size
from lumina_quant.strategies.artifact_portfolio_mode import (
    _BarsSubsetProxy,
    _SignalCaptureQueue,
)
from lumina_quant.strategies.external_alpha_sleeves import (
    _extract_feature,
    _market_snapshot,
    _window_snapshot,
)
from lumina_quant.strategy import Strategy
from lumina_quant.strategies import resolve_strategy_class
from lumina_quant.tuning import HyperParam, resolve_params_from_schema

_STRATEGY_ID = "vol_managed_risk_overlay"
_STRATEGY_NAME = "VolManagedRiskOverlayStrategy"
_EPS = 1e-12


def _rolling_z(values: list[float]) -> float:
    """Sample z-score of the last value vs its trailing window (0.0 if degenerate)."""
    if len(values) < 2:
        return 0.0
    mean_value = sum(values) / float(len(values))
    variance = sum((value - mean_value) ** 2 for value in values) / float(len(values) - 1)
    sigma = variance**0.5
    if sigma <= _EPS:
        return 0.0
    return float((values[-1] - mean_value) / sigma)


@dataclass(slots=True)
class _OverlayState:
    """Benchmark closed-bar closes plus a rolling realized-vol history."""

    closes: deque[float]
    vol_history: deque[float]
    last_time_key: str = ""


@register("strategy", "VolManagedRiskOverlayStrategy", interface="event_driven")
class VolManagedRiskOverlayStrategy(Strategy):
    """Vol-target + crash-gate WRAPPER that rescales a child strategy's signals.

    The child is instantiated on a bars proxy with a signal-capture queue; on each
    event the child computes its signals, they are drained, and each non-EXIT
    signal's notional metadata and strength are multiplied by ``overlay_scale``
    (EXIT signals pass through unscaled).  ``overlay_scale`` is the vol-target
    clamp times the crash gate, both derived from closed benchmark bars only.
    """

    decision_cadence_seconds = 60
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "child_strategy_class": HyperParam.string(
                "child_strategy_class",
                default="DiversifiedMultiFactorEnsembleStrategy",
                tunable=False,
            ),
            "target_vol_per_bar": HyperParam.floating(
                "target_vol_per_bar", default=0.01, low=0.0, high=1.0
            ),
            "vol_window": HyperParam.integer("vol_window", default=60, low=5, high=500),
            "min_scale": HyperParam.floating("min_scale", default=0.25, low=0.0, high=1.0),
            "max_scale": HyperParam.floating("max_scale", default=1.0, low=0.0, high=1.0),
            "crash_benchmark_symbol": HyperParam.string(
                "crash_benchmark_symbol", default="BTC/USDT", tunable=False
            ),
            "crash_vol_window": HyperParam.integer("crash_vol_window", default=60, low=5, high=500),
            "crash_drawdown_window": HyperParam.integer(
                "crash_drawdown_window", default=120, low=5, high=2000
            ),
            "crash_drawdown_threshold": HyperParam.floating(
                "crash_drawdown_threshold", default=-0.20, low=-1.0, high=0.0
            ),
            "crash_vol_z_threshold": HyperParam.floating(
                "crash_vol_z_threshold", default=1.5, low=0.0, high=10.0
            ),
            "crash_scale": HyperParam.floating("crash_scale", default=0.0, low=0.0, high=0.3),
            "crash_vol_z_window": HyperParam.integer(
                "crash_vol_z_window", default=120, low=5, high=2000
            ),
            "rebalance_band": HyperParam.floating(
                "rebalance_band", default=0.15, low=0.0, high=2.0
            ),
        }

    def __init__(self, bars: Any, events: Any, **params: Any) -> None:
        self.bars = bars
        self.events = events
        resolved = resolve_params_from_schema(self.get_param_schema(), params, keep_unknown=False)

        self.child_strategy_class = str(resolved["child_strategy_class"]).strip()
        # ``child_params``/``child_symbols`` are not part of the tunable schema --
        # they are structural wiring, read defensively from the raw params.
        raw_child_params = params.get("child_params")
        self.child_params: dict[str, Any] = (
            dict(raw_child_params) if isinstance(raw_child_params, dict) else {}
        )
        raw_child_symbols = params.get("child_symbols")
        configured_symbols = (
            [str(symbol) for symbol in raw_child_symbols if str(symbol).strip()]
            if isinstance(raw_child_symbols, (list, tuple))
            else list(getattr(self.bars, "symbol_list", []) or [])
        )

        self.target_vol_per_bar = max(0.0, float(resolved["target_vol_per_bar"]))
        self.vol_window = max(2, int(resolved["vol_window"]))
        self.min_scale = max(0.0, float(resolved["min_scale"]))
        self.max_scale = max(self.min_scale, min(1.0, float(resolved["max_scale"])))
        self.crash_benchmark_symbol = str(resolved["crash_benchmark_symbol"]).strip()
        self.crash_vol_window = max(2, int(resolved["crash_vol_window"]))
        self.crash_drawdown_window = max(2, int(resolved["crash_drawdown_window"]))
        self.crash_drawdown_threshold = min(0.0, float(resolved["crash_drawdown_threshold"]))
        self.crash_vol_z_threshold = max(0.0, float(resolved["crash_vol_z_threshold"]))
        self.crash_scale = max(0.0, min(0.3, float(resolved["crash_scale"])))
        self.crash_vol_z_window = max(2, int(resolved["crash_vol_z_window"]))
        self.rebalance_band = max(0.0, float(resolved["rebalance_band"]))

        # Instantiate the child on a bars proxy with a signal-capture queue, reusing
        # the ArtifactPortfolioModeStrategy machinery.
        child_cls = resolve_strategy_class(
            self.child_strategy_class, default_name=self.child_strategy_class
        )
        self._child_queue = _SignalCaptureQueue()
        child_bars = _BarsSubsetProxy(self.bars, list(configured_symbols))
        self._child = child_cls(child_bars, self._child_queue, **dict(self.child_params))
        self.symbol_list = list(getattr(self._child, "symbol_list", configured_symbols) or [])

        # Propagate child requirements so the runtime feeds the right inputs.
        self.required_features = tuple(getattr(self._child, "required_features", ()) or ())
        self.required_timeframes = tuple(getattr(self._child, "required_timeframes", ()) or ())
        self.uses_timeframe_aggregator = bool(
            getattr(self._child, "uses_timeframe_aggregator", False)
        )
        child_cadence = getattr(self._child, "decision_cadence_seconds", None)
        if isinstance(child_cadence, (int, float)) and child_cadence > 0:
            self.decision_cadence_seconds = int(child_cadence)
        child_contract = getattr(self._child, "preferred_contract", None)
        if isinstance(child_contract, str) and child_contract:
            self.preferred_contract = child_contract

        size = _state_size(
            self.vol_window,
            self.crash_vol_window,
            self.crash_drawdown_window,
            self.crash_vol_z_window,
        )
        self._overlay = _OverlayState(
            closes=deque(maxlen=size),
            vol_history=deque(maxlen=max(2, self.crash_vol_z_window)),
        )
        # Per-symbol last forwarded record {type, alloc, scale} — the band gate
        # only suppresses a redundant re-size of the SAME child intent.
        self._last_forward: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------ state

    def get_state(self) -> dict[str, Any]:
        child_getter = getattr(self._child, "get_state", None)
        child_state = dict(child_getter() or {}) if callable(child_getter) else {}
        return {
            "child": child_state,
            "overlay": {
                "closes": list(self._overlay.closes),
                "vol_history": list(self._overlay.vol_history),
                "last_time_key": self._overlay.last_time_key,
            },
            "last_forward": {k: dict(v) for k, v in self._last_forward.items()},
        }

    def set_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        child_state = state.get("child")
        setter = getattr(self._child, "set_state", None)
        if callable(setter) and isinstance(child_state, dict):
            setter(child_state)
        overlay = state.get("overlay")
        if isinstance(overlay, dict):
            self._overlay.closes.clear()
            for value in list(overlay.get("closes") or [])[
                -int(self._overlay.closes.maxlen or 0) :
            ]:
                parsed = safe_float(value)
                if parsed is not None:
                    self._overlay.closes.append(parsed)
            self._overlay.vol_history.clear()
            for value in list(overlay.get("vol_history") or [])[
                -int(self._overlay.vol_history.maxlen or 0) :
            ]:
                parsed = safe_float(value)
                if parsed is not None:
                    self._overlay.vol_history.append(parsed)
            self._overlay.last_time_key = str(overlay.get("last_time_key") or "")
        raw_fwd = state.get("last_forward")
        if isinstance(raw_fwd, dict):
            restored: dict[str, dict[str, Any]] = {}
            for symbol, rec in raw_fwd.items():
                if not isinstance(rec, dict):
                    continue
                restored[str(symbol)] = {
                    "type": str(rec.get("type") or ""),
                    "alloc": safe_float(rec.get("alloc")),
                    "scale": safe_float(rec.get("scale")),
                }
            self._last_forward = restored

    # ----------------------------------------------------------- benchmark IO

    def _benchmark_close(self, event: Any) -> float | None:
        """Resolve the benchmark close from the event (3-tier cascade / snapshots)."""
        symbol = self.crash_benchmark_symbol
        # MARKET_WINDOW: prefer the per-symbol window snapshot for the benchmark.
        snapshot = _window_snapshot(event, symbol)
        if snapshot is not None and snapshot.close is not None:
            return float(snapshot.close)
        # Single-symbol MARKET event that IS the benchmark.
        if str(getattr(event, "symbol", "")) == symbol:
            market = _market_snapshot(event)
            if market is not None and market.close is not None:
                return float(market.close)
        # Fall back to the feature cascade (event attr / bars lookups).
        for field in ("close", "mark_price", "index_price"):
            value = _extract_feature(self.bars, event, symbol, field)
            if value is not None and value > 0.0:
                return float(value)
        return None

    def _ingest_benchmark(self, event: Any) -> None:
        """Append the benchmark close once per closed bar (append-then-compute)."""
        key = time_key(getattr(event, "time", None))
        if key and key == self._overlay.last_time_key:
            return
        close = self._benchmark_close(event)
        if close is None or close <= 0.0:
            return
        self._overlay.last_time_key = key
        self._overlay.closes.append(float(close))
        vol = realized_volatility(list(self._overlay.closes), window=self.crash_vol_z_window)
        if vol is not None:
            self._overlay.vol_history.append(float(vol))

    def _overlay_scale(self) -> float:
        """Compute the vol-target clamp times the crash gate on closed bars only."""
        closes = list(self._overlay.closes)
        realized = realized_volatility(closes, window=self.vol_window)
        if realized is None:
            vol_scale = self.max_scale
        else:
            raw = self.target_vol_per_bar / max(realized, _EPS)
            vol_scale = max(self.min_scale, min(self.max_scale, raw))
        return float(vol_scale * self._crash_gate(closes))

    def _crash_gate(self, closes: list[float]) -> float:
        """Daniel-Moskowitz conjunction: deep drawdown AND elevated vol-z together."""
        drawdown = drawdown_from_peak(closes, window=self.crash_drawdown_window)
        if drawdown is None:
            return 1.0
        in_drawdown = drawdown <= self.crash_drawdown_threshold
        vol_z = _rolling_z(list(self._overlay.vol_history))
        elevated_vol = vol_z >= self.crash_vol_z_threshold
        if in_drawdown and elevated_vol:
            return self.crash_scale
        return 1.0

    # ------------------------------------------------------------- forwarding

    def _forward_child_signal(self, signal: SignalEvent, overlay_scale: float) -> None:
        signal_type = str(getattr(signal, "signal_type", "")).upper()
        metadata = dict(signal.metadata or {})
        is_exit = signal_type == "EXIT"

        if is_exit:
            # De-risking must never be blocked or rescaled.
            forwarded = self._rebuild_signal(signal, metadata, strength_scale=1.0)
            self.events.put(forwarded)
            return

        sym = str(signal.symbol)
        child_alloc = safe_float(metadata.get("target_allocation"))
        # rebalance_band: only suppress a REDUNDANT re-size of the SAME child
        # intent (same side + same target_allocation) when the overlay multiplier
        # barely moved. A new/changed child trade (different side or allocation)
        # must always be forwarded — the band must never drop child intent.
        prev = self._last_forward.get(sym)
        if prev is not None:
            prev_alloc = prev.get("alloc")
            alloc_same = (prev_alloc is None and child_alloc is None) or (
                prev_alloc is not None
                and child_alloc is not None
                and abs(float(prev_alloc) - float(child_alloc))
                <= 1e-9 + 1e-9 * max(abs(float(prev_alloc)), abs(float(child_alloc)))
            )
            prev_scale = prev.get("scale")
            if (
                prev.get("type") == signal_type
                and alloc_same
                and prev_scale is not None
                and abs(overlay_scale - float(prev_scale)) < self.rebalance_band
            ):
                return

        # Scale sizing NOTIONAL + strength only. max_symbol_exposure_pct is a
        # min-clamp cap (per the authoring contract) — do NOT scale/inflate it.
        for field in ("target_allocation", "max_order_value"):
            value = safe_float(metadata.get(field))
            if value is not None and value > 0.0:
                metadata[field] = float(value) * overlay_scale
        metadata["overlay_scale"] = float(overlay_scale)
        metadata["child_strategy_class"] = self.child_strategy_class

        forwarded = self._rebuild_signal(signal, metadata, strength_scale=overlay_scale)
        self._last_forward[sym] = {
            "type": signal_type,
            "alloc": child_alloc,
            "scale": float(overlay_scale),
        }
        self.events.put(forwarded)

    def _rebuild_signal(
        self, signal: SignalEvent, metadata: dict[str, Any], *, strength_scale: float
    ) -> SignalEvent:
        base_strength = float(getattr(signal, "strength", 1.0) or 1.0)
        return SignalEvent(
            strategy_id=f"{_STRATEGY_ID}::{self.child_strategy_class}",
            symbol=str(signal.symbol),
            datetime=signal.datetime,
            signal_type=str(signal.signal_type),
            strength=base_strength * float(strength_scale),
            price=getattr(signal, "price", None),
            stop_loss=getattr(signal, "stop_loss", None),
            take_profit=getattr(signal, "take_profit", None),
            position_side=getattr(signal, "position_side", None),
            client_order_id=getattr(signal, "client_order_id", None),
            time_in_force=getattr(signal, "time_in_force", None),
            metadata=metadata,
            trailing_percent=getattr(signal, "trailing_percent", None),
        )

    def _drain(self, overlay_scale: float) -> None:
        for item in self._child_queue.drain():
            if isinstance(item, SignalEvent):
                self._forward_child_signal(item, overlay_scale)

    # ------------------------------------------------------------- dispatchers

    def calculate_signals(self, event: Any) -> None:
        try:
            self._ingest_benchmark(event)
            overlay_scale = self._overlay_scale()
            self._child.calculate_signals(event)
            self._drain(overlay_scale)
        except Exception:
            # The overlay must never raise from calculate_signals.
            self._child_queue.drain()

    def calculate_signals_window(self, event: Any, aggregator: Any) -> None:
        try:
            self._ingest_benchmark(event)
            overlay_scale = self._overlay_scale()
            handler = getattr(self._child, "calculate_signals_window", None)
            if callable(handler):
                handler(event, aggregator)
            else:
                self._child.calculate_signals(event)
            self._drain(overlay_scale)
        except Exception:
            self._child_queue.drain()

    def calculate_signals_context(self, context: Any) -> None:
        try:
            event = getattr(context, "event", context)
            self._ingest_benchmark(event)
            overlay_scale = self._overlay_scale()
            handler = getattr(self._child, "calculate_signals_context", None)
            if callable(handler):
                handler(context)
            else:
                window = getattr(self._child, "calculate_signals_window", None)
                if callable(window):
                    window(event, getattr(context, "aggregator", None))
                else:
                    self._child.calculate_signals(event)
            self._drain(overlay_scale)
        except Exception:
            self._child_queue.drain()


__all__ = ["VolManagedRiskOverlayStrategy"]

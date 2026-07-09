"""Average-pairwise-correlation crash-guard risk overlay (a de-risk-only WRAPPER).

``AvgCorrelationCrashGuardOverlayStrategy`` instantiates a CHILD return-seeking
strategy and de-risks its book when the child basket's average pairwise
correlation z-spikes above its own regime band (Pollet-Wilson 2010 average
correlation as a deleveraging gauge).  Like ``VolManagedRiskOverlayStrategy`` it
introduces ZERO new return-seeking signal -- the guard scale is ALWAYS ``<= 1.0``
so it can only down-weight (never lever beyond) the child's intent -- so it
cannot worsen the child's overfitting.  The thesis is drawdown / Calmar variance
reduction, NEVER a Sharpe-lift claim.

Gauge: per completed decision bar, append closes for the guard basket (default =
the child ``symbol_list``, ``min_symbols`` floor) into bounded deques and compute
the trailing average pairwise correlation of log returns over ``corr_window`` via
the Engle-Kelly (2012) equicorrelation identity
``rho_bar = (Var(sum of per-symbol z-scored returns) - N) / (N*(N-1))`` (O(N*T),
fixed symbol order, deterministic).  ``rho_bar`` history over ``corr_z_window`` is
z-scored against its own trailing band.  A two-state hysteresis guard:

- ENGAGE (``scale = derisk_scale``) when ``z >= z_enter`` AND
  ``rho_bar >= corr_abs_floor`` -- so a noise spike in a decorrelated regime
  cannot trigger the guard;
- RELEASE (``scale = 1.0``) only when ``z <= z_exit`` AND ``min_dwell_bars`` have
  elapsed since engagement.

Non-EXIT child signals have their ``target_allocation``, ``max_order_value`` and
``strength`` multiplied by the guard scale; EXIT signals are forwarded UNSCALED so
de-risking is never blocked.  A ``rebalance_band`` suppresses re-emit of a non-EXIT
signal whose multiplier has not moved beyond the band since the prior emit for that
symbol.  Warmup / degenerate input (sub-``min_symbols``, NaN, zero-variance) leaves
the scale neutral at ``1.0`` and never raises -- the scale at bar ``t`` never reads
``t+1``.

NOTE: authoring lane module -- intentionally NOT ``@register``-ed (an unregistered
module under ``strategies/`` is inert); the integration owner wires registration,
the ``research_only`` tier hint, candidate builders and the manifest/baseline
re-pin in one atomic commit.  The overlay-family naming deviation from
``*_alpha_sleeves.py`` mirrors the ``vol_managed_risk_overlay.py`` sibling.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np

from lumina_quant.core.events import SignalEvent
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.strategies import resolve_strategy_class
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
from lumina_quant.tuning import HyperParam, resolve_params_from_schema

_STRATEGY_ID = "correlation_crash_guard_overlay"
_STRATEGY_NAME = "AvgCorrelationCrashGuardOverlayStrategy"
_EPS = 1e-12


def _average_pairwise_correlation(return_series: Any) -> float | None:
    """Engle-Kelly equicorrelation of aligned per-symbol return vectors.

    Each row is a per-symbol return vector; rows are aligned to their common
    trailing length, standardized (mean 0, unit sample variance, ddof=1), and the
    average pairwise correlation is recovered from the variance of their sum:
    ``rho_bar = (Var(sum z) - N) / (N*(N-1))``.  ``None`` on fewer than two rows
    with valid variance or a degenerate window.  Fixed-order, never raises.
    """
    rows: list[np.ndarray] = []
    for row in return_series:
        arr = np.asarray([float(v) for v in row], dtype=float)
        if arr.size >= 2 and bool(np.all(np.isfinite(arr))):
            rows.append(arr)
    if len(rows) < 2:
        return None
    length = min(int(arr.size) for arr in rows)
    if length < 2:
        return None
    z_rows: list[np.ndarray] = []
    for arr in rows:
        window = arr[-length:]
        mean = float(window.mean())
        var = float(((window - mean) ** 2).sum()) / float(length - 1)
        if var <= _EPS:
            continue
        z_rows.append((window - mean) / math.sqrt(var))
    n = len(z_rows)
    if n < 2:
        return None
    stacked = np.vstack(z_rows)
    summed = stacked.sum(axis=0)
    s_mean = float(summed.mean())
    var_s = float(((summed - s_mean) ** 2).sum()) / float(length - 1)
    rho = (var_s - float(n)) / (float(n) * float(n - 1))
    return float(rho) if math.isfinite(rho) else None


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


def _restore_bounded_deque(target: deque[float], values: Any) -> None:
    """Refill a bounded deque from a serialized list, dropping non-finite values."""
    target.clear()
    for value in list(values or [])[-int(target.maxlen or 0) :]:
        parsed = safe_float(value)
        if parsed is not None:
            target.append(parsed)


@dataclass(slots=True)
class _GuardState:
    """Per-symbol closes, the rho-bar history, and the two-state guard flag."""

    closes: dict[str, deque[float]]
    rho_history: deque[float]
    last_time_key: str = ""
    engaged: bool = False
    dwell: int = 0
    current_scale: float = 1.0


class AvgCorrelationCrashGuardOverlayStrategy(Strategy):
    """Average-correlation crash-guard WRAPPER that de-risks a child's signals."""

    decision_cadence_seconds = 86400
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
            "corr_window": HyperParam.integer("corr_window", default=96, low=8, high=2000),
            "corr_z_window": HyperParam.integer("corr_z_window", default=240, low=8, high=4000),
            "z_enter": HyperParam.floating("z_enter", default=1.5, low=0.0, high=10.0),
            "z_exit": HyperParam.floating("z_exit", default=0.5, low=0.0, high=10.0),
            "corr_abs_floor": HyperParam.floating(
                "corr_abs_floor", default=0.35, low=0.0, high=1.0
            ),
            "derisk_scale": HyperParam.floating("derisk_scale", default=0.35, low=0.0, high=1.0),
            "min_dwell_bars": HyperParam.integer("min_dwell_bars", default=24, low=0, high=2000),
            "min_symbols": HyperParam.integer("min_symbols", default=4, low=2, high=512),
            "rebalance_band": HyperParam.floating(
                "rebalance_band", default=0.15, low=0.0, high=2.0
            ),
        }

    def __init__(self, bars: Any, events: Any, **params: Any) -> None:
        self.bars = bars
        self.events = events
        resolved = resolve_params_from_schema(self.get_param_schema(), params, keep_unknown=False)

        self.child_strategy_class = str(resolved["child_strategy_class"]).strip()
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

        self.corr_window = max(2, int(resolved["corr_window"]))
        self.corr_z_window = max(2, int(resolved["corr_z_window"]))
        self.z_enter = max(0.0, float(resolved["z_enter"]))
        self.z_exit = min(self.z_enter, max(0.0, float(resolved["z_exit"])))
        self.corr_abs_floor = max(0.0, min(1.0, float(resolved["corr_abs_floor"])))
        self.derisk_scale = max(0.0, min(1.0, float(resolved["derisk_scale"])))
        self.min_dwell_bars = max(0, int(resolved["min_dwell_bars"]))
        self.min_symbols = max(2, int(resolved["min_symbols"]))
        self.rebalance_band = max(0.0, float(resolved["rebalance_band"]))

        child_cls = resolve_strategy_class(
            self.child_strategy_class, default_name=self.child_strategy_class
        )
        self._child_queue = _SignalCaptureQueue()
        child_bars = _BarsSubsetProxy(self.bars, list(configured_symbols))
        self._child = child_cls(child_bars, self._child_queue, **dict(self.child_params))
        self.symbol_list = list(getattr(self._child, "symbol_list", configured_symbols) or [])

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

        size = _state_size(self.corr_window)
        self._guard = _GuardState(
            closes={symbol: deque(maxlen=size) for symbol in self.symbol_list},
            rho_history=deque(maxlen=max(2, self.corr_z_window)),
        )
        self._last_forward: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------ state

    def get_state(self) -> dict[str, Any]:
        child_getter = getattr(self._child, "get_state", None)
        child_state = dict(child_getter() or {}) if callable(child_getter) else {}
        guard: dict[str, Any] = {
            "closes": {symbol: list(dq) for symbol, dq in self._guard.closes.items()},
            "rho_history": list(self._guard.rho_history),
            "last_time_key": self._guard.last_time_key,
            "engaged": bool(self._guard.engaged),
            "dwell": int(self._guard.dwell),
            "current_scale": float(self._guard.current_scale),
        }
        return {
            "child": child_state,
            "guard": guard,
            "last_forward": {k: dict(v) for k, v in self._last_forward.items()},
        }

    def set_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        child_state = state.get("child")
        setter = getattr(self._child, "set_state", None)
        if callable(setter) and isinstance(child_state, dict):
            setter(child_state)
        guard = state.get("guard")
        if isinstance(guard, dict):
            raw_closes = guard.get("closes")
            if isinstance(raw_closes, dict):
                for symbol, dq in self._guard.closes.items():
                    _restore_bounded_deque(dq, raw_closes.get(symbol))
            _restore_bounded_deque(self._guard.rho_history, guard.get("rho_history"))
            self._guard.last_time_key = str(guard.get("last_time_key") or "")
            self._guard.engaged = bool(guard.get("engaged", False))
            dwell = safe_float(guard.get("dwell"))
            self._guard.dwell = int(dwell) if dwell is not None and dwell >= 0 else 0
            scale = safe_float(guard.get("current_scale"))
            self._guard.current_scale = (
                max(0.0, min(1.0, float(scale))) if scale is not None else 1.0
            )
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

    # -------------------------------------------------------------- gauge IO

    def _symbol_close(self, event: Any, symbol: str) -> float | None:
        """Resolve a symbol close from the event (window / market / feature cascade)."""
        snapshot = _window_snapshot(event, symbol)
        if snapshot is not None and snapshot.close is not None:
            return float(snapshot.close)
        if str(getattr(event, "symbol", "")) == symbol:
            market = _market_snapshot(event)
            if market is not None and market.close is not None:
                return float(market.close)
        for field_name in ("close", "mark_price", "index_price"):
            value = _extract_feature(self.bars, event, symbol, field_name)
            if value is not None and value > 0.0:
                return float(value)
        return None

    def _ingest(self, event: Any) -> None:
        """Advance closed-bar state once per new bar: closes, rho-bar, guard flag."""
        key = time_key(getattr(event, "time", None))
        if key and key == self._guard.last_time_key:
            return
        updated = False
        for symbol, dq in self._guard.closes.items():
            close = self._symbol_close(event, symbol)
            if close is not None and close > 0.0:
                dq.append(float(close))
                updated = True
        if not updated:
            return
        self._guard.last_time_key = key
        self._update_guard()

    def _basket_returns(self) -> list[list[float]]:
        """Trailing ``corr_window`` log returns per symbol (fixed symbol order)."""
        series: list[list[float]] = []
        need = self.corr_window + 1
        for symbol in self.symbol_list:
            closes = list(self._guard.closes.get(symbol, ()))
            if len(closes) < need:
                continue
            window = closes[-need:]
            returns: list[float] = []
            ok = True
            for prev, cur in pairwise(window):
                if prev <= 0.0 or cur <= 0.0:
                    ok = False
                    break
                returns.append(math.log(cur / prev))
            if ok and len(returns) == self.corr_window:
                series.append(returns)
        return series

    def _update_guard(self) -> None:
        """Recompute rho-bar, its z-score, and step the two-state hysteresis guard."""
        series = self._basket_returns()
        if len(series) < self.min_symbols:
            # Warmup / insufficient breadth: hold the current guard state, stay
            # never-raise. A never-engaged overlay is thus neutral at 1.0.
            self._guard.current_scale = self.derisk_scale if self._guard.engaged else 1.0
            return
        rho_bar = _average_pairwise_correlation(series)
        if rho_bar is None:
            self._guard.current_scale = self.derisk_scale if self._guard.engaged else 1.0
            return
        self._guard.rho_history.append(float(rho_bar))
        z = _rolling_z(list(self._guard.rho_history))
        if self._guard.engaged:
            self._guard.dwell += 1
            if z <= self.z_exit and self._guard.dwell >= self.min_dwell_bars:
                self._guard.engaged = False
                self._guard.dwell = 0
        else:
            if z >= self.z_enter and rho_bar >= self.corr_abs_floor:
                self._guard.engaged = True
                self._guard.dwell = 0
        self._guard.current_scale = self.derisk_scale if self._guard.engaged else 1.0

    def _overlay_scale(self) -> float:
        return float(self._guard.current_scale)

    # ------------------------------------------------------------- forwarding

    def _forward_child_signal(self, signal: SignalEvent, overlay_scale: float) -> None:
        signal_type = str(getattr(signal, "signal_type", "")).upper()
        metadata = dict(signal.metadata or {})
        is_exit = signal_type == "EXIT"

        if is_exit:
            forwarded = self._rebuild_signal(signal, metadata, strength_scale=1.0)
            self.events.put(forwarded)
            return

        sym = str(signal.symbol)
        child_alloc = safe_float(metadata.get("target_allocation"))
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

        for meta_field in ("target_allocation", "max_order_value"):
            value = safe_float(metadata.get(meta_field))
            if value is not None and value > 0.0:
                metadata[meta_field] = float(value) * overlay_scale
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
            self._ingest(event)
            overlay_scale = self._overlay_scale()
            self._child.calculate_signals(event)
            self._drain(overlay_scale)
        except Exception:
            self._child_queue.drain()

    def calculate_signals_window(self, event: Any, aggregator: Any) -> None:
        try:
            self._ingest(event)
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
            self._ingest(event)
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


# --------------------------------------------------------------------------- #
# Candidate-wiring hints for the integrator (this lane does NOT wire candidates
# itself -- new-file-only, no shared-file edits).  Admission is the overlay A/B
# route: same child wrapped in the incumbent vol-managed overlay vs this one, on
# identical WF windows / cost model; the thesis is MDD/Calmar reduction only.
# --------------------------------------------------------------------------- #
_SUGGESTED_FAMILY = "overlay"
_SUGGESTED_CANDIDATE_TAGS: tuple[str, ...] = (
    "overlay",
    "risk_overlay",
    "correlation_crash_guard",
    "pollet_wilson",
    "de_risk",
    "crypto",
)

_CORRELATION_CRASH_GUARD_OVERLAY_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "1d": (
        {
            "variant": "equicorr_guard",
            "corr_window": 96,
            "corr_z_window": 240,
            "z_enter": 1.5,
            "z_exit": 0.5,
            "corr_abs_floor": 0.35,
            "derisk_scale": 0.35,
            "min_dwell_bars": 24,
        },
        {
            "variant": "equicorr_guard_tight",
            "corr_window": 72,
            "corr_z_window": 180,
            "z_enter": 1.75,
            "z_exit": 0.4,
            "corr_abs_floor": 0.40,
            "derisk_scale": 0.30,
            "min_dwell_bars": 18,
        },
    ),
}

__all__ = [
    "AvgCorrelationCrashGuardOverlayStrategy",
    "_average_pairwise_correlation",
]

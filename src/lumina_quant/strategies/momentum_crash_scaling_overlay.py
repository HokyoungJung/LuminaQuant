"""Momentum-crash dynamic-scaling risk overlay (a de-risk-only WRAPPER).

``MomentumCrashDynamicScalingOverlayStrategy`` instantiates a CHILD momentum
strategy and rescales the notional of every non-EXIT signal the child emits by a
CONTINUOUS Daniel-Moskowitz (2016) momentum-crash weight.  Like
``VolManagedRiskOverlayStrategy`` it introduces ZERO new return-seeking signal --
it can only down-weight (never lever beyond) the child's own intent -- so it
cannot worsen the child's overfitting.  The thesis is drawdown / Calmar
variance reduction of an existing momentum book, NEVER a Sharpe-lift claim.

The residual this overlay occupies vs the incumbent vol-managed overlay (which
already carries a Daniel-Moskowitz-cited BINARY conjunction crash gate):

- CONTINUOUS mean-variance weight, not a binary cliff.  Per completed benchmark
  bar a bear/rebound state ``(I_bear, I_rebound)`` and a child-own variance
  ``sigma^2`` (Barroso-Santa-Clara 2015 own-vol denominator) drive
  ``mu_hat = mu_base * (1 - b_bear*I_bear - b_panic*I_bear*I_rebound)`` (floored at
  0) and ``raw_scale = clamp(lambda * mu_hat / max(sigma^2, eps), 0, 1)``.
- A bear x REBOUND interaction the incumbent's gate cannot see: the incumbent has
  ZERO rebound logic, so on a deep-drawdown panic-rebound (WML behaves like a
  written call, DM2016) it stays at full exposure while this overlay throttles.
- The Barroso-Santa-Clara OWN vol of the momentum book (last forwarded per-symbol
  allocations x per-bar symbol returns), which spikes when the short leg turns
  convex even when benchmark vol is moderate; never-raise fallback to benchmark
  realized vol on thin history.

``raw_scale`` is QUANTIZED to a 5-level ladder ``{0, 0.25, 0.5, 0.75, 1.0}`` with
min-hold hysteresis on bucket transitions: de-risking (a lower bucket) applies
immediately, re-risking (a higher bucket) waits ``bucket_min_hold_bars``.  Non-EXIT
child signals have their ``target_allocation``, ``max_order_value`` and ``strength``
multiplied by the bucket; EXIT signals are forwarded UNSCALED so de-risking is
never blocked.  A ``rebalance_band`` suppresses re-emit of a non-EXIT signal whose
multiplier has not moved beyond the band since the prior emit for that symbol.

The overlay is data-local: it tracks only closed-bar closes (benchmark + the child
book) in bounded deques and computes the state on closed bars (no-lookahead -- the
scale at bar ``t`` never reads ``t+1``).  ``calculate_signals`` never raises.

NOTE: authoring lane module -- intentionally NOT ``@register``-ed (an unregistered
module under ``strategies/`` is inert); the integration owner wires registration,
the ``research_only`` tier hint, candidate builders and the manifest/baseline
re-pin in one atomic commit.  The overlay-family naming deviation from
``*_alpha_sleeves.py`` mirrors the ``vol_managed_risk_overlay.py`` sibling.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from lumina_quant.core.events import SignalEvent
from lumina_quant.indicators.alpha_features import (
    drawdown_from_peak,
    realized_volatility,
    simple_return,
)
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.indicators.rolling_stats import ewma_volatility
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

_STRATEGY_ID = "momentum_crash_scaling_overlay"
_STRATEGY_NAME = "MomentumCrashDynamicScalingOverlayStrategy"
_EPS = 1e-12

# The de-risk ladder: quantized scale buckets, never above 1.0 (de-risk only).
_LADDER: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
_LADDER_STEP = 0.25


def bear_rebound_state(
    closes: Any,
    *,
    bear_lookback: int,
    rebound_lookback: int,
    rebound_threshold: float,
    dd_window: int,
    dd_threshold: float,
) -> tuple[int, int] | None:
    """Return the Daniel-Moskowitz ``(I_bear, I_rebound)`` state on closed bars.

    ``I_bear`` fires when the trailing ``bear_lookback`` benchmark return is
    negative OR the ``dd_window`` drawdown-from-peak is at or below
    ``dd_threshold`` (Cooper-Gutierrez-Hameed / DM2016 bear state).  ``I_rebound``
    fires when the short-window ``rebound_lookback`` return is at or above
    ``rebound_threshold`` (the DM2016 panic-rebound state).  ``None`` on
    insufficient / non-finite history; pure list-of-float arithmetic, never raises.
    """
    values = [float(v) for v in closes if (p := safe_float(v)) is not None and p > 0.0]
    need = max(int(bear_lookback), int(rebound_lookback), int(dd_window)) + 1
    if len(values) < need:
        return None
    bear_ret = simple_return(values, lookback=max(1, int(bear_lookback)))
    drawdown = drawdown_from_peak(values, window=max(2, int(dd_window)))
    in_bear = (bear_ret is not None and bear_ret < 0.0) or (
        drawdown is not None and drawdown <= float(dd_threshold)
    )
    rebound_ret = simple_return(values, lookback=max(1, int(rebound_lookback)))
    in_rebound = rebound_ret is not None and rebound_ret >= float(rebound_threshold)
    return (1 if in_bear else 0, 1 if in_rebound else 0)


def _quantize_to_ladder(raw: float) -> float:
    """Snap a raw scale in ``[0, 1]`` to the nearest 5-level ladder bucket."""
    clamped = max(0.0, min(1.0, float(raw)))
    idx = int(clamped / _LADDER_STEP + 0.5)  # round-half-up, deterministic
    idx = max(0, min(len(_LADDER) - 1, idx))
    return _LADDER[idx]


def _restore_bounded_deque(target: deque[float], values: Any) -> None:
    """Refill a bounded deque from a serialized list, dropping non-finite values."""
    target.clear()
    for value in list(values or [])[-int(target.maxlen or 0) :]:
        parsed = safe_float(value)
        if parsed is not None:
            target.append(parsed)


@dataclass(slots=True)
class _OverlayState:
    """Benchmark closes, child-proxy returns, and the held ladder bucket."""

    closes: deque[float]
    proxy_returns: deque[float]
    last_close: dict[str, float] = field(default_factory=dict)
    last_signed_alloc: dict[str, float] = field(default_factory=dict)
    last_time_key: str = ""
    current_bucket: float = 1.0
    bars_since_bucket_change: int = 0


class MomentumCrashDynamicScalingOverlayStrategy(Strategy):
    """Continuous momentum-crash de-risk WRAPPER that rescales a child's signals."""

    decision_cadence_seconds = 86400
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "child_strategy_class": HyperParam.string(
                "child_strategy_class",
                default="LowTurnoverTrendPersistenceStrategy",
                tunable=False,
            ),
            "bear_lookback_bars": HyperParam.integer(
                "bear_lookback_bars", default=90, low=5, high=2000
            ),
            "dd_window_bars": HyperParam.integer("dd_window_bars", default=180, low=5, high=4000),
            "dd_threshold": HyperParam.floating("dd_threshold", default=-0.20, low=-1.0, high=0.0),
            "rebound_lookback_bars": HyperParam.integer(
                "rebound_lookback_bars", default=10, low=1, high=500
            ),
            "rebound_threshold": HyperParam.floating(
                "rebound_threshold", default=0.05, low=0.0, high=1.0
            ),
            "mu_base": HyperParam.floating("mu_base", default=1.0, low=0.0, high=2.0),
            "b_bear": HyperParam.floating("b_bear", default=0.5, low=0.0, high=1.0),
            "b_panic": HyperParam.floating("b_panic", default=0.5, low=0.0, high=1.0),
            "sigma_half_life_bars": HyperParam.integer(
                "sigma_half_life_bars", default=20, low=2, high=500
            ),
            "lambda_scale": HyperParam.floating("lambda_scale", default=1.0e-3, low=0.0, high=1.0),
            "vol_window": HyperParam.integer("vol_window", default=60, low=5, high=500),
            "bucket_min_hold_bars": HyperParam.integer(
                "bucket_min_hold_bars", default=24, low=0, high=2000
            ),
            "benchmark_symbol": HyperParam.string(
                "benchmark_symbol", default="BTC/USDT", tunable=False
            ),
            "rebalance_band": HyperParam.floating(
                "rebalance_band", default=0.10, low=0.0, high=2.0
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

        self.bear_lookback_bars = max(1, int(resolved["bear_lookback_bars"]))
        self.dd_window_bars = max(2, int(resolved["dd_window_bars"]))
        self.dd_threshold = min(0.0, float(resolved["dd_threshold"]))
        self.rebound_lookback_bars = max(1, int(resolved["rebound_lookback_bars"]))
        self.rebound_threshold = max(0.0, float(resolved["rebound_threshold"]))
        self.mu_base = max(0.0, float(resolved["mu_base"]))
        self.b_bear = max(0.0, min(1.0, float(resolved["b_bear"])))
        self.b_panic = max(0.0, min(1.0, float(resolved["b_panic"])))
        self.sigma_half_life_bars = max(2, int(resolved["sigma_half_life_bars"]))
        self.lambda_scale = max(0.0, float(resolved["lambda_scale"]))
        self.vol_window = max(2, int(resolved["vol_window"]))
        self.bucket_min_hold_bars = max(0, int(resolved["bucket_min_hold_bars"]))
        self.benchmark_symbol = str(resolved["benchmark_symbol"]).strip()
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

        size = _state_size(self.bear_lookback_bars, self.dd_window_bars, self.vol_window)
        self._overlay = _OverlayState(
            closes=deque(maxlen=size),
            proxy_returns=deque(maxlen=max(4, 8 * self.sigma_half_life_bars)),
        )
        self._last_forward: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------ state

    def get_state(self) -> dict[str, Any]:
        child_getter = getattr(self._child, "get_state", None)
        child_state = dict(child_getter() or {}) if callable(child_getter) else {}
        overlay: dict[str, Any] = {
            "closes": list(self._overlay.closes),
            "proxy_returns": list(self._overlay.proxy_returns),
            "last_close": {k: float(v) for k, v in self._overlay.last_close.items()},
            "last_signed_alloc": {k: float(v) for k, v in self._overlay.last_signed_alloc.items()},
            "last_time_key": self._overlay.last_time_key,
            "current_bucket": float(self._overlay.current_bucket),
            "bars_since_bucket_change": int(self._overlay.bars_since_bucket_change),
        }
        return {
            "child": child_state,
            "overlay": overlay,
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
            _restore_bounded_deque(self._overlay.closes, overlay.get("closes"))
            _restore_bounded_deque(self._overlay.proxy_returns, overlay.get("proxy_returns"))
            self._overlay.last_close = _restore_float_map(overlay.get("last_close"), positive=True)
            self._overlay.last_signed_alloc = _restore_float_map(
                overlay.get("last_signed_alloc"), positive=False
            )
            self._overlay.last_time_key = str(overlay.get("last_time_key") or "")
            bucket = safe_float(overlay.get("current_bucket"))
            self._overlay.current_bucket = (
                _quantize_to_ladder(bucket) if bucket is not None else 1.0
            )
            held = safe_float(overlay.get("bars_since_bucket_change"))
            self._overlay.bars_since_bucket_change = (
                int(held) if held is not None and held >= 0 else 0
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

    # ----------------------------------------------------------- benchmark IO

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
        """Advance closed-bar state once per new bar: closes, proxy returns, bucket."""
        key = time_key(getattr(event, "time", None))
        if key and key == self._overlay.last_time_key:
            return
        bench_close = self._symbol_close(event, self.benchmark_symbol)
        if bench_close is None or bench_close <= 0.0:
            return
        self._overlay.last_time_key = key
        self._overlay.closes.append(float(bench_close))

        # Child-proxy portfolio return = sum_i (last signed alloc_i * per-bar return_i).
        proxy_return = 0.0
        saw_return = False
        symbols = set(self.symbol_list)
        symbols.add(self.benchmark_symbol)
        for symbol in symbols:
            close = (
                bench_close
                if symbol == self.benchmark_symbol
                else self._symbol_close(event, symbol)
            )
            if close is None or close <= 0.0:
                continue
            prev = self._overlay.last_close.get(symbol)
            if prev is not None and prev > 0.0:
                weight = self._overlay.last_signed_alloc.get(symbol, 0.0)
                if weight != 0.0:
                    proxy_return += weight * (close / prev - 1.0)
                saw_return = True
            self._overlay.last_close[symbol] = float(close)
        if saw_return:
            self._overlay.proxy_returns.append(float(proxy_return))

        self._advance_bucket()

    def _child_variance(self) -> float | None:
        """Barroso-Santa-Clara child-own variance; benchmark-vol fallback on thin history."""
        proxy = list(self._overlay.proxy_returns)
        if len(proxy) >= 2:
            vol = ewma_volatility(proxy, half_life=self.sigma_half_life_bars)
            if vol is not None:
                return float(vol) * float(vol)
        bench_vol = realized_volatility(list(self._overlay.closes), window=self.vol_window)
        if bench_vol is not None:
            return float(bench_vol) * float(bench_vol)
        return None

    def _mu_hat(self) -> float:
        state = bear_rebound_state(
            list(self._overlay.closes),
            bear_lookback=self.bear_lookback_bars,
            rebound_lookback=self.rebound_lookback_bars,
            rebound_threshold=self.rebound_threshold,
            dd_window=self.dd_window_bars,
            dd_threshold=self.dd_threshold,
        )
        if state is None:
            return self.mu_base
        i_bear, i_rebound = state
        mu = self.mu_base * (
            1.0 - self.b_bear * float(i_bear) - self.b_panic * float(i_bear) * float(i_rebound)
        )
        return max(0.0, mu)

    def _raw_scale(self, mu_hat: float, sigma_sq: float | None) -> float:
        """``clamp(lambda * mu_hat / max(sigma^2, eps), 0, 1)`` (vol throttle inert if unknown)."""
        mu = max(0.0, float(mu_hat))
        if sigma_sq is None or sigma_sq <= _EPS:
            raw = mu  # no vol information -> vol throttle is inert, mu_hat governs
        else:
            raw = self.lambda_scale * mu / float(sigma_sq)
        return max(0.0, min(1.0, raw))

    def _advance_bucket(self) -> None:
        """Recompute the target ladder bucket with immediate de-risk / delayed re-risk."""
        raw = self._raw_scale(self._mu_hat(), self._child_variance())
        target = _quantize_to_ladder(raw)
        current = self._overlay.current_bucket
        if target < current:
            # De-risking is never blocked -- apply immediately.
            self._overlay.current_bucket = target
            self._overlay.bars_since_bucket_change = 0
        elif target > current:
            # Re-risking waits out the min-hold hysteresis window.
            if self._overlay.bars_since_bucket_change >= self.bucket_min_hold_bars:
                self._overlay.current_bucket = target
                self._overlay.bars_since_bucket_change = 0
            else:
                self._overlay.bars_since_bucket_change += 1
        else:
            self._overlay.bars_since_bucket_change += 1

    def _overlay_scale(self) -> float:
        return float(self._overlay.current_bucket)

    # ------------------------------------------------------------- forwarding

    def _forward_child_signal(self, signal: SignalEvent, overlay_scale: float) -> None:
        signal_type = str(getattr(signal, "signal_type", "")).upper()
        metadata = dict(signal.metadata or {})
        is_exit = signal_type == "EXIT"

        if is_exit:
            forwarded = self._rebuild_signal(signal, metadata, strength_scale=1.0)
            self._record_signed_alloc(signal, signal_type, 0.0)
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
                # Redundant re-size of the same intent -- but still refresh the
                # proxy weight so the own-vol estimate tracks the held book.
                self._record_signed_alloc(signal, signal_type, child_alloc)
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
        self._record_signed_alloc(signal, signal_type, child_alloc)
        self.events.put(forwarded)

    def _record_signed_alloc(
        self, signal: SignalEvent, signal_type: str, child_alloc: float | None
    ) -> None:
        """Track the child's signed target allocation for the own-vol proxy."""
        sym = str(signal.symbol)
        if signal_type == "EXIT" or child_alloc is None:
            self._overlay.last_signed_alloc[sym] = 0.0
            return
        side = str(getattr(signal, "position_side", "") or "").upper()
        sign = -1.0 if signal_type == "SHORT" or side == "SHORT" else 1.0
        self._overlay.last_signed_alloc[sym] = sign * abs(float(child_alloc))

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


def _restore_float_map(raw: Any, *, positive: bool) -> dict[str, float]:
    """Restore a ``{symbol: float}`` map from serialized state, dropping bad entries."""
    out: dict[str, float] = {}
    if isinstance(raw, dict):
        for symbol, value in raw.items():
            parsed = safe_float(value)
            if parsed is None:
                continue
            if positive and parsed <= 0.0:
                continue
            out[str(symbol)] = float(parsed)
    return out


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
    "momentum_crash",
    "daniel_moskowitz",
    "de_risk",
    "crypto",
)

_MOMENTUM_CRASH_SCALING_OVERLAY_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "1d": (
        {
            "variant": "dm_continuous",
            "bear_lookback_bars": 90,
            "dd_window_bars": 180,
            "dd_threshold": -0.20,
            "rebound_lookback_bars": 10,
            "rebound_threshold": 0.05,
            "sigma_half_life_bars": 20,
            "bucket_min_hold_bars": 24,
        },
        {
            "variant": "dm_fast_rebound",
            "bear_lookback_bars": 60,
            "dd_window_bars": 120,
            "dd_threshold": -0.15,
            "rebound_lookback_bars": 7,
            "rebound_threshold": 0.04,
            "sigma_half_life_bars": 14,
            "bucket_min_hold_bars": 16,
        },
    ),
}

__all__ = [
    "MomentumCrashDynamicScalingOverlayStrategy",
    "bear_rebound_state",
]

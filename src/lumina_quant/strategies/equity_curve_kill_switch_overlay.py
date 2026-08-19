"""Equity-curve kill-switch governance overlay (a de-risk-only WRAPPER).

``EquityCurveKillSwitchOverlayStrategy`` instantiates a CHILD return-seeking
strategy and throttles (or fully halts) the notional of every signal the child
emits, based on the CHILD'S OWN PROXY EQUITY CURVE.  Like the incumbent
``VolManagedRiskOverlayStrategy`` / ``MomentumCrashDynamicScalingOverlayStrategy``
it introduces ZERO new return-seeking signal -- it can only down-weight (never
lever beyond) the child's own intent -- so it cannot worsen the child's
overfitting.  The thesis is drawdown / ruin-risk containment ONLY, never a
Sharpe-lift claim.

Lineage (public sources, independent adaptation -- NOT a reproduction, NOT an
endorsement, NOT a performance claim):

- The publicly stated capital-management principles of the Korean retail trader
  known as "알바트로스" (성필규): a hard stop on every trade, a hard ceiling on
  ACCOUNT loss (not just per-trade loss), and mechanical adherence to those
  rules (the rule, not the feeling, decides).  Only the SHAPE of the rule is borrowed (ladder the exposure down as the
  account drawdown deepens; stop entirely at a stated account loss); no numeric
  parameter of any real trader's system is known to this module.
- FlightF / AOA public statements about capping ACCOUNT loss -- e.g. stopping
  new entries once the account is down roughly 30% -- motivate the terminal
  0-rung of the default ladder.

RESIDUAL vs the incumbent overlays (why this is not a duplicate):

- ``vol_managed_risk_overlay`` keys on a BENCHMARK price series (realized vol +
  benchmark drawdown/vol-z conjunction).
- ``momentum_crash_scaling_overlay`` keys on benchmark bear/rebound state plus a
  proxy-return VARIANCE estimate.
- THIS overlay keys on the LEVEL of the child's proxy EQUITY CURVE: peak-to-
  trough drawdown of the child book itself, the child's consecutive-losing-trade
  streak, a calendar-month loss limit, and an equity-curve moving-average filter.
  A child can be in a deep own-equity drawdown while the benchmark is calm (the
  incumbents stay at full exposure); that is exactly the state this overlay
  governs.

Composition -- ``overlay_scale`` is the PRODUCT of four de-risk-only components,
each in ``[0, 1]``:

1. DRAWDOWN LADDER (``ladder``): a ``"depth:scale,..."`` string, e.g. the default
   ``"0.05:0.75,0.10:0.50,0.15:0.25,0.20:0.0"``.  The scale of the DEEPEST
   breached rung applies (``depth >= threshold``); 1.0 when nothing is breached.
2. CONSECUTIVE-LOSS HALVING: once the child has closed
   ``consecutive_loss_halving_from`` losing trades in a row, the scale halves for
   that loss and again for every further consecutive loss, floored at
   ``min_scale``; a non-losing closed trade resets the streak.
3. MONTHLY LOSS LIMIT: if proxy equity has fallen ``month_loss_limit`` or more
   since the FIRST bar of the current UTC calendar month, the scale is 0 (kill)
   until the next calendar month.  ``0.0`` disables it.
4. EQUITY-CURVE MA FILTER: if ``equity_ma_window > 0`` and proxy equity is below
   its own SMA, the scale is multiplied by ``below_ma_scale``.  ``0`` disables it.

HYSTERESIS: de-risking applies on the bar it is detected; re-risking (a scale
INCREASE) waits ``rerisk_min_bars`` bars after the last de-risk so the book does
not flip-flop around a rung boundary.

SIZING-MODE CAVEAT (shared with the incumbent overlays): the scale multiplies
the child's ``target_allocation`` / ``max_order_value`` / ``strength`` only.  The
child's protective ``stop_loss`` / ``take_profit`` are forwarded UNTOUCHED (a
de-risk wrapper must never strip protection).  Under the default
``target_allocation_mode=legacy_notional_cap`` the portfolio sizes
``qty = equity * risk_per_trade / |price - stop|`` and uses the (scaled)
allocation only as a notional CAP, so with a wide child stop an intermediate
rung may leave the entry size unchanged (dollar risk-at-stop is governed by
``risk_per_trade``); ``notional_fraction`` / ``isolated_margin_fraction`` modes
scale fully.  Kill (scale 0) always drops entries regardless of mode.

KILL STATE (``overlay_scale == 0``): the overlay emits one EXIT per symbol that
carries a nonzero child target (once per kill episode) and then SUPPRESSES the
child's LONG/SHORT signals; child EXITs are always forwarded.  A ladder-caused
kill lifts when the proxy drawdown recovers above ``recover_drawdown``; a
month-limit kill lifts at the next UTC calendar month.

PROXY EQUITY.  The overlay never sees real fills.  It reconstructs a SHADOW book
from the child's intercepted signals -- LONG adds ``+target_allocation``, SHORT
adds ``-target_allocation`` (that is the portfolio's pyramiding semantics), EXIT
zeroes the symbol -- and compounds ``1 + sum_i w_i(prev) * (close_i/prev_close_i
- 1)`` once per closed bar.  Weights are recorded from the child's INTENT even
while signals are suppressed, so the shadow curve keeps tracking the child
unimpeded and a kill can detect the child's recovery instead of locking itself
out on a frozen curve.  No-lookahead: the scale used at bar ``t`` is computed
from bars ``<= t`` before the child is invoked, and the bar-``t`` return uses the
weights as of bar ``t-1``.

AUTHOR'S CHOICES (nothing below is a published number of any named trader): the
ladder rungs and their scales, ``consecutive_loss_halving_from=3``, the 0.5
halving factor, ``month_loss_limit=0.10``, ``recover_drawdown=0.10``,
``rerisk_min_bars=5``, ``min_scale=0.10``, and the per-symbol price-based
definition of a "losing trade".  The PUBLIC sources state only the qualitative
rules (stop losses, cap account loss, obey the rule mechanically).

``drawdown_ladder_scale`` and ``kill_switch_scale`` are pure, importable
functions with no dependency on the class, so the same governance can be applied
to a real portfolio equity curve at the book level.  research_only.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from lumina_quant.core.events import SignalEvent
from lumina_quant.core.plugin_registry import register
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.indicators.moving_average import simple_moving_average
from lumina_quant.strategies import resolve_strategy_class
from lumina_quant.strategies.artifact_portfolio_mode import (
    _BarsSubsetProxy,
    _SignalCaptureQueue,
)
from lumina_quant.strategies.external_alpha_sleeves import (
    _event_datetime_utc,
    _extract_feature,
    _market_snapshot,
    _window_snapshot,
)
from lumina_quant.strategy import Strategy
from lumina_quant.tuning import HyperParam, resolve_params_from_schema

_STRATEGY_ID = "equity_curve_kill_switch_overlay"
_EPS = 1e-12

#: Default drawdown ladder: shallow trim, then progressively harder throttling,
#: terminating in a full stop.  AUTHOR'S CHOICE (see the module docstring).
DEFAULT_LADDER = "0.05:0.75,0.10:0.50,0.15:0.25,0.20:0.0"

#: Minimum equity samples retained regardless of ``equity_ma_window``.
_MIN_EQUITY_HISTORY = 400


# --------------------------------------------------------------------------- #
# Pure, importable governance functions (no class needed to use them).
# --------------------------------------------------------------------------- #


def parse_drawdown_ladder(ladder: Any) -> tuple[tuple[float, float], ...]:
    """Parse a ladder spec into ascending ``((depth, scale), ...)`` rungs.

    Accepts either the ``"0.05:0.75,0.10:0.50"`` string form or an already
    parsed sequence of ``(depth, scale)`` pairs.  Malformed entries, negative
    depths and non-finite values are dropped rather than raised on; an empty or
    unparseable spec yields no rungs (the ladder is then inert).
    """
    rungs: list[tuple[float, float]] = []
    if ladder is None:
        return ()
    if isinstance(ladder, str):
        items: list[Any] = [chunk for chunk in ladder.split(",") if chunk.strip()]
    elif isinstance(ladder, Sequence):
        items = list(ladder)
    else:
        return ()
    for item in items:
        if isinstance(item, str):
            parts = item.split(":")
            if len(parts) != 2:
                continue
            depth = safe_float(parts[0])
            scale = safe_float(parts[1])
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            pair = list(item)
            if len(pair) != 2:
                continue
            depth = safe_float(pair[0])
            scale = safe_float(pair[1])
        else:
            continue
        if depth is None or scale is None or depth < 0.0:
            continue
        rungs.append((float(depth), float(max(0.0, min(1.0, scale)))))
    rungs.sort(key=lambda rung: rung[0])
    return tuple(rungs)


def drawdown_ladder_scale(drawdown: Any, ladder: Any) -> float:
    """Return the exposure scale for ``drawdown`` under ``ladder``.

    ``drawdown`` is a DEPTH; either sign convention is accepted (``-0.12`` and
    ``0.12`` both mean "12% below the peak").  The scale of the deepest breached
    rung wins, breach being ``depth >= threshold``; 1.0 when nothing is breached
    or the ladder is empty.  Never raises.
    """
    rungs = parse_drawdown_ladder(ladder)
    if not rungs:
        return 1.0
    depth_value = safe_float(drawdown)
    if depth_value is None:
        return 1.0
    depth = abs(float(depth_value))
    scale = 1.0
    for threshold, rung_scale in rungs:
        if depth + _EPS >= threshold:
            scale = rung_scale
        else:
            break
    return float(max(0.0, min(1.0, scale)))


def _split_equity_curve(equity_curve: Any) -> tuple[list[float], list[Any]]:
    """Normalize an equity curve into ``(values, stamps)``.

    Accepts a plain sequence of equity levels, or a sequence of
    ``(timestamp, equity)`` pairs (``stamps`` is empty in the plain case).
    """
    values: list[float] = []
    stamps: list[Any] = []
    if not isinstance(equity_curve, Sequence) or isinstance(equity_curve, (str, bytes)):
        try:
            equity_curve = list(equity_curve)
        except Exception:
            return [], []
    for item in equity_curve:
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) == 2:
            pair = list(item)
            parsed = safe_float(pair[1])
            if parsed is None:
                continue
            stamps.append(pair[0])
            values.append(float(parsed))
            continue
        parsed = safe_float(item)
        if parsed is None:
            continue
        values.append(float(parsed))
    if stamps and len(stamps) != len(values):
        return values, []
    return values, stamps


def _month_key(raw_time: Any) -> str:
    """UTC calendar-month key (``"YYYY-MM"``) for a raw event timestamp."""
    parsed = _event_datetime_utc(raw_time)
    if parsed is None:
        return ""
    return f"{parsed.year:04d}-{parsed.month:02d}"


def _month_anchor_from_stamps(values: list[float], stamps: list[Any]) -> float | None:
    """Equity at the first sample of the LAST sample's UTC calendar month."""
    if not values or not stamps:
        return None
    current = _month_key(stamps[-1])
    if not current:
        return None
    anchor: float | None = None
    for value, stamp in zip(values, stamps, strict=False):
        if _month_key(stamp) == current:
            anchor = float(value)
            break
    return anchor


def _trailing_loss_streak(values: list[float]) -> int:
    """Consecutive strictly-negative increments at the tail of an equity curve."""
    streak = 0
    for idx in range(len(values) - 1, 0, -1):
        if values[idx] < values[idx - 1]:
            streak += 1
        else:
            break
    return streak


def kill_switch_scale(
    equity_curve: Any,
    *,
    ladder: Any = DEFAULT_LADDER,
    consecutive_loss_halving_from: int = 3,
    month_loss_limit: float = 0.10,
    equity_ma_window: int = 0,
    below_ma_scale: float = 0.5,
    min_scale: float = 0.10,
    loss_streak: int | None = None,
    month_start_equity: float | None = None,
    peak: float | None = None,
) -> tuple[float, dict[str, Any]]:
    """Governance scale for an equity curve, plus a diagnostics dict.

    ``equity_curve`` is a sequence of equity LEVELS (any positive base), or of
    ``(timestamp, equity)`` pairs -- the pair form lets the monthly loss limit
    find its own calendar anchor.

    The optional keywords let a caller that already tracks richer state override
    the curve-derived estimates: ``loss_streak`` (closed-TRADE losing streak,
    versus the curve's consecutive down-bars), ``month_start_equity`` (the
    calendar anchor), and ``peak`` (a running all-time peak, for callers whose
    retained window is shorter than the full history).

    Returns ``(scale, diagnostics)`` where ``scale`` is in ``{0.0} U [min_scale,
    1.0]``: a 0 from the ladder's terminal rung or the monthly limit is a KILL
    and is never lifted to ``min_scale``.  Never raises.
    """
    diagnostics: dict[str, Any] = {
        "equity": None,
        "peak": None,
        "drawdown": 0.0,
        "ladder_scale": 1.0,
        "loss_streak": 0,
        "streak_scale": 1.0,
        "month_loss": None,
        "month_scale": 1.0,
        "equity_ma": None,
        "ma_scale": 1.0,
        "reasons": [],
        "reason": "",
        "scale": 1.0,
    }
    values, stamps = _split_equity_curve(equity_curve)
    if not values:
        return 1.0, diagnostics

    equity = values[-1]
    peak_value = safe_float(peak)
    running_peak = float(peak_value) if peak_value is not None else max(values)
    running_peak = max(running_peak, equity)
    drawdown = 0.0
    if running_peak > _EPS:
        drawdown = max(0.0, 1.0 - equity / running_peak)
    diagnostics["equity"] = float(equity)
    diagnostics["peak"] = float(running_peak)
    diagnostics["drawdown"] = float(drawdown)

    floor = max(0.0, min(1.0, float(min_scale)))

    # (1) drawdown ladder
    ladder_scale = drawdown_ladder_scale(drawdown, ladder)
    diagnostics["ladder_scale"] = float(ladder_scale)

    # (2) consecutive-loss halving
    streak = int(loss_streak) if loss_streak is not None else _trailing_loss_streak(values)
    streak = max(0, streak)
    trigger = int(consecutive_loss_halving_from)
    streak_scale = 1.0
    if trigger > 0 and streak >= trigger:
        streak_scale = max(floor, 0.5 ** (streak - trigger + 1))
    diagnostics["loss_streak"] = streak
    diagnostics["streak_scale"] = float(streak_scale)

    # (3) monthly loss limit
    month_scale = 1.0
    limit = max(0.0, float(month_loss_limit))
    if limit > 0.0:
        anchor = safe_float(month_start_equity)
        if anchor is None:
            anchor = _month_anchor_from_stamps(values, stamps)
        if anchor is not None and float(anchor) > _EPS:
            month_loss = 1.0 - equity / float(anchor)
            diagnostics["month_loss"] = float(month_loss)
            if month_loss + _EPS >= limit:
                month_scale = 0.0
    diagnostics["month_scale"] = float(month_scale)

    # (4) equity-curve MA filter
    ma_scale = 1.0
    window = int(equity_ma_window)
    if window > 0 and len(values) >= window:
        moving_average = simple_moving_average(values, window)
        if moving_average is not None:
            diagnostics["equity_ma"] = float(moving_average)
            if equity < float(moving_average):
                ma_scale = max(0.0, min(1.0, float(below_ma_scale)))
    diagnostics["ma_scale"] = float(ma_scale)

    reasons: list[str] = []
    if ladder_scale < 1.0:
        reasons.append("drawdown_ladder")
    if streak_scale < 1.0:
        reasons.append("loss_streak")
    if month_scale < 1.0:
        reasons.append("month_limit")
    if ma_scale < 1.0:
        reasons.append("below_equity_ma")
    diagnostics["reasons"] = reasons
    diagnostics["reason"] = ",".join(reasons) if reasons else "none"

    product = ladder_scale * streak_scale * month_scale * ma_scale
    scale = 0.0 if product <= _EPS else max(floor, min(1.0, product))
    diagnostics["scale"] = float(scale)
    return float(scale), diagnostics


# --------------------------------------------------------------------------- #
# The wrapper strategy.
# --------------------------------------------------------------------------- #


def _sign(value: float) -> int:
    if value > _EPS:
        return 1
    if value < -_EPS:
        return -1
    return 0


@dataclass(slots=True)
class _OverlayState:
    """Shadow-book weights, the proxy equity curve, and the kill-switch latch."""

    equity_curve: deque[float]
    equity: float = 1.0
    peak: float = 1.0
    weights: dict[str, float] = field(default_factory=dict)
    last_close: dict[str, float] = field(default_factory=dict)
    open_trades: dict[str, dict[str, float]] = field(default_factory=dict)
    loss_streak: int = 0
    month_key: str = ""
    month_start_equity: float | None = None
    killed: bool = False
    kill_source: str = ""
    kill_exits_sent: bool = False
    effective_scale: float = 1.0
    bars_since_derisk: int = 0
    last_time_key: str = ""


@register("strategy", "EquityCurveKillSwitchOverlayStrategy", interface="event_driven")
class EquityCurveKillSwitchOverlayStrategy(Strategy):
    """Equity-curve kill-switch WRAPPER that throttles or halts a child strategy.

    The child is instantiated on a bars proxy with a signal-capture queue; on each
    event the overlay advances its proxy equity curve, derives ``overlay_scale``,
    lets the child compute, then rescales (or suppresses) the drained signals.
    EXIT signals are always forwarded unscaled so de-risking is never blocked.
    """

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
            "ladder": HyperParam.string("ladder", default=DEFAULT_LADDER, tunable=False),
            "consecutive_loss_halving_from": HyperParam.integer(
                "consecutive_loss_halving_from", default=3, low=0, high=20
            ),
            "month_loss_limit": HyperParam.floating(
                "month_loss_limit", default=0.10, low=0.0, high=1.0
            ),
            "equity_ma_window": HyperParam.integer("equity_ma_window", default=0, low=0, high=500),
            "below_ma_scale": HyperParam.floating("below_ma_scale", default=0.5, low=0.0, high=1.0),
            "rerisk_min_bars": HyperParam.integer("rerisk_min_bars", default=5, low=0, high=500),
            "recover_drawdown": HyperParam.floating(
                "recover_drawdown", default=0.10, low=0.0, high=1.0
            ),
            "min_scale": HyperParam.floating("min_scale", default=0.10, low=0.0, high=1.0),
        }

    def __init__(self, bars: Any, events: Any, **params: Any) -> None:
        self.bars = bars
        self.events = events
        resolved = resolve_params_from_schema(self.get_param_schema(), params, keep_unknown=False)

        # ``child_strategy`` is accepted as an alias of the incumbent overlays'
        # ``child_strategy_class`` so either wiring spelling resolves the child.
        alias = str(params.get("child_strategy") or "").strip()
        self.child_strategy_class = alias or str(resolved["child_strategy_class"]).strip()
        # ``child_params``/``child_symbols`` are structural wiring, not tunables --
        # read defensively from the raw params (as the incumbent overlays do).
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

        self.ladder = str(resolved["ladder"])
        self._ladder_rungs = parse_drawdown_ladder(self.ladder)
        self.consecutive_loss_halving_from = max(0, int(resolved["consecutive_loss_halving_from"]))
        self.month_loss_limit = max(0.0, float(resolved["month_loss_limit"]))
        self.equity_ma_window = max(0, int(resolved["equity_ma_window"]))
        self.below_ma_scale = max(0.0, min(1.0, float(resolved["below_ma_scale"])))
        self.rerisk_min_bars = max(0, int(resolved["rerisk_min_bars"]))
        self.recover_drawdown = max(0.0, float(resolved["recover_drawdown"]))
        self.min_scale = max(0.0, min(1.0, float(resolved["min_scale"])))

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
        self.decision_cadence_seconds = (
            int(child_cadence)
            if isinstance(child_cadence, (int, float)) and child_cadence > 0
            else None
        )
        child_contract = getattr(self._child, "preferred_contract", None)
        if isinstance(child_contract, str) and child_contract:
            self.preferred_contract = child_contract
        # Fallback allocation when a child signal carries no explicit target.
        self._child_default_alloc = safe_float(getattr(self._child, "target_allocation", None))

        self._overlay = _OverlayState(
            equity_curve=deque(maxlen=max(self.equity_ma_window, _MIN_EQUITY_HISTORY)),
        )
        self._last_event_time: Any = None
        self._last_diagnostics: dict[str, Any] = {}

    # ------------------------------------------------------------------ state

    def get_state(self) -> dict[str, Any]:
        child_getter = getattr(self._child, "get_state", None)
        child_state = dict(child_getter() or {}) if callable(child_getter) else {}
        overlay = self._overlay
        return {
            "child": child_state,
            "overlay": {
                "equity_curve": list(overlay.equity_curve),
                "equity": float(overlay.equity),
                "peak": float(overlay.peak),
                "weights": {k: float(v) for k, v in overlay.weights.items()},
                "last_close": {k: float(v) for k, v in overlay.last_close.items()},
                "open_trades": {
                    k: {"close": float(v.get("close", 0.0)), "side": float(v.get("side", 0.0))}
                    for k, v in overlay.open_trades.items()
                },
                "loss_streak": int(overlay.loss_streak),
                "month_key": str(overlay.month_key),
                "month_start_equity": (
                    None
                    if overlay.month_start_equity is None
                    else float(overlay.month_start_equity)
                ),
                "killed": bool(overlay.killed),
                "kill_source": str(overlay.kill_source),
                "kill_exits_sent": bool(overlay.kill_exits_sent),
                "effective_scale": float(overlay.effective_scale),
                "bars_since_derisk": int(overlay.bars_since_derisk),
                "last_time_key": str(overlay.last_time_key),
            },
        }

    def set_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        child_state = state.get("child")
        setter = getattr(self._child, "set_state", None)
        if callable(setter) and isinstance(child_state, dict):
            setter(child_state)
        raw = state.get("overlay")
        if not isinstance(raw, dict):
            return
        overlay = self._overlay
        overlay.equity_curve.clear()
        for value in list(raw.get("equity_curve") or [])[-int(overlay.equity_curve.maxlen or 0) :]:
            parsed = safe_float(value)
            if parsed is not None:
                overlay.equity_curve.append(float(parsed))
        restored_equity = safe_float(raw.get("equity"))
        overlay.equity = 1.0 if restored_equity is None else float(restored_equity)
        restored_peak = safe_float(raw.get("peak"))
        overlay.peak = overlay.equity if restored_peak is None else float(restored_peak)
        overlay.weights = _restore_float_map(raw.get("weights"))
        overlay.last_close = {
            key: value
            for key, value in _restore_float_map(raw.get("last_close")).items()
            if value > 0.0
        }
        overlay.open_trades = _restore_trade_map(raw.get("open_trades"))
        overlay.loss_streak = max(0, int(safe_float(raw.get("loss_streak")) or 0))
        overlay.month_key = str(raw.get("month_key") or "")
        overlay.month_start_equity = safe_float(raw.get("month_start_equity"))
        overlay.killed = bool(raw.get("killed"))
        overlay.kill_source = str(raw.get("kill_source") or "")
        overlay.kill_exits_sent = bool(raw.get("kill_exits_sent"))
        restored_scale = safe_float(raw.get("effective_scale"))
        overlay.effective_scale = 1.0 if restored_scale is None else float(restored_scale)
        overlay.bars_since_derisk = max(0, int(safe_float(raw.get("bars_since_derisk")) or 0))
        overlay.last_time_key = str(raw.get("last_time_key") or "")

    # ----------------------------------------------------------------- prices

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

    # ---------------------------------------------------------------- ingest

    def _ingest(self, event: Any) -> None:
        """Advance the proxy equity curve and the kill latch once per closed bar."""
        key = time_key(getattr(event, "time", None))
        if key and key == self._overlay.last_time_key:
            return
        overlay = self._overlay
        overlay.last_time_key = key

        # Proxy portfolio return uses the PREVIOUS bar's shadow weights.
        proxy_return = 0.0
        saw_bar = False
        for symbol in sorted(set(self.symbol_list) | set(overlay.weights)):
            close = self._symbol_close(event, symbol)
            if close is None or close <= 0.0:
                continue
            saw_bar = True
            prev = overlay.last_close.get(symbol)
            if prev is not None and prev > 0.0:
                weight = overlay.weights.get(symbol, 0.0)
                if weight != 0.0:
                    proxy_return += weight * (close / prev - 1.0)
            overlay.last_close[symbol] = float(close)
        if not saw_bar:
            return

        overlay.equity = max(_EPS, overlay.equity * (1.0 + proxy_return))
        overlay.equity_curve.append(float(overlay.equity))
        overlay.peak = max(overlay.peak, overlay.equity)
        overlay.bars_since_derisk += 1

        self._advance_month(event)
        self._advance_scale()

    def _advance_month(self, event: Any) -> None:
        """Roll the UTC calendar-month anchor; a month roll lifts a month kill."""
        overlay = self._overlay
        key = _month_key(getattr(event, "time", None))
        if not key:
            return
        if key == overlay.month_key:
            return
        overlay.month_key = key
        overlay.month_start_equity = float(overlay.equity)
        if overlay.killed and overlay.kill_source == "month":
            overlay.killed = False
            overlay.kill_source = ""
            overlay.kill_exits_sent = False

    def _advance_scale(self) -> None:
        """Recompute the raw scale, apply the kill latch, then the hysteresis."""
        overlay = self._overlay
        raw_scale, diagnostics = kill_switch_scale(
            list(overlay.equity_curve),
            ladder=self._ladder_rungs,
            consecutive_loss_halving_from=self.consecutive_loss_halving_from,
            month_loss_limit=self.month_loss_limit,
            equity_ma_window=self.equity_ma_window,
            below_ma_scale=self.below_ma_scale,
            min_scale=self.min_scale,
            loss_streak=overlay.loss_streak,
            month_start_equity=overlay.month_start_equity,
            peak=overlay.peak,
        )
        self._last_diagnostics = diagnostics
        drawdown = float(safe_float(diagnostics.get("drawdown")) or 0.0)

        # Latch: enter the kill on any zero component, recording its source.
        # NOTE: a 0.0 component is falsy -- resolve it explicitly, never via
        # ``or``, or a month kill gets mislabelled as a ladder kill and then
        # waits for a drawdown recovery that the month rule never promises.
        if raw_scale <= _EPS and not overlay.killed:
            month_scale = safe_float(diagnostics.get("month_scale"))
            overlay.killed = True
            overlay.kill_source = (
                "month" if month_scale is not None and month_scale <= _EPS else "ladder"
            )
            overlay.kill_exits_sent = False
        elif overlay.killed and overlay.kill_source == "ladder":
            # A month roll clears a month kill (see ``_advance_month``); a ladder
            # kill clears only once the proxy drawdown has actually healed.
            if drawdown < self.recover_drawdown:
                overlay.killed = False
                overlay.kill_source = ""
                overlay.kill_exits_sent = False
        if overlay.killed:
            raw_scale = 0.0
            reasons = [r for r in (diagnostics.get("reasons") or []) if r != "none"]
            reasons.append(f"kill_switch_{overlay.kill_source or 'latched'}")
            diagnostics["reasons"] = reasons
            diagnostics["reason"] = ",".join(reasons)

        # Hysteresis: de-risk immediately, re-risk only after a quiet period.
        if raw_scale < overlay.effective_scale - _EPS:
            overlay.effective_scale = float(raw_scale)
            overlay.bars_since_derisk = 0
        elif raw_scale > overlay.effective_scale + _EPS:
            if overlay.bars_since_derisk >= self.rerisk_min_bars:
                overlay.effective_scale = float(raw_scale)
        diagnostics["effective_scale"] = float(overlay.effective_scale)

    # ------------------------------------------------------------- kill exits

    def _emit_kill_exits(self) -> None:
        """Emit one EXIT per symbol carrying a nonzero child target (once per kill)."""
        overlay = self._overlay
        if not overlay.killed or overlay.kill_exits_sent:
            return
        overlay.kill_exits_sent = True
        for symbol in sorted(overlay.weights):
            if abs(overlay.weights.get(symbol, 0.0)) <= _EPS:
                continue
            self.events.put(
                SignalEvent(
                    strategy_id=f"{_STRATEGY_ID}::{self.child_strategy_class}",
                    symbol=str(symbol),
                    datetime=self._last_event_time,
                    signal_type="EXIT",
                    strength=1.0,
                    metadata={
                        "strategy": _STRATEGY_ID,
                        "kill_switch": True,
                        "overlay_scale": 0.0,
                        "overlay_reason": str(self._last_diagnostics.get("reason") or "kill"),
                        "proxy_drawdown": self._diagnostic_drawdown(),
                        "child_strategy_class": self.child_strategy_class,
                    },
                )
            )

    # ------------------------------------------------------------- forwarding

    def _diagnostic_drawdown(self) -> float:
        parsed = safe_float(self._last_diagnostics.get("drawdown"))
        return float(parsed) if parsed is not None else 0.0

    def _child_alloc(self, metadata: dict[str, Any]) -> float | None:
        alloc = safe_float(metadata.get("target_allocation"))
        if alloc is not None:
            return float(alloc)
        return self._child_default_alloc

    def _record_intent(self, signal: SignalEvent, signal_type: str, alloc: float | None) -> None:
        """Update the shadow book from the child's INTENT (even when suppressed).

        LONG/SHORT ADD signed allocation (the portfolio's pyramiding semantics);
        EXIT zeroes the symbol.  Trade bookkeeping for the losing-streak counter
        uses the per-symbol proxy price move between entry and exit.
        """
        overlay = self._overlay
        symbol = str(signal.symbol)
        prev_weight = overlay.weights.get(symbol, 0.0)
        if signal_type == "EXIT":
            new_weight = 0.0
        else:
            side = str(getattr(signal, "position_side", "") or "").upper()
            sign = -1.0 if signal_type == "SHORT" or side == "SHORT" else 1.0
            magnitude = abs(float(alloc)) if alloc is not None else 0.0
            new_weight = prev_weight + sign * magnitude
        if abs(new_weight) <= _EPS:
            new_weight = 0.0
        overlay.weights[symbol] = float(new_weight)

        close = overlay.last_close.get(symbol)
        prev_sign = _sign(prev_weight)
        new_sign = _sign(new_weight)
        if prev_sign != 0 and new_sign != prev_sign:
            self._close_trade(symbol, close)
        if new_sign != 0 and new_sign != prev_sign and close is not None and close > 0.0:
            overlay.open_trades[symbol] = {"close": float(close), "side": float(new_sign)}

    def _close_trade(self, symbol: str, close: float | None) -> None:
        """Settle a shadow trade and advance/reset the consecutive-loss streak."""
        trade = self._overlay.open_trades.pop(symbol, None)
        if trade is None or close is None or close <= 0.0:
            return
        entry = float(trade.get("close", 0.0))
        side = float(trade.get("side", 0.0))
        if entry <= 0.0 or side == 0.0:
            return
        # ponytail: per-symbol price move is the proxy trade PnL -- fills, fees and
        # partial sizing are not modeled, only the SIGN feeds the streak counter.
        pnl = (close / entry - 1.0) * side
        if pnl < 0.0:
            self._overlay.loss_streak += 1
        else:
            self._overlay.loss_streak = 0

    def _forward_child_signal(self, signal: SignalEvent, overlay_scale: float) -> None:
        signal_type = str(getattr(signal, "signal_type", "")).upper()
        metadata = dict(signal.metadata or {})
        alloc = self._child_alloc(metadata)
        self._record_intent(signal, signal_type, alloc)

        if signal_type == "EXIT":
            # De-risking must never be blocked, rescaled or suppressed.
            self.events.put(self._rebuild_signal(signal, metadata, strength_scale=1.0))
            return
        if overlay_scale <= _EPS:
            # Kill state: the child's entries are dropped, not merely shrunk.
            return

        # Scale sizing NOTIONAL + strength only. max_symbol_exposure_pct is a
        # min-clamp cap (per the authoring contract) -- do NOT scale/inflate it.
        for meta_field in ("target_allocation", "max_order_value"):
            value = safe_float(metadata.get(meta_field))
            if value is not None and value > 0.0:
                metadata[meta_field] = float(value) * overlay_scale
        metadata["overlay_scale"] = float(overlay_scale)
        metadata["overlay_reason"] = str(self._last_diagnostics.get("reason") or "none")
        metadata["proxy_drawdown"] = self._diagnostic_drawdown()
        metadata["child_strategy_class"] = self.child_strategy_class
        self.events.put(self._rebuild_signal(signal, metadata, strength_scale=overlay_scale))

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
            # The child's protective levels are part of its de-risking; the
            # overlay must forward them untouched (dropping them would REMOVE
            # protection, the opposite of this overlay's mandate).
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

    def _drain_exits_only(self) -> None:
        """Salvage the child's EXITs after it raised, discarding everything else.

        The child may already have queued signals BEFORE it blew up, and those
        can include de-risking EXITs.  Dropping the whole queue would be the
        wrong failure direction for a de-risk-only overlay: a child crash must
        never be able to STRAND an open position.  Entries are still discarded
        -- a half-computed child is not trusted to open risk.  Never raises.
        """
        for item in self._child_queue.drain():
            if not isinstance(item, SignalEvent):
                continue
            if str(getattr(item, "signal_type", "")).upper() != "EXIT":
                continue
            try:
                self._forward_child_signal(item, 0.0)
            except Exception:
                continue

    # ------------------------------------------------------------ dispatchers

    def _pre(self, event: Any) -> float:
        self._last_event_time = getattr(event, "time", None)
        self._ingest(event)
        self._emit_kill_exits()
        return float(self._overlay.effective_scale)

    def calculate_signals(self, event: Any) -> None:
        try:
            overlay_scale = self._pre(event)
            self._child.calculate_signals(event)
            self._drain(overlay_scale)
        except Exception:
            # The overlay must never raise from calculate_signals.
            self._drain_exits_only()

    def calculate_signals_window(self, event: Any, aggregator: Any = None) -> None:
        try:
            overlay_scale = self._pre(event)
            handler = getattr(self._child, "calculate_signals_window", None)
            if callable(handler):
                handler(event, aggregator)
            else:
                self._child.calculate_signals(event)
            self._drain(overlay_scale)
        except Exception:
            self._drain_exits_only()

    def calculate_signals_context(self, context: Any) -> None:
        try:
            event = getattr(context, "event", context)
            overlay_scale = self._pre(event)
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
            self._drain_exits_only()


def _restore_float_map(raw: Any) -> dict[str, float]:
    """Restore a ``{symbol: float}`` map from serialized state, dropping bad entries."""
    out: dict[str, float] = {}
    if isinstance(raw, dict):
        for symbol, value in raw.items():
            parsed = safe_float(value)
            if parsed is None:
                continue
            out[str(symbol)] = float(parsed)
    return out


def _restore_trade_map(raw: Any) -> dict[str, dict[str, float]]:
    """Restore the ``{symbol: {close, side}}`` open-trade map from serialized state."""
    out: dict[str, dict[str, float]] = {}
    if not isinstance(raw, dict):
        return out
    for symbol, record in raw.items():
        if not isinstance(record, dict):
            continue
        close = safe_float(record.get("close"))
        side = safe_float(record.get("side"))
        if close is None or side is None or close <= 0.0 or side == 0.0:
            continue
        out[str(symbol)] = {"close": float(close), "side": float(side)}
    return out


# --------------------------------------------------------------------------- #
# Candidate-wiring hints for the integrator (this lane does NOT wire candidates
# itself -- new-file-only, no shared-file edits).  Admission is the overlay A/B
# route: the SAME child bare vs wrapped, on identical walk-forward windows and
# cost model; the thesis is max-drawdown / ruin-risk reduction ONLY, never a
# Sharpe lift.  research_only.
# --------------------------------------------------------------------------- #
_SUGGESTED_FAMILY = "overlay"
_SUGGESTED_CANDIDATE_TAGS: tuple[str, ...] = (
    "overlay",
    "risk_overlay",
    "kill_switch",
    "equity_curve",
    "governance",
    "de_risk",
    "research_only",
)

__all__ = [
    "DEFAULT_LADDER",
    "EquityCurveKillSwitchOverlayStrategy",
    "drawdown_ladder_scale",
    "kill_switch_scale",
    "parse_drawdown_ladder",
]

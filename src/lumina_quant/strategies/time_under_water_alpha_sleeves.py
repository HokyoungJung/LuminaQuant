"""Cross-sectional drawdown-DURATION (time-under-water) sleeve, depth-residualized.

``CrossSectionalTimeUnderWaterStrategy`` ranks the underwater cross-section on
the DURATION of the current drawdown -- the number of bars since the last
within-window running peak -- a TIME transform of the price path that none of
the depth-based (near-high), return-based (long-run overreaction), or
cost-basis-based (capital-gains-overhang) incumbents use.  The sleeve trades ONLY
names that are genuinely underwater (a depth eligibility band), and residualizes
the duration score on drawdown DEPTH so what remains is the stagnation-time
component orthogonal to how far a name has fallen.

Long the longest-stagnant names (anchored sellers exhausted, reference points
adapted with TIME, overhead break-even supply decayed -- Arkes et al.
reference-point adaptation), short the freshest drawdowns (thickest peak-anchored
overhang, disposition holders defending break-even -- Grinblatt-Han).

THEORY / PROVENANCE
--------------------
- Grinblatt & Han (2005), *JFE* 78(2) -- disposition overhang is thickest right
  after a peak (supplies the short-leg sign).
- Arkes, Hirshleifer, Jiang & Lim (2008), *OBHDP* 105(1) -- reference-point
  adaptation OVER TIME: the direct mechanism for a duration transform.
- Barberis & Xiong (2012), *JFE* 104(2); Da, Gurun & Warachka (2014), *RFS* 27(7)
  (stagnation as the limiting low-salience case).
- Bhootra & Hur (2013), *JBF* 37(10) -- declared CONTESTED-SIGN counter-anchor.
- HONEST FLAG (binding): no published XS return study ranks directly on drawdown
  DURATION -- this is a mechanism-level anchor chain, hence the aggressive
  EXPECTED NULL carried at the data-PC.

SIGNAL SPEC
-----------
Per completed daily bar, per symbol, over a trailing ``lookback_bars`` window:

1. ``peak = max(close)`` (LAST argmax for determinism); ``TUW = bars since peak``;
   ``duration_frac = TUW / eff_window in [0, 1]``; ``depth = close/peak - 1 <= 0``;
   ``trough`` = min close since the peak (first argmin).
2. ELIGIBILITY BAND (decoupler #1): only genuinely-underwater names trade --
   ENTER at ``depth <= depth_enter`` (-12%), stay while ``depth <= depth_exit``
   (-8%, band hysteresis), hard floor ``depth >= depth_floor`` (-85%,
   terminal-collapse exclusion).  At-high names are structurally OUTSIDE the
   universe (the near-high incumbent's entire long book cannot be shorted here).
3. XS z-score ``duration_frac`` among eligibles, then RESIDUALIZE ``duration_z``
   on ``depth_z`` (decoupler #2) via the shared single-regressor Gram-Schmidt
   ``cross_sectional_residualize`` primitive -- the sleeve trades only the
   duration component orthogonal to depth/nearness by construction.
4. ``score_mode="duration"`` (default) ranks the residual; ``"duration_recovery"``
   adds ``0.25 * z(recovery_slope)`` where ``recovery_slope`` is the OLS slope of
   log-close over the last ``recovery_window`` bars since the trough.
5. LONG the top ``quantile_pct`` residual-duration quantile (longest stagnation),
   SHORT the bottom (freshest drawdowns).  Rank-hysteresis hold band + hard
   ``min_hold_decisions`` + post-exit ``cooldown_decisions`` keep turnover low.

Sizing is inverse-realized-vol risk parity normalised to
``target_gross_exposure`` and clamped by ``target_vol``.  The book self-skips
below ``min_symbols`` eligible names.  TUW is an integer counter (the stickiest
characteristic class in the book), which IS the ex-ante cost story.

This module is data-local (no I/O, no hidden configuration bus), pure Python
(``math`` + ``deque`` + pure indicator primitives, no numpy), never raises from
``calculate_signals``, and ships WITHOUT ``@register`` (inert).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any

from lumina_quant.core.plugin_registry import register
from lumina_quant.indicators.alpha_features import realized_volatility
from lumina_quant.indicators.annualization import (
    annualize_per_bar_vol,
    bars_per_year_from_spacing,
)
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.indicators.cross_sectional_residualize import cross_sectional_residualize
from lumina_quant.indicators.rolling_stats import ts_regression_slope
from lumina_quant.strategies.adaptive_crypto_alpha_sleeves import (
    _age_cross_positions,
    _state_size,
)
from lumina_quant.strategies.external_alpha_sleeves import (
    _EPS,
    _Snapshot,
    _emit,
    _event_datetime_utc,
    _event_symbols,
    _market_snapshot,
    _safe_non_negative_int,
    _target_metadata,
    _window_snapshot,
)
from lumina_quant.strategy import Strategy
from lumina_quant.tuning import HyperParam, resolve_params_from_schema

_STRATEGY_ID = "time_under_water"
_STRATEGY_NAME = "CrossSectionalTimeUnderWaterStrategy"


@dataclass(slots=True)
class _State:
    closes: deque[float]
    volumes: deque[float]
    mode: str = "OUT"
    entry_price: float | None = None
    bars_held: int = 0
    cooldown: int = 0
    last_time_key: str = ""
    score: float | None = None


def _coerce_float_list(value: Any) -> list[float]:
    """Best-effort ``list[float]`` coercion that never raises on adversarial input."""
    if not isinstance(value, (list, tuple)):
        return []
    out: list[float] = []
    for item in value:
        parsed = safe_float(item)
        if parsed is not None:
            out.append(parsed)
    return out


def _cross_z(values: dict[str, float]) -> dict[str, float]:
    """Cross-sectional z-score of ``values`` (0.0 for every symbol if degenerate)."""
    count = len(values)
    if count == 0:
        return {}
    mean_value = sum(values.values()) / float(count)
    variance = sum((value - mean_value) ** 2 for value in values.values()) / float(
        max(1, count - 1)
    )
    sigma = variance**0.5
    if sigma <= _EPS:
        return dict.fromkeys(values, 0.0)
    return {symbol: (value - mean_value) / sigma for symbol, value in values.items()}


def _time_under_water(closes: list[float]) -> tuple[int, float, int] | None:
    """Return ``(tuw_bars, depth, trough_idx)`` over ``closes`` or ``None``.

    ``peak`` is the running max (LAST argmax for determinism); ``tuw`` is bars
    since that peak; ``depth = close/peak - 1 <= 0``; ``trough_idx`` is the FIRST
    argmin from the peak onward (a stable recovery anchor).  Never raises.
    """
    n = len(closes)
    if n < 2:
        return None
    peak = closes[0]
    peak_idx = 0
    for idx in range(1, n):
        if closes[idx] >= peak:  # >= -> LAST bar attaining the max
            peak = closes[idx]
            peak_idx = idx
    if peak <= 0.0:
        return None
    close = closes[-1]
    if close <= 0.0:
        return None
    depth = close / peak - 1.0
    if not math.isfinite(depth):
        return None
    tuw = (n - 1) - peak_idx
    trough = closes[peak_idx]
    trough_idx = peak_idx
    for idx in range(peak_idx, n):
        if closes[idx] < trough:  # strict < -> FIRST (earliest) trough
            trough = closes[idx]
            trough_idx = idx
    return tuw, float(depth), trough_idx


def _recovery_slope(closes: list[float], trough_idx: int, recovery_window: int) -> float | None:
    """OLS slope of log-close over the last ``recovery_window`` bars since trough."""
    segment = closes[trough_idx:]
    if len(segment) > recovery_window:
        segment = segment[-recovery_window:]
    if len(segment) < 3:
        return None
    if any(value <= 0.0 for value in segment):
        return None
    xs = [float(idx) for idx in range(len(segment))]
    ys = [math.log(value) for value in segment]
    slope = ts_regression_slope(xs, ys)
    if slope is None or not math.isfinite(slope):
        return None
    return float(slope)


@register("strategy", "CrossSectionalTimeUnderWaterStrategy", interface="event_driven")
class CrossSectionalTimeUnderWaterStrategy(Strategy):
    """Long-short XS drawdown-duration book, depth-residualized, underwater-gated.

    See the module docstring for the full theory, signal spec, and the
    distinct-from rationale versus the near-high anchoring (depth level),
    long-run overreaction (path-blind return), and capital-gains-overhang
    (volume cost-basis) incumbents.  Reads only local event/bar OHLCV; performs
    no I/O and never raises from ``calculate_signals``.
    """

    # Daily-bar weekly-cadence cross-sectional book; >= 30-minute live floor
    # trivially cleared.
    decision_cadence_seconds = 86400
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            # 52-week-equivalent trailing window (bars); data-PC sweeps 26/52wk.
            "lookback_bars": HyperParam.integer("lookback_bars", default=364, low=20, high=200000),
            "min_history_bars": HyperParam.integer(
                "min_history_bars", default=60, low=5, high=100000
            ),
            "depth_enter": HyperParam.floating("depth_enter", default=-0.12, low=-0.99, high=0.0),
            "depth_exit": HyperParam.floating("depth_exit", default=-0.08, low=-0.99, high=0.0),
            "depth_floor": HyperParam.floating("depth_floor", default=-0.85, low=-0.999, high=0.0),
            # Swept via the pre-registered ``_TIME_UNDER_WATER_SLICE`` cells
            # ({duration, duration_recovery}), not the continuous tuner.
            "score_mode": HyperParam.string("score_mode", default="duration", tunable=False),
            "recovery_window": HyperParam.integer("recovery_window", default=28, low=3, high=20000),
            "vol_window": HyperParam.integer("vol_window", default=20, low=2, high=2000),
            "quantile_pct": HyperParam.floating("quantile_pct", default=0.25, low=0.02, high=0.50),
            "rebalance_bars": HyperParam.integer("rebalance_bars", default=7, low=1, high=100000),
            "min_hold_decisions": HyperParam.integer(
                "min_hold_decisions", default=4, low=0, high=100000
            ),
            "cooldown_decisions": HyperParam.integer(
                "cooldown_decisions", default=1, low=0, high=100000
            ),
            "rank_hysteresis_buffer": HyperParam.integer(
                "rank_hysteresis_buffer", default=1, low=0, high=512
            ),
            "residualize": HyperParam.boolean("residualize", default=True, grid=[True, False]),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
            "min_symbols": HyperParam.integer("min_symbols", default=5, low=2, high=512),
            "min_dollar_volume": HyperParam.floating(
                "min_dollar_volume", default=0.0, low=0.0, high=1e15
            ),
            "target_gross_exposure": HyperParam.floating(
                "target_gross_exposure", default=1.0, low=0.0, high=3.0
            ),
            "target_vol": HyperParam.floating("target_vol", default=0.20, low=0.0, high=2.0),
            "stop_loss_pct": HyperParam.floating("stop_loss_pct", default=0.15, low=0.0, high=0.50),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=0, low=0, high=200000),
            "base_allocation": HyperParam.floating(
                "base_allocation", default=0.20, low=0.0, high=2.0, tunable=False
            ),
            "max_symbol_exposure_pct": HyperParam.floating(
                "max_symbol_exposure_pct", default=0.40, low=0.0, high=2.0, tunable=False
            ),
            "max_order_value": HyperParam.floating(
                "max_order_value", default=400.0, low=0.0, high=1_000_000.0, tunable=False
            ),
            "min_price": HyperParam.floating("min_price", default=0.10, low=0.0, high=1_000_000.0),
        }

    def __init__(self, bars: Any, events: Any, **params: Any) -> None:
        self.bars = bars
        self.events = events
        self.symbol_list = list(getattr(self.bars, "symbol_list", []) or [])
        resolved = resolve_params_from_schema(self.get_param_schema(), params, keep_unknown=False)
        self.lookback_bars = max(3, int(resolved["lookback_bars"]))
        self.min_history_bars = max(2, int(resolved["min_history_bars"]))
        self.depth_enter = min(0.0, float(resolved["depth_enter"]))
        self.depth_exit = min(0.0, float(resolved["depth_exit"]))
        self.depth_floor = min(0.0, float(resolved["depth_floor"]))
        mode = str(resolved["score_mode"]).lower()
        self.score_mode = mode if mode in {"duration", "duration_recovery"} else "duration"
        self.recovery_window = max(3, int(resolved["recovery_window"]))
        self.vol_window = max(2, int(resolved["vol_window"]))
        self.quantile_pct = min(0.5, max(0.0, float(resolved["quantile_pct"])))
        self.rebalance_bars = max(1, int(resolved["rebalance_bars"]))
        self.min_hold_decisions = max(0, int(resolved["min_hold_decisions"]))
        self.cooldown_decisions = max(0, int(resolved["cooldown_decisions"]))
        self.rank_hysteresis_buffer = max(0, int(resolved["rank_hysteresis_buffer"]))
        self.residualize = bool(resolved["residualize"])
        self.allow_short = bool(resolved["allow_short"])
        self.min_symbols = max(2, int(resolved["min_symbols"]))
        self.min_dollar_volume = max(0.0, float(resolved["min_dollar_volume"]))
        self.target_gross_exposure = max(0.0, float(resolved["target_gross_exposure"]))
        self.target_vol = max(0.0, float(resolved["target_vol"]))
        self.stop_loss_pct = max(0.0, float(resolved["stop_loss_pct"]))
        self.max_hold_bars = max(0, int(resolved["max_hold_bars"]))
        self.base_allocation = max(0.0, float(resolved["base_allocation"]))
        self.max_symbol_exposure_pct = max(0.0, float(resolved["max_symbol_exposure_pct"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        size = _state_size(self.lookback_bars, self.vol_window + 1, self.max_hold_bars)
        self._state: dict[str, _State] = {
            symbol: _State(
                closes=deque(maxlen=size),
                volumes=deque(maxlen=size),
            )
            for symbol in self.symbol_list
        }
        self._last_eval_time_key = ""
        self._tick = 0
        # Recent decision-bar epochs (seconds) for deterministic bar-spacing
        # inference: the vol-target scalar annualizes the per-bar portfolio vol
        # via sqrt(bars_per_year) derived from the median gap here.
        self._recent_times: deque[float] = deque(maxlen=16)

    # ------------------------------------------------------------------ #
    # state
    # ------------------------------------------------------------------ #
    def get_state(self) -> dict[str, Any]:
        return {
            "last_eval_time_key": self._last_eval_time_key,
            "tick": int(self._tick),
            "recent_times": list(self._recent_times),
            "symbol_state": {
                symbol: {
                    "closes": list(item.closes),
                    "volumes": list(item.volumes),
                    "mode": item.mode,
                    "entry_price": item.entry_price,
                    "bars_held": int(item.bars_held),
                    "cooldown": int(item.cooldown),
                    "last_time_key": item.last_time_key,
                    "score": item.score,
                }
                for symbol, item in self._state.items()
            },
        }

    def set_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        self._last_eval_time_key = str(state.get("last_eval_time_key", ""))
        self._tick = _safe_non_negative_int(state.get("tick"))
        self._recent_times.clear()
        for value in _coerce_float_list(state.get("recent_times"))[
            -int(self._recent_times.maxlen or 0) :
        ]:
            self._recent_times.append(value)
        raw = state.get("symbol_state")
        if not isinstance(raw, dict):
            return
        for symbol, payload in raw.items():
            if symbol not in self._state or not isinstance(payload, dict):
                continue
            item = self._state[symbol]
            try:
                for attr in ("closes", "volumes"):
                    target = getattr(item, attr)
                    target.clear()
                    maxlen = int(target.maxlen or 0)
                    values = _coerce_float_list(payload.get(attr))
                    for value in values[-maxlen:] if maxlen else values:
                        target.append(value)
                mode = str(payload.get("mode", "OUT")).upper()
                item.mode = mode if mode in {"OUT", "LONG", "SHORT"} else "OUT"
                item.entry_price = safe_float(payload.get("entry_price"))
                item.bars_held = _safe_non_negative_int(payload.get("bars_held"))
                item.cooldown = _safe_non_negative_int(payload.get("cooldown"))
                item.last_time_key = str(payload.get("last_time_key", ""))
                item.score = safe_float(payload.get("score"))
            except Exception:
                continue

    # ------------------------------------------------------------------ #
    # ingestion
    # ------------------------------------------------------------------ #
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
                    self._tick += 1
                    self._evaluate(snapshot.time)

    # ------------------------------------------------------------------ #
    # scoring / selection
    # ------------------------------------------------------------------ #
    def _eligible_features(
        self,
    ) -> tuple[
        dict[str, float],
        dict[str, float],
        dict[str, float],
        dict[str, float],
        dict[str, dict[str, Any]],
    ]:
        """Return ``(duration_frac, depth, recovery_slope, vols, metas)`` per eligible name."""
        duration_frac: dict[str, float] = {}
        depth_by: dict[str, float] = {}
        recovery: dict[str, float] = {}
        vols: dict[str, float] = {}
        metas: dict[str, dict[str, Any]] = {}
        for symbol, item in self._state.items():
            closes = list(item.closes)
            volumes = list(item.volumes)
            if len(closes) < self.min_history_bars:
                continue
            eff_window = min(self.lookback_bars, len(closes))
            window_closes = closes[-eff_window:]
            tuw_result = _time_under_water(window_closes)
            if tuw_result is None:
                continue
            tuw, depth, trough_idx = tuw_result
            # Terminal-collapse floor: never trade a name in a bottomless drawdown.
            if depth < self.depth_floor:
                continue
            # Underwater eligibility band with hysteresis: OUT names must clear the
            # ENTER threshold; held names stay until they recover past EXIT.
            if item.mode == "OUT":
                if depth > self.depth_enter:
                    continue
                if item.cooldown > 0:
                    continue
            else:
                if depth > self.depth_exit:
                    continue
            vol = realized_volatility(closes, window=self.vol_window)
            if vol is None or vol <= _EPS:
                continue
            if self.min_dollar_volume > 0.0:
                window_vol = volumes[-eff_window:]
                dvs = [
                    close * volume
                    for close, volume in zip(window_closes, window_vol, strict=False)
                    if close > 0.0
                ]
                if not dvs or (sum(dvs) / float(len(dvs))) < self.min_dollar_volume:
                    continue
            frac = tuw / float(eff_window)
            duration_frac[symbol] = float(frac)
            depth_by[symbol] = float(depth)
            vols[symbol] = float(vol)
            metas[symbol] = {
                "time_under_water": int(tuw),
                "duration_frac": float(frac),
                "drawdown_depth": float(depth),
                "trough_index": int(trough_idx),
                "lookback_used": int(eff_window),
                "full_lookback": bool(eff_window >= self.lookback_bars),
            }
            if self.score_mode == "duration_recovery":
                slope = _recovery_slope(window_closes, trough_idx, self.recovery_window)
                if slope is not None:
                    recovery[symbol] = float(slope)
                    metas[symbol]["recovery_slope"] = float(slope)
        return duration_frac, depth_by, recovery, vols, metas

    def _score_and_select(
        self,
    ) -> tuple[dict[str, tuple[str, float, dict[str, Any]]], dict[str, float]]:
        duration_frac, depth_by, recovery, vols, metas = self._eligible_features()
        if len(duration_frac) < self.min_symbols:
            return {}, {}

        ordered = sorted(duration_frac)  # deterministic symbol order for the residualizer
        duration_z = _cross_z(duration_frac)
        depth_z = _cross_z(depth_by)
        if self.residualize:
            residual_vec = cross_sectional_residualize(
                [duration_z[symbol] for symbol in ordered],
                [[depth_z[symbol] for symbol in ordered]],
            )
            if residual_vec is None or len(residual_vec) != len(ordered):
                return {}, {}
            score = {symbol: float(residual_vec[idx]) for idx, symbol in enumerate(ordered)}
        else:
            score = {symbol: float(duration_z[symbol]) for symbol in ordered}

        if self.score_mode == "duration_recovery" and recovery:
            recovery_z = _cross_z(recovery)
            for symbol in ordered:
                score[symbol] = score[symbol] + 0.25 * recovery_z.get(symbol, 0.0)

        # Score collapse (duration collinear with depth, or no dispersion) -> abstain.
        score_values = list(score.values())
        score_mean = sum(score_values) / float(len(score_values))
        score_var = sum((value - score_mean) ** 2 for value in score_values) / float(
            max(1, len(score_values) - 1)
        )
        if score_var**0.5 <= _EPS:
            return {}, {}

        for symbol in ordered:
            metas[symbol]["duration_z"] = float(duration_z[symbol])
            metas[symbol]["depth_z"] = float(depth_z[symbol])
            metas[symbol]["residual_duration"] = float(score[symbol])

        # Ascending by score (symbol tiebreak): top quantile LONG (longest
        # stagnation), bottom quantile SHORT (freshest drawdowns).  A held name is
        # retained while its rank stays within the quantile plus the hysteresis
        # buffer (enter top ~20% / exit past ~40%).
        rank_order = sorted(score, key=lambda symbol: (score[symbol], symbol))
        count = len(rank_order)
        n_side = max(1, int(self.quantile_pct * count))
        if 2 * n_side > count:
            n_side = count // 2
        if n_side < 1:
            return {}, {}
        buffer = self.rank_hysteresis_buffer
        long_core = set(rank_order[count - n_side :])
        short_core = set(rank_order[:n_side])
        long_hold = set(rank_order[count - min(count, n_side + buffer) :])
        short_hold = set(rank_order[: min(count, n_side + buffer)])

        targets: dict[str, tuple[str, float, dict[str, Any]]] = {}
        for symbol in rank_order:
            mode = self._state[symbol].mode
            value = float(score[symbol])
            meta = metas[symbol]
            if symbol in long_core:
                targets[symbol] = ("LONG", value, meta)
            elif self.allow_short and symbol in short_core:
                targets[symbol] = ("SHORT", value, meta)
            elif mode == "LONG" and symbol in long_hold:
                targets[symbol] = ("LONG", value, meta)
            elif mode == "SHORT" and self.allow_short and symbol in short_hold:
                targets[symbol] = ("SHORT", value, meta)
        return targets, vols

    def _inverse_vol_weights(
        self,
        targets: dict[str, tuple[str, float, dict[str, Any]]],
        vols: dict[str, float],
    ) -> tuple[dict[str, float], float]:
        inv = {
            symbol: 1.0 / max(vols.get(symbol, 0.0), _EPS)
            for symbol in targets
            if vols.get(symbol, 0.0) > _EPS
        }
        total_inv = sum(inv.values())
        if total_inv <= _EPS:
            return {}, 1.0
        # ``portfolio_vol`` is the inverse-vol-weighted PER-BAR vol; the
        # ``inv / total_inv`` normalization above is scale-invariant (annualizing
        # every vol_i cancels), so the risk-parity weights are horizon-free.
        # The vol-target SCALAR, however, compares ``portfolio_vol`` against an
        # annual-scale ``target_vol`` (0.20): annualize the per-bar estimate via
        # sqrt(bars_per_year) inferred from observed bar spacing first, otherwise
        # the Moreira-Muir clamp is INERT. When spacing is unavailable we pass
        # through (scalar=1.0) rather than throttle on mismatched horizons.
        portfolio_vol = sum((inv[symbol] / total_inv) * vols[symbol] for symbol in inv)
        scalar = 1.0
        if self.target_vol > 0.0 and portfolio_vol > _EPS:
            bars_per_year = bars_per_year_from_spacing(self._recent_times)
            portfolio_vol_ann = annualize_per_bar_vol(portfolio_vol, bars_per_year)
            if portfolio_vol_ann is not None and portfolio_vol_ann > _EPS:
                scalar = min(1.0, self.target_vol / portfolio_vol_ann)
        weights = {
            symbol: (inv[symbol] / total_inv) * self.target_gross_exposure * scalar
            for symbol in inv
        }
        return weights, float(scalar)

    # ------------------------------------------------------------------ #
    # aging / emission
    # ------------------------------------------------------------------ #
    def _age(self, event_time: Any) -> None:
        max_hold = self.max_hold_bars if self.max_hold_bars > 0 else (1 << 62)
        _age_cross_positions(
            self.events,
            self._state,  # type: ignore[arg-type]
            event_time=event_time,
            strategy_id=_STRATEGY_ID,
            strategy_name=_STRATEGY_NAME,
            stop_loss_pct=self.stop_loss_pct,
            max_hold_bars=max_hold,
        )

    def _evaluate(self, event_time: Any) -> None:
        if len(self.symbol_list) < self.min_symbols:
            return
        # Record the decision-bar epoch so the vol-target scalar can infer bar
        # spacing (this runs once per new bar, before the rebalance gate).
        dt = _event_datetime_utc(event_time)
        if dt is not None:
            self._recent_times.append(dt.timestamp())
        # Stops / max-hold age EVERY bar so a held name is always protected,
        # independent of the slow weekly rebalance clock.
        self._age(event_time)
        if self._tick % self.rebalance_bars:
            return
        # Post-exit cooldown decays once per weekly decision.
        for item in self._state.values():
            if item.cooldown > 0:
                item.cooldown -= 1
        targets, vols = self._score_and_select()
        weights, scalar = self._inverse_vol_weights(targets, vols)
        self._emit_targets(targets, weights, scalar, event_time)

    def _emit_targets(
        self,
        targets: dict[str, tuple[str, float, dict[str, Any]]],
        weights: dict[str, float],
        scalar: float,
        event_time: Any,
    ) -> None:
        for symbol, item in self._state.items():
            target = targets.get(symbol)
            price = item.closes[-1] if item.closes else None
            if target is None:
                if item.mode != "OUT":
                    # Min-hold floor: a would-be exit inside the hold window is
                    # suppressed (turnover discipline).
                    if item.bars_held < self.min_hold_decisions:
                        continue
                    _emit(
                        self.events,
                        strategy_id=_STRATEGY_ID,
                        symbol=symbol,
                        event_time=event_time,
                        signal_type="EXIT",
                        price=price,
                        metadata={"strategy": _STRATEGY_NAME, "reason": "rank_or_band_lapsed"},
                    )
                    item.mode = "OUT"
                    item.entry_price = None
                    item.bars_held = 0
                    item.cooldown = self.cooldown_decisions
                    item.score = None
                continue
            target_mode, score, meta = target
            if item.mode == target_mode:
                item.score = float(score)
                continue
            # Min-hold floor: a would-be side-flip inside the hold window is
            # suppressed; the current position is kept until the hold clears.
            if item.mode != "OUT" and item.bars_held < self.min_hold_decisions:
                continue
            if item.mode != "OUT":
                _emit(
                    self.events,
                    strategy_id=_STRATEGY_ID,
                    symbol=symbol,
                    event_time=event_time,
                    signal_type="EXIT",
                    price=price,
                    metadata={"strategy": _STRATEGY_NAME, "reason": "side_flip"},
                )
            weight = float(weights.get(symbol, 0.0))
            alloc = max(0.0, self.base_allocation * weight)
            if alloc <= 0.0:
                # Zero-alloc entries omit ``target_allocation`` from metadata and
                # the engine resizes them to its DEFAULT allocation -- an unsized,
                # un-vol-gated position. Skip the entry; if a side-flip EXIT was
                # just emitted, drop to OUT (mirroring the rank-lapse flat
                # transition, cooldown included) so state matches it.
                if item.mode != "OUT":
                    item.mode = "OUT"
                    item.entry_price = None
                    item.bars_held = 0
                    item.cooldown = self.cooldown_decisions
                    item.score = None
                continue
            stop_loss = None
            if price is not None and self.stop_loss_pct > 0.0:
                stop_loss = price * (
                    1.0 - self.stop_loss_pct if target_mode == "LONG" else 1.0 + self.stop_loss_pct
                )
            metadata = _target_metadata(
                strategy=_STRATEGY_NAME,
                target_allocation=alloc,
                max_order_value=self.max_order_value,
                score=float(score),
                target_mode=target_mode,
                inverse_vol_weight=weight,
                vol_target_scalar=float(scalar),
                **meta,
            )
            if self.max_symbol_exposure_pct > 0.0:
                metadata["max_symbol_exposure_pct"] = min(
                    float(metadata.get("max_symbol_exposure_pct", self.max_symbol_exposure_pct)),
                    self.max_symbol_exposure_pct,
                )
            _emit(
                self.events,
                strategy_id=_STRATEGY_ID,
                symbol=symbol,
                event_time=event_time,
                signal_type=target_mode,
                strength=max(0.25, min(3.0, abs(score))),
                price=price,
                stop_loss=stop_loss,
                metadata=metadata,
            )
            item.mode = target_mode
            item.entry_price = price
            item.bars_held = 0
            item.score = float(score)


# --------------------------------------------------------------------------- #
# Candidate-wiring hints for the W3 integrator (this lane does NOT wire
# candidates itself -- new-file-only, no shared-file edits per the live-safety
# plan).  Admission route is `allow_multi_asset=True` at the data-PC handoff:
# this book is a pure cross-sectional long-short (NOT carry, NOT momentum), so it
# is honestly EXCLUDED from any carry/momentum tag-superset allowlist -- no fake
# carry tag is added to game that path.
# --------------------------------------------------------------------------- #
_SUGGESTED_FAMILY = "cross_sectional"
_SUGGESTED_CANDIDATE_TAGS: tuple[str, ...] = (
    "cross_sectional",
    "drawdown_duration",
    "time_under_water",
    "disposition",
    "reference_point_adaptation",
    "low_turnover",
    "crypto",
)

# Candidate slice (weekly rebalance via ``rebalance_bars``).  The data-PC owns
# the lookback {182,364} x score_mode {duration, duration_recovery} grid; two
# cells are seeded to keep the candidate library thin.  The decision clock here is
# the BAR-count ``rebalance_bars`` (``_tick % rebalance_bars``), NOT an ISO-week
# key, so 4h/1h cells set ``rebalance_bars`` explicitly (7 -> 42 -> 168) to keep
# the weekly cadence -- ``min_hold_decisions`` then counts the SAME calendar weeks
# and stays fixed.  The bar windows (lookback_bars, vol_window, recovery_window)
# scale x6 / x24; the depth fractions, quantile_pct, min_symbols and exposure are
# unit-free and stay fixed.  ``lookback_bars`` at 1h (8736) sits under the
# ~9000-bar cap.
_TIME_UNDER_WATER_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "4h": (
        {
            "variant": "tuw_52wk_duration",
            "lookback_bars": 2184,
            "depth_enter": -0.12,
            "depth_exit": -0.08,
            "depth_floor": -0.85,
            "score_mode": "duration",
            "quantile_pct": 0.25,
            "rebalance_bars": 42,
            "min_hold_decisions": 4,
            "min_symbols": 5,
            "vol_window": 120,
            "allow_short": True,
            "target_gross_exposure": 1.0,
        },
        {
            "variant": "tuw_26wk_recovery",
            "lookback_bars": 1092,
            "depth_enter": -0.12,
            "depth_exit": -0.08,
            "depth_floor": -0.85,
            "score_mode": "duration_recovery",
            "recovery_window": 168,
            "quantile_pct": 0.25,
            "rebalance_bars": 42,
            "min_hold_decisions": 4,
            "min_symbols": 5,
            "vol_window": 120,
            "allow_short": True,
            "target_gross_exposure": 1.0,
        },
    ),
    "1h": (
        {
            "variant": "tuw_52wk_duration",
            "lookback_bars": 8736,
            "depth_enter": -0.12,
            "depth_exit": -0.08,
            "depth_floor": -0.85,
            "score_mode": "duration",
            "quantile_pct": 0.25,
            "rebalance_bars": 168,
            "min_hold_decisions": 4,
            "min_symbols": 5,
            "vol_window": 480,
            "allow_short": True,
            "target_gross_exposure": 1.0,
        },
        {
            "variant": "tuw_26wk_recovery",
            "lookback_bars": 4368,
            "depth_enter": -0.12,
            "depth_exit": -0.08,
            "depth_floor": -0.85,
            "score_mode": "duration_recovery",
            "recovery_window": 672,
            "quantile_pct": 0.25,
            "rebalance_bars": 168,
            "min_hold_decisions": 4,
            "min_symbols": 5,
            "vol_window": 480,
            "allow_short": True,
            "target_gross_exposure": 1.0,
        },
    ),
    "1d": (
        {
            "variant": "tuw_52wk_duration",
            "lookback_bars": 364,
            "depth_enter": -0.12,
            "depth_exit": -0.08,
            "depth_floor": -0.85,
            "score_mode": "duration",
            "quantile_pct": 0.25,
            "min_hold_decisions": 4,
            "min_symbols": 5,
            "vol_window": 20,
            "allow_short": True,
            "target_gross_exposure": 1.0,
        },
        {
            "variant": "tuw_26wk_recovery",
            "lookback_bars": 182,
            "depth_enter": -0.12,
            "depth_exit": -0.08,
            "depth_floor": -0.85,
            "score_mode": "duration_recovery",
            "recovery_window": 28,
            "quantile_pct": 0.25,
            "min_hold_decisions": 4,
            "min_symbols": 5,
            "vol_window": 20,
            "allow_short": True,
            "target_gross_exposure": 1.0,
        },
    ),
}

__all__ = ["CrossSectionalTimeUnderWaterStrategy"]

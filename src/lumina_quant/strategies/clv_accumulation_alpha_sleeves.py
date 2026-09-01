"""Cross-sectional close-location-value ACCUMULATION sleeve (doubly residualized).

``CrossSectionalCloseLocationAccumulationStrategy`` ranks the cross-section on a
multi-week volume-weighted CLOSE-LOCATION-VALUE footprint -- exactly the pure
``chaikin_money_flow`` primitive (``indicators/volume.py``): ``CLV = ((C-L) -
(H-C)) / (H-L)`` volume-weighted over a trailing window, with zero-range bars
contributing 0.  The raw CMF is then DOUBLY residualized cross-sectionally (via
the pure ``cross_sectional_residualize`` Gram-Schmidt primitive) on BOTH a
same-window trailing-return momentum z-score AND a nearness-to-high z-score, so
the sleeve trades ONLY the signed intra-bar flow footprint that is explained by
NEITHER momentum NOR nearness.  Long the top-residual quantile (stealth
accumulation), short the bottom (distribution).

THEORY / PROVENANCE
--------------------
- Chordia & Subrahmanyam (2004), *JFE* 72(3): autocorrelated order-flow
  imbalances create slow, predictable price pressure -- patient traders split
  large orders over days-to-weeks.
- Easley, Lopez de Prado & O'Hara (2016), *JFE* 120(2): bar-level price geometry
  recovers signed informed flow without tick data (bulk-volume classification).
- Silantyev (2019), *Digital Finance* 1(1): trade-flow imbalance is the strongest
  driver of crypto-perp price changes; the repo's direct taker-flow fields are
  coverage-blocked (~5%), so CLV x volume is the 100%-coverage OHLCV proxy.

Persistent one-sided absorption leaves a footprint in completed-bar geometry:
closes cluster near bar highs (accumulation) or lows (distribution) EVEN WHEN
the net close-to-close return is flat.  The counterparty is the impatient
liquidity demander whose flow marks within-bar extremes (disposition sellers,
lottery-preference FOMO buyers, and mechanically forced liquidation/stop-run
flow printing the wicks patient accumulators absorb).

SIGNAL SPEC
-----------
Per completed decision bar (OHLCV), per symbol, on a weekly decision clock
(``rebalance_bars``) with a hard ``min_hold_decisions`` floor + a rank-hysteresis
band:

1. ``cmf`` = ``chaikin_money_flow`` over ``accumulation_window_bars`` (4wk of 1d).
2. ``momentum`` = ``simple_return`` over ``momentum_window_bars`` (same window).
3. ``nearness`` = ``close / max(high)`` over ``min(nearness_window_bars, avail)``
   -- the same anchor variable, and the same ``max_available`` fallback window,
   the near-high anchoring incumbent uses.
4. Cross-sectionally z-score ``cmf``, ``momentum`` and ``nearness`` across the
   eligible universe, then ``residual = cross_sectional_residualize(cmf_z,
   [momentum_z, nearness_z])`` -- the flow footprint orthogonal to both axes.
5. Rank by residual; the top ``quantile_pct`` fraction are LONG, the bottom
   ``quantile_pct`` fraction SHORT, inverse-realized-vol sized and normalized to
   ``target_gross_exposure`` (clamped by ``target_vol``).  A held name is
   retained while its rank stays within ``quantile + rank_hysteresis_buffer``.

DISTINCT-FROM
-------------
The load-bearing decoupling: the DOUBLE residualization removes the momentum and
the nearness components, so the sleeve is orthogonal both to every trend sleeve
AND to ``CrossSectionalNearHighAnchoringStrategy`` (whose anchor is nearness).
It is signed (unlike unsigned VPIN toxicity) and location-based (unlike the
volume-MAGNITUDE ``CrossSectionalFlowShareRotationStrategy`` share).  These are
the divergences the build-gate test pins by RUNNING those incumbents.

This module is data-local (no I/O, no hidden configuration bus), pure Python
(``math`` only), and never raises from ``calculate_signals``.  It ships WITHOUT
``@register`` (inert until the integration wave wires it).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any

from lumina_quant.core.plugin_registry import register
from lumina_quant.indicators.alpha_features import realized_volatility, simple_return
from lumina_quant.indicators.annualization import (
    annualize_per_bar_vol,
    bars_per_year_from_spacing,
)
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.indicators.cross_sectional_residualize import cross_sectional_residualize
from lumina_quant.indicators.volume import chaikin_money_flow
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

_STRATEGY_ID = "clv_accumulation"
_STRATEGY_NAME = "CrossSectionalCloseLocationAccumulationStrategy"


@dataclass(slots=True)
class _State:
    closes: deque[float]
    highs: deque[float]
    lows: deque[float]
    volumes: deque[float]
    mode: str = "OUT"
    entry_price: float | None = None
    bars_held: int = 0
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


# NOTE: ships WITHOUT ``@register`` (inert).  The atomic integration commit adds
# ``@register`` + tier hint ``research_only`` + candidate builders + manifest /
# baseline re-pin for all surviving wave-2 lanes at once (single owner).
@register("strategy", "CrossSectionalCloseLocationAccumulationStrategy", interface="event_driven")
class CrossSectionalCloseLocationAccumulationStrategy(Strategy):
    """Long-short XS close-location flow accumulation, doubly residualized.

    See the module docstring for the full theory, signal spec, and distinct-from
    rationale versus the near-high anchoring, flow-share, and VPIN-toxicity
    incumbents.  This class only reads local event/bar OHLCV; it performs no I/O
    and never raises from ``calculate_signals``.
    """

    decision_cadence_seconds = 1800
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            # ~4wk of 1d bars; the data-PC sweeps 1/2/4/8wk.
            "accumulation_window_bars": HyperParam.integer(
                "accumulation_window_bars", default=28, low=3, high=20000
            ),
            "momentum_window_bars": HyperParam.integer(
                "momentum_window_bars", default=28, low=2, high=20000
            ),
            # min(52wk, max_available) trailing-high window -- mirrors the
            # near-high anchoring incumbent's anchor variable.
            "nearness_window_bars": HyperParam.integer(
                "nearness_window_bars", default=364, low=5, high=200000
            ),
            "min_history_bars": HyperParam.integer(
                "min_history_bars", default=30, low=3, high=100000
            ),
            "vol_window": HyperParam.integer("vol_window", default=30, low=2, high=2000),
            "quantile_pct": HyperParam.floating("quantile_pct", default=0.25, low=0.02, high=0.50),
            "rebalance_bars": HyperParam.integer("rebalance_bars", default=7, low=1, high=100000),
            "min_hold_decisions": HyperParam.integer(
                "min_hold_decisions", default=2, low=0, high=100000
            ),
            "rank_hysteresis_buffer": HyperParam.integer(
                "rank_hysteresis_buffer", default=1, low=0, high=512
            ),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
            "min_symbols": HyperParam.integer("min_symbols", default=6, low=2, high=512),
            "target_gross_exposure": HyperParam.floating(
                "target_gross_exposure", default=1.0, low=0.0, high=3.0
            ),
            "target_vol": HyperParam.floating("target_vol", default=0.20, low=0.0, high=2.0),
            "stop_loss_pct": HyperParam.floating("stop_loss_pct", default=0.10, low=0.0, high=0.50),
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
        self.accumulation_window_bars = max(3, int(resolved["accumulation_window_bars"]))
        self.momentum_window_bars = max(2, int(resolved["momentum_window_bars"]))
        self.nearness_window_bars = max(5, int(resolved["nearness_window_bars"]))
        self.min_history_bars = max(3, int(resolved["min_history_bars"]))
        self.vol_window = max(2, int(resolved["vol_window"]))
        self.quantile_pct = min(0.5, max(0.0, float(resolved["quantile_pct"])))
        self.rebalance_bars = max(1, int(resolved["rebalance_bars"]))
        self.min_hold_decisions = max(0, int(resolved["min_hold_decisions"]))
        self.rank_hysteresis_buffer = max(0, int(resolved["rank_hysteresis_buffer"]))
        self.allow_short = bool(resolved["allow_short"])
        self.min_symbols = max(2, int(resolved["min_symbols"]))
        self.target_gross_exposure = max(0.0, float(resolved["target_gross_exposure"]))
        self.target_vol = max(0.0, float(resolved["target_vol"]))
        self.stop_loss_pct = max(0.0, float(resolved["stop_loss_pct"]))
        self.max_hold_bars = max(0, int(resolved["max_hold_bars"]))
        self.base_allocation = max(0.0, float(resolved["base_allocation"]))
        self.max_symbol_exposure_pct = max(0.0, float(resolved["max_symbol_exposure_pct"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        # Retain the longest window any statistic needs (the nearness window can
        # be the full 52wk; a young symbol keeps only what it has).
        size = _state_size(
            self.nearness_window_bars,
            self.accumulation_window_bars + 1,
            self.momentum_window_bars + 1,
            self.vol_window + 1,
        )
        self._state: dict[str, _State] = {
            symbol: _State(
                closes=deque(maxlen=size),
                highs=deque(maxlen=size),
                lows=deque(maxlen=size),
                volumes=deque(maxlen=size),
            )
            for symbol in self.symbol_list
        }
        self._last_eval_time_key = ""
        self._tick = 0
        # Recent decision-bar epochs (seconds) for deterministic bar-spacing
        # inference: the vol-target scalar annualizes the per-bar portfolio vol
        # via sqrt(bars_per_year) so the Moreira-Muir clamp is not left inert.
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
                    "highs": list(item.highs),
                    "lows": list(item.lows),
                    "volumes": list(item.volumes),
                    "mode": item.mode,
                    "entry_price": item.entry_price,
                    "bars_held": int(item.bars_held),
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
                for attr in ("closes", "highs", "lows", "volumes"):
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
        high = safe_float(snapshot.high)
        if high is None or high < close:
            high = close
        low = safe_float(snapshot.low)
        if low is None or low > close:
            low = close
        item.closes.append(close)
        item.highs.append(high)
        item.lows.append(low)
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
    def _residual_scores(
        self,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, dict[str, Any]]]:
        """Return ``(residual_by_symbol, vols, metas)`` for the eligible universe."""
        cmf: dict[str, float] = {}
        momentum: dict[str, float] = {}
        nearness: dict[str, float] = {}
        vols: dict[str, float] = {}
        for symbol, item in self._state.items():
            closes = list(item.closes)
            highs = list(item.highs)
            lows = list(item.lows)
            volumes = list(item.volumes)
            if len(closes) < self.min_history_bars:
                continue
            close = closes[-1]
            if close is None or close <= 0.0:
                continue
            cmf_value = chaikin_money_flow(
                highs, lows, closes, volumes, period=self.accumulation_window_bars
            )
            mom_value = simple_return(closes, lookback=self.momentum_window_bars)
            eff_lookback = min(self.nearness_window_bars, len(highs))
            trailing_high = max(highs[-eff_lookback:]) if eff_lookback > 0 else None
            vol = realized_volatility(closes, window=self.vol_window)
            if cmf_value is None or mom_value is None or trailing_high is None:
                continue
            if trailing_high <= _EPS or vol is None or vol <= _EPS:
                continue
            near = close / trailing_high
            if not math.isfinite(near):
                continue
            cmf[symbol] = float(cmf_value)
            momentum[symbol] = float(mom_value)
            nearness[symbol] = float(near)
            vols[symbol] = float(vol)

        if len(cmf) < self.min_symbols:
            return {}, {}, {}

        ordered = sorted(cmf)
        cmf_z = _cross_z(cmf)
        mom_z = _cross_z(momentum)
        near_z = _cross_z(nearness)
        residual_vec = cross_sectional_residualize(
            [cmf_z[symbol] for symbol in ordered],
            [
                [mom_z[symbol] for symbol in ordered],
                [near_z[symbol] for symbol in ordered],
            ],
        )
        if residual_vec is None or len(residual_vec) != len(ordered):
            return {}, {}, {}
        residual = {symbol: float(residual_vec[idx]) for idx, symbol in enumerate(ordered)}
        metas: dict[str, dict[str, Any]] = {
            symbol: {
                "cmf": float(cmf[symbol]),
                "cmf_z": float(cmf_z[symbol]),
                "momentum_z": float(mom_z[symbol]),
                "nearness_z": float(near_z[symbol]),
                "flow_residual": float(residual[symbol]),
            }
            for symbol in ordered
        }
        return residual, vols, metas

    def _score_and_select(
        self,
    ) -> tuple[dict[str, tuple[str, float, dict[str, Any]]], dict[str, float]]:
        residual, vols, metas = self._residual_scores()
        if not residual:
            return {}, {}

        # Ascending by residual (symbol tiebreak): bottom quantile is short, top
        # quantile is long.  A held name is retained while its rank stays within
        # the quantile boundary plus the hysteresis buffer.
        ordered = sorted(residual, key=lambda symbol: (residual[symbol], symbol))
        count = len(ordered)
        n_side = max(1, int(self.quantile_pct * count))
        if 2 * n_side > count:
            n_side = count // 2
        if n_side < 1:
            return {}, {}
        buffer = self.rank_hysteresis_buffer
        long_core = set(ordered[count - n_side :])
        short_core = set(ordered[:n_side])
        long_hold_n = min(count, n_side + buffer)
        long_hold = set(ordered[count - long_hold_n :])
        short_hold = set(ordered[: min(count, n_side + buffer)])

        targets: dict[str, tuple[str, float, dict[str, Any]]] = {}
        for symbol in ordered:
            mode = self._state[symbol].mode
            score = residual[symbol]
            meta = metas[symbol]
            if symbol in long_core:
                targets[symbol] = ("LONG", score, meta)
            elif self.allow_short and symbol in short_core:
                targets[symbol] = ("SHORT", score, meta)
            elif mode == "LONG" and symbol in long_hold:
                targets[symbol] = ("LONG", score, meta)
            elif mode == "SHORT" and self.allow_short and symbol in short_hold:
                targets[symbol] = ("SHORT", score, meta)
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
        # independent of the slow weekly rebalance clock.  This DOES advance the
        # min-hold decision counter as well (a hard turnover floor).
        self._age(event_time)
        if self._tick % self.rebalance_bars:
            return
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
                        metadata={"strategy": _STRATEGY_NAME, "reason": "rank_lapsed"},
                    )
                    item.mode = "OUT"
                    item.entry_price = None
                    item.bars_held = 0
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
                # just emitted, drop to OUT so state matches it.
                if item.mode != "OUT":
                    item.mode = "OUT"
                    item.entry_price = None
                    item.bars_held = 0
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
# this book is a pure cross-sectional long-short (NOT carry, NOT momentum -- the
# momentum component is residualized OUT), so it is honestly EXCLUDED from any
# carry/momentum tag-superset allowlist -- no fake carry tag is added.
# --------------------------------------------------------------------------- #
_SUGGESTED_FAMILY = "cross_sectional"
_SUGGESTED_CANDIDATE_TAGS: tuple[str, ...] = (
    "cross_sectional",
    "close_location",
    "accumulation",
    "flow",
    "long_short",
    "zscore",
    "crypto",
)

# Candidate slice.  The data-PC owns the 1/2/4/8wk accumulation-window factor_ic
# sweep, so we seed only two windows (~2wk and ~4wk) per timeframe to keep the
# candidate library thin.  The bar-denominated windows (accumulation / momentum /
# nearness / min-history / vol) scale x6 (4h) / x24 (1h) to hold their wall-clock
# span, and ``rebalance_bars`` -- the bar-count weekly clock -- scales the same way
# (7 -> 42 -> 168) to keep the weekly rebalance cadence.  The ``min_hold_decisions``
# floor, the integer ``rank_hysteresis_buffer`` (rank positions), the quantile, and
# the vol-target / exposure fractions are timeframe-invariant and stay fixed.
_CLV_ACCUMULATION_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "4h": (
        {
            "variant": "acc4wk",
            "accumulation_window_bars": 168,
            "momentum_window_bars": 168,
            "nearness_window_bars": 2184,
            "min_history_bars": 180,
            "vol_window": 180,
            "quantile_pct": 0.25,
            "rebalance_bars": 42,
            "min_hold_decisions": 2,
            "rank_hysteresis_buffer": 1,
            "min_symbols": 6,
            "allow_short": True,
            "target_gross_exposure": 1.0,
            "target_vol": 0.20,
            "stop_loss_pct": 0.10,
        },
        {
            "variant": "acc2wk",
            "accumulation_window_bars": 84,
            "momentum_window_bars": 84,
            "nearness_window_bars": 1260,
            "min_history_bars": 180,
            "vol_window": 180,
            "quantile_pct": 0.25,
            "rebalance_bars": 42,
            "min_hold_decisions": 2,
            "rank_hysteresis_buffer": 1,
            "min_symbols": 6,
            "allow_short": True,
            "target_gross_exposure": 1.0,
            "target_vol": 0.20,
            "stop_loss_pct": 0.10,
        },
    ),
    "1h": (
        {
            "variant": "acc4wk",
            "accumulation_window_bars": 672,
            "momentum_window_bars": 672,
            "nearness_window_bars": 8736,
            "min_history_bars": 720,
            "vol_window": 720,
            "quantile_pct": 0.25,
            "rebalance_bars": 168,
            "min_hold_decisions": 2,
            "rank_hysteresis_buffer": 1,
            "min_symbols": 6,
            "allow_short": True,
            "target_gross_exposure": 1.0,
            "target_vol": 0.20,
            "stop_loss_pct": 0.10,
        },
        {
            "variant": "acc2wk",
            "accumulation_window_bars": 336,
            "momentum_window_bars": 336,
            "nearness_window_bars": 5040,
            "min_history_bars": 720,
            "vol_window": 720,
            "quantile_pct": 0.25,
            "rebalance_bars": 168,
            "min_hold_decisions": 2,
            "rank_hysteresis_buffer": 1,
            "min_symbols": 6,
            "allow_short": True,
            "target_gross_exposure": 1.0,
            "target_vol": 0.20,
            "stop_loss_pct": 0.10,
        },
    ),
    "1d": (
        {
            "variant": "acc4wk",
            "accumulation_window_bars": 28,
            "momentum_window_bars": 28,
            "nearness_window_bars": 364,
            "min_history_bars": 30,
            "vol_window": 30,
            "quantile_pct": 0.25,
            "rebalance_bars": 7,
            "min_hold_decisions": 2,
            "rank_hysteresis_buffer": 1,
            "min_symbols": 6,
            "allow_short": True,
            "target_gross_exposure": 1.0,
            "target_vol": 0.20,
            "stop_loss_pct": 0.10,
        },
        {
            "variant": "acc2wk",
            "accumulation_window_bars": 14,
            "momentum_window_bars": 14,
            "nearness_window_bars": 210,
            "min_history_bars": 30,
            "vol_window": 30,
            "quantile_pct": 0.25,
            "rebalance_bars": 7,
            "min_hold_decisions": 2,
            "rank_hysteresis_buffer": 1,
            "min_symbols": 6,
            "allow_short": True,
            "target_gross_exposure": 1.0,
            "target_vol": 0.20,
            "stop_loss_pct": 0.10,
        },
    ),
}

__all__ = ["CrossSectionalCloseLocationAccumulationStrategy"]

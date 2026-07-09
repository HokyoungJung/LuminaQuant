"""Forecast-free rebalancing-premium harvest on a diversity-weighted basket.

``RebalancingPremiumHarvestStrategy`` holds a diversity-weighted basket of the
most liquid names in the cross-section and rebalances it back to fixed target
weights on a SLOW clock (monthly by default; a weekly variant via a param).
The only edge it targets is the rebalancing / diversification premium -- a
mechanical excess-growth benefit, NOT a return forecast.

FORECAST-FREE (the defining property)
-------------------------------------
The target weights are a pure function of the diversity-weighting rule
(``softmax_weights`` over -log dollar-volume rank, or configurable
equal-weight).  They never depend on a predicted return, a momentum reading, a
z-score, or any directional signal.  Two inputs that are identical up to some
time ``T`` produce identical target weights at ``T`` regardless of how they
diverge afterwards: the sleeve looks only at trailing liquidity to decide
WEIGHTS, and looks only at realized price drift to decide the mechanical
harvest TRADE at a rebalance.  This is what distinguishes it from every
forecast sleeve in the book.

THEORY / PROVENANCE
--------------------
- Fernholz (2002) Stochastic Portfolio Theory: a periodically rebalanced,
  diversity-weighted portfolio earns an ``excess growth rate`` of roughly
  ``1/2 * (average variance - average covariance)`` over its own buy-and-hold
  -- the "rebalancing premium" / "diversification return".
- Booth & Fama (1992) "Diversification Returns and Asset Contributions"; and
  Willenbrock (2011, FAJ) "Diversification Return, Portfolio Rebalancing, and
  the Commodity Return Puzzle": rebalancing mechanically sells relative
  winners and buys relative losers, harvesting variance without a forecast.

This mechanism is asset-class-general (it is NOT one of the crypto-specific
alpha axes in the external sweep), and is honestly flagged as such: the edge is
a growth-rate / variance benefit whose magnitude SHRINKS as the basket's
pairwise correlation rises (a real caveat in a BTC-dominated crypto universe),
not a high-Sharpe directional bet.

SIGNAL SPEC
-----------
Per completed decision bar (OHLCV including volume):

1. Ingest ``close`` and ``volume`` per symbol; maintain trailing deques.
2. Derive the current ``rebalance clock`` bucket from the bar time -- a
   ``(year, month)`` key for the monthly clock, an ISO ``(year, week)`` key for
   the weekly clock.  A REBALANCE fires only when the bucket changes (the first
   eligible bar of a new period); every other bar is drift-only.
3. At a rebalance, for each eligible symbol compute a trailing-mean dollar
   volume ``mean(close * volume)`` over ``liquidity_window`` bars as the
   liquidity proxy, ``dense_rank_desc`` it, and turn ``-log(rank)`` into target
   weights via ``softmax_weights`` (or assign an equal weight), scaled to
   ``target_gross_exposure`` (gross <= 1).
4. Compute each held name's DRIFTED weight since the last rebalance
   (``held_weight * current_price / anchor_price``, renormalized to the same
   gross).  The mechanical harvest TRADE is ``target_weight - drifted_weight``:
   a relative WINNER (price grew faster than the basket) has drifted above its
   target and is TRIMMED (negative trade); a relative LOSER is ADDED (positive
   trade).  This trade -- not the target weight -- is where the diversification
   return is captured, and it is a pure function of realized drift.
5. Emit a ``LONG`` per basket member carrying the target weight, the drifted
   weight, and the ``rebalance_trade``; emit ``EXIT`` for names that drop out
   of the basket.  Between rebalances the book emits NOTHING (drift-only).

The book self-skips (emits nothing) whenever fewer than ``min_symbols`` names
carry a valid price and positive dollar volume, and never emits a directional
forecast: it is long-only the basket, always.

DISTINCT-FROM
-------------
No other sleeve is forecast-free.  The cross-sectional momentum / reversal /
flow-share books all RANK symbols by a predicted-return proxy (returns, trend
sign, turnover-share z-score) and go long/short on that forecast; this book
ranks only by liquidity to set weights and trades only against realized drift.
It is also distinct from ``CrossSectionalFlowShareRotationStrategy`` (turnover
SHARE as a directional attention signal) -- here dollar volume is used solely
to shape a diversified hold, never as an entry signal.

This module is data-local (no I/O, no hidden configuration bus), pure Python
(``math``/``deque`` only, no numpy/scipy/sklearn/statsmodels), and never raises
from ``calculate_signals``.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any

from lumina_quant.core.plugin_registry import register
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.indicators.ensemble_weights import softmax_weights
from lumina_quant.indicators.flow_share import dense_rank_desc
from lumina_quant.strategies.adaptive_crypto_alpha_sleeves import _state_size
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

_STRATEGY_ID = "rebalancing_premium_harvest"
_STRATEGY_NAME = "RebalancingPremiumHarvestStrategy"

_MONTHLY = "monthly"
_WEEKLY = "weekly"
_DIVERSITY = "diversity"
_EQUAL = "equal"


@dataclass(slots=True)
class _State:
    closes: deque[float]
    volumes: deque[float]
    mode: str = "OUT"
    held_weight: float | None = None
    anchor_price: float | None = None
    last_time_key: str = ""


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


def _period_key(raw_time: Any, period: str) -> str:
    """Return the slow-clock bucket for ``raw_time`` ("" when it cannot be parsed).

    Monthly -> ``YYYY-MM``; weekly -> ISO ``YYYY-Www``.  An unparseable time
    yields ``""``, which never triggers a rebalance (drift-only, never-raise).
    """
    dt = _event_datetime_utc(raw_time)
    if dt is None:
        return ""
    if period == _WEEKLY:
        iso = dt.isocalendar()
        return f"{int(iso[0]):04d}-W{int(iso[1]):02d}"
    return f"{dt.year:04d}-{dt.month:02d}"


def _mean_dollar_volume(closes: deque[float], volumes: deque[float], window: int) -> float | None:
    """Trailing-mean ``close * volume`` over the last ``window`` bars (liquidity proxy).

    Returns ``None`` when there is no usable overlap; a non-positive mean is
    left for the caller to exclude (illiquid names never enter the basket).
    """
    if not closes or not volumes:
        return None
    depth = min(len(closes), len(volumes), max(1, window))
    if depth <= 0:
        return None
    close_tail = list(closes)[-depth:]
    volume_tail = list(volumes)[-depth:]
    total = 0.0
    count = 0
    for close, volume in zip(close_tail, volume_tail, strict=False):
        if close is None or volume is None:
            continue
        if not (math.isfinite(close) and math.isfinite(volume)):
            continue
        if close <= 0.0 or volume < 0.0:
            continue
        total += close * volume
        count += 1
    if count == 0:
        return None
    return total / float(count)


@register("strategy", _STRATEGY_NAME, interface="event_driven")
class RebalancingPremiumHarvestStrategy(Strategy):
    """Harvest the rebalancing premium on a forecast-free diversity-weighted basket.

    See the module docstring for the full theory, the forecast-free property,
    and the mechanical-harvest signal spec.  This class only reads local
    event/bar OHLCV (including volume); it performs no I/O and never raises from
    ``calculate_signals``.
    """

    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "rebalance_period": HyperParam.categorical(
                "rebalance_period",
                default=_MONTHLY,
                choices=(_MONTHLY, _WEEKLY),
                grid=[_MONTHLY, _WEEKLY],
            ),
            "weighting": HyperParam.categorical(
                "weighting",
                default=_DIVERSITY,
                choices=(_DIVERSITY, _EQUAL),
                grid=[_DIVERSITY, _EQUAL],
            ),
            "diversity_temperature": HyperParam.floating(
                "diversity_temperature", default=1.0, low=0.05, high=10.0
            ),
            "liquidity_window": HyperParam.integer(
                "liquidity_window", default=20, low=1, high=4000
            ),
            "max_basket_size": HyperParam.integer("max_basket_size", default=0, low=0, high=512),
            "min_symbols": HyperParam.integer("min_symbols", default=4, low=2, high=512),
            "target_gross_exposure": HyperParam.floating(
                "target_gross_exposure", default=1.0, low=0.0, high=1.0
            ),
            "min_price": HyperParam.floating("min_price", default=0.0, low=0.0, high=1_000_000.0),
            "max_symbol_exposure_pct": HyperParam.floating(
                "max_symbol_exposure_pct", default=0.40, low=0.0, high=2.0, tunable=False
            ),
            "max_order_value": HyperParam.floating(
                "max_order_value", default=400.0, low=0.0, high=1_000_000.0, tunable=False
            ),
        }

    def __init__(self, bars: Any, events: Any, **params: Any) -> None:
        self.bars = bars
        self.events = events
        self.symbol_list = list(getattr(self.bars, "symbol_list", []) or [])
        resolved = resolve_params_from_schema(self.get_param_schema(), params, keep_unknown=False)
        period = str(resolved["rebalance_period"]).lower()
        self.rebalance_period = period if period in {_MONTHLY, _WEEKLY} else _MONTHLY
        weighting = str(resolved["weighting"]).lower()
        self.weighting = weighting if weighting in {_DIVERSITY, _EQUAL} else _DIVERSITY
        self.diversity_temperature = max(1e-6, float(resolved["diversity_temperature"]))
        self.liquidity_window = max(1, int(resolved["liquidity_window"]))
        self.max_basket_size = max(0, int(resolved["max_basket_size"]))
        self.min_symbols = max(2, int(resolved["min_symbols"]))
        self.target_gross_exposure = max(0.0, min(1.0, float(resolved["target_gross_exposure"])))
        self.min_price = max(0.0, float(resolved["min_price"]))
        self.max_symbol_exposure_pct = max(0.0, float(resolved["max_symbol_exposure_pct"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self._symbol_index = {symbol: idx for idx, symbol in enumerate(self.symbol_list)}
        size = _state_size(self.liquidity_window + 1, 2)
        self._state: dict[str, _State] = {
            symbol: _State(
                closes=deque(maxlen=size),
                volumes=deque(maxlen=size),
            )
            for symbol in self.symbol_list
        }
        self._last_eval_time_key = ""
        self._last_rebalance_key = ""
        self._tick = 0

    # ------------------------------------------------------------------ #
    # state
    # ------------------------------------------------------------------ #
    def get_state(self) -> dict[str, Any]:
        return {
            "last_eval_time_key": self._last_eval_time_key,
            "last_rebalance_key": self._last_rebalance_key,
            "tick": int(self._tick),
            "symbol_state": {
                symbol: {
                    "closes": list(item.closes),
                    "volumes": list(item.volumes),
                    "mode": item.mode,
                    "held_weight": item.held_weight,
                    "anchor_price": item.anchor_price,
                    "last_time_key": item.last_time_key,
                }
                for symbol, item in self._state.items()
            },
        }

    def set_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        self._last_eval_time_key = str(state.get("last_eval_time_key", ""))
        self._last_rebalance_key = str(state.get("last_rebalance_key", ""))
        self._tick = _safe_non_negative_int(state.get("tick"))
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
                item.mode = mode if mode in {"OUT", "LONG"} else "OUT"
                held = safe_float(payload.get("held_weight"))
                item.held_weight = held if held is not None and held >= 0.0 else None
                anchor = safe_float(payload.get("anchor_price"))
                item.anchor_price = anchor if anchor is not None and anchor > 0.0 else None
                item.last_time_key = str(payload.get("last_time_key", ""))
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
    # weighting (forecast-free)
    # ------------------------------------------------------------------ #
    def _target_weights(self, dvs: dict[str, float]) -> dict[str, float]:
        """Forecast-free target weights from dollar-volume rank only.

        ``equal`` assigns ``gross / n`` to every member; ``diversity`` softmaxes
        ``-log(dense_rank_desc(dollar_volume))`` (the most-liquid rank-1 name
        gets the heaviest weight, spread by ``diversity_temperature``) and
        scales the result to ``target_gross_exposure``.  No return, momentum, or
        directional input enters here.
        """
        symbols = list(dvs.keys())
        n = len(symbols)
        if n == 0:
            return {}
        if self.weighting == _EQUAL:
            weight = self.target_gross_exposure / float(n)
            return dict.fromkeys(symbols, weight)
        dv_values = [dvs[symbol] for symbol in symbols]
        scores: list[float] = []
        for symbol in symbols:
            rank = dense_rank_desc(dv_values, dvs[symbol])
            rank = n if rank is None else int(rank)
            scores.append(-math.log(float(max(1, rank))))
        weights = softmax_weights(scores, temperature=self.diversity_temperature)
        if weights is None:
            weight = self.target_gross_exposure / float(n)
            return dict.fromkeys(symbols, weight)
        return {
            symbol: self.target_gross_exposure * float(weight)
            for symbol, weight in zip(symbols, weights, strict=False)
        }

    def _eligible(self) -> tuple[dict[str, float], dict[str, float]]:
        """Return ``(current_close, dollar_volume)`` for every basket-eligible symbol."""
        current: dict[str, float] = {}
        dvs: dict[str, float] = {}
        for symbol, item in self._state.items():
            if not item.closes:
                continue
            close = item.closes[-1]
            if close is None or close <= self.min_price:
                continue
            dv = _mean_dollar_volume(item.closes, item.volumes, self.liquidity_window)
            if dv is None or dv <= 0.0:
                continue
            current[symbol] = float(close)
            dvs[symbol] = float(dv)
        if self.max_basket_size > 0 and len(dvs) > self.max_basket_size:
            ranked = sorted(
                dvs.items(),
                key=lambda kv: (-kv[1], self._symbol_index.get(kv[0], 1 << 30)),
            )
            keep = {symbol for symbol, _dv in ranked[: self.max_basket_size]}
            current = {symbol: value for symbol, value in current.items() if symbol in keep}
            dvs = {symbol: value for symbol, value in dvs.items() if symbol in keep}
        return current, dvs

    def _drifted_weights(self, current: dict[str, float]) -> dict[str, float]:
        """Weights the held basket has drifted to since the last rebalance.

        ``held_weight * current_price / anchor_price`` for every prior member
        that still has a valid price, renormalized to ``target_gross_exposure``.
        Empty until the first rebalance has anchored the basket.
        """
        drift_values: dict[str, float] = {}
        for symbol, item in self._state.items():
            if item.held_weight is None or not item.anchor_price:
                continue
            price = current.get(symbol)
            if price is None:
                price = item.closes[-1] if item.closes else None
            if price is None or price <= 0.0 or item.anchor_price <= 0.0:
                continue
            drift_values[symbol] = item.held_weight * (price / item.anchor_price)
        total = sum(drift_values.values())
        if total <= _EPS:
            return {}
        return {
            symbol: self.target_gross_exposure * value / total
            for symbol, value in drift_values.items()
        }

    # ------------------------------------------------------------------ #
    # rebalance / emission
    # ------------------------------------------------------------------ #
    def _evaluate(self, event_time: Any) -> None:
        if len(self.symbol_list) < self.min_symbols:
            return
        period_key = _period_key(event_time, self.rebalance_period)
        if not period_key or period_key == self._last_rebalance_key:
            # Unparseable time OR still inside the current period -> drift-only.
            return
        current, dvs = self._eligible()
        if len(dvs) < self.min_symbols:
            # Not enough of a cross-section to diversify: skip WITHOUT anchoring
            # the clock, so the first eligible bar of the period still rebalances.
            return
        targets = self._target_weights(dvs)
        drifted = self._drifted_weights(current)
        self._emit_rebalance(current, dvs, targets, drifted, period_key, event_time)
        self._last_rebalance_key = period_key

    def _emit_rebalance(
        self,
        current: dict[str, float],
        dvs: dict[str, float],
        targets: dict[str, float],
        drifted: dict[str, float],
        period_key: str,
        event_time: Any,
    ) -> None:
        dv_values = list(dvs.values())
        for symbol, item in self._state.items():
            target = targets.get(symbol)
            price = current.get(symbol)
            if price is None:
                price = item.closes[-1] if item.closes else None
            if target is None:
                # Dropped from the basket -> flatten what was held.
                if item.mode != "OUT":
                    drifted_weight = drifted.get(symbol)
                    _emit(
                        self.events,
                        strategy_id=_STRATEGY_ID,
                        symbol=symbol,
                        event_time=event_time,
                        signal_type="EXIT",
                        price=price,
                        metadata={
                            "strategy": _STRATEGY_NAME,
                            "reason": "dropped_from_basket",
                            "rebalance_key": period_key,
                            "is_rebalance": True,
                            "drifted_weight": (
                                float(drifted_weight) if drifted_weight is not None else None
                            ),
                            "rebalance_trade": (
                                -float(drifted_weight) if drifted_weight is not None else None
                            ),
                        },
                    )
                    item.mode = "OUT"
                    item.held_weight = None
                    item.anchor_price = None
                continue
            drifted_weight = drifted.get(symbol)
            trade = (target - drifted_weight) if drifted_weight is not None else None
            rank = dense_rank_desc(dv_values, dvs[symbol])
            metadata = _target_metadata(
                strategy=_STRATEGY_NAME,
                target_allocation=float(target),
                max_order_value=self.max_order_value,
                target_mode="LONG",
                target_weight=float(target),
                drifted_weight=(float(drifted_weight) if drifted_weight is not None else None),
                rebalance_trade=(float(trade) if trade is not None else None),
                dollar_volume=float(dvs[symbol]),
                rank=int(rank) if rank is not None else None,
                weighting=self.weighting,
                rebalance_period=self.rebalance_period,
                rebalance_key=period_key,
                is_rebalance=True,
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
                signal_type="LONG",
                strength=1.0,
                price=price,
                metadata=metadata,
            )
            item.mode = "LONG"
            item.held_weight = float(target)
            item.anchor_price = float(price) if price is not None and price > 0.0 else None


# --------------------------------------------------------------------------- #
# Candidate-wiring hints for the W3 integrator (this lane does NOT wire
# candidates itself -- new-file-only, no shared-file edits per the live-safety
# plan).  Admission route is `allow_multi_asset=True` at the data-PC handoff:
# this book is a forecast-free cross-sectional basket-hold, NOT carry, so it is
# honestly EXCLUDED from any carry tag -- no fake carry tag is added.
# --------------------------------------------------------------------------- #
_SUGGESTED_FAMILY = "cross_sectional"
_SUGGESTED_CANDIDATE_TAGS: tuple[str, ...] = (
    "cross_sectional",
    "rebalancing_premium",
    "diversification_return",
    "forecast_free",
    "low_turnover",
    "crypto",
)

_REBALANCING_PREMIUM_HARVEST_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "1d": (
        {
            "variant": "monthly_diversity",
            "rebalance_period": _MONTHLY,
            "weighting": _DIVERSITY,
            "diversity_temperature": 1.0,
            "liquidity_window": 20,
            "max_basket_size": 12,
            "min_symbols": 5,
            "target_gross_exposure": 1.0,
        },
        {
            "variant": "weekly_equal",
            "rebalance_period": _WEEKLY,
            "weighting": _EQUAL,
            "diversity_temperature": 1.0,
            "liquidity_window": 14,
            "max_basket_size": 10,
            "min_symbols": 5,
            "target_gross_exposure": 1.0,
        },
    ),
}


__all__ = ["RebalancingPremiumHarvestStrategy"]

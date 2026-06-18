"""Diversified cross-FACTOR ensemble book (managed-futures / AQR style).

``DiversifiedMultiFactorEnsembleStrategy`` is a single coordinated cross-sectional
book that combines four lowly-correlated, decades-documented premia under one
vol-targeted, inverse-vol risk-parity, low-turnover wrapper:

- TREND (time-series momentum, volatility-adjusted),
- CROSS-SECTIONAL MOMENTUM (12-1 skip-period relative momentum),
- LOW VOLATILITY (the low-risk anomaly, negated realized vol so calm names score
  high),
- CARRY (negated funding z plus a negated basis component, so calm/negative
  funding regimes earn a long tailwind).

Each leg is computed RAW per symbol, then Z-SCORED ACROSS THE CROSS-SECTION at
the SAME event bar (mean/std over the symbols that print a finite value this
bar).  The legs are blended into a single ``composite`` via tunable
``weight_*`` factors and ranked by the shared cross-sectional ranker.  Sizing is
INVERSE-VOL risk parity (per-name weight proportional to ``1 / realized_vol``)
normalized to ``target_gross_exposure`` and then scaled by
``min(1, target_vol / portfolio_realized_vol)`` where the portfolio vol is the
inverse-vol-weighted realized vol of the selected names — matching the vol
conventions of ``CrossAssetDiversifiedTrendStrategy``.

This is DISTINCT from every existing single-factor sleeve: it is the only book
that diversifies ACROSS factors (trend + xs-momentum + low-vol + carry) inside
one risk-budgeted, vol-targeted portfolio.  It is data-local (no I/O, no hidden
configuration bus), reuses the shared cross-sectional rebalancing helpers and
the 3-tier no-lookahead feature cascade, and never raises from
``calculate_signals``.
"""

from __future__ import annotations

import math
from collections import deque
from itertools import pairwise
from dataclasses import dataclass
from typing import Any

from lumina_quant.core.plugin_registry import register
from lumina_quant.indicators.alpha_features import (
    basis_bps,
    realized_volatility,
)
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.indicators.fast_ops import volatility_adjusted_momentum
from lumina_quant.strategies.adaptive_crypto_alpha_sleeves import (
    _age_cross_positions,
    _ranked_targets,
    _state_size,
)
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

_STRATEGY_ID = "diversified_multifactor_ensemble"
_STRATEGY_NAME = "DiversifiedMultiFactorEnsembleStrategy"
_LEGS = ("trend", "xs_momentum", "low_vol", "carry")


@dataclass(slots=True)
class _State:
    closes: deque[float]
    volumes: deque[float]
    funding_rate: deque[float]
    basis_bps: deque[float]
    open_interest: deque[float]
    mode: str = "OUT"
    entry_price: float | None = None
    bars_held: int = 0
    last_time_key: str = ""
    composite: float | None = None


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


def _zscore_across(values: dict[str, float]) -> dict[str, float]:
    """Z-score a cross-section of values keyed by symbol (no-lookahead: same bar)."""
    if len(values) < 2:
        return dict.fromkeys(values, 0.0)
    series = list(values.values())
    mean_value = sum(series) / float(len(series))
    variance = sum((value - mean_value) ** 2 for value in series) / float(len(series) - 1)
    sigma = variance**0.5
    if sigma <= _EPS:
        return dict.fromkeys(values, 0.0)
    return {symbol: float((value - mean_value) / sigma) for symbol, value in values.items()}


@register("strategy", "DiversifiedMultiFactorEnsembleStrategy", interface="event_driven")
class DiversifiedMultiFactorEnsembleStrategy(Strategy):
    """Cross-factor diversified book: trend + xs-momentum + low-vol + carry.

    Per symbol the four raw legs are computed, cross-sectionally z-scored at the
    same event bar, blended into a single ``composite``, and ranked by the shared
    cross-sectional ranker.  Selected names are sized by inverse realized
    volatility (risk parity) normalized to ``target_gross_exposure`` then scaled
    by a portfolio vol-target clamp.  Turnover is suppressed by a
    ``rebalance_band`` on the composite delta (held names only flip when their
    composite drifts beyond the band or their side flips); stops and max-hold
    aging run every bar.
    """

    decision_cadence_seconds = 60
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False
    required_features = ("funding_rate", "open_interest", "mark_price", "index_price")

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "trend_lookback": HyperParam.integer("trend_lookback", default=120, low=20, high=400),
            "vol_window": HyperParam.integer("vol_window", default=60, low=10, high=200),
            "funding_z_window": HyperParam.integer("funding_z_window", default=48, low=8, high=200),
            "skip_bars": HyperParam.integer("skip_bars", default=1, low=0, high=10),
            "weight_trend": HyperParam.floating("weight_trend", default=0.35, low=0.0, high=1.0),
            "weight_xs_momentum": HyperParam.floating(
                "weight_xs_momentum", default=0.35, low=0.0, high=1.0
            ),
            "weight_low_vol": HyperParam.floating(
                "weight_low_vol", default=0.20, low=0.0, high=1.0
            ),
            "weight_carry": HyperParam.floating("weight_carry", default=0.10, low=0.0, high=1.0),
            "rebalance_bars": HyperParam.integer("rebalance_bars", default=16, low=1, high=240),
            "rebalance_band": HyperParam.floating(
                "rebalance_band", default=0.20, low=0.0, high=2.0
            ),
            "threshold": HyperParam.floating("threshold", default=0.5, low=0.0, high=3.0),
            "max_longs": HyperParam.integer("max_longs", default=5, low=1, high=50),
            "max_shorts": HyperParam.integer("max_shorts", default=5, low=0, high=50),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
            "min_symbols": HyperParam.integer("min_symbols", default=4, low=2, high=512),
            "target_gross_exposure": HyperParam.floating(
                "target_gross_exposure", default=1.0, low=0.0, high=3.0
            ),
            "target_vol": HyperParam.floating("target_vol", default=0.20, low=0.0, high=2.0),
            "stop_loss_pct": HyperParam.floating("stop_loss_pct", default=0.08, low=0.0, high=0.50),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=480, low=0, high=5000),
            "base_allocation": HyperParam.floating(
                "base_allocation", default=0.20, low=0.0, high=2.0, tunable=False
            ),
            "use_shrunk_cov_vol": HyperParam.boolean(
                "use_shrunk_cov_vol", default=False, grid=[True, False]
            ),
            "max_symbol_exposure_pct": HyperParam.floating(
                "max_symbol_exposure_pct", default=0.40, low=0.0, high=2.0, tunable=False
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
        self.trend_lookback = max(2, int(resolved["trend_lookback"]))
        self.vol_window = max(2, int(resolved["vol_window"]))
        self.funding_z_window = max(3, int(resolved["funding_z_window"]))
        self.skip_bars = max(0, int(resolved["skip_bars"]))
        self.weight_trend = max(0.0, float(resolved["weight_trend"]))
        self.weight_xs_momentum = max(0.0, float(resolved["weight_xs_momentum"]))
        self.weight_low_vol = max(0.0, float(resolved["weight_low_vol"]))
        self.weight_carry = max(0.0, float(resolved["weight_carry"]))
        self.rebalance_bars = max(1, int(resolved["rebalance_bars"]))
        self.rebalance_band = max(0.0, float(resolved["rebalance_band"]))
        self.threshold = max(0.0, float(resolved["threshold"]))
        self.max_longs = max(1, int(resolved["max_longs"]))
        self.max_shorts = max(0, int(resolved["max_shorts"]))
        self.allow_short = bool(resolved["allow_short"])
        self.min_symbols = max(2, int(resolved["min_symbols"]))
        self.target_gross_exposure = max(0.0, float(resolved["target_gross_exposure"]))
        self.target_vol = max(0.0, float(resolved["target_vol"]))
        self.use_shrunk_cov_vol = bool(resolved["use_shrunk_cov_vol"])
        self.stop_loss_pct = max(0.0, float(resolved["stop_loss_pct"]))
        self.max_hold_bars = max(0, int(resolved["max_hold_bars"]))
        self.base_allocation = max(0.0, float(resolved["base_allocation"]))
        self.max_symbol_exposure_pct = max(0.0, float(resolved["max_symbol_exposure_pct"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        self.factor_weights = {
            "trend": self.weight_trend,
            "xs_momentum": self.weight_xs_momentum,
            "low_vol": self.weight_low_vol,
            "carry": self.weight_carry,
        }
        size = _state_size(
            self.trend_lookback + self.skip_bars,
            self.vol_window,
            self.funding_z_window,
            self.max_hold_bars,
        )
        self._state = {
            symbol: _State(
                closes=deque(maxlen=size),
                volumes=deque(maxlen=size),
                funding_rate=deque(maxlen=size),
                basis_bps=deque(maxlen=size),
                open_interest=deque(maxlen=size),
            )
            for symbol in self.symbol_list
        }
        self._last_eval_time_key = ""
        self._tick = 0

    def get_state(self) -> dict[str, Any]:
        return {
            "last_eval_time_key": self._last_eval_time_key,
            "tick": int(self._tick),
            "symbol_state": {
                symbol: {
                    "closes": list(item.closes),
                    "volumes": list(item.volumes),
                    "funding_rate": list(item.funding_rate),
                    "basis_bps": list(item.basis_bps),
                    "open_interest": list(item.open_interest),
                    "mode": item.mode,
                    "entry_price": item.entry_price,
                    "bars_held": int(item.bars_held),
                    "last_time_key": item.last_time_key,
                    "composite": item.composite,
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
        if not isinstance(raw, dict):
            return
        for symbol, payload in raw.items():
            if symbol not in self._state or not isinstance(payload, dict):
                continue
            item = self._state[symbol]
            for attr in ("closes", "volumes", "funding_rate", "basis_bps", "open_interest"):
                target = getattr(item, attr)
                target.clear()
                for value in list(payload.get(attr) or [])[-int(target.maxlen or 0) :]:
                    parsed = safe_float(value)
                    if parsed is not None:
                        target.append(parsed)
            mode = str(payload.get("mode", "OUT")).upper()
            item.mode = mode if mode in {"OUT", "LONG", "SHORT"} else "OUT"
            item.entry_price = safe_float(payload.get("entry_price"))
            item.bars_held = _safe_non_negative_int(payload.get("bars_held"))
            item.last_time_key = str(payload.get("last_time_key", ""))
            item.composite = safe_float(payload.get("composite"))

    def _extract_feature(self, event: Any, symbol: str, field: str) -> float | None:
        direct = safe_float(getattr(event, field, None))
        if direct is not None:
            return direct
        getter = getattr(self.bars, "get_latest_feature_value", None)
        if callable(getter):
            try:
                parsed = safe_float(getter(symbol, field))
            except Exception:
                parsed = None
            if parsed is not None:
                return parsed
        getter = getattr(self.bars, "get_latest_bar_value", None)
        if callable(getter):
            try:
                return safe_float(getter(symbol, field))
            except Exception:
                return None
        return None

    def _update_symbol(self, event: Any, symbol: str, snapshot: _Snapshot) -> bool:
        close = safe_float(snapshot.close)
        if close is None or close <= self.min_price:
            return False
        item = self._state[symbol]
        key = time_key(snapshot.time)
        if key and key == item.last_time_key:
            return False
        # All-or-nothing feature collection at the closed bar (3-tier cascade).
        funding = self._extract_feature(event, symbol, "funding_rate")
        mark = self._extract_feature(event, symbol, "mark_price")
        index = self._extract_feature(event, symbol, "index_price")
        oi = self._extract_feature(event, symbol, "open_interest")
        basis = basis_bps(mark, index)
        feature_map = {
            "funding_rate": funding,
            "basis_bps": basis,
            "open_interest": oi,
        }
        if any(value is None for value in feature_map.values()):
            return False
        item.last_time_key = key
        item.closes.append(close)
        item.volumes.append(max(0.0, float(snapshot.volume or 0.0)))
        item.funding_rate.append(float(funding))
        item.basis_bps.append(float(basis))
        item.open_interest.append(float(oi))
        return True

    def calculate_signals_window(self, event: Any, aggregator: Any = None) -> None:
        _ = aggregator
        event_key = time_key(getattr(event, "time", None))
        updated = False
        for symbol in _event_symbols(event, self.symbol_list):
            snapshot = _window_snapshot(event, symbol)
            if snapshot is not None and self._update_symbol(event, symbol, snapshot):
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
            if snapshot is not None and self._update_symbol(event, str(symbol), snapshot):
                key = time_key(snapshot.time)
                if key and key != self._last_eval_time_key:
                    self._last_eval_time_key = key
                    self._tick += 1
                    self._evaluate(snapshot.time)

    def _raw_legs(self, item: _State) -> dict[str, float] | None:
        """Compute the four raw factor legs for a single symbol, or ``None``."""
        closes = list(item.closes)
        required = self.trend_lookback + self.skip_bars + 1
        if len(closes) < required:
            return None
        vol = realized_volatility(item.closes, window=self.vol_window)
        if vol is None or vol <= _EPS:
            return None
        trend = volatility_adjusted_momentum(
            item.closes, lookback=self.trend_lookback, volatility_window=self.vol_window
        )
        if trend is None:
            return None
        # 12-1 style cross-sectional momentum: total return over the lookback
        # window ending skip_bars ago, divided by realized vol.
        recent = closes[-(self.skip_bars + 1)]
        base = closes[-(self.trend_lookback + self.skip_bars + 1)]
        if base <= 0.0 or recent <= 0.0:
            return None
        xs_mom = (recent / base - 1.0) / max(vol, _EPS)
        # Carry: negate funding z-score (calm/negative funding => long tailwind)
        # plus a negated basis z-score (rich basis => headwind).  BOTH legs are
        # z-scored over the same window so neither dominates by raw units before
        # the cross-sectional combine: basis_bps is in basis points (~50-100) and
        # would otherwise swamp the O(1) funding z-score if added raw.
        funding_tail = list(item.funding_rate)[-self.funding_z_window :]
        if len(funding_tail) < 2:
            return None
        funding_z = _rolling_z(funding_tail)
        basis_tail = list(item.basis_bps)[-self.funding_z_window :]
        basis_now = float(item.basis_bps[-1]) if item.basis_bps else 0.0
        basis_z = _rolling_z(basis_tail) if len(basis_tail) >= 2 else 0.0
        carry = -funding_z - basis_z
        low_vol = -float(vol)
        return {
            "trend": float(trend),
            "xs_momentum": float(xs_mom),
            "low_vol": float(low_vol),
            "carry": float(carry),
            # Carried-through scalars used downstream for sizing/metadata.
            "_realized_vol": float(vol),
            "_funding_z": float(funding_z),
            "_basis_bps": float(basis_now),
        }

    def _composite_rows(
        self,
    ) -> tuple[list[tuple[float, str, dict[str, Any]]], dict[str, float]]:
        """Cross-sectionally z-score each leg at this bar and blend a composite."""
        raw: dict[str, dict[str, float]] = {}
        for symbol, item in self._state.items():
            legs = self._raw_legs(item)
            if legs is not None:
                raw[symbol] = legs
        if len(raw) < self.min_symbols:
            return [], {}
        # Z-score EACH leg across the cross-section at THIS bar (no-lookahead).
        leg_z = {
            leg: _zscore_across({symbol: legs[leg] for symbol, legs in raw.items()})
            for leg in _LEGS
        }
        rows: list[tuple[float, str, dict[str, Any]]] = []
        vols: dict[str, float] = {}
        for symbol, legs in raw.items():
            composite = sum(self.factor_weights[leg] * leg_z[leg][symbol] for leg in _LEGS)
            vols[symbol] = float(legs["_realized_vol"])
            rows.append(
                (
                    float(composite),
                    symbol,
                    {
                        "composite": float(composite),
                        "z_trend": float(leg_z["trend"][symbol]),
                        "z_xs_momentum": float(leg_z["xs_momentum"][symbol]),
                        "z_low_vol": float(leg_z["low_vol"][symbol]),
                        "z_carry": float(leg_z["carry"][symbol]),
                        "realized_vol": float(legs["_realized_vol"]),
                        "funding_z": float(legs["_funding_z"]),
                        "basis_bps": float(legs["_basis_bps"]),
                    },
                )
            )
        return rows, vols

    def _inverse_vol_weights(
        self,
        targets: dict[str, tuple[str, float, dict[str, Any]]],
        vols: dict[str, float],
    ) -> tuple[dict[str, float], float]:
        """Inverse-vol risk parity weights + portfolio vol-target scalar."""
        inv = {
            symbol: 1.0 / max(vols.get(symbol, 0.0), _EPS)
            for symbol in targets
            if vols.get(symbol, 0.0) > _EPS
        }
        total_inv = sum(inv.values())
        if total_inv <= _EPS:
            return {}, 1.0
        # Inverse-vol-weighted realized vol of the selected names.
        portfolio_vol = sum((inv[symbol] / total_inv) * vols[symbol] for symbol in inv)
        if self.use_shrunk_cov_vol:
            shrunk = self._shrunk_portfolio_vol(inv, total_inv)
            if shrunk is not None and shrunk > _EPS:
                portfolio_vol = shrunk
        scalar = 1.0
        if self.target_vol > 0.0 and portfolio_vol > _EPS:
            scalar = min(1.0, self.target_vol / portfolio_vol)
        weights = {
            symbol: (inv[symbol] / total_inv) * self.target_gross_exposure * scalar
            for symbol in inv
        }
        return weights, float(scalar)

    def _shrunk_portfolio_vol(self, inv: dict[str, float], total_inv: float) -> float | None:
        """Ledoit-Wolf shrunk portfolio vol of the selected names (gross scalar only).

        Builds an aligned ``(T, N)`` matrix of closed-bar log returns over the
        common trailing window of the selected names, weights them by the SAME
        normalized inverse-vol risk-parity weights used for the naive estimate, and
        delegates to :func:`shrunk_portfolio_vol`.  Returns ``None`` when the matrix
        cannot be built (e.g. fewer than two aligned closed bars), so the caller
        falls back to the naive ``sum(w_i * vol_i)``.
        """
        from lumina_quant.portfolio.optimizer_core import shrunk_portfolio_vol

        symbols = list(inv.keys())
        if not symbols:
            return None
        # Per-name closed-bar log-return series (past bars only; no-lookahead).
        series: dict[str, list[float]] = {}
        min_len = None
        for symbol in symbols:
            closes = list(self._state[symbol].closes)
            rets: list[float] = []
            for prev, curr in pairwise(closes):
                if prev > 0.0 and curr > 0.0:
                    rets.append(math.log(curr / prev))
            if len(rets) < 1:
                return None
            series[symbol] = rets
            min_len = len(rets) if min_len is None else min(min_len, len(rets))
        if not min_len or min_len < 2:
            return None
        matrix = [[series[symbol][-min_len + row] for symbol in symbols] for row in range(min_len)]
        weights = [inv[symbol] / total_inv for symbol in symbols]
        return float(shrunk_portfolio_vol(matrix, weights, window=self.vol_window))

    def _evaluate(self, event_time: Any) -> None:
        if len(self.symbol_list) < self.min_symbols:
            return
        # Enforce stop-loss / max-hold on EVERY bar (not only off-ticks) so a held
        # name that is re-selected or stays inside rebalance_band on a rebalance
        # tick is still protected; ranking below may then re-enter on a fresh signal.
        self._age(event_time)
        if self._tick % self.rebalance_bars:
            return
        rows, vols = self._composite_rows()
        if len(rows) < self.min_symbols:
            return
        targets = _ranked_targets(
            rows,
            threshold=self.threshold,
            max_longs=self.max_longs,
            max_shorts=self.max_shorts,
            allow_short=self.allow_short,
        )
        weights, scalar = self._inverse_vol_weights(targets, vols)
        self._emit_weighted(targets, weights, scalar, event_time)

    def _age(self, event_time: Any) -> None:
        # max_hold_bars == 0 disables max-hold aging; the shared helper treats it
        # as "exit immediately", so pass an effectively-unbounded cap in that case
        # and let only the stop-loss branch fire.
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

    def _emit_weighted(
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
                    _emit(
                        self.events,
                        strategy_id=_STRATEGY_ID,
                        symbol=symbol,
                        event_time=event_time,
                        signal_type="EXIT",
                        price=price,
                        metadata={"strategy": _STRATEGY_NAME, "reason": "rebalance_removed"},
                    )
                    item.mode = "OUT"
                    item.entry_price = None
                    item.bars_held = 0
                    item.composite = None
                continue
            target_mode, composite, meta = target
            weight = float(weights.get(symbol, 0.0))
            alloc = max(0.0, self.base_allocation * weight)
            if item.mode == target_mode:
                # Turnover gate: hold without a new order when the composite drift
                # stays inside the band (and the side has not flipped).
                prev = item.composite
                if prev is not None and abs(composite - prev) < self.rebalance_band:
                    item.bars_held += 1
                    continue
                item.composite = float(composite)
                item.bars_held += 1
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
            stop_loss = None
            if price is not None and self.stop_loss_pct > 0.0:
                stop_loss = price * (
                    1.0 - self.stop_loss_pct if target_mode == "LONG" else 1.0 + self.stop_loss_pct
                )
            metadata = _target_metadata(
                strategy=_STRATEGY_NAME,
                target_allocation=alloc,
                max_order_value=self.max_order_value,
                score=float(composite),
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
                strength=max(0.25, min(3.0, abs(composite) / max(self.threshold, _EPS))),
                price=price,
                stop_loss=stop_loss,
                metadata=metadata,
            )
            item.mode = target_mode
            item.entry_price = price
            item.bars_held = 0
            item.composite = float(composite)
        # Stop/max-hold housekeeping for names that stayed OUT or were not selected.
        for symbol, item in self._state.items():
            if item.mode == "OUT" or symbol in targets:
                continue
            price = item.closes[-1] if item.closes else None
            item.bars_held += 1
            stop = False
            if price is not None and item.entry_price is not None and self.stop_loss_pct > 0.0:
                if item.mode == "LONG" and price <= item.entry_price * (1.0 - self.stop_loss_pct):
                    stop = True
                if item.mode == "SHORT" and price >= item.entry_price * (1.0 + self.stop_loss_pct):
                    stop = True
            if stop or (self.max_hold_bars > 0 and item.bars_held >= self.max_hold_bars):
                _emit(
                    self.events,
                    strategy_id=_STRATEGY_ID,
                    symbol=symbol,
                    event_time=event_time,
                    signal_type="EXIT",
                    price=price,
                    metadata={
                        "strategy": _STRATEGY_NAME,
                        "reason": "stop_loss" if stop else "max_hold",
                    },
                )
                item.mode = "OUT"
                item.entry_price = None
                item.bars_held = 0
                item.composite = None


__all__ = ["DiversifiedMultiFactorEnsembleStrategy"]

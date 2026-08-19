"""Cross-sectional FUNDING-RATE MOMENTUM carry book (delta-funding premium).

``CrossSectionalFundingMomentumCarryStrategy`` ranks the perpetual-futures
universe by the *momentum* of the funding rate — the time-series slope and
EWMA of the FIRST DIFFERENCE of funding — rather than by the funding *level*
that every existing funding sleeve harvests.

Funding is highly autocorrelated, so its first difference (delta-funding) is a
slow, decorrelated, lowest-turnover signal.  A symbol whose funding is steadily
rising is paying longs an ever-larger premium to short it (and vice versa); the
edge is the funding accrual cash flow, which survives slippage because the book
trades at most once per 8h funding interval with multi-day holds.

Per symbol the realized funding rate is captured at each closed funding interval
via the shared 3-tier no-lookahead ``_extract_feature`` cascade (all-or-nothing:
when funding is absent the symbol simply contributes no entries, mirroring
``FundingHarvestCarry``'s graceful fallback).  The raw per-symbol momentum score
is the EWMA of the funding first-difference plus the rolling OLS slope of funding
over the trailing ``funding_slope_window`` intervals.  Raw scores are
CROSS-SECTIONALLY z-scored at the same interval via ``_zscore_across`` (zeros if
fewer than two symbols print), ranked long-top / short-bottom by the shared
ranker, and sized by INVERSE-VOL risk parity normalized to
``target_gross_exposure`` then scaled by ``min(1, target_vol / portfolio_vol)``.
A wide ``rebalance_band`` hysteresis on the score keeps held names in place
without new orders; stops and max-hold aging run every evaluation.

This is DISTINCT from every existing funding sleeve, which all trade the funding
LEVEL (carry sign): this book trades the funding TREND.  It is data-local (no
I/O, no hidden configuration bus), reuses the shared cross-sectional helpers and
the no-lookahead feature cascade, and never raises from ``calculate_signals``.

LABELING NOTE (accurate naming).  The ranking signal here is funding-rate
MOMENTUM / positioning-momentum (the TREND of the funding curve), NOT carry
harvest.  "Carry" in the name refers only to the funding-accrual cash flow that
accrues while a name is HELD across funding intervals -- it does not mean the
side is chosen to collect funding.  In fact, going LONG a name whose funding is
rising (top of the momentum rank) typically PAYS funding rather than collecting
it.  A genuine funding-sign-aware carry variant is available opt-in and default
OFF via ``true_carry_sign``: when ON, a ranked target is kept only if its side
actually COLLECTS the funding it is exposed to (LONG only when the latest funding
level is negative; SHORT only when it is positive).  With the flag OFF the book
is byte-identical to the pure momentum ranking.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from lumina_quant.core.plugin_registry import register
from lumina_quant.indicators.alpha_features import realized_volatility
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.strategies.adaptive_crypto_alpha_sleeves import (
    _age_cross_positions,
    _ranked_targets,
    _state_size,
)
from lumina_quant.strategies.diversified_multifactor_ensemble import _zscore_across
from lumina_quant.strategies.external_alpha_sleeves import (
    _EPS,
    _Snapshot,
    _annualize_per_bar_vol,
    _emit,
    _event_datetime_utc,
    _event_symbols,
    _extract_feature,
    _market_snapshot,
    _safe_non_negative_int,
    _target_metadata,
    _window_snapshot,
)
from lumina_quant.strategy import Strategy
from lumina_quant.tuning import HyperParam, resolve_params_from_schema

_STRATEGY_ID = "cross_sectional_funding_momentum_carry"
_STRATEGY_NAME = "CrossSectionalFundingMomentumCarryStrategy"
# 8h funding cadence floor (seconds).  decision_cadence_seconds must be >= this.
_FUNDING_CADENCE_SECONDS = 28800


@dataclass(slots=True)
class _State:
    closes: deque[float]
    volumes: deque[float]
    funding_rate: deque[float]
    mode: str = "OUT"
    entry_price: float | None = None
    bars_held: int = 0
    last_time_key: str = ""
    score: float | None = None


def _funding_momentum(
    funding: list[float], *, diff_window: int, slope_window: int, ewma_span: int
) -> float | None:
    """Delta-funding momentum: EWMA of first differences + rolling OLS slope.

    Uses strictly the funding prints provided (the just-closed interval is the
    last element — observable, no lookahead).  Returns ``None`` when there is too
    little history to form a first difference.
    """
    if len(funding) < 2:
        return None
    diffs = [funding[i] - funding[i - 1] for i in range(1, len(funding))]
    diff_tail = diffs[-max(1, diff_window) :]
    # EWMA of the first differences (most-recent weighted highest).
    span = max(1, int(ewma_span))
    alpha = 2.0 / (span + 1.0)
    ewma = diff_tail[0]
    for value in diff_tail[1:]:
        ewma = alpha * value + (1.0 - alpha) * ewma
    # Rolling OLS slope of the funding LEVEL over the trailing slope window.
    level_tail = funding[-max(2, slope_window) :]
    slope = _ols_slope(level_tail)
    return float(ewma) + float(slope)


def _ols_slope(values: list[float]) -> float:
    """Least-squares slope of ``values`` vs an index 0..n-1 (0.0 if degenerate)."""
    n = len(values)
    if n < 2:
        return 0.0
    mean_x = (n - 1) / 2.0
    mean_y = sum(values) / float(n)
    num = 0.0
    den = 0.0
    for i, value in enumerate(values):
        dx = i - mean_x
        num += dx * (value - mean_y)
        den += dx * dx
    if den <= _EPS:
        return 0.0
    return num / den


@register("strategy", _STRATEGY_NAME, interface="event_driven")
class CrossSectionalFundingMomentumCarryStrategy(Strategy):
    """Cross-sectional funding-rate MOMENTUM (delta-funding) carry book.

    Per symbol the realized funding rate is captured at each closed interval; the
    raw momentum score is the EWMA of the funding first-difference plus the
    rolling slope of funding.  Raw scores are cross-sectionally z-scored at the
    same interval, ranked long-top / short-bottom, and sized by inverse-vol risk
    parity normalized to ``target_gross_exposure`` then scaled by a portfolio
    vol-target clamp.  A wide ``rebalance_band`` on the score suppresses turnover
    (held names only flip when the score drifts beyond the band or the side
    flips); stops and max-hold aging run every evaluation.  Trades at most once
    per funding interval (``decision_cadence_seconds >= 8h``).
    """

    decision_cadence_seconds = _FUNDING_CADENCE_SECONDS
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False
    required_features = ("funding_rate",)

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "funding_diff_window": HyperParam.integer(
                "funding_diff_window", default=6, low=1, high=240
            ),
            "funding_slope_window": HyperParam.integer(
                "funding_slope_window", default=8, low=2, high=240
            ),
            "ewma_span": HyperParam.integer("ewma_span", default=4, low=1, high=120),
            "threshold": HyperParam.floating("threshold", default=0.5, low=0.0, high=3.0),
            "max_longs": HyperParam.integer("max_longs", default=5, low=1, high=50),
            "max_shorts": HyperParam.integer("max_shorts", default=5, low=0, high=50),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
            "true_carry_sign": HyperParam.boolean("true_carry_sign", default=False, tunable=False),
            "min_symbols": HyperParam.integer("min_symbols", default=4, low=2, high=512),
            "target_gross_exposure": HyperParam.floating(
                "target_gross_exposure", default=1.0, low=0.0, high=3.0
            ),
            "target_vol": HyperParam.floating("target_vol", default=0.20, low=0.0, high=2.0),
            "rebalance_band": HyperParam.floating(
                "rebalance_band", default=0.30, low=0.0, high=3.0
            ),
            "stop_loss_pct": HyperParam.floating("stop_loss_pct", default=0.08, low=0.0, high=0.50),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=21, low=0, high=2000),
            "vol_window": HyperParam.integer("vol_window", default=24, low=2, high=400),
            "base_allocation": HyperParam.floating(
                "base_allocation", default=0.20, low=0.0, high=2.0, tunable=False
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
        self.funding_diff_window = max(1, int(resolved["funding_diff_window"]))
        self.funding_slope_window = max(2, int(resolved["funding_slope_window"]))
        self.ewma_span = max(1, int(resolved["ewma_span"]))
        self.threshold = max(0.0, float(resolved["threshold"]))
        self.max_longs = max(1, int(resolved["max_longs"]))
        self.max_shorts = max(0, int(resolved["max_shorts"]))
        self.allow_short = bool(resolved["allow_short"])
        self.min_symbols = max(2, int(resolved["min_symbols"]))
        self.target_gross_exposure = max(0.0, float(resolved["target_gross_exposure"]))
        self.target_vol = max(0.0, float(resolved["target_vol"]))
        self.rebalance_band = max(0.0, float(resolved["rebalance_band"]))
        self.stop_loss_pct = max(0.0, float(resolved["stop_loss_pct"]))
        self.max_hold_bars = max(0, int(resolved["max_hold_bars"]))
        self.vol_window = max(2, int(resolved["vol_window"]))
        self.base_allocation = max(0.0, float(resolved["base_allocation"]))
        self.max_symbol_exposure_pct = max(0.0, float(resolved["max_symbol_exposure_pct"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        # STRATEGY-IMPROVE (strategy-local, config-gated, default OFF => byte-
        # identical). Opt-in genuine funding-sign-aware carry: keep a ranked
        # target only if its side actually COLLECTS the funding it holds (LONG
        # iff latest funding < 0, SHORT iff latest funding > 0). Read from raw
        # resolved params so manifests, class schema and runtime behavior agree.
        self.true_carry_sign = bool(resolved["true_carry_sign"])
        # Enforce the 8h funding cadence floor in code (instances cannot decide
        # faster than the underlying funding accrual interval).
        self.decision_cadence_seconds = max(
            int(type(self).decision_cadence_seconds), _FUNDING_CADENCE_SECONDS
        )
        size = _state_size(
            self.funding_diff_window + 1,
            self.funding_slope_window,
            self.vol_window,
            self.max_hold_bars,
        )
        self._state = {
            symbol: _State(
                closes=deque(maxlen=size),
                volumes=deque(maxlen=size),
                funding_rate=deque(maxlen=size),
            )
            for symbol in self.symbol_list
        }
        self._last_eval_time_key = ""
        self._tick = 0
        # Recent decision-interval epochs (seconds) for deterministic bar-spacing
        # inference: the vol-target scalar annualizes the per-bar portfolio vol
        # via sqrt(bars_per_year) derived from the median gap here.
        self._recent_times: deque[float] = deque(maxlen=16)

    def get_state(self) -> dict[str, Any]:
        return {
            "last_eval_time_key": self._last_eval_time_key,
            "tick": int(self._tick),
            "recent_times": list(self._recent_times),
            "symbol_state": {
                symbol: {
                    "closes": list(item.closes),
                    "volumes": list(item.volumes),
                    "funding_rate": list(item.funding_rate),
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
        for value in list(state.get("recent_times") or [])[-int(self._recent_times.maxlen or 0) :]:
            parsed = safe_float(value)
            if parsed is not None:
                self._recent_times.append(parsed)
        raw = state.get("symbol_state")
        if not isinstance(raw, dict):
            return
        for symbol, payload in raw.items():
            if symbol not in self._state or not isinstance(payload, dict):
                continue
            item = self._state[symbol]
            for attr in ("closes", "volumes", "funding_rate"):
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
            item.score = safe_float(payload.get("score"))

    def _extract_feature(self, event: Any, symbol: str, field: str) -> float | None:
        return _extract_feature(self.bars, event, symbol, field)

    def _update_symbol(self, event: Any, symbol: str, snapshot: _Snapshot) -> bool:
        close = safe_float(snapshot.close)
        if close is None or close <= self.min_price:
            return False
        item = self._state[symbol]
        key = time_key(snapshot.time)
        if key and key == item.last_time_key:
            return False
        # All-or-nothing funding capture at the closed interval (3-tier cascade).
        # When funding is absent we DROP the bar entirely so this symbol simply
        # contributes no entries (graceful FundingHarvestCarry-style fallback).
        funding = self._extract_feature(event, symbol, "funding_rate")
        if funding is None:
            return False
        item.last_time_key = key
        item.closes.append(close)
        item.volumes.append(max(0.0, float(snapshot.volume or 0.0)))
        item.funding_rate.append(float(funding))
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

    def _raw_score(self, item: _State) -> tuple[float, float] | None:
        """Return ``(funding_momentum, realized_vol)`` for a symbol, or ``None``."""
        funding = list(item.funding_rate)
        momentum = _funding_momentum(
            funding,
            diff_window=self.funding_diff_window,
            slope_window=self.funding_slope_window,
            ewma_span=self.ewma_span,
        )
        if momentum is None:
            return None
        vol = realized_volatility(item.closes, window=self.vol_window)
        if vol is None or vol <= _EPS:
            return None
        return float(momentum), float(vol)

    def _score_rows(
        self,
    ) -> tuple[list[tuple[float, str, dict[str, Any]]], dict[str, float]]:
        """Cross-sectionally z-score the raw funding-momentum at this interval."""
        raw: dict[str, float] = {}
        vols: dict[str, float] = {}
        for symbol, item in self._state.items():
            scored = self._raw_score(item)
            if scored is not None:
                raw[symbol], vols[symbol] = scored
        if len(raw) < self.min_symbols:
            return [], {}
        z = _zscore_across(raw)
        rows: list[tuple[float, str, dict[str, Any]]] = []
        for symbol, score in z.items():
            rows.append(
                (
                    float(score),
                    symbol,
                    {
                        "funding_momentum": float(raw[symbol]),
                        "realized_vol": float(vols[symbol]),
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
        # ``portfolio_vol`` is the inverse-vol-weighted PER-BAR vol; the
        # ``inv / total_inv`` normalization above is scale-invariant (annualizing
        # every vol_i cancels), so the risk-parity weights are horizon-free. The
        # vol-target SCALAR, however, compares ``portfolio_vol`` against an
        # annual-scale ``target_vol`` (~0.20): annualize the per-bar estimate via
        # sqrt(bars_per_year) inferred from observed interval spacing first,
        # otherwise the Moreira-Muir clamp is INERT. When spacing is unavailable
        # we pass through (scalar=1.0) rather than throttle on mismatched horizons.
        portfolio_vol = sum((inv[symbol] / total_inv) * vols[symbol] for symbol in inv)
        scalar = 1.0
        if self.target_vol > 0.0 and portfolio_vol > _EPS:
            portfolio_vol_ann = _annualize_per_bar_vol(portfolio_vol, self._recent_times)
            if portfolio_vol_ann is not None and portfolio_vol_ann > _EPS:
                scalar = min(1.0, self.target_vol / portfolio_vol_ann)
        weights = {
            symbol: (inv[symbol] / total_inv) * self.target_gross_exposure * scalar
            for symbol in inv
        }
        return weights, float(scalar)

    def _filter_carry_sign(
        self, targets: dict[str, tuple[str, float, dict[str, Any]]]
    ) -> dict[str, tuple[str, float, dict[str, Any]]]:
        """Keep only targets whose side actually COLLECTS funding.

        LONG collects when the latest funding level is negative; SHORT collects
        when it is positive.  Names printing no funding (empty history) contribute
        nothing and are dropped.  Only invoked when ``true_carry_sign`` is ON.
        """
        kept: dict[str, tuple[str, float, dict[str, Any]]] = {}
        for symbol, entry in targets.items():
            mode = entry[0]
            item = self._state.get(symbol)
            funding = float(item.funding_rate[-1]) if item and item.funding_rate else 0.0
            long_collects = mode == "LONG" and funding < 0.0
            short_collects = mode == "SHORT" and funding > 0.0
            if long_collects or short_collects:
                kept[symbol] = entry
        return kept

    def _evaluate(self, event_time: Any) -> None:
        if len(self.symbol_list) < self.min_symbols:
            return
        # Record the decision-interval epoch so the vol-target scalar can infer
        # bar spacing.  This lane ranks EVERY evaluation (each funding interval)
        # with no coarser rebalance gate, so this per-interval hook already
        # reflects the true bar cadence.
        dt = _event_datetime_utc(event_time)
        if dt is not None:
            self._recent_times.append(dt.timestamp())
        # Stops / max-hold run on EVERY evaluation (each funding interval) before
        # the ranker so a held name is always protected.
        self._age(event_time)
        rows, vols = self._score_rows()
        if len(rows) < self.min_symbols:
            return
        targets = _ranked_targets(
            rows,
            threshold=self.threshold,
            max_longs=self.max_longs,
            max_shorts=self.max_shorts,
            allow_short=self.allow_short,
        )
        if self.true_carry_sign:
            targets = self._filter_carry_sign(targets)
        weights, scalar = self._inverse_vol_weights(targets, vols)
        self._emit_weighted(targets, weights, scalar, event_time)

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
                    item.score = None
                continue
            target_mode, score, meta = target
            weight = float(weights.get(symbol, 0.0))
            alloc = max(0.0, self.base_allocation * weight)
            if alloc <= 0.0:
                # Zero-alloc entries omit ``target_allocation`` from metadata and
                # the engine resizes them to its DEFAULT allocation.  Treat a
                # zero-weight target as removed: flatten if held, never emit an
                # unsized entry (a held name outside the rebalance band would
                # otherwise re-emit with alloc=0).
                if item.mode != "OUT":
                    _emit(
                        self.events,
                        strategy_id=_STRATEGY_ID,
                        symbol=symbol,
                        event_time=event_time,
                        signal_type="EXIT",
                        price=price,
                        metadata={"strategy": _STRATEGY_NAME, "reason": "zero_allocation"},
                    )
                    item.mode = "OUT"
                    item.entry_price = None
                    item.bars_held = 0
                    item.score = None
                continue
            if item.mode == target_mode:
                # Turnover gate: hold without a new order while the score drift
                # stays inside the band (and the side has not flipped). NOTE: do
                # NOT increment bars_held here — _age_cross_positions already ages
                # every non-OUT name once per evaluation (double-count would fire
                # max_hold ~2x early).
                prev = item.score
                if prev is not None and abs(score - prev) < self.rebalance_band:
                    continue
                item.score = float(score)
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
                strength=max(0.25, min(3.0, abs(score) / max(self.threshold, _EPS))),
                price=price,
                stop_loss=stop_loss,
                metadata=metadata,
            )
            item.mode = target_mode
            item.entry_price = price
            item.bars_held = 0
            item.score = float(score)
        # Stop / max-hold housekeeping for names that stayed OUT or were not
        # selected this interval.
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
                item.score = None


__all__ = ["CrossSectionalFundingMomentumCarryStrategy"]

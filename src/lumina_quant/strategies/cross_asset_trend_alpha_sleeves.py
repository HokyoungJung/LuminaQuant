"""Cross-asset diversified trend and intermarket lead-lag continuation sleeves.

These two event-driven sleeves are deliberately data-local (no I/O, no hidden
configuration bus) and reuse the shared cross-sectional rebalancer
(``_ranked_targets`` / ``_emit_rebalance_targets`` / ``_age_cross_positions``)
together with ``_CrossSectionalState`` so they slot into the existing candidate
pipeline as drop-in research hypotheses:

- ``CrossAssetDiversifiedTrendStrategy``: a single COORDINATED multi-asset trend
  book spanning the whole normalized universe (crypto + equity + commodity +
  metals).  Per symbol it scores a time-series-momentum trend (sign of the
  simple return over ``trend_lookback``, optionally confirmed by an SMA slope),
  keeps only names whose trend magnitude clears ``min_trend``, sizes each
  selected name by INVERSE realized volatility (risk parity over ``vol_window``)
  and scales the whole book to a portfolio ``target_vol`` (sum of per-name risk
  contributions -> single clamped scalar).  This is the canonical robust
  managed-futures edge: diversified cross-asset trend with inverse-vol risk
  budgeting + portfolio vol-targeting; diversification across weakly-correlated
  asset classes raises risk-adjusted return.  It is DISTINCT from
  ``TopCapTimeSeriesMomentumStrategy`` (crypto top-cap, single-timeframe, no
  cross-asset inverse-vol risk budgeting and no portfolio vol-target), from the
  ``CompositeTrend`` family (per-symbol regime composite, not one risk-budgeted
  book) and from the per-symbol return riders (independent positions rather than
  one coordinated risk-parity book).

- ``IntermarketLeadLagContinuationStrategy``: a COMMODITY -> EQUITY intermarket
  lead-lag continuation sleeve.  Configurable leader -> follower pairs (default
  oil -> energy: ``CL/USDT``/``BZ/USDT`` leads ``XLE/USDT``; optional copper ->
  industrial proxy).  When a leader's recent return over ``lead_lookback``
  clears an entry z-threshold the follower is entered in the SAME direction
  (delayed-reaction continuation); the leg exits on follower catch-up (the
  follower's own move converges to the leader move), ``max_hold`` or a stop.
  This is DISTINCT from ``SemisLeadLagRotationStrategy`` (SOXL -> single-name
  semis, intra-equity), ``BenchmarkLeadLagContinuationStrategy`` (crypto
  benchmark -> name), ``LeadLagSpilloverStrategy`` (generic ridge spillover) and
  ``MetalEquityDivergenceReversalStrategy`` (divergence FADE) — it is
  specifically a directional COMMODITY -> EQUITY continuation.

Both sleeves are >= 30m only (1d primary, 4h variant), trade through
``_emit``/``_target_metadata`` and never raise from ``calculate_signals``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from lumina_quant.core.plugin_registry import register
from lumina_quant.indicators.alpha_features import (
    realized_volatility,
    rolling_zscore,
    simple_return,
)
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.indicators.moving_average import simple_moving_average
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
from lumina_quant.strategies.robust_alpha_sleeves import (
    _CrossSectionalState,
    _mode,
    _restore_deque,
)
from lumina_quant.strategy import Strategy
from lumina_quant.tuning import HyperParam, resolve_params_from_schema


def _pack_cross(item: _CrossSectionalState) -> dict[str, Any]:
    return {
        "closes": list(item.closes),
        "volumes": list(item.volumes),
        "mode": item.mode,
        "entry_price": item.entry_price,
        "bars_held": int(item.bars_held),
        "last_time_key": item.last_time_key,
    }


def _restore_cross(item: _CrossSectionalState, payload: dict[str, Any]) -> None:
    _restore_deque(item.closes, payload.get("closes"))
    _restore_deque(item.volumes, payload.get("volumes"))
    item.mode = _mode(payload.get("mode"))
    item.entry_price = safe_float(payload.get("entry_price"))
    item.bars_held = _safe_non_negative_int(payload.get("bars_held"))
    item.last_time_key = str(payload.get("last_time_key", ""))


@register("strategy", "CrossAssetDiversifiedTrendStrategy", interface="event_driven")
class CrossAssetDiversifiedTrendStrategy(Strategy):
    """Risk-budgeted cross-asset trend book with inverse-vol + portfolio vol-target.

    Per symbol a TSMOM trend signal is the simple return over ``trend_lookback``
    (optionally confirmed by an SMA slope over the same lookback).  Names whose
    absolute trend clears ``min_trend`` become candidates; each candidate is
    sized by inverse realized volatility (risk parity over ``vol_window``) and
    the whole book is scaled to a portfolio ``target_vol`` by dividing the
    per-name allocation through the summed risk contributions (clamped).  The
    long leg rides up-trenders and, when ``allow_short`` is set, the short leg
    rides down-trenders.  Rebalance is banded: a name only flips when its
    inverse-vol weight drifts by more than ``rebalance_band`` (or its side
    flips), keeping turnover low.
    """

    decision_cadence_seconds = 86400
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "trend_lookback": HyperParam.integer("trend_lookback", default=90, low=4, high=20000),
            "vol_window": HyperParam.integer("vol_window", default=60, low=4, high=4096),
            "min_trend": HyperParam.floating("min_trend", default=0.02, low=0.0, high=5.0),
            "max_positions": HyperParam.integer("max_positions", default=10, low=1, high=256),
            "target_vol": HyperParam.floating("target_vol", default=0.02, low=0.0, high=2.0),
            "rebalance_band": HyperParam.floating(
                "rebalance_band", default=0.25, low=0.0, high=1.0
            ),
            "confirm_sma_slope": HyperParam.boolean(
                "confirm_sma_slope", default=True, grid=[True, False]
            ),
            "rebalance_bars": HyperParam.integer("rebalance_bars", default=5, low=1, high=10080),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
            "stop_loss_pct": HyperParam.floating("stop_loss_pct", default=0.12, low=0.0, high=0.50),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=252, low=1, high=200000),
            "min_symbols": HyperParam.integer("min_symbols", default=6, low=2, high=512),
            "target_allocation": HyperParam.floating(
                "target_allocation", default=0.40, low=0.0, high=5.0, tunable=False
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
        self.min_trend = max(0.0, float(resolved["min_trend"]))
        self.max_positions = max(1, int(resolved["max_positions"]))
        self.target_vol = max(0.0, float(resolved["target_vol"]))
        self.rebalance_band = max(0.0, min(1.0, float(resolved["rebalance_band"])))
        self.confirm_sma_slope = bool(resolved["confirm_sma_slope"])
        self.rebalance_bars = max(1, int(resolved["rebalance_bars"]))
        self.allow_short = bool(resolved["allow_short"])
        self.stop_loss_pct = max(0.0, float(resolved["stop_loss_pct"]))
        self.max_hold_bars = max(1, int(resolved["max_hold_bars"]))
        self.min_symbols = max(2, int(resolved["min_symbols"]))
        self.target_allocation = max(0.0, float(resolved["target_allocation"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        size = _state_size(self.trend_lookback, self.vol_window, self.max_hold_bars)
        self._state = {
            symbol: _CrossSectionalState(deque(maxlen=size), deque(maxlen=size))
            for symbol in self.symbol_list
        }
        # Persisted per-symbol inverse-vol target weight (for the rebalance band).
        self._weights: dict[str, float] = dict.fromkeys(self.symbol_list, 0.0)
        self._last_eval_time_key = ""
        self._tick = 0

    def get_state(self) -> dict[str, Any]:
        return {
            "last_eval_time_key": self._last_eval_time_key,
            "tick": int(self._tick),
            "weights": dict(self._weights),
            "symbol_state": {symbol: _pack_cross(item) for symbol, item in self._state.items()},
        }

    def set_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        self._last_eval_time_key = str(state.get("last_eval_time_key", ""))
        self._tick = _safe_non_negative_int(state.get("tick"))
        raw_weights = state.get("weights")
        if isinstance(raw_weights, dict):
            for symbol, value in raw_weights.items():
                parsed = safe_float(value)
                if symbol in self._weights and parsed is not None:
                    self._weights[symbol] = max(0.0, float(parsed))
        raw = state.get("symbol_state")
        if isinstance(raw, dict):
            for symbol, payload in raw.items():
                if symbol in self._state and isinstance(payload, dict):
                    _restore_cross(self._state[symbol], payload)

    def calculate_signals_window(self, event: Any, aggregator: Any = None) -> None:
        _ = aggregator
        event_key = time_key(getattr(event, "time", None))
        updated = False
        for symbol in _event_symbols(event, self.symbol_list):
            snapshot = _window_snapshot(event, symbol)
            if snapshot is not None and self._update(symbol, snapshot):
                updated = True
        if updated and event_key and event_key != self._last_eval_time_key:
            self._last_eval_time_key = event_key
            self._tick += 1
            self._rebalance(getattr(event, "time", None))

    def calculate_signals(self, event: Any) -> None:
        if str(getattr(event, "type", "")).upper() == "MARKET_WINDOW":
            self.calculate_signals_window(event, None)
            return
        if getattr(event, "type", None) != "MARKET":
            return
        symbol = getattr(event, "symbol", None)
        if symbol in self._state:
            snapshot = _market_snapshot(event)
            if snapshot is not None and self._update(str(symbol), snapshot):
                key = time_key(snapshot.time)
                if key and key != self._last_eval_time_key:
                    self._last_eval_time_key = key
                    self._tick += 1
                    self._rebalance(snapshot.time)

    def _update(self, symbol: str, snapshot: _Snapshot) -> bool:
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

    def _trend_rows(
        self,
    ) -> tuple[list[tuple[float, str, dict[str, Any]]], dict[str, float]]:
        """Score each symbol's trend and inverse-vol weight (pre-scaling)."""
        scored: list[tuple[float, str, float, float, dict[str, Any]]] = []
        for symbol, item in self._state.items():
            trend = simple_return(item.closes, lookback=self.trend_lookback)
            vol = realized_volatility(item.closes, window=self.vol_window)
            if trend is None or vol is None or vol <= _EPS:
                continue
            if abs(trend) < self.min_trend:
                continue
            if self.confirm_sma_slope:
                sma_now = simple_moving_average(item.closes, self.trend_lookback)
                closes = list(item.closes)
                # SMA one lookback ago confirms the trend's direction (slope).
                if len(closes) <= self.trend_lookback or sma_now is None:
                    continue
                prior = closes[: -self.trend_lookback]
                sma_prior = simple_moving_average(prior, self.trend_lookback)
                if sma_prior is None:
                    continue
                slope = sma_now - sma_prior
                if (trend > 0.0 and slope <= 0.0) or (trend < 0.0 and slope >= 0.0):
                    continue
            inv_vol = 1.0 / max(vol, _EPS)
            scored.append((float(trend), symbol, float(vol), float(inv_vol), {}))
        if not scored:
            return [], {}
        # Keep the strongest |trend| names up to max_positions.
        scored.sort(key=lambda row: abs(row[0]), reverse=True)
        selected = scored[: self.max_positions]
        total_inv_vol = sum(row[3] for row in selected)
        if total_inv_vol <= _EPS:
            return [], {}
        # Portfolio vol-target scalar: each name contributes ~weight * vol of risk;
        # with inverse-vol weights every name contributes the same unit, so the
        # book risk is proportional to the count.  Scale so the summed risk
        # contribution hits target_vol, then clamp to keep gross bounded.
        risk_sum = sum((row[3] / total_inv_vol) * row[2] for row in selected)
        scalar = 1.0
        if self.target_vol > 0.0 and risk_sum > _EPS:
            scalar = min(2.0, self.target_vol / risk_sum)
        rows: list[tuple[float, str, dict[str, Any]]] = []
        weights: dict[str, float] = {}
        for trend, symbol, vol, inv_vol, _meta in selected:
            weight = (inv_vol / total_inv_vol) * scalar
            weights[symbol] = float(weight)
            # The cross-sectional ranker keys on a signed score; carry the trend
            # sign so up-trenders rank long and down-trenders rank short, while
            # the inverse-vol weight rides in the metadata for sizing.
            rows.append(
                (
                    float(trend),
                    symbol,
                    {
                        "trend_return": trend,
                        "realized_vol": vol,
                        "inverse_vol_weight": float(weight),
                        "vol_target_scalar": float(scalar),
                    },
                )
            )
        return rows, weights

    def _rebalance(self, event_time: Any) -> None:
        if len(self.symbol_list) < self.min_symbols:
            return
        if self._tick % self.rebalance_bars:
            _age_cross_positions(
                self.events,
                self._state,
                event_time=event_time,
                strategy_id="cross_asset_diversified_trend",
                strategy_name="CrossAssetDiversifiedTrendStrategy",
                stop_loss_pct=self.stop_loss_pct,
                max_hold_bars=self.max_hold_bars,
            )
            return
        rows, weights = self._trend_rows()
        if len(rows) < 1:
            _age_cross_positions(
                self.events,
                self._state,
                event_time=event_time,
                strategy_id="cross_asset_diversified_trend",
                strategy_name="CrossAssetDiversifiedTrendStrategy",
                stop_loss_pct=self.stop_loss_pct,
                max_hold_bars=self.max_hold_bars,
            )
            return
        targets = _ranked_targets(
            rows,
            threshold=self.min_trend,
            max_longs=self.max_positions,
            max_shorts=self.max_positions if self.allow_short else 0,
            allow_short=self.allow_short,
        )
        # Turnover is already suppressed by _emit_weighted's hold branch, which
        # keeps a same-side held name WITHOUT a new order. Do NOT delete held
        # names from `targets` here: _emit_weighted treats a missing target as a
        # removal and would EXIT->re-ENTER every tick (churn) and reset aging.
        # `rebalance_band` is reserved for a future banded weight-drift RESIZE.
        # Per-name allocation is the inverse-vol weight * target_allocation; emit
        # via the shared rebalancer (which evenly splits target_gross_exposure
        # across selected names — we override per-name sizing through metadata).
        self._emit_weighted(targets, weights, event_time)

    def _emit_weighted(
        self,
        targets: dict[str, tuple[str, float, dict[str, Any]]],
        weights: dict[str, float],
        event_time: Any,
    ) -> None:
        target_symbols = set(targets)
        for symbol, item in self._state.items():
            target = targets.get(symbol)
            price = item.closes[-1] if item.closes else None
            if target is None:
                if item.mode != "OUT":
                    _emit(
                        self.events,
                        strategy_id="cross_asset_diversified_trend",
                        symbol=symbol,
                        event_time=event_time,
                        signal_type="EXIT",
                        price=price,
                        metadata={
                            "strategy": "CrossAssetDiversifiedTrendStrategy",
                            "reason": "rebalance_removed",
                        },
                    )
                    item.mode = "OUT"
                    item.entry_price = None
                    item.bars_held = 0
                    self._weights[symbol] = 0.0
                continue
            target_mode, score, meta = target
            weight = float(weights.get(symbol, 0.0))
            alloc = max(0.0, self.target_allocation * weight)
            if item.mode == target_mode:
                item.bars_held += 1
                self._weights[symbol] = weight
                continue
            if item.mode != "OUT":
                _emit(
                    self.events,
                    strategy_id="cross_asset_diversified_trend",
                    symbol=symbol,
                    event_time=event_time,
                    signal_type="EXIT",
                    price=price,
                    metadata={
                        "strategy": "CrossAssetDiversifiedTrendStrategy",
                        "reason": "side_flip",
                    },
                )
            stop_loss = None
            if price is not None and self.stop_loss_pct > 0.0:
                stop_loss = price * (
                    1.0 - self.stop_loss_pct if target_mode == "LONG" else 1.0 + self.stop_loss_pct
                )
            metadata = _target_metadata(
                strategy="CrossAssetDiversifiedTrendStrategy",
                target_allocation=alloc,
                max_order_value=self.max_order_value,
                score=score,
                target_mode=target_mode,
                **meta,
            )
            _emit(
                self.events,
                strategy_id="cross_asset_diversified_trend",
                symbol=symbol,
                event_time=event_time,
                signal_type=target_mode,
                strength=max(0.25, min(3.0, abs(score) / max(self.min_trend, _EPS))),
                price=price,
                stop_loss=stop_loss,
                metadata=metadata,
            )
            item.mode = target_mode
            item.entry_price = price
            item.bars_held = 0
            self._weights[symbol] = weight
        # Age the names that stayed OUT (stop/max-hold housekeeping is a no-op
        # for OUT names but keeps the contract identical to the sibling sleeves).
        for symbol, item in self._state.items():
            if item.mode == "OUT" or symbol in target_symbols:
                continue
            price = item.closes[-1] if item.closes else None
            item.bars_held += 1
            stop = False
            if price is not None and item.entry_price is not None and self.stop_loss_pct > 0.0:
                if item.mode == "LONG" and price <= item.entry_price * (1.0 - self.stop_loss_pct):
                    stop = True
                if item.mode == "SHORT" and price >= item.entry_price * (1.0 + self.stop_loss_pct):
                    stop = True
            if stop or item.bars_held >= self.max_hold_bars:
                _emit(
                    self.events,
                    strategy_id="cross_asset_diversified_trend",
                    symbol=symbol,
                    event_time=event_time,
                    signal_type="EXIT",
                    price=price,
                    metadata={
                        "strategy": "CrossAssetDiversifiedTrendStrategy",
                        "reason": "stop_loss" if stop else "max_hold",
                    },
                )
                item.mode = "OUT"
                item.entry_price = None
                item.bars_held = 0
                self._weights[symbol] = 0.0


@dataclass(slots=True)
class _PairLegState:
    leader: str
    follower: str
    leader_closes: deque[float]
    follower_closes: deque[float]
    leader_returns: deque[float]
    mode: str = "OUT"
    entry_price: float | None = None
    entry_leader_return: float | None = None
    bars_held: int = 0
    last_eval_key: str = ""


def _parse_pair_spec(raw: str) -> list[tuple[str, str]]:
    """Parse ``leader>follower`` pairs from a comma/semicolon separated string."""
    out: list[tuple[str, str]] = []
    for token in str(raw or "").replace(";", ",").split(","):
        chunk = token.strip()
        if not chunk or ">" not in chunk:
            continue
        leader, _, follower = chunk.partition(">")
        leader = leader.strip()
        follower = follower.strip()
        if leader and follower:
            out.append((leader, follower))
    return out


@register("strategy", "IntermarketLeadLagContinuationStrategy", interface="event_driven")
class IntermarketLeadLagContinuationStrategy(Strategy):
    """Commodity -> equity directional lead-lag continuation across configured pairs.

    For each configured ``leader>follower`` pair the leader's return over
    ``lead_lookback`` is z-scored against its own trailing window; when the
    z-score clears ``entry_z`` (and the raw move clears ``min_leader_move``) and
    the follower has NOT yet chased (its own ``follow_lookback`` return stays
    under ``max_follower_move``), the follower is entered in the SAME direction
    as the leader.  The leg exits once the follower catches up (its move
    converges toward the leader move within ``catch_up_fraction``), on a stop, or
    on ``max_hold_bars``.  Per-pair state holds bounded leader/follower close
    deques plus the leader-return history for the rolling z-score.
    """

    decision_cadence_seconds = 86400
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "pair_spec": HyperParam.string(
                "pair_spec",
                default="CL/USDT>XLE/USDT,BZ/USDT>XLE/USDT",
                tunable=False,
            ),
            "lead_lookback": HyperParam.integer("lead_lookback", default=5, low=1, high=4096),
            "follow_lookback": HyperParam.integer("follow_lookback", default=3, low=1, high=4096),
            "z_window": HyperParam.integer("z_window", default=60, low=4, high=20000),
            "entry_z": HyperParam.floating("entry_z", default=1.5, low=0.0, high=10.0),
            "min_leader_move": HyperParam.floating(
                "min_leader_move", default=0.01, low=0.0, high=1.0
            ),
            "max_follower_move": HyperParam.floating(
                "max_follower_move", default=0.01, low=0.0, high=1.0
            ),
            "catch_up_fraction": HyperParam.floating(
                "catch_up_fraction", default=0.70, low=0.05, high=1.0
            ),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
            "stop_loss_pct": HyperParam.floating("stop_loss_pct", default=0.04, low=0.0, high=0.50),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=20, low=1, high=200000),
            "target_allocation": HyperParam.floating(
                "target_allocation", default=0.02, low=0.0, high=2.0, tunable=False
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
        self.lead_lookback = max(1, int(resolved["lead_lookback"]))
        self.follow_lookback = max(1, int(resolved["follow_lookback"]))
        self.z_window = max(3, int(resolved["z_window"]))
        self.entry_z = max(0.0, float(resolved["entry_z"]))
        self.min_leader_move = max(0.0, float(resolved["min_leader_move"]))
        self.max_follower_move = max(0.0, float(resolved["max_follower_move"]))
        self.catch_up_fraction = max(0.01, min(1.0, float(resolved["catch_up_fraction"])))
        self.allow_short = bool(resolved["allow_short"])
        self.stop_loss_pct = max(0.0, float(resolved["stop_loss_pct"]))
        self.max_hold_bars = max(1, int(resolved["max_hold_bars"]))
        self.target_allocation = max(0.0, float(resolved["target_allocation"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        size = _state_size(
            self.lead_lookback, self.follow_lookback, self.z_window, self.max_hold_bars
        )
        # Build per-pair state only for pairs whose leader AND follower are live.
        self._legs: list[_PairLegState] = []
        seen: set[tuple[str, str]] = set()
        symbols = set(self.symbol_list)
        for leader, follower in _parse_pair_spec(str(resolved["pair_spec"])):
            key = (leader, follower)
            if leader in symbols and follower in symbols and leader != follower and key not in seen:
                seen.add(key)
                self._legs.append(
                    _PairLegState(
                        leader,
                        follower,
                        deque(maxlen=size),
                        deque(maxlen=size),
                        deque(maxlen=size),
                    )
                )
        self._latest: dict[str, float] = {}
        self._last_seen: dict[str, str] = {}
        self._tracked = {symbol for leg in self._legs for symbol in (leg.leader, leg.follower)}

    def get_state(self) -> dict[str, Any]:
        return {
            "latest": dict(self._latest),
            "last_seen": dict(self._last_seen),
            "legs": [
                {
                    "leader_closes": list(leg.leader_closes),
                    "follower_closes": list(leg.follower_closes),
                    "leader_returns": list(leg.leader_returns),
                    "mode": leg.mode,
                    "entry_price": leg.entry_price,
                    "entry_leader_return": leg.entry_leader_return,
                    "bars_held": int(leg.bars_held),
                    "last_eval_key": leg.last_eval_key,
                }
                for leg in self._legs
            ],
        }

    def set_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        self._latest = {
            str(k): float(v)
            for k, v in dict(state.get("latest") or {}).items()
            if safe_float(v) is not None
        }
        self._last_seen = {str(k): str(v) for k, v in dict(state.get("last_seen") or {}).items()}
        payload = state.get("legs")
        if not isinstance(payload, list):
            return
        for leg, leg_payload in zip(self._legs, payload, strict=False):
            if not isinstance(leg_payload, dict):
                continue
            _restore_deque(leg.leader_closes, leg_payload.get("leader_closes"))
            _restore_deque(leg.follower_closes, leg_payload.get("follower_closes"))
            _restore_deque(leg.leader_returns, leg_payload.get("leader_returns"))
            leg.mode = _mode(leg_payload.get("mode"))
            leg.entry_price = safe_float(leg_payload.get("entry_price"))
            leg.entry_leader_return = safe_float(leg_payload.get("entry_leader_return"))
            leg.bars_held = _safe_non_negative_int(leg_payload.get("bars_held"))
            leg.last_eval_key = str(leg_payload.get("last_eval_key", ""))

    def calculate_signals_window(self, event: Any, aggregator: Any = None) -> None:
        _ = aggregator
        event_key = time_key(getattr(event, "time", None))
        for symbol in _event_symbols(event, list(self._tracked)):
            snapshot = _window_snapshot(event, symbol)
            if snapshot is not None:
                self._update(symbol, snapshot)
        self._evaluate(getattr(event, "time", None), event_key)

    def calculate_signals(self, event: Any) -> None:
        if str(getattr(event, "type", "")).upper() == "MARKET_WINDOW":
            self.calculate_signals_window(event, None)
            return
        if getattr(event, "type", None) != "MARKET":
            return
        symbol = getattr(event, "symbol", None)
        if symbol in self._tracked:
            snapshot = _market_snapshot(event)
            if snapshot is not None:
                self._update(str(symbol), snapshot)
                self._evaluate(snapshot.time, time_key(snapshot.time))

    def _update(self, symbol: str, snapshot: _Snapshot) -> None:
        close = safe_float(snapshot.close)
        if close is None or close <= self.min_price:
            return
        key = time_key(snapshot.time)
        if key and key == self._last_seen.get(symbol):
            return
        self._last_seen[symbol] = key
        self._latest[symbol] = close

    def _evaluate(self, event_time: Any, event_key: str = "") -> None:
        if not self._legs:
            return
        for leg in self._legs:
            self._evaluate_leg(leg, event_time, event_key)

    def _evaluate_leg(self, leg: _PairLegState, event_time: Any, event_key: str) -> None:
        if event_key:
            if leg.last_eval_key == event_key:
                return
            # Only advance the leg once both members have printed this bar.
            if (
                self._last_seen.get(leg.leader) != event_key
                or self._last_seen.get(leg.follower) != event_key
            ):
                return
            leg.last_eval_key = event_key
        leader_price = self._latest.get(leg.leader)
        follower_price = self._latest.get(leg.follower)
        if leader_price is None or follower_price is None:
            return
        leg.leader_closes.append(float(leader_price))
        leg.follower_closes.append(float(follower_price))
        leader_ret = simple_return(leg.leader_closes, lookback=self.lead_lookback)
        follower_ret = simple_return(leg.follower_closes, lookback=self.follow_lookback)
        if leader_ret is not None:
            leg.leader_returns.append(float(leader_ret))
        if leg.mode != "OUT":
            self._maybe_exit_leg(leg, event_time, leader_ret, follower_ret)
            return
        if leader_ret is None or follower_ret is None:
            return
        leader_z = rolling_zscore(leg.leader_returns, window=self.z_window)
        if leader_z is None or abs(leader_z) < self.entry_z:
            return
        if abs(leader_ret) < self.min_leader_move:
            return
        signal_type = ""
        # Continuation: follow the leader's direction only if the follower has
        # NOT already chased the move in that direction.
        if leader_ret > 0.0 and follower_ret <= self.max_follower_move:
            signal_type = "LONG"
        elif self.allow_short and leader_ret < 0.0 and follower_ret >= -self.max_follower_move:
            signal_type = "SHORT"
        if not signal_type:
            return
        stop_loss = None
        if self.stop_loss_pct > 0.0:
            stop_loss = follower_price * (
                1.0 - self.stop_loss_pct if signal_type == "LONG" else 1.0 + self.stop_loss_pct
            )
        metadata = _target_metadata(
            strategy="IntermarketLeadLagContinuationStrategy",
            target_allocation=self.target_allocation,
            max_order_value=self.max_order_value,
            leader_symbol=leg.leader,
            follower_symbol=leg.follower,
            leader_return=leader_ret,
            follower_return=follower_ret,
            leader_z=leader_z,
            target_mode=signal_type,
            reason="leader_continuation",
        )
        _emit(
            self.events,
            strategy_id="intermarket_lead_lag_continuation",
            symbol=leg.follower,
            event_time=event_time,
            signal_type=signal_type,
            strength=max(0.25, min(3.0, abs(leader_z) / max(self.entry_z, _EPS))),
            price=follower_price,
            stop_loss=stop_loss,
            metadata=metadata,
        )
        leg.mode = signal_type
        leg.entry_price = float(follower_price)
        leg.entry_leader_return = float(leader_ret)
        leg.bars_held = 0

    def _maybe_exit_leg(
        self,
        leg: _PairLegState,
        event_time: Any,
        leader_ret: float | None,
        follower_ret: float | None,
    ) -> None:
        leg.bars_held += 1
        price = leg.follower_closes[-1] if leg.follower_closes else None
        reason = ""
        stopped = (
            price is not None
            and leg.entry_price is not None
            and self.stop_loss_pct > 0.0
            and (
                (leg.mode == "LONG" and price <= leg.entry_price * (1.0 - self.stop_loss_pct))
                or (leg.mode == "SHORT" and price >= leg.entry_price * (1.0 + self.stop_loss_pct))
            )
        )
        if stopped:
            reason = "stop_loss"
        # Catch-up: the follower has converged to a fraction of the original
        # leader move (delayed reaction has played out).
        if (
            not reason
            and follower_ret is not None
            and leg.entry_leader_return is not None
            and abs(leg.entry_leader_return) > _EPS
        ):
            caught = follower_ret / leg.entry_leader_return
            if caught >= self.catch_up_fraction:
                reason = "follower_caught_up"
        if not reason and leg.bars_held >= self.max_hold_bars:
            reason = "max_hold"
        if not reason:
            return
        _emit(
            self.events,
            strategy_id="intermarket_lead_lag_continuation",
            symbol=leg.follower,
            event_time=event_time,
            signal_type="EXIT",
            price=price,
            metadata={
                "strategy": "IntermarketLeadLagContinuationStrategy",
                "leader_symbol": leg.leader,
                "follower_symbol": leg.follower,
                "reason": reason,
                "leader_return": leader_ret,
                "follower_return": follower_ret,
            },
        )
        leg.mode = "OUT"
        leg.entry_price = None
        leg.entry_leader_return = None
        leg.bars_held = 0


__all__ = [
    "CrossAssetDiversifiedTrendStrategy",
    "IntermarketLeadLagContinuationStrategy",
]

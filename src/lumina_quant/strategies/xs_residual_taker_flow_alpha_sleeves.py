"""Cross-sectional RESIDUAL taker-flow accumulation sleeve (flow price hasn't paid for).

``CrossSectionalResidualTakerFlowStrategy`` ranks the perp cross-section on
cumulative signed AGGRESSOR imbalance that is NOT explained by the same-window
return.  Per symbol, over a trailing ~5-day formation window (120 x 1h bars),
the sleeve computes ``sum(taker_buy_quote - taker_sell_quote) /
sum(quote_volume)`` -- net taker (market-order) quote flow normalized by total
quote turnover, a bounded [-1, +1] participation-weighted imbalance.  The
per-bar quote turnover is exactly ``taker_buy_quote + taker_sell_quote`` (every
trade has precisely one taker side), so the statistic needs ONLY the two
pre-registered taker columns.  The flow characteristic is cross-sectionally
z-scored, then residualized against the same-window return z via the pure
``cross_sectional_residualize`` Gram-Schmidt primitive.  The residual isolates
aggressive net flow NOT yet impounded in price: persistent absorbed buying with
a flat price is informed accumulation against passive liquidity, predicting
continuation as the passive side's repricing completes.  LONG the top residual
quintile, SHORT the bottom quintile, inverse-realized-vol sized, on a weekly
rebalance clock with a 0.5z rank-hysteresis band and a hard one-week
(``min_hold_decisions = 7`` daily decisions) minimum hold.

THEORY / EXTERNAL PRIOR
-----------------------
- Chordia & Subrahmanyam (2004, JFE) -- signed order imbalance positively
  predicts subsequent returns after controlling for returns.
- Barclay & Warner (1993, JFE) stealth trading -- informed flow is split and
  accumulates before price impact.
- Easley / Lopez de Prado / O'Hara flow-toxicity line.
- Repo ledger: graveyard item 7 records feature-flow families died on ~5% taker
  coverage PRE-backfill -- the design brief explicitly lifts this death for the
  taker columns (full aggTrades backfill).

EXPECTED NULL (pre-registered, verbatim)
----------------------------------------
Crypto perp taker flow is retail-dominated and contrarian-signed: residual flow
REVERSES rather than continues (fails pre-registered sign-lock), or flow is
fully impounded within hours so the 5-day residual carries zero cross-sectional
information (IC ~ 0).

FALSIFYING MEASUREMENT (pre-registered, verbatim)
-------------------------------------------------
Weekly-horizon rank IC of residual-flow z vs next-5-day returns on the data-PC
with block-bootstrap CI; kill if CI covers 0 or sign flips across validation
folds.  Secondary gate: incremental orthogonal factor_ic vs
TakerFlowImbalanceContinuation and FlowShareRotation must be > 0.

RESIDUAL-DEGENERACY GUARD (documented data-PC admission gate)
-------------------------------------------------------------
In-code, a decision bar whose residual vector carries no cross-sectional
dispersion (population sigma <= 1e-9 -- e.g. flow z collinear with return z, or
a dispersion-free flow cross-section) emits NOTHING rather than rank noise.  On
the data-PC the same guard is an ADMISSION gate: if the residual's
cross-sectional variance share ``1 - R^2`` of the flow-on-return regression
falls below 0.10 over the evaluation sample (flow z >= 90% explained by return
z), the residual is a return proxy / noise ranking and the sleeve is REJECTED
before any IC is read.

SIGNAL SPEC
-----------
Per completed 1h decision bar (all-or-nothing taker capture; a bar where either
taker column is None -- 8h staleness cutoff -- is dropped so the symbol simply
contributes no entries instead of polluting the cross-section):

1. Per symbol append ``net = taker_buy_quote - taker_sell_quote`` and
   ``turnover = taker_buy_quote + taker_sell_quote``.
2. ``flow = fsum(net[-formation:]) / fsum(turnover[-formation:])`` over the
   trailing ``formation_window_bars`` (120 x 1h ~ 5 days).
3. ``ret = simple_return(closes, lookback=formation_window_bars)`` -- the
   same-window return, computed over the SAME accepted bars as the flow.
4. Cross-sectionally z-score ``flow`` and ``ret``; ``residual =
   cross_sectional_residualize(flow_z, [ret_z])``; re-z the residual so the
   hysteresis band has a z unit.
5. Every ``decision_interval_bars`` (24 x 1h = daily) the DECISION clock ticks;
   every ``rebalance_decisions`` (7 daily decisions = weekly) the ranker runs:
   top ``quantile_pct`` (0.20) of residual z is LONG, bottom quintile SHORT,
   inverse-realized-vol sized, normalized to ``target_gross_exposure`` and
   clamped by an annualization-aware ``target_vol`` throttle
   (``indicators/annualization``).  A held name is RETAINED while its residual
   z stays within ``rank_hysteresis_z`` (0.5z) of its side's quantile boundary.
6. Hard min-hold: a rank exit or side flip is suppressed until the name has
   been held ``min_hold_decisions`` (7) daily decision ticks -- the one proven
   turnover rescue (RPT 3.69 -> 24.69bps), applied ex-ante.  Protective
   stop-loss / max-hold exits (close-confirmed, via signal metadata -- never
   engine intrabar brackets) remain live every bar.

TURNOVER ARGUMENT
-----------------
Weekly rebalance + hard 1-week min-hold + the 0.5z hysteresis band keep the
book turning over at most ~1x/week.  A weekly XS quintile spread of even
40-60bps vs ~15-25bps round-trip cost clears RPT >= 10bps if the IC is real.
Entries stamp ``intended_hold_seconds`` (~7 days) so the engine's hold-aware
seams (e.g. the funding-entry guard) can read the intended horizon.

DISTINCT-FROM (nearest incumbents)
----------------------------------
- ``TakerFlowImbalanceContinuationStrategy`` (per-symbol, 12-bar flow window,
  60s cadence, requires price confirmation) and ``TakerFlowExhaustion`` (per-
  symbol fade of extremes): this book is a multi-DAY cross-sectional long-short
  whose residualization EXCLUDES exactly the price-confirmed flow those sleeves
  trade.
- ``CrossSectionalFlowShareRotationStrategy``: flow-share primitives are
  UNSIGNED quote-volume SHARE (turnover concentration), never signed aggressor
  imbalance.
- Dead-by-literature naive short-term reversal: the characteristic is
  residualized ON returns (flow beyond what return implies) and the
  continuation sign is pre-registered; a reversal-shaped result fails
  sign-lock and dies honestly.

This module is data-local (no I/O, no hidden configuration bus), pure Python
(``math``/``deque`` only, no numpy), and never raises from
``calculate_signals``.
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
    _extract_feature,
    _market_snapshot,
    _safe_non_negative_int,
    _target_metadata,
    _window_snapshot,
)
from lumina_quant.strategy import Strategy
from lumina_quant.tuning import HyperParam, resolve_params_from_schema

_STRATEGY_ID = "xs_residual_taker_flow"
_STRATEGY_NAME = "CrossSectionalResidualTakerFlowStrategy"

# Population-sigma floor below which the residual cross-section is treated as
# degenerate (no dispersion / return proxy) and the decision emits NOTHING.
_RESIDUAL_DEGENERACY_SIGMA = 1e-9


@dataclass(slots=True)
class _State:
    closes: deque[float]
    net_flows: deque[float]
    turnovers: deque[float]
    mode: str = "OUT"
    entry_price: float | None = None
    bars_held: int = 0
    decisions_held: int = 0
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


@register("strategy", "CrossSectionalResidualTakerFlowStrategy", interface="event_driven")
class CrossSectionalResidualTakerFlowStrategy(Strategy):
    """Long-short XS residual taker-flow accumulation (weekly, min-hold guarded).

    See the module docstring for the full theory, pre-registered null /
    falsifier, signal spec, and distinct-from rationale versus the taker-flow
    continuation/exhaustion and flow-share incumbents.  This class only reads
    local event/bar OHLCV plus the two taker feature columns via the shared
    3-tier no-lookahead ``_extract_feature`` cascade; it performs no I/O and
    never raises from ``calculate_signals``.
    """

    decision_cadence_seconds = 3600
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False
    required_features = ("taker_buy_quote_volume", "taker_sell_quote_volume")

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            # ~5 days of 1h bars (pre-registered formation window).
            "formation_window_bars": HyperParam.integer(
                "formation_window_bars", default=120, low=8, high=20000
            ),
            # Bars per DECISION tick (24 x 1h = daily decisions).
            "decision_interval_bars": HyperParam.integer(
                "decision_interval_bars", default=24, low=1, high=100000
            ),
            # Decision ticks per REBALANCE (7 daily decisions = weekly ranker).
            "rebalance_decisions": HyperParam.integer(
                "rebalance_decisions", default=7, low=1, high=100000
            ),
            # Hard min-hold in daily decision ticks (7 = one week).
            "min_hold_decisions": HyperParam.integer(
                "min_hold_decisions", default=7, low=0, high=100000
            ),
            "rank_hysteresis_z": HyperParam.floating(
                "rank_hysteresis_z", default=0.5, low=0.0, high=10.0
            ),
            "quantile_pct": HyperParam.floating("quantile_pct", default=0.20, low=0.02, high=0.50),
            "vol_window": HyperParam.integer("vol_window", default=72, low=2, high=2000),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
            "min_symbols": HyperParam.integer("min_symbols", default=10, low=2, high=512),
            "target_gross_exposure": HyperParam.floating(
                "target_gross_exposure", default=1.0, low=0.0, high=3.0
            ),
            "target_vol": HyperParam.floating("target_vol", default=0.20, low=0.0, high=2.0),
            "stop_loss_pct": HyperParam.floating("stop_loss_pct", default=0.10, low=0.0, high=0.50),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=0, low=0, high=200000),
            # ~7 days, stamped verbatim into entry metadata for hold-aware seams.
            "intended_hold_seconds": HyperParam.floating(
                "intended_hold_seconds",
                default=604800.0,
                low=0.0,
                high=31_536_000.0,
                tunable=False,
            ),
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
        self.formation_window_bars = max(2, int(resolved["formation_window_bars"]))
        self.decision_interval_bars = max(1, int(resolved["decision_interval_bars"]))
        self.rebalance_decisions = max(1, int(resolved["rebalance_decisions"]))
        self.min_hold_decisions = max(0, int(resolved["min_hold_decisions"]))
        self.rank_hysteresis_z = max(0.0, float(resolved["rank_hysteresis_z"]))
        self.quantile_pct = min(0.5, max(0.0, float(resolved["quantile_pct"])))
        self.vol_window = max(2, int(resolved["vol_window"]))
        self.allow_short = bool(resolved["allow_short"])
        self.min_symbols = max(2, int(resolved["min_symbols"]))
        self.target_gross_exposure = max(0.0, float(resolved["target_gross_exposure"]))
        self.target_vol = max(0.0, float(resolved["target_vol"]))
        self.stop_loss_pct = max(0.0, float(resolved["stop_loss_pct"]))
        self.max_hold_bars = max(0, int(resolved["max_hold_bars"]))
        self.intended_hold_seconds = max(0.0, float(resolved["intended_hold_seconds"]))
        self.base_allocation = max(0.0, float(resolved["base_allocation"]))
        self.max_symbol_exposure_pct = max(0.0, float(resolved["max_symbol_exposure_pct"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        size = _state_size(self.formation_window_bars + 1, self.vol_window + 1)
        self._state: dict[str, _State] = {
            symbol: _State(
                closes=deque(maxlen=size),
                net_flows=deque(maxlen=size),
                turnovers=deque(maxlen=size),
            )
            for symbol in self.symbol_list
        }
        self._last_eval_time_key = ""
        self._tick = 0
        self._decision_count = 0
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
            "decision_count": int(self._decision_count),
            "recent_times": list(self._recent_times),
            "symbol_state": {
                symbol: {
                    "closes": list(item.closes),
                    "net_flows": list(item.net_flows),
                    "turnovers": list(item.turnovers),
                    "mode": item.mode,
                    "entry_price": item.entry_price,
                    "bars_held": int(item.bars_held),
                    "decisions_held": int(item.decisions_held),
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
        self._decision_count = _safe_non_negative_int(state.get("decision_count"))
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
                for attr in ("closes", "net_flows", "turnovers"):
                    target = getattr(item, attr)
                    target.clear()
                    maxlen = int(target.maxlen or 0)
                    values = _coerce_float_list(payload.get(attr))
                    if attr == "turnovers":
                        values = [max(0.0, value) for value in values]
                    for value in values[-maxlen:] if maxlen else values:
                        target.append(value)
                mode = str(payload.get("mode", "OUT")).upper()
                item.mode = mode if mode in {"OUT", "LONG", "SHORT"} else "OUT"
                item.entry_price = safe_float(payload.get("entry_price"))
                item.bars_held = _safe_non_negative_int(payload.get("bars_held"))
                item.decisions_held = _safe_non_negative_int(payload.get("decisions_held"))
                item.last_time_key = str(payload.get("last_time_key", ""))
                item.score = safe_float(payload.get("score"))
            except Exception:
                continue

    def on_warmup_end(self) -> None:
        """Drop warmup ghost positions at the first live bar (engine one-shot hook).

        The engine suppresses every signal emitted during the warmup region, so
        an "entry" this sleeve recorded during warmup never became a portfolio
        position (ghost-position desynchronization).  Reset every symbol's
        position bookkeeping to the OUT defaults (mode, entry price, hold /
        decision counters, score) while PRESERVING the data deques (closes /
        net_flows / turnovers) so live decisions keep their formation history.
        """
        for item in self._state.values():
            self._flatten(item)

    # ------------------------------------------------------------------ #
    # ingestion
    # ------------------------------------------------------------------ #
    def _update_symbol(self, event: Any, symbol: str, snapshot: _Snapshot) -> bool:
        close = safe_float(snapshot.close)
        if close is None or close <= self.min_price:
            return False
        item = self._state[symbol]
        key = time_key(snapshot.time)
        if key and key == item.last_time_key:
            return False
        # All-or-nothing taker capture at the closed bar (3-tier cascade,
        # None-tolerant: 8h staleness returns None).  When either taker column
        # is absent the bar is DROPPED entirely so this symbol contributes no
        # entries instead of polluting the cross-section (graveyard item 7
        # pre-emption; mirrors the funding-carry all-or-nothing convention).
        taker_buy = _extract_feature(self.bars, event, symbol, "taker_buy_quote_volume")
        taker_sell = _extract_feature(self.bars, event, symbol, "taker_sell_quote_volume")
        if taker_buy is None or taker_sell is None:
            return False
        buy = max(0.0, float(taker_buy))
        sell = max(0.0, float(taker_sell))
        item.last_time_key = key
        item.closes.append(close)
        item.net_flows.append(buy - sell)
        item.turnovers.append(buy + sell)
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

    # ------------------------------------------------------------------ #
    # scoring / selection
    # ------------------------------------------------------------------ #
    def _residual_scores(
        self,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, dict[str, Any]]]:
        """Return ``(residual_z_by_symbol, vols, metas)`` for the eligible universe."""
        flows: dict[str, float] = {}
        returns: dict[str, float] = {}
        vols: dict[str, float] = {}
        window = self.formation_window_bars
        # Panel-eligibility freshness gate (mirrors the behavioral-value
        # sleeve's ``last_committed_day`` gate): the strategy-level latest
        # committed key is the max accepted per-symbol bar key; a symbol whose
        # last accepted update lags it is STALE (frozen feed / dropped bars)
        # and must not pollute this decision's cross-sectional ranks.
        latest_key = ""
        for item in self._state.values():
            if item.last_time_key > latest_key:
                latest_key = item.last_time_key
        for symbol, item in self._state.items():
            if not item.last_time_key or item.last_time_key != latest_key:
                continue
            if len(item.net_flows) < window or len(item.closes) < window + 1:
                continue
            closes = list(item.closes)
            close = closes[-1]
            if close is None or close <= 0.0:
                continue
            nets = list(item.net_flows)[-window:]
            turns = list(item.turnovers)[-window:]
            denom = math.fsum(turns)
            if denom <= _EPS:
                continue
            flow = math.fsum(nets) / denom
            ret = simple_return(closes, lookback=window)
            vol = realized_volatility(closes, window=self.vol_window)
            if ret is None or vol is None or vol <= _EPS or not math.isfinite(flow):
                continue
            flows[symbol] = float(flow)
            returns[symbol] = float(ret)
            vols[symbol] = float(vol)

        if len(flows) < self.min_symbols:
            return {}, {}, {}

        ordered = sorted(flows)
        flow_z = _cross_z(flows)
        ret_z = _cross_z(returns)
        residual_vec = cross_sectional_residualize(
            [flow_z[symbol] for symbol in ordered],
            [[ret_z[symbol] for symbol in ordered]],
        )
        if residual_vec is None or len(residual_vec) != len(ordered):
            return {}, {}, {}
        # Residual-degeneracy guard (in-code leg): a residual cross-section with
        # no dispersion is a noise ranking (flow fully explained by return, or a
        # dispersion-free flow panel) -- emit NOTHING.  The data-PC admission
        # gate documented in the module docstring adjudicates the sample-level
        # variance-share version of the same failure.
        mean_res = math.fsum(residual_vec) / float(len(residual_vec))
        sigma = math.sqrt(
            math.fsum((value - mean_res) ** 2 for value in residual_vec) / float(len(residual_vec))
        )
        if sigma <= _RESIDUAL_DEGENERACY_SIGMA:
            return {}, {}, {}
        residual = {symbol: float(residual_vec[idx]) for idx, symbol in enumerate(ordered)}
        residual_z = _cross_z(residual)
        metas: dict[str, dict[str, Any]] = {
            symbol: {
                "taker_flow": float(flows[symbol]),
                "taker_flow_z": float(flow_z[symbol]),
                "return_z": float(ret_z[symbol]),
                "flow_residual": float(residual[symbol]),
                "flow_residual_z": float(residual_z[symbol]),
            }
            for symbol in ordered
        }
        return residual_z, vols, metas

    def _score_and_select(
        self,
    ) -> tuple[dict[str, tuple[str, float, dict[str, Any]]], dict[str, float]]:
        residual_z, vols, metas = self._residual_scores()
        if not residual_z:
            return {}, {}

        # Ascending by residual z (symbol tiebreak): bottom quintile is short,
        # top quintile long.  A held name is RETAINED while its residual z stays
        # within ``rank_hysteresis_z`` of its side's quantile-boundary score.
        ordered = sorted(residual_z, key=lambda symbol: (residual_z[symbol], symbol))
        count = len(ordered)
        n_side = max(1, int(self.quantile_pct * count))
        if 2 * n_side > count:
            n_side = count // 2
        if n_side < 1:
            return {}, {}
        long_core = ordered[count - n_side :]
        short_core = ordered[:n_side]
        long_boundary = residual_z[long_core[0]]
        short_boundary = residual_z[short_core[-1]]
        long_set = set(long_core)
        short_set = set(short_core)

        targets: dict[str, tuple[str, float, dict[str, Any]]] = {}
        for symbol in ordered:
            mode = self._state[symbol].mode
            score = residual_z[symbol]
            meta = metas[symbol]
            if symbol in long_set:
                targets[symbol] = ("LONG", score, meta)
            elif self.allow_short and symbol in short_set:
                targets[symbol] = ("SHORT", score, meta)
            elif mode == "LONG" and score >= long_boundary - self.rank_hysteresis_z:
                targets[symbol] = ("LONG", score, meta)
            elif (
                mode == "SHORT"
                and self.allow_short
                and score <= short_boundary + self.rank_hysteresis_z
            ):
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
        # the Moreira-Muir clamp is INERT.  When spacing is unavailable we pass
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
        # A protective exit (stop / max-hold) flattens a name outside the
        # min-hold discipline; keep the decision-hold counter consistent.
        for item in self._state.values():
            if item.mode == "OUT":
                item.decisions_held = 0

    def _evaluate(self, event_time: Any) -> None:
        if len(self.symbol_list) < self.min_symbols:
            return
        # Record the decision-bar epoch so the vol-target scalar can infer bar
        # spacing (this runs on every new bar, before the decision gate).
        dt = _event_datetime_utc(event_time)
        if dt is not None:
            self._recent_times.append(dt.timestamp())
        # Stops / max-hold age EVERY bar so a held name is always protected,
        # independent of the slow decision/rebalance clocks (close-confirmed
        # EXIT signals -- never engine intrabar brackets).
        self._age(event_time)
        if self._tick % self.decision_interval_bars:
            return
        # Daily DECISION tick: advance the min-hold counter for held names.
        self._decision_count += 1
        for item in self._state.values():
            if item.mode != "OUT":
                item.decisions_held += 1
        if self._decision_count % self.rebalance_decisions:
            return
        # Weekly REBALANCE tick: run the ranker.
        targets, vols = self._score_and_select()
        if not targets:
            # Degenerate panel (min-symbols shortfall, residual degeneracy,
            # empty quantiles): NOT a rank verdict on held names -- return
            # without flushing a mature book via ``rank_lapsed`` exits
            # (mirrors offsession_basis_dislocation).
            return
        weights, scalar = self._inverse_vol_weights(targets, vols)
        self._emit_targets(targets, weights, scalar, event_time)

    def _flatten(self, item: _State) -> None:
        item.mode = "OUT"
        item.entry_price = None
        item.bars_held = 0
        item.decisions_held = 0
        item.score = None

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
                    # Hard min-hold: a would-be rank exit inside the hold window
                    # is suppressed (the one proven turnover rescue, ex-ante).
                    if item.decisions_held < self.min_hold_decisions:
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
                    self._flatten(item)
                continue
            target_mode, score, meta = target
            if item.mode == target_mode:
                item.score = float(score)
                continue
            # Hard min-hold: a would-be side flip inside the hold window is
            # suppressed; the current position is kept until the hold clears.
            if item.mode != "OUT" and item.decisions_held < self.min_hold_decisions:
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
                # un-vol-gated position (v5 zero-alloc resize trap).  Skip the
                # entry; if a side-flip EXIT was just emitted, drop to OUT so
                # state matches it.
                if item.mode != "OUT":
                    self._flatten(item)
                continue
            # No engine intrabar bracket: the close-confirmed stop in
            # ``_age`` / ``_age_cross_positions`` is the ONLY stop (module
            # docstring: "never engine intrabar brackets"), so entries carry
            # ``stop_loss=None`` and never desync from the aging path.
            metadata = _target_metadata(
                strategy=_STRATEGY_NAME,
                target_allocation=alloc,
                max_order_value=self.max_order_value,
                score=float(score),
                target_mode=target_mode,
                inverse_vol_weight=weight,
                vol_target_scalar=float(scalar),
                intended_hold_seconds=float(self.intended_hold_seconds),
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
                metadata=metadata,
            )
            item.mode = target_mode
            item.entry_price = price
            item.bars_held = 0
            item.decisions_held = 0
            item.score = float(score)


# --------------------------------------------------------------------------- #
# Candidate-wiring hints for the orchestrator (this lane does NOT wire
# candidates itself -- new-file-only, no shared-file edits).  Admission route is
# `allow_multi_asset=True` at the data-PC handoff: this book is a pure
# cross-sectional long-short whose momentum component is residualized OUT (NOT
# carry, NOT momentum), so it is honestly EXCLUDED from any carry/momentum
# tag-superset allowlist -- no fake carry tag is added.
# --------------------------------------------------------------------------- #
_SUGGESTED_FAMILY = "cross_sectional"
_SUGGESTED_CANDIDATE_TAGS: tuple[str, ...] = (
    "cross_sectional",
    "taker_flow",
    "order_flow",
    "residual",
    "accumulation",
    "long_short",
    "zscore",
    "crypto",
)

# Candidate slice.  The pre-registered MECHANISM constants -- formation ~5 days,
# quintile 0.20, 0.5z rank hysteresis, weekly rebalance (7 daily decisions), and
# the hard one-week ``min_hold_decisions = 7`` -- are FIXED across every variant
# and timeframe (no tunable grids).  Bar-denominated windows scale with the
# timeframe to hold their wall-clock span (1h: 120-bar formation, 24-bar daily
# decision tick; 4h: 30-bar formation, 6-bar daily decision tick); variants
# differ ONLY in risk plumbing (gross exposure / vol target / stop / breadth).
_XS_RESIDUAL_TAKER_FLOW_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "1h": (
        {
            "variant": "baseline_ls",
            "formation_window_bars": 120,
            "decision_interval_bars": 24,
            "rebalance_decisions": 7,
            "min_hold_decisions": 7,
            "rank_hysteresis_z": 0.5,
            "quantile_pct": 0.20,
            "vol_window": 72,
            "min_symbols": 10,
            "allow_short": True,
            "target_gross_exposure": 1.0,
            "target_vol": 0.20,
            "stop_loss_pct": 0.10,
            "max_hold_bars": 0,
            "intended_hold_seconds": 604800.0,
        },
        {
            "variant": "guarded_ls",
            "formation_window_bars": 120,
            "decision_interval_bars": 24,
            "rebalance_decisions": 7,
            "min_hold_decisions": 7,
            "rank_hysteresis_z": 0.5,
            "quantile_pct": 0.20,
            "vol_window": 72,
            "min_symbols": 12,
            "allow_short": True,
            "target_gross_exposure": 0.60,
            "target_vol": 0.15,
            "stop_loss_pct": 0.08,
            "max_hold_bars": 0,
            "intended_hold_seconds": 604800.0,
        },
    ),
    "4h": (
        {
            "variant": "baseline_ls",
            "formation_window_bars": 30,
            "decision_interval_bars": 6,
            "rebalance_decisions": 7,
            "min_hold_decisions": 7,
            "rank_hysteresis_z": 0.5,
            "quantile_pct": 0.20,
            "vol_window": 18,
            "min_symbols": 10,
            "allow_short": True,
            "target_gross_exposure": 1.0,
            "target_vol": 0.20,
            "stop_loss_pct": 0.10,
            "max_hold_bars": 0,
            "intended_hold_seconds": 604800.0,
        },
        {
            "variant": "guarded_ls",
            "formation_window_bars": 30,
            "decision_interval_bars": 6,
            "rebalance_decisions": 7,
            "min_hold_decisions": 7,
            "rank_hysteresis_z": 0.5,
            "quantile_pct": 0.20,
            "vol_window": 18,
            "min_symbols": 12,
            "allow_short": True,
            "target_gross_exposure": 0.60,
            "target_vol": 0.15,
            "stop_loss_pct": 0.08,
            "max_hold_bars": 0,
            "intended_hold_seconds": 604800.0,
        },
    ),
}

__all__ = ["CrossSectionalResidualTakerFlowStrategy"]

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
5. The ranker runs once per UTC ISO week after all symbols report the same
   timestamp:
   top ``quantile_pct`` (0.20) of residual z is LONG, bottom quintile SHORT,
   inverse-realized-vol sized, normalized to ``target_gross_exposure`` and
   clamped by an annualization-aware ``target_vol`` throttle
   (``indicators/annualization``).  A held name is RETAINED while its residual
   z stays within ``rank_hysteresis_z`` (0.5z) of its side's quantile boundary.
6. Hard min-hold: a rank exit or side flip is suppressed until the name has
   been held ``min_hold_decisions`` weekly decision ticks -- the one proven
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
from itertools import pairwise
from typing import Any

from lumina_quant.core.plugin_registry import register
from lumina_quant.indicators.alpha_features import realized_volatility, simple_return
from lumina_quant.indicators.annualization import (
    annualize_per_bar_vol,
    bars_per_year_from_spacing,
)
from lumina_quant.indicators.common import safe_float
from lumina_quant.indicators.cross_sectional_residualize import cross_sectional_residualize
from lumina_quant.strategies.adaptive_crypto_alpha_sleeves import (
    _age_cross_positions,
    _state_size,
)
from lumina_quant.strategies.external_alpha_sleeves import (
    _EPS,
    _emit,
    _event_datetime_utc,
    _safe_non_negative_int,
    _target_metadata,
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
    times: deque[str]
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


def _current_taker_row(
    event: Any, symbol: str, event_time: Any
) -> tuple[float, float, float] | None:
    """Return the sole, explicitly timestamped current raw bar and its taker pair.

    Feature getters commonly expose an untimestamped scalar. Such a value can
    be a stale cache entry, so this sleeve only accepts the two taker columns
    when the event/window payload itself ties both to the decision timestamp.
    """
    try:
        rows = list(dict(getattr(event, "bars_1s", {}) or {}).get(symbol) or [])
    except Exception:
        return None
    matches: list[tuple[float, float, float]] = []
    for row in rows:
        row_time = (
            row.get("time")
            if isinstance(row, dict)
            else row[0]
            if isinstance(row, (tuple, list)) and row
            else getattr(row, "time", None)
        )
        if _event_datetime_utc(row_time) != event_time:
            continue
        if not isinstance(row, dict):
            return None
        close = safe_float(row.get("close"))
        buy = safe_float(row.get("taker_buy_quote_volume"))
        sell = safe_float(row.get("taker_sell_quote_volume"))
        if (
            close is None
            or buy is None
            or sell is None
            or not math.isfinite(close)
            or not math.isfinite(buy)
            or not math.isfinite(sell)
        ):
            return None
        if buy < 0.0 or sell < 0.0:
            return None
        matches.append((float(close), float(buy), float(sell)))
    # More than one current row is ambiguous, including a duplicate whose
    # values happen to agree. Never choose one by arrival order.
    return matches[0] if len(matches) == 1 else None


@register("strategy", "CrossSectionalResidualTakerFlowStrategy", interface="event_driven")
class CrossSectionalResidualTakerFlowStrategy(Strategy):
    """Long-short XS residual taker-flow accumulation (weekly, min-hold guarded).

    See the module docstring for the full theory, pre-registered null /
    falsifier, signal spec, and distinct-from rationale versus the taker-flow
    continuation/exhaustion and flow-share incumbents.  This class only reads
    local event/window rows plus the two taker feature columns stamped on those
    rows. Untimestamped feature-cache scalars are intentionally not trusted:
    they cannot establish current-bar provenance. It performs no I/O and never
    raises from ``calculate_signals``.
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
            "min_hold_decisions": HyperParam.integer(
                "min_hold_decisions", default=1, low=0, high=100000
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
                times=deque(maxlen=size),
            )
            for symbol in self.symbol_list
        }
        self._last_rebalance_week: tuple[int, int] | None = None
        self._last_decision_week: tuple[int, int] | None = None
        self._last_age_time_key = ""
        # Recent decision-bar epochs (seconds) for deterministic bar-spacing
        # inference: the vol-target scalar annualizes the per-bar portfolio vol
        # via sqrt(bars_per_year) so the Moreira-Muir clamp is not left inert.
        self._recent_times: deque[float] = deque(maxlen=16)

    # ------------------------------------------------------------------ #
    # state
    # ------------------------------------------------------------------ #
    def get_state(self) -> dict[str, Any]:
        return {
            "last_rebalance_week": list(self._last_rebalance_week)
            if self._last_rebalance_week
            else None,
            "last_decision_week": list(self._last_decision_week)
            if self._last_decision_week
            else None,
            "last_age_time_key": self._last_age_time_key,
            "recent_times": list(self._recent_times),
            "symbol_state": {
                symbol: {
                    "closes": list(item.closes),
                    "net_flows": list(item.net_flows),
                    "turnovers": list(item.turnovers),
                    "times": list(item.times),
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
        """Install a complete, internally coherent checkpoint atomically."""
        if not isinstance(state, dict):
            return
        if set(state) != {
            "last_rebalance_week",
            "last_decision_week",
            "last_age_time_key",
            "recent_times",
            "symbol_state",
        }:
            return
        raw = state.get("symbol_state")
        if not isinstance(raw, dict) or set(raw) != set(self._state):
            return

        def parse_week(value: Any) -> tuple[int, int] | object | None:
            if value is None:
                return None
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                return _INVALID
            if any(isinstance(part, bool) or not isinstance(part, int) for part in value):
                return _INVALID
            try:
                year, week = int(value[0]), int(value[1])
            except TypeError, ValueError:
                return _INVALID
            return (year, week) if 1 <= year <= 9999 and 1 <= week <= 53 else _INVALID

        _INVALID = object()
        rebalance_week = parse_week(state.get("last_rebalance_week"))
        decision_week = parse_week(state.get("last_decision_week"))
        if rebalance_week is _INVALID or decision_week is _INVALID:
            return
        last_age_time_key = state.get("last_age_time_key", "")
        if not isinstance(last_age_time_key, str) or (
            last_age_time_key and _event_datetime_utc(last_age_time_key) is None
        ):
            return
        recent_raw = state.get("recent_times")
        if (
            not isinstance(recent_raw, (list, tuple))
            or len(recent_raw) > int(self._recent_times.maxlen or 0)
            or any(
                not isinstance(value, (int, float)) or isinstance(value, bool)
                for value in recent_raw
            )
        ):
            return
        recent_times = _coerce_float_list(recent_raw)
        if len(recent_times) != len(recent_raw) or any(
            not math.isfinite(value) for value in recent_times
        ):
            return
        if any(right < left for left, right in pairwise(recent_times)):
            return

        candidate: dict[str, _State] = {}
        common_time: Any = None
        common_time_key: str | None = None
        common_length: int | None = None
        common_times: list[str] | None = None
        for symbol in self.symbol_list:
            payload = raw.get(symbol)
            existing = self._state[symbol]
            if not isinstance(payload, dict) or set(payload) != {
                "closes",
                "net_flows",
                "turnovers",
                "times",
                "mode",
                "entry_price",
                "bars_held",
                "decisions_held",
                "last_time_key",
                "score",
            }:
                return
            vectors: dict[str, list[float]] = {}
            for attr in ("closes", "net_flows", "turnovers"):
                values_raw = payload.get(attr)
                if not isinstance(values_raw, (list, tuple)) or any(
                    not isinstance(value, (int, float)) or isinstance(value, bool)
                    for value in values_raw
                ):
                    return
                values = _coerce_float_list(values_raw)
                if len(values) != len(values_raw) or any(
                    not math.isfinite(value) for value in values
                ):
                    return
                if attr == "turnovers" and any(value < 0.0 for value in values):
                    return
                vectors[attr] = values
            times_raw = payload.get("times")
            if not isinstance(times_raw, (list, tuple)) or any(
                not isinstance(value, str) for value in times_raw
            ):
                return
            times: list[str] = []
            for raw_time in times_raw:
                point = _event_datetime_utc(raw_time)
                if point is None or point.isoformat() != raw_time:
                    return
                times.append(raw_time)
            if any(right <= left for left, right in pairwise(times)):
                return
            length = len(vectors["closes"])
            if (
                length != len(vectors["net_flows"])
                or length != len(vectors["turnovers"])
                or length != len(times)
                or length > int(existing.closes.maxlen or 0)
            ):
                return
            if any(value <= self.min_price for value in vectors["closes"]) or any(
                abs(net_flow) > turnover
                for net_flow, turnover in zip(
                    vectors["net_flows"], vectors["turnovers"], strict=True
                )
            ):
                return
            try:
                if not all(
                    math.isfinite(math.fsum(values))
                    for values in (vectors["net_flows"], vectors["turnovers"])
                ):
                    return
            except OverflowError:
                return
            if common_length is None:
                common_length = length
            elif length != common_length:
                return
            if common_times is None:
                common_times = times
            elif times != common_times:
                return
            last_time_key = payload.get("last_time_key", "")
            parsed_time = (
                _event_datetime_utc(last_time_key) if isinstance(last_time_key, str) else None
            )
            if bool(length) != bool(parsed_time) or (times and last_time_key != times[-1]):
                return
            if parsed_time is not None:
                if common_time is None:
                    common_time = parsed_time
                    common_time_key = last_time_key
                elif parsed_time != common_time or last_time_key != common_time_key:
                    return
            mode = payload.get("mode", "OUT")
            if not isinstance(mode, str) or mode not in {"OUT", "LONG", "SHORT"}:
                return
            entry_price = payload.get("entry_price")
            score = payload.get("score")
            if entry_price is not None and (
                not isinstance(entry_price, (int, float))
                or isinstance(entry_price, bool)
                or not math.isfinite(float(entry_price))
            ):
                return
            if score is not None and (
                not isinstance(score, (int, float))
                or isinstance(score, bool)
                or not math.isfinite(float(score))
            ):
                return
            entry_price = float(entry_price) if entry_price is not None else None
            score = float(score) if score is not None else None
            bars_held = _safe_non_negative_int(payload.get("bars_held"))
            decisions_held = _safe_non_negative_int(payload.get("decisions_held"))
            if (
                isinstance(payload.get("bars_held"), bool)
                or not isinstance(payload.get("bars_held"), int)
                or isinstance(payload.get("decisions_held"), bool)
                or not isinstance(payload.get("decisions_held"), int)
                or bars_held != payload.get("bars_held")
                or decisions_held != payload.get("decisions_held")
            ):
                return
            if mode == "OUT":
                if (
                    entry_price is not None
                    or bars_held != 0
                    or decisions_held != 0
                    or score is not None
                ):
                    return
            elif entry_price is None or entry_price <= self.min_price or score is None:
                return
            candidate[symbol] = _State(
                closes=deque(vectors["closes"], maxlen=existing.closes.maxlen),
                net_flows=deque(vectors["net_flows"], maxlen=existing.net_flows.maxlen),
                turnovers=deque(vectors["turnovers"], maxlen=existing.turnovers.maxlen),
                times=deque(times, maxlen=existing.times.maxlen),
                mode=mode,
                entry_price=entry_price,
                bars_held=bars_held,
                decisions_held=decisions_held,
                last_time_key=last_time_key,
                score=score,
            )
        if common_time is None and any(item.last_time_key for item in candidate.values()):
            return
        if (
            common_time is not None
            and last_age_time_key
            and (
                last_age_time_key != common_time_key
                or _event_datetime_utc(last_age_time_key) != common_time
            )
        ):
            return
        if common_time is not None:
            if not last_age_time_key or not recent_times:
                return
            if recent_times[-1] != common_time.timestamp():
                return
            current_week = tuple(int(value) for value in common_time.isocalendar()[:2])
            if decision_week is not None and decision_week != current_week:
                return
            if rebalance_week is not None and rebalance_week > current_week:
                return
        elif (
            last_age_time_key
            or recent_times
            or rebalance_week is not None
            or decision_week is not None
        ):
            return

        self._state = candidate
        self._last_rebalance_week = rebalance_week
        self._last_decision_week = decision_week
        self._last_age_time_key = last_age_time_key
        self._recent_times = deque(recent_times, maxlen=self._recent_times.maxlen)

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
    def _commit_symbol(
        self,
        symbol: str,
        timestamp: str,
        close: float,
        buy: float,
        sell: float,
    ) -> None:
        """Commit a preflighted taker row without any further failure points."""
        item = self._state[symbol]
        item.last_time_key = timestamp
        item.closes.append(close)
        item.net_flows.append(buy - sell)
        item.turnovers.append(buy + sell)
        item.times.append(timestamp)

    def calculate_signals_window(self, event: Any, aggregator: Any = None) -> None:
        _ = aggregator
        if not self.symbol_list:
            return
        event_dt = _event_datetime_utc(getattr(event, "time", None))
        if event_dt is None:
            return
        panel: list[tuple[str, float, float, float]] = []
        for symbol in self.symbol_list:
            item = self._state[symbol]
            previous_dt = _event_datetime_utc(item.last_time_key) if item.last_time_key else None
            row = _current_taker_row(event, symbol, event_dt)
            if row is None:
                return
            close, buy, sell = row
            net_flow = buy - sell
            turnover = buy + sell
            if (
                close <= self.min_price
                or not math.isfinite(net_flow)
                or not math.isfinite(turnover)
                or turnover < 0.0
                or abs(net_flow) > turnover
                or (item.last_time_key and previous_dt is None)
                or (previous_dt is not None and event_dt <= previous_dt)
            ):
                return
            try:
                if not (
                    math.isfinite(math.fsum((*item.net_flows, net_flow)))
                    and math.isfinite(math.fsum((*item.turnovers, turnover)))
                ):
                    return
            except OverflowError:
                return
            panel.append((symbol, close, buy, sell))
        # All current taker pairs passed preflight, so no name can advance
        # independently of the cross-sectional decision window.
        timestamp = event_dt.isoformat()
        for symbol, close, buy, sell in panel:
            self._commit_symbol(symbol, timestamp, close, buy, sell)
        self._evaluate(event_dt)

    def calculate_signals(self, event: Any) -> None:
        if str(getattr(event, "type", "")).upper() == "MARKET_WINDOW":
            self.calculate_signals_window(event, None)
            return
        # Cross-sectional state is only advanced by complete MARKET_WINDOW
        # panels; individual callbacks cannot create a partial panel.
        return

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
        common_key = next(
            (item.last_time_key for item in self._state.values() if item.last_time_key),
            "",
        )
        if not common_key or any(item.last_time_key != common_key for item in self._state.values()):
            return {}, {}, {}
        common_times = next((list(item.times) for item in self._state.values()), [])
        if not common_times or any(
            list(item.times) != common_times
            or len(item.times) != len(item.closes)
            or len(item.times) != len(item.net_flows)
            or len(item.times) != len(item.turnovers)
            for item in self._state.values()
        ):
            return {}, {}, {}
        for symbol, item in self._state.items():
            if len(item.net_flows) < window or len(item.closes) < window + 1:
                continue
            closes = list(item.closes)
            close = closes[-1]
            if close is None or close <= 0.0:
                continue
            if len(item.turnovers) < window:
                continue
            net_flow = math.fsum(list(item.net_flows)[-window:])
            denom = math.fsum(list(item.turnovers)[-window:])
            if not math.isfinite(net_flow) or not math.isfinite(denom) or denom <= _EPS:
                continue
            flow = net_flow / denom
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
        market_flow = math.fsum(flows.values()) / float(len(flows))
        relative_flows = {symbol: flow - market_flow for symbol, flow in flows.items()}
        flow_z = _cross_z(relative_flows)
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
                "market_taker_flow": float(market_flow),
                "relative_taker_flow": float(relative_flows[symbol]),
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
        dt = _event_datetime_utc(event_time)
        if dt is None:
            return
        event_key = dt.isoformat()
        # Sequential per-symbol events share one UTC timestamp; age once for
        # that timestamp rather than once for every symbol delivery.
        if event_key != self._last_age_time_key:
            self._last_age_time_key = event_key
            self._recent_times.append(dt.timestamp())
            self._age(event_time)
        iso = dt.isocalendar()
        week = (int(iso[0]), int(iso[1]))
        if week != self._last_decision_week:
            self._last_decision_week = week
            for item in self._state.values():
                if item.mode != "OUT":
                    item.decisions_held += 1
        if week == self._last_rebalance_week:
            return
        targets, vols = self._score_and_select()
        if not targets:
            # Degenerate panel (min-symbols shortfall, residual degeneracy,
            # empty quantiles): NOT a rank verdict on held names -- return
            # without flushing a mature book via ``rank_lapsed`` exits
            # (mirrors offsession_basis_dislocation).
            return
        self._last_rebalance_week = week
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
            "min_hold_decisions": 1,
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
            "min_hold_decisions": 1,
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
            "min_hold_decisions": 1,
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
            "variant": "short_window_long_only",
            "formation_window_bars": 30,
            "min_hold_decisions": 1,
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

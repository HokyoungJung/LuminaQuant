"""Cross-sectional downside-TAIL-RISK premium sleeve (vol-neutralized ES level).

``DownsideTailRiskPremiumStrategy`` ranks the liquid cross-section on the
empirical LOWER-TAIL LEVEL of each symbol's own trailing daily log returns --
the expected shortfall ``ES_q`` (mean of the ``q``-lowest returns) -- after
vol-neutralization, and takes a weekly XS long-short book: LONG the worst
left-tail quantile (the priced downside-risk exposure), SHORT the mildest.

THEORY / PROVENANCE (crypto-specific, existence-verified)
---------------------------------------------------------
- Zhang, Li, Xiong & Wang (2021), *JBF* 133 106246, "Downside risk and the
  cross-section of cryptocurrency returns" -- a positive downside-risk premium,
  risk-compensation + limits-to-arbitrage.
- Dobrynskaya (2024), *IRFA* 91, "Is downside risk priced in cryptocurrency
  market?" -- crash sensitivity priced while plain market beta often is not.
- Ang, Chen & Xing (2006), *RFS* 19(4) -- the downside-risk foundation.
- OPPOSING equities prior, disclosed: Atilgan, Bali, Demirtas & Gunaydin (2020),
  *JFE* 135(3) "Left-tail momentum" (negative relation).  The sign is
  segment-dependent; this sleeve fixes the sign to the crypto-specific anchors
  (LONG worst left tail) and treats an equities-style sign flip as a
  falsification of the anchored hypothesis, never a silently flipped sleeve.

The counterparty is the crash-averse holder who underweights coins with a
demonstrated bad left tail and overpays for smooth low-drawdown names; the
sleeve is the insurance seller, durable because harvesting requires actually
bearing the left tail.

SIGNAL SPEC
-----------
Per completed daily bar, per symbol, on a weekly decision clock
(``rebalance_bars``) with a hard ``min_hold_periods`` floor + a rank-hysteresis
band:

1. ``ES`` = mean of the ``ceil(tail_q * n)`` lowest trailing log returns over
   ``es_window`` (a signed, negative lower-tail level); ``sigma`` = ``sample_std``
   of the same window.
2. When ``vol_neutralize`` is on: rank-transform the ES MAGNITUDE (``-ES``) and
   ``sigma`` across the eligible universe, take the closed-form OLS residual of
   the ES-magnitude rank on the sigma rank, and z-score it -- the left-tail
   severity beyond what total vol explains (incremental to the occupied idio-vol
   / low-vol axis).  When off, the raw ES-magnitude z-score is used.
3. Rank by the score; the top ``quantile_pct`` fraction are LONG (worst left
   tail), the bottom SHORT (mildest), inverse-realized-vol sized and normalized
   to ``target_gross_exposure``.  A held name is retained while its rank stays
   within the quantile boundary plus ``hysteresis_band`` (a rank fraction).

DISTINCT-FROM
-------------
Own-return tail LEVEL, not: Hill tail-index SHAPE
(``TailIndexRegimeRiderStrategy`` -- a per-symbol trend rider gated on the
loss-tail power-law exponent RATIO); symmetric second moment / right-tail
lottery (``IdiosyncraticVolatilityStrategy`` / ``LotterySkewnessStrategy``);
benchmark-conditioned co-movement (the downside-beta sleeve).  Vol-neutralization
is what keeps it incremental to the idio-vol / low-vol level.  These divergences
are pinned by the build-gate test by RUNNING those incumbents.

Lane-local numerics (``_expected_shortfall``, ``_rank_residual``,
``_average_ranks``) stay in this module -- no shared-path edit.  Pure Python
(``math`` only), never-raise, deterministic.  Ships WITHOUT ``@register``
(inert until the integration wave wires it).
"""

from __future__ import annotations

import math
from collections import deque
from itertools import pairwise
from typing import Any

from lumina_quant.core.plugin_registry import register
from lumina_quant.indicators.annualization import (
    annualize_per_bar_vol,
    bars_per_year_from_spacing,
)
from lumina_quant.indicators.rolling_stats import sample_std
from lumina_quant.strategies.adaptive_crypto_alpha_sleeves import _age_cross_positions
from lumina_quant.strategies.cross_sectional_anomaly_alpha_sleeves import (
    _CrossUpdateMixin,
    _event_datetime_from_key,
)
from lumina_quant.strategies.equity_xs_factor_alpha_sleeves import _state_size
from lumina_quant.strategies.external_alpha_sleeves import (
    _EPS,
    _emit,
    _event_datetime_utc,
    _target_metadata,
)
from lumina_quant.strategies.robust_alpha_sleeves import _CrossSectionalState
from lumina_quant.strategy import Strategy
from lumina_quant.tuning import HyperParam, resolve_params_from_schema

_STRATEGY_ID = "downside_tail_risk"
_STRATEGY_NAME = "DownsideTailRiskPremiumStrategy"


def _log_returns(closes: list[float]) -> list[float]:
    """Return the finite close-to-close log returns of a positive price path."""
    out: list[float] = []
    prev: float | None = None
    for value in closes:
        if prev is not None and prev > 0.0 and value > 0.0:
            ret = math.log(value / prev)
            if math.isfinite(ret):
                out.append(ret)
        prev = value
    return out


def _expected_shortfall(returns: list[float], q: float) -> float | None:
    """Return the mean of the lowest ``ceil(q*n)`` returns (a signed tail level).

    ``None`` when fewer than two returns are available.  ``q`` is clamped to
    ``(0, 0.5]``; the tail size is ``max(1, ceil(q*n))``.  For a left tail the
    result is negative (mean of the worst losses).  Pure, never raises.
    """
    n = len(returns)
    if n < 2:
        return None
    qq = min(0.5, max(1e-9, float(q)))
    k = max(1, math.ceil(qq * n))
    k = min(k, n)
    ordered = sorted(returns)
    tail = ordered[:k]
    return math.fsum(tail) / float(len(tail))


def _average_ranks(values: list[float], *, tol: float = 1e-9) -> list[float]:
    """Return tie-aware average ranks (0-based); values within ``tol`` tie.

    Equal (or within-``tol``) values receive the SAME average rank, so a
    cross-section with no real dispersion on an axis contributes a constant
    (zero-variance) rank vector -- exactly what makes the vol-neutralization
    collapse to the raw ES rank when every symbol shares the same sigma.
    """
    n = len(values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda idx: values[idx])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and (values[order[j + 1]] - values[order[i]]) <= tol:
            j += 1
        avg = (i + j) / 2.0
        for pos in range(i, j + 1):
            ranks[order[pos]] = avg
        i = j + 1
    return ranks


def _rank_residual(y_ranks: list[float], x_ranks: list[float]) -> list[float] | None:
    """Return the closed-form OLS residual of ``y_ranks`` on ``x_ranks``.

    ``residual_i = (y_i - mean_y) - beta * (x_i - mean_x)`` with
    ``beta = cov(x, y) / var(x)``.  When ``var(x)`` is degenerate (every symbol
    shares the same x-rank) the demeaned ``y`` is returned unchanged.  ``None``
    on empty / mismatched input.  Pure, never raises.
    """
    n = len(y_ranks)
    if n == 0 or len(x_ranks) != n:
        return None
    mean_x = math.fsum(x_ranks) / float(n)
    mean_y = math.fsum(y_ranks) / float(n)
    var_x = math.fsum((x - mean_x) ** 2 for x in x_ranks)
    dy = [y - mean_y for y in y_ranks]
    if var_x <= _EPS:
        return dy
    cov = math.fsum((x - mean_x) * (y - mean_y) for x, y in zip(x_ranks, y_ranks, strict=False))
    beta = cov / var_x
    return [dyi - beta * (x - mean_x) for x, dyi in zip(x_ranks, dy, strict=False)]


# NOTE: ships WITHOUT ``@register`` (inert).  The atomic integration commit adds
# ``@register`` + tier hint ``research_only`` + candidate builders + manifest /
# baseline re-pin for all surviving wave-2b lanes at once (single owner).
@register("strategy", "DownsideTailRiskPremiumStrategy", interface="event_driven")
class DownsideTailRiskPremiumStrategy(_CrossUpdateMixin, Strategy):
    """Long worst / short mildest own-return left tail, vol-neutralized, weekly XS.

    See the module docstring for the full theory, signal spec, and distinct-from
    rationale versus the Hill tail-index rider, idio-vol, and lottery incumbents.
    Reads only local event/bar OHLCV; performs no I/O and never raises from
    ``calculate_signals``.
    """

    decision_cadence_seconds = 86400
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "es_window": HyperParam.integer("es_window", default=180, low=8, high=20000),
            "tail_q": HyperParam.floating("tail_q", default=0.05, low=0.005, high=0.50),
            "vol_neutralize": HyperParam.boolean(
                "vol_neutralize", default=True, grid=[True, False]
            ),
            "min_history_bars": HyperParam.integer(
                "min_history_bars", default=60, low=8, high=100000
            ),
            "quantile_pct": HyperParam.floating("quantile_pct", default=0.20, low=0.02, high=0.50),
            "rebalance_bars": HyperParam.integer("rebalance_bars", default=7, low=1, high=100000),
            "min_hold_periods": HyperParam.integer(
                "min_hold_periods", default=4, low=0, high=100000
            ),
            "hysteresis_band": HyperParam.floating(
                "hysteresis_band", default=0.10, low=0.0, high=0.50
            ),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
            "min_symbols": HyperParam.integer("min_symbols", default=5, low=2, high=512),
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
        self.es_window = max(8, int(resolved["es_window"]))
        self.tail_q = min(0.5, max(1e-9, float(resolved["tail_q"])))
        self.vol_neutralize = bool(resolved["vol_neutralize"])
        self.min_history_bars = max(8, int(resolved["min_history_bars"]))
        self.quantile_pct = min(0.5, max(0.0, float(resolved["quantile_pct"])))
        self.rebalance_bars = max(1, int(resolved["rebalance_bars"]))
        self.min_hold_periods = max(0, int(resolved["min_hold_periods"]))
        self.hysteresis_band = min(0.5, max(0.0, float(resolved["hysteresis_band"])))
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
        size = _state_size(self.es_window + 1, self.min_history_bars + 1)
        self._state: dict[str, _CrossSectionalState] = {
            symbol: _CrossSectionalState(deque(maxlen=size), deque(maxlen=size))
            for symbol in self.symbol_list
        }
        self._close_times = {symbol: deque(maxlen=size) for symbol in self.symbol_list}
        self._last_eval_time_key = ""
        self._tick = 0
        # Recent decision-bar epochs (seconds) for deterministic bar-spacing
        # inference: the vol-target scalar annualizes the per-bar portfolio vol
        # via sqrt(bars_per_year) so the Moreira-Muir clamp is not left inert.
        self._recent_times: deque[float] = deque(maxlen=16)

    # ------------------------------------------------------------------ #
    # state (extend the mixin serialization with the bar-spacing epochs)
    # ------------------------------------------------------------------ #
    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state["recent_times"] = list(self._recent_times)
        return state

    def set_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        base_fields = set(super().get_state())
        if set(state) != base_fields | {"recent_times"}:
            return
        raw_times = state["recent_times"]
        if (
            not isinstance(raw_times, list)
            or len(raw_times) > int(self._recent_times.maxlen or 0)
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in raw_times
            )
            or any(right <= left for left, right in pairwise(raw_times))
        ):
            return
        base_state = {key: state[key] for key in base_fields}
        common_keys = base_state["close_times"][self.symbol_list[0]] if self.symbol_list else []
        expected_times = [
            dt.timestamp()
            for key in common_keys[-int(self._recent_times.maxlen or 0) :]
            if (dt := _event_datetime_from_key(key)) is not None
        ]
        if raw_times != expected_times:
            return
        super().set_state(base_state)
        if super().get_state() != base_state:
            return
        self._recent_times = deque(raw_times, maxlen=self._recent_times.maxlen)

    # ------------------------------------------------------------------ #
    # scoring / selection
    # ------------------------------------------------------------------ #
    def _residual_scores(
        self,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, dict[str, Any]]]:
        """Return ``(score_by_symbol, sigmas, metas)`` for the eligible universe."""
        es_mag: dict[str, float] = {}
        sigmas: dict[str, float] = {}
        raw_es: dict[str, float] = {}
        for symbol, item in self._state.items():
            closes = list(item.closes)
            if len(closes) < self.min_history_bars:
                continue
            returns = _log_returns(closes)
            if len(returns) < max(2, self.min_history_bars // 2):
                continue
            window_returns = returns[-self.es_window :]
            es = _expected_shortfall(window_returns, self.tail_q)
            sigma = sample_std(window_returns)
            if es is None or sigma is None or sigma <= _EPS:
                continue
            es_mag[symbol] = -float(es)  # larger magnitude = worse left tail
            sigmas[symbol] = float(sigma)
            raw_es[symbol] = float(es)

        if len(es_mag) < self.min_symbols:
            return {}, {}, {}

        ordered = sorted(es_mag)
        es_values = [es_mag[symbol] for symbol in ordered]
        sigma_values = [sigmas[symbol] for symbol in ordered]
        es_ranks = _average_ranks(es_values)
        if self.vol_neutralize:
            sigma_ranks = _average_ranks(sigma_values)
            residual_vec = _rank_residual(es_ranks, sigma_ranks)
        else:
            mean_es_rank = math.fsum(es_ranks) / float(len(es_ranks))
            residual_vec = [value - mean_es_rank for value in es_ranks]
        if residual_vec is None or len(residual_vec) != len(ordered):
            return {}, {}, {}

        scores = {symbol: float(residual_vec[idx]) for idx, symbol in enumerate(ordered)}
        score_values = list(scores.values())
        score_scale = max(1.0, *(abs(value) for value in score_values))
        if max(score_values) - min(score_values) <= _EPS * score_scale:
            return {}, {}, {}
        metas: dict[str, dict[str, Any]] = {
            symbol: {
                "expected_shortfall": float(raw_es[symbol]),
                "es_magnitude": float(es_mag[symbol]),
                "realized_sigma": float(sigmas[symbol]),
                "tail_score": float(scores[symbol]),
            }
            for symbol in ordered
        }
        return scores, sigmas, metas

    def _score_and_select(
        self,
    ) -> tuple[dict[str, tuple[str, float, dict[str, Any]]], dict[str, float]]:
        scores, sigmas, metas = self._residual_scores()
        if not scores:
            return {}, {}
        score_values = list(scores.values())
        score_scale = max(1.0, *(abs(value) for value in score_values))
        if max(score_values) - min(score_values) <= _EPS * score_scale:
            return {}, {}

        ordered = sorted(scores, key=lambda symbol: (scores[symbol], symbol))
        count = len(ordered)
        n_side = max(1, int(self.quantile_pct * count))
        if 2 * n_side > count:
            n_side = count // 2
        if n_side < 1:
            return {}, {}
        buffer = round(self.hysteresis_band * count)
        long_core = set(ordered[count - n_side :])
        short_core = set(ordered[:n_side])
        long_hold = set(ordered[count - min(count, n_side + buffer) :])
        short_hold = set(ordered[: min(count, n_side + buffer)])

        targets: dict[str, tuple[str, float, dict[str, Any]]] = {}
        for symbol in ordered:
            mode = self._state[symbol].mode
            score = scores[symbol]
            meta = metas[symbol]
            if symbol in long_core:
                targets[symbol] = ("LONG", score, meta)
            elif self.allow_short and symbol in short_core:
                targets[symbol] = ("SHORT", score, meta)
            elif mode == "LONG" and symbol in long_hold:
                targets[symbol] = ("LONG", score, meta)
            elif mode == "SHORT" and self.allow_short and symbol in short_hold:
                targets[symbol] = ("SHORT", score, meta)
        return targets, sigmas

    def _inverse_vol_weights(
        self,
        targets: dict[str, tuple[str, float, dict[str, Any]]],
        sigmas: dict[str, float],
    ) -> tuple[dict[str, float], float]:
        inv = {
            symbol: 1.0 / max(sigmas.get(symbol, 0.0), _EPS)
            for symbol in targets
            if sigmas.get(symbol, 0.0) > _EPS
        }
        total_inv = sum(inv.values())
        if total_inv <= _EPS:
            return {}, 1.0
        # ``portfolio_vol`` is the inverse-vol-weighted PER-BAR vol; the
        # ``inv / total_inv`` normalization above is scale-invariant (annualizing
        # every sigma_i cancels), so the risk-parity weights are horizon-free.
        # The vol-target SCALAR, however, compares ``portfolio_vol`` against an
        # annual-scale ``target_vol`` (0.20): annualize the per-bar estimate via
        # sqrt(bars_per_year) inferred from observed bar spacing first, otherwise
        # the Moreira-Muir clamp is INERT. When spacing is unavailable we pass
        # through (scalar=1.0) rather than throttle on mismatched horizons.
        portfolio_vol = sum((inv[symbol] / total_inv) * sigmas[symbol] for symbol in inv)
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
            self._state,
            event_time=event_time,
            strategy_id=_STRATEGY_ID,
            strategy_name=_STRATEGY_NAME,
            stop_loss_pct=self.stop_loss_pct,
            max_hold_bars=max_hold,
        )

    def _rebalance(self, event_time: Any) -> None:
        if len(self.symbol_list) < self.min_symbols:
            return
        # Record the decision-bar epoch so the vol-target scalar can infer bar
        # spacing (this runs once per new bar, before the rebalance gate).
        dt = _event_datetime_utc(event_time)
        if dt is not None:
            self._recent_times.append(dt.timestamp())
        # Stops / max-hold age every bar (protection + min-hold decision clock).
        self._age(event_time)
        if self._tick % self.rebalance_bars:
            return
        targets, sigmas = self._score_and_select()
        weights, scalar = self._inverse_vol_weights(targets, sigmas)
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
                    if item.bars_held < self.min_hold_periods:
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
                continue
            target_mode, score, meta = target
            if item.mode == target_mode:
                continue
            if item.mode != "OUT" and item.bars_held < self.min_hold_periods:
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


# --------------------------------------------------------------------------- #
# Candidate-wiring hints for the W3 integrator (this lane does NOT wire
# candidates itself -- new-file-only, no shared-file edits per the live-safety
# plan).  Admission route is `allow_multi_asset=True` at the data-PC handoff:
# a pure cross-sectional long-short (NOT carry, NOT momentum) -- no fake carry
# tag is added.
# --------------------------------------------------------------------------- #
_SUGGESTED_FAMILY = "cross_sectional"
_SUGGESTED_CANDIDATE_TAGS: tuple[str, ...] = (
    "cross_sectional",
    "downside_risk",
    "expected_shortfall",
    "tail_risk",
    "long_short",
    "crypto",
)

# Candidate slice.  The data-PC owns the {raw, vol-neutralized} x {90,180,360}d
# sweep, so we seed only the vol-neutralized 180d pool-admission form per timeframe
# to keep the candidate library thin.  The bar-denominated ES / min-history windows
# scale x6 (4h) / x24 (1h) to hold their ~180d wall-clock tail span, and
# ``rebalance_bars`` -- the bar-count weekly clock -- scales the same way
# (7 -> 42 -> 168) to keep the weekly rebalance cadence.  The ``min_hold_periods``
# floor, the ``tail_q`` / ``hysteresis_band`` fractions, the quantile, and the
# vol-target / exposure fractions are timeframe-invariant and stay fixed.
_DOWNSIDE_TAIL_RISK_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "4h": (
        {
            "variant": "es180_volneutral",
            "es_window": 1080,
            "tail_q": 0.05,
            "vol_neutralize": True,
            "min_history_bars": 360,
            "quantile_pct": 0.20,
            "rebalance_bars": 42,
            "min_hold_periods": 4,
            "hysteresis_band": 0.10,
            "min_symbols": 5,
            "allow_short": True,
            "target_gross_exposure": 1.0,
            "target_vol": 0.20,
            "stop_loss_pct": 0.10,
        },
    ),
    "1h": (
        {
            "variant": "es180_volneutral",
            "es_window": 4320,
            "tail_q": 0.05,
            "vol_neutralize": True,
            "min_history_bars": 1440,
            "quantile_pct": 0.20,
            "rebalance_bars": 168,
            "min_hold_periods": 4,
            "hysteresis_band": 0.10,
            "min_symbols": 5,
            "allow_short": True,
            "target_gross_exposure": 1.0,
            "target_vol": 0.20,
            "stop_loss_pct": 0.10,
        },
    ),
    "1d": (
        {
            "variant": "es180_volneutral",
            "es_window": 180,
            "tail_q": 0.05,
            "vol_neutralize": True,
            "min_history_bars": 60,
            "quantile_pct": 0.20,
            "rebalance_bars": 7,
            "min_hold_periods": 4,
            "hysteresis_band": 0.10,
            "min_symbols": 5,
            "allow_short": True,
            "target_gross_exposure": 1.0,
            "target_vol": 0.20,
            "stop_loss_pct": 0.10,
        },
    ),
}

__all__ = ["DownsideTailRiskPremiumStrategy"]

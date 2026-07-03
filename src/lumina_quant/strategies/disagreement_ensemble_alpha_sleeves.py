"""Disagreement-gated ensemble combiner over four OHLCV-only directional components.

THEORY / PROVENANCE.  Forecast combination (Bates & Granger 1969): a combined
forecast from several imperfect models can beat the single best model when the
components' errors are not perfectly correlated -- the combination need not be
"smarter" than any one member, only diverse in its mistakes.  Timmermann (2006,
"Forecast Combinations") surveys the empirical robustness of simple/adaptive
combination weights and the conditions under which combining helps versus
hurts.  Ensemble diversity (Krogh & Vedelsby 1995): the ensemble's squared
error is the weighted-average member error MINUS the members' ambiguity
(disagreement) around the ensemble mean -- disagreement is therefore not
merely a diversity bonus but a signal about how much the combination is
currently "paying for" agreement.  This sleeve inverts that relationship into
a real-time RELIABILITY gate: when the four components disagree sharply on
DIRECTION (not just magnitude), that is precisely when a blended forecast is
least trustworthy, so the strategy stands fully aside rather than averaging
through the contradiction.  The combination arithmetic itself -- inverse-error
weighting, the dispersion/disagreement coefficient, and the rolling directional
hit-rate used to score each component's reliability -- is
:mod:`lumina_quant.indicators.ensemble_weights`, absorbed from the
DeepLearning repo's online ensemble engine (``ensemble_strategies/components/
weights.py`` and its disagreement-switching rule).

ENTRY.  Per symbol, four internal directional component scores are maintained,
each bounded to ``[-1, 1]``, all OHLCV-only and causal (computed from data
through the current decision bar's close):

  (i)   EMA-slope TSMOM sign/strength -- the fractional slope of an
        exponential moving average over a lookback, squashed through
        :func:`~lumina_quant.indicators.alpha_features.clipped_tanh_score`.
  (ii)  Rolling z-score reversion (fade sign) -- the NEGATIVE of the
        tanh-squashed rolling z-score of price: a price far above its recent
        mean scores negative (a fade), and vice versa.
  (iii) Donchian channel position -- where the close sits between the
        trailing Donchian upper/lower band, linearly mapped to ``[-1, 1]``
        (``+1`` at the upper band, ``-1`` at the lower band).
  (iv)  Kaufman efficiency-ratio-signed trend -- the (unsigned) Kaufman
        efficiency ratio over a lookback, signed by the direction of the
        price change over the same window.

On each accepted decision bar the sleeve first scores the PRIOR bar's four
component predictions against the bar's just-realized return (sign agreement,
via :func:`~lumina_quant.indicators.ensemble_weights.direction_hit_rate` over a
trailing window) -- this is causal: the prediction was formed at bar ``t-1``
and is graded against the return realized over bar ``t``.  Each component's
trailing error (``1 - hit_rate``; a component with no scored history yet is
treated as maximally unreliable, i.e. error ``1.0`` -- a conservative,
never-favor-the-untested default) feeds
:func:`~lumina_quant.indicators.ensemble_weights.inverse_error_weights` to
produce adaptive combination weights.  The sleeve then computes this bar's
fresh component scores and the
:func:`~lumina_quant.indicators.ensemble_weights.disagreement_coefficient`
(coefficient of variation) across them.  When the disagreement exceeds
``disagreement_gate`` -- OR is ``None`` (the coefficient is undefined when the
components' mean is ~0, which happens precisely when they are split roughly
evenly for/against a direction; treated here as maximal disagreement, i.e.
BLOCKED, the conservative reading) -- no entry/pyramid signal is produced.
Otherwise the weighted composite ``sum(w_i * score_i)`` is compared against
``+/-entry_band``: LONG when ``composite >= entry_band``, SHORT when
``composite <= -entry_band``, flat otherwise.  Pyramiding re-confirms the same
gate and the same-direction composite threshold on every subsequent bar.  The
position itself, once armed, rides on the inherited ATR trailing stop with
pyramiding and vol-targeted sizing (:class:`_ReturnRiderBase`) -- the ensemble
machinery only decides WHETHER and WHICH WAY to enter/add, never the exit.

DISTINCT FROM :class:`DiversifiedMultiFactorEnsembleStrategy`: that strategy is
a CROSS-SECTIONAL factor book -- it combines factor exposures across the
symbol universe at a point in time to build a portfolio.  This strategy is a
PER-SYMBOL TIME-SERIES signal combiner; its novelty is the DISAGREEMENT GATE --
it trades a single symbol only when its four internal directional views agree
closely enough, standing fully aside whenever they contradict each other
regardless of how strong any individual component's score is.  No other sleeve
in the registry combines multiple internal directional views with an explicit
consensus/disagreement admission gate.

OHLCV-only, per-symbol single-asset, pure Python (no numpy/scipy), never-raise
graceful guards throughout, ``decision_cadence_seconds >= 1800`` (inherited
>=30m floor).  Data-local: no fetching, no artifact writes, no hidden
environment state.  Must still be validated on the data-bearing machine.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

from lumina_quant.indicators.alpha_features import clipped_tanh_score, rolling_zscore
from lumina_quant.indicators.bands import donchian_channel
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.indicators.ensemble_weights import (
    direction_hit_rate,
    disagreement_coefficient,
    inverse_error_weights,
)
from lumina_quant.indicators.momentum import kaufman_efficiency_ratio
from lumina_quant.indicators.moving_average import exponential_moving_average_series
from lumina_quant.strategies.external_alpha_sleeves import _EPS, _Snapshot
from lumina_quant.strategies.return_rider_alpha_sleeves import (
    _restore_deque,
    _ReturnRiderBase,
    _RiderCommon,
    _RiderState,
)
from lumina_quant.tuning import HyperParam

# The four internal directional component scores, in a fixed, stable order.
_COMPONENT_NAMES: tuple[str, str, str, str] = (
    "tsmom",
    "reversion",
    "donchian",
    "trend_efficiency",
)
_EQUAL_WEIGHTS: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)

_ComponentScores = tuple[float, float, float, float]


def _tsmom_score(closes: list[float], *, window: int, lookback: int, scale: float) -> float:
    """Return the EMA-slope TSMOM sign/strength score in ``[-1, 1]``."""
    series = exponential_moving_average_series(closes, window)
    if series is None or len(series) <= lookback:
        return 0.0
    now = series[-1]
    prev = series[-1 - lookback]
    if abs(prev) <= _EPS:
        return 0.0
    slope = (now - prev) / abs(prev)
    if not math.isfinite(slope):
        return 0.0
    return clipped_tanh_score(slope, scale=scale)


def _reversion_score(closes: list[float], *, window: int, scale: float) -> float:
    """Return the rolling z-score reversion (fade sign) score in ``[-1, 1]``."""
    z = rolling_zscore(closes, window=window)
    if z is None:
        return 0.0
    return -clipped_tanh_score(z, scale=scale)


def _donchian_score(
    closes: list[float], highs: list[float], lows: list[float], *, window: int
) -> float:
    """Return the Donchian channel position score in ``[-1, 1]``."""
    upper, middle, lower = donchian_channel(highs, lows, window=window)
    if upper is None or lower is None or middle is None or not closes:
        return 0.0
    span = upper - lower
    if span <= _EPS:
        return 0.0
    position = 2.0 * (closes[-1] - middle) / span
    return max(-1.0, min(1.0, position))


def _trend_efficiency_score(closes: list[float], *, period: int) -> float:
    """Return the Kaufman efficiency-ratio-signed trend score in ``[-1, 1]``."""
    er = kaufman_efficiency_ratio(closes, period=period)
    if er is None or len(closes) <= period:
        return 0.0
    direction = closes[-1] - closes[-1 - period]
    if direction > 0.0:
        sign = 1.0
    elif direction < 0.0:
        sign = -1.0
    else:
        sign = 0.0
    return max(-1.0, min(1.0, sign * er))


# @register("strategy", "DisagreementGatedEnsembleStrategy", interface="event_driven")  # applied atomically with the research_only tier hint in W3 integration — do NOT apply earlier (live-safety)
class DisagreementGatedEnsembleStrategy(_ReturnRiderBase):
    """Combine four directional components; trade only when they agree.

    Weights are adaptive inverse-error weights over each component's trailing
    directional hit-rate; the entry/pyramid gate is the cross-component
    disagreement coefficient (see module docstring).  The winner is ridden with
    the inherited ATR trailing stop + pyramiding; vol-targeted sizing.
    """

    decision_cadence_seconds = 1800
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    strategy_name = "DisagreementGatedEnsembleStrategy"
    strategy_id = "disagreement_gated_ensemble"

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "tsmom_window": HyperParam.integer("tsmom_window", default=20, low=2, high=4096),
            "tsmom_lookback": HyperParam.integer("tsmom_lookback", default=5, low=1, high=512),
            "tsmom_scale": HyperParam.floating("tsmom_scale", default=0.02, low=1e-6, high=1.0),
            "reversion_window": HyperParam.integer(
                "reversion_window", default=20, low=3, high=4096
            ),
            "reversion_scale": HyperParam.floating(
                "reversion_scale", default=8.0, low=0.01, high=100.0
            ),
            "donchian_window": HyperParam.integer("donchian_window", default=20, low=2, high=4096),
            "efficiency_period": HyperParam.integer(
                "efficiency_period", default=14, low=2, high=1024
            ),
            "error_window": HyperParam.integer("error_window", default=60, low=2, high=4096),
            "return_deadband": HyperParam.floating(
                "return_deadband", default=0.0, low=0.0, high=0.10
            ),
            "disagreement_gate": HyperParam.floating(
                "disagreement_gate", default=1.0, low=0.0, high=50.0
            ),
            "entry_band": HyperParam.floating("entry_band", default=0.15, low=0.0, high=1.0),
            "trail_atr_mult": HyperParam.floating(
                "trail_atr_mult", default=3.0, low=0.1, high=20.0
            ),
            "atr_period": HyperParam.integer("atr_period", default=14, low=2, high=1024),
            "max_adds": HyperParam.integer("max_adds", default=3, low=0, high=32),
            "add_step_atr": HyperParam.floating("add_step_atr", default=1.0, low=0.0, high=20.0),
            "vol_window": HyperParam.integer("vol_window", default=48, low=2, high=4096),
            "target_vol": HyperParam.floating("target_vol", default=0.02, low=0.0, high=1.0),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=240, low=1, high=200000),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
            "add_alloc_fraction": HyperParam.floating(
                "add_alloc_fraction", default=0.5, low=0.0, high=1.0
            ),
            "target_allocation": HyperParam.floating(
                "target_allocation", default=0.30, low=0.0, high=5.0, tunable=False
            ),
            "max_order_value": HyperParam.floating(
                "max_order_value", default=5000.0, low=0.0, high=1_000_000.0, tunable=False
            ),
            "min_price": HyperParam.floating("min_price", default=0.10, low=0.0, high=1_000_000.0),
        }

    def __init__(self, bars: Any, events: Any, **params: Any) -> None:
        super().__init__(bars, events, **params)
        self._scores: dict[str, _ComponentScores | None] = dict.fromkeys(self.symbol_list)
        self._realized: dict[str, deque[float]] = {
            symbol: deque(maxlen=self.error_window) for symbol in self.symbol_list
        }
        self._predicted: dict[str, tuple[deque[float], ...]] = {
            symbol: tuple(deque(maxlen=self.error_window) for _ in range(4))
            for symbol in self.symbol_list
        }

    def _bind_params(self, resolved: dict[str, Any]) -> None:
        self.tsmom_window = max(2, int(resolved["tsmom_window"]))
        self.tsmom_lookback = max(1, int(resolved["tsmom_lookback"]))
        self.tsmom_scale = max(1e-9, float(resolved["tsmom_scale"]))
        self.reversion_window = max(3, int(resolved["reversion_window"]))
        self.reversion_scale = max(1e-9, float(resolved["reversion_scale"]))
        self.donchian_window = max(2, int(resolved["donchian_window"]))
        self.efficiency_period = max(2, int(resolved["efficiency_period"]))
        self.error_window = max(2, int(resolved["error_window"]))
        self.return_deadband = max(0.0, float(resolved["return_deadband"]))
        self.disagreement_gate = max(0.0, float(resolved["disagreement_gate"]))
        self.entry_band = max(0.0, float(resolved["entry_band"]))

    def _common_config(self, resolved: dict[str, Any]) -> _RiderCommon:
        return self._resolve_common(
            resolved,
            extra_window=max(
                self.tsmom_window + self.tsmom_lookback,
                self.reversion_window,
                self.donchian_window,
                self.efficiency_period,
            )
            + 2,
        )

    # -- causal advance: score-error bookkeeping + fresh component scores ----
    def _process_symbol(self, symbol: str, snapshot: _Snapshot) -> None:
        # Advance the ensemble bookkeeping on this bar's close BEFORE the base
        # runs ride+entry, mirroring the base's own time-key dedup so a
        # repeated boundary firing never double-scores a bar.  Both the
        # outcome-scoring (against the PRIOR bar's predictions) and the fresh
        # component scores use only the current and past closes, so deciding
        # on them at this bar's close is not look-ahead.
        item = self._state.get(symbol)
        if item is not None:
            key = time_key(snapshot.time)
            if not (key and key == item.last_time_key):
                self._advance_ensemble(symbol, item, snapshot)
        super()._process_symbol(symbol, snapshot)

    def _advance_ensemble(self, symbol: str, item: _RiderState, snapshot: _Snapshot) -> None:
        close = safe_float(snapshot.close)
        if close is None or close <= self.min_price:
            return
        prior_closes = list(item.closes)
        prev_scores = self._scores.get(symbol)
        if prior_closes and prior_closes[-1] > _EPS and prev_scores is not None:
            realized_return = math.log(close / prior_closes[-1])
            if math.isfinite(realized_return):
                self._append_outcome(symbol, prev_scores, realized_return)
        high = safe_float(snapshot.high)
        if high is None:
            high = close
        low = safe_float(snapshot.low)
        if low is None:
            low = close
        working_closes = [*prior_closes, close]
        working_highs = [*item.highs, high]
        working_lows = [*item.lows, low]
        self._scores[symbol] = self._component_scores(working_closes, working_highs, working_lows)

    def _append_outcome(
        self, symbol: str, prev_scores: _ComponentScores, realized_return: float
    ) -> None:
        self._realized[symbol].append(realized_return)
        for idx, history in enumerate(self._predicted[symbol]):
            history.append(prev_scores[idx])

    def _component_scores(
        self, closes: list[float], highs: list[float], lows: list[float]
    ) -> _ComponentScores:
        return (
            _tsmom_score(
                closes,
                window=self.tsmom_window,
                lookback=self.tsmom_lookback,
                scale=self.tsmom_scale,
            ),
            _reversion_score(closes, window=self.reversion_window, scale=self.reversion_scale),
            _donchian_score(closes, highs, lows, window=self.donchian_window),
            _trend_efficiency_score(closes, period=self.efficiency_period),
        )

    def _trailing_errors(self, symbol: str) -> list[float]:
        """Return the 4 components' trailing ``1 - hit_rate`` errors.

        A component with no scored (predicted, realized) pairs yet is treated
        as maximally unreliable (error ``1.0``) -- a conservative default that
        never lets an untested component dominate the combination weights.
        """
        realized_list = list(self._realized.get(symbol, ()))
        errors: list[float] = []
        for history in self._predicted.get(symbol, ()):
            hit_rate = direction_hit_rate(
                list(history), realized_list, deadband=self.return_deadband
            )
            errors.append(1.0 - hit_rate if hit_rate is not None else 1.0)
        if len(errors) != 4:
            errors = [1.0, 1.0, 1.0, 1.0]
        return errors

    def _decision_state(self, symbol: str) -> dict[str, Any] | None:
        """Return the combined weights/disagreement/composite for this bar, or ``None``."""
        scores = self._scores.get(symbol)
        if scores is None:
            return None
        errors = self._trailing_errors(symbol)
        weights = inverse_error_weights(errors)
        if weights is None or len(weights) != 4:
            weights = list(_EQUAL_WEIGHTS)
        disagreement = disagreement_coefficient(scores)
        composite = sum(w * s for w, s in zip(weights, scores, strict=True))
        return {
            "scores": scores,
            "weights": weights,
            "disagreement": disagreement,
            "composite": composite,
        }

    def _gated_direction(self, symbol: str) -> str:
        """Return ``"LONG"``/``"SHORT"``/``""`` after applying the disagreement gate."""
        state = self._decision_state(symbol)
        if state is None:
            return ""
        disagreement = state["disagreement"]
        # None means the coefficient is undefined (near-zero consensus mean) --
        # treated as maximal disagreement, i.e. blocked (the conservative read).
        if disagreement is None or disagreement > self.disagreement_gate:
            return ""
        composite = state["composite"]
        if composite >= self.entry_band:
            return "LONG"
        if composite <= -self.entry_band:
            return "SHORT"
        return ""

    def _symbol_for(self, item: _RiderState) -> str | None:
        for symbol, candidate in self._state.items():
            if candidate is item:
                return symbol
        return None

    def _entry_decision(self, item: _RiderState) -> str:
        symbol = self._symbol_for(item)
        if symbol is None:
            return ""
        return self._gated_direction(symbol)

    def _should_pyramid(self, item: _RiderState) -> bool:
        symbol = self._symbol_for(item)
        if symbol is None:
            return False
        return self._gated_direction(symbol) == item.mode

    def _entry_metadata(self, item: _RiderState) -> dict[str, Any]:
        symbol = self._symbol_for(item)
        state = self._decision_state(symbol) if symbol is not None else None
        if state is None:
            return {
                "composite": None,
                "disagreement": None,
                "weights": None,
                "component_scores": None,
            }
        disagreement = state["disagreement"]
        return {
            "composite": float(state["composite"]),
            "disagreement": float(disagreement) if disagreement is not None else None,
            "weights": dict(
                zip(_COMPONENT_NAMES, (float(w) for w in state["weights"]), strict=True)
            ),
            "component_scores": dict(
                zip(_COMPONENT_NAMES, (float(s) for s in state["scores"]), strict=True)
            ),
        }

    # -- state -----------------------------------------------------------------
    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state["ensemble_scores"] = {
            symbol: (list(scores) if scores is not None else None)
            for symbol, scores in self._scores.items()
        }
        state["ensemble_history"] = {
            symbol: {
                "realized": list(self._realized.get(symbol, ())),
                "predicted": [list(history) for history in self._predicted.get(symbol, ())],
            }
            for symbol in self.symbol_list
        }
        return state

    def set_state(self, state: dict[str, Any]) -> None:
        super().set_state(state)
        raw_scores = state.get("ensemble_scores") if isinstance(state, dict) else None
        if isinstance(raw_scores, dict):
            for symbol, payload in raw_scores.items():
                if symbol not in self._scores:
                    continue
                if payload is None:
                    self._scores[symbol] = None
                    continue
                try:
                    seq = [float(v) for v in list(payload)]
                except Exception:
                    continue
                if len(seq) == 4 and all(math.isfinite(v) for v in seq):
                    self._scores[symbol] = (seq[0], seq[1], seq[2], seq[3])
        raw_history = state.get("ensemble_history") if isinstance(state, dict) else None
        if isinstance(raw_history, dict):
            for symbol, payload in raw_history.items():
                if symbol not in self._realized or not isinstance(payload, dict):
                    continue
                self._safe_restore_deque(self._realized[symbol], payload.get("realized"))
                predicted_payload = payload.get("predicted")
                targets = self._predicted.get(symbol)
                if targets is not None and isinstance(predicted_payload, list):
                    for idx, history in enumerate(targets):
                        if idx < len(predicted_payload):
                            self._safe_restore_deque(history, predicted_payload[idx])

    @staticmethod
    def _safe_restore_deque(target: deque[float], payload: Any) -> None:
        """Call :func:`_restore_deque`, swallowing non-iterable adversarial payloads.

        ``_restore_deque`` does ``list(payload or [])``, which raises
        ``TypeError`` for a truthy non-iterable payload (e.g. a bare ``int`` or
        ``float``); this wrapper keeps the never-raise contract for arbitrary
        deserialized input without altering the shared helper.
        """
        try:
            _restore_deque(target, payload)
        except Exception:
            target.clear()


# One variant per timeframe so the W3 integrator can build candidates without
# touching this module's logic; shape mirrors the existing rider slices
# (e.g. ``_ADAPTIVE_TREND_RIDER_SLICE`` in candidate_library.py).
_DISAGREEMENT_ENSEMBLE_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "30m": (
        {
            "variant": "fast_ls",
            "tsmom_window": 12,
            "tsmom_lookback": 3,
            "tsmom_scale": 0.015,
            "reversion_window": 16,
            "reversion_scale": 6.0,
            "donchian_window": 16,
            "efficiency_period": 10,
            "error_window": 60,
            "return_deadband": 0.0,
            "disagreement_gate": 1.0,
            "entry_band": 0.12,
            "trail_atr_mult": 3.0,
            "atr_period": 14,
            "max_adds": 3,
            "add_step_atr": 1.0,
            "vol_window": 48,
            "target_vol": 0.020,
            "max_hold_bars": 200,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
    "1h": (
        {
            "variant": "core_ls",
            "tsmom_window": 20,
            "tsmom_lookback": 5,
            "tsmom_scale": 0.02,
            "reversion_window": 20,
            "reversion_scale": 8.0,
            "donchian_window": 20,
            "efficiency_period": 14,
            "error_window": 72,
            "return_deadband": 0.0,
            "disagreement_gate": 1.0,
            "entry_band": 0.15,
            "trail_atr_mult": 3.2,
            "atr_period": 14,
            "max_adds": 3,
            "add_step_atr": 1.0,
            "vol_window": 48,
            "target_vol": 0.020,
            "max_hold_bars": 180,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
    "4h": (
        {
            "variant": "swing_ls",
            "tsmom_window": 28,
            "tsmom_lookback": 6,
            "tsmom_scale": 0.03,
            "reversion_window": 28,
            "reversion_scale": 10.0,
            "donchian_window": 24,
            "efficiency_period": 16,
            "error_window": 60,
            "return_deadband": 0.0,
            "disagreement_gate": 1.05,
            "entry_band": 0.18,
            "trail_atr_mult": 3.5,
            "atr_period": 14,
            "max_adds": 2,
            "add_step_atr": 1.0,
            "vol_window": 36,
            "target_vol": 0.030,
            "max_hold_bars": 120,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
    "1d": (
        {
            "variant": "macro_ls",
            "tsmom_window": 36,
            "tsmom_lookback": 8,
            "tsmom_scale": 0.05,
            "reversion_window": 36,
            "reversion_scale": 12.0,
            "donchian_window": 20,
            "efficiency_period": 20,
            "error_window": 48,
            "return_deadband": 0.0,
            "disagreement_gate": 1.1,
            "entry_band": 0.20,
            "trail_atr_mult": 4.0,
            "atr_period": 10,
            "max_adds": 2,
            "add_step_atr": 1.0,
            "vol_window": 24,
            "target_vol": 0.040,
            "max_hold_bars": 60,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
}

__all__ = [
    "DisagreementGatedEnsembleStrategy",
]

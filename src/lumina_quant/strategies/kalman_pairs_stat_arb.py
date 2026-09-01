"""Kalman-filter pairs statistical arbitrage traded on the hedge-implied residual z.

Independent adaptation of publicly described rules; not a reproduction,
endorsement or performance claim.  Public-source lineage:

* The textbook Kalman-filter pairs trade (E. Chan lineage): the log-price
  relation ``y_t = beta_t * x_t + alpha_t + eps_t`` is filtered online with a
  random-walk state.  Two deviation measures fall out of it: the *standardized
  innovation* ``e_t / sqrt(S_t)`` -- the surprise of today's ``y`` given
  yesterday's filtered hedge, scaled by the filter's own innovation variance --
  and the z-score of the Engle-Granger residual that today's posterior implies
  over the trailing window.  The residual z is the default here; the innovation
  z stays available as ``signal_mode="innovation"``.
* The residual-ADF / mean-reversion-half-life discipline of the classic pairs
  and statistical-arbitrage curriculum published by the Korean quant educator
  "amateur quant" (Cho Sung-hyun), within the scope of that public research
  writing: only trade a pair while a preregistered residual-ADF heuristic
  rejects a unit root in the spread, and bound the holding period by the
  estimated OU/AR(1) half-life.

Hypothesis: for a genuinely cointegrated pair the Kalman posterior is a
self-updating hedge that needs no fixed beta lookback, so fading large
deviations of the residual it implies -- while an uncalibrated residual-ADF
heuristic on that same residual rejects a unit root -- and closing on reversion or
on a half-life clock harvests the spread's mean reversion with fewer stale-beta
artefacts than a rolling-window hedge.

Why every gate reads the Engle-Granger residual and not the filter's own
residual: the a-posteriori Kalman residual ``y_t - beta_t*x_t - alpha_t`` is
white noise by construction (the update step has already absorbed the
surprise), so an ADF test on it rejects a unit root even for two INDEPENDENT
random walks, and an AR(1) fit on it returns a sub-bar half-life.  The ADF
gate, the half-life cap and the default z therefore all read

    ``resid_tau = log_y_tau - beta_t * log_x_tau - alpha_t``

over the trailing window: the CURRENT hedge applied to HISTORICAL prices, i.e.
the Engle-Granger residual of the pair under today's posterior.

How this differs from the repo's existing pair books: ``pair_trading_zscore``
and ``pair_spread_zscore`` estimate the hedge with rolling OLS or a scalar RLS
and z-score the *spread level*.  This sleeve carries the full 2-state Kalman
posterior from ``lumina_quant.indicators.stat_arb`` (slope AND intercept, with
explicit process noise ``delta`` and observation noise ``obs_noise``), z-scores
and z-scores the Engle-Granger residual that posterior implies (with the filter-normalised
innovation z kept as an option), gates new entries on an uncalibrated residual-ADF heuristic
of that residual, and caps the hold with its OU half-life.

Public source vs. author's choices:

* Public source: the Kalman regression pair trade, the innovation as the trade
  trigger, the ADF cointegration screen, and the AR(1) half-life as the natural
  holding horizon.
* AUTHOR's choices (arbitrary, unvalidated, never fitted to data here): every
  numeric default -- ``kalman_delta=1e-4``, ``kalman_obs_noise=1e-5``,
  ``entry_z=2.0``, ``exit_z=0.5``, ``stop_z=4.0``, ``min_updates=60``,
  ``adf_window=90`` with the preregistered generic ADF threshold,
  ``half_life_multiple=2.0``,
  ``max_hold_bars=120``, the ``leg_allocation`` / ``max_leg_allocation`` sizing
  and the 1-hour decision cadence.

``kalman_obs_noise`` is a VARIANCE in log-price^2 units, not a standard
deviation: the ``1e-5`` default is an observation sd of ~0.32% per bar, which
is the order of magnitude of a liquid pair's residual noise.  A value such as
``1e-3`` is an sd of ~3.2%, wide enough that the standardized innovation of a
genuinely cointegrated pair never reaches ``entry_z``.

research_only: no backtest is claimed, no live wiring, no performance claim.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

from lumina_quant.core.plugin_registry import register
from lumina_quant.indicators import (
    KalmanHedgeState,
    adf_critical_value,
    adf_t_statistic,
    ar1_half_life,
    kalman_hedge_ratio_step,
    zscore,
)
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.strategies.external_alpha_sleeves import (
    _Snapshot,
    _emit,
    _event_symbols,
    _market_snapshot,
    _target_metadata,
    _window_snapshot,
)
from lumina_quant.strategy import Strategy
from lumina_quant.tuning import HyperParam, resolve_params_from_schema

_STRATEGY_ID = "kalman_pairs_stat_arb"
_SIGNAL_MODES = ("spread_z", "innovation")
_FLAT = "FLAT"
_LONG_SPREAD = "LONG_SPREAD"
_SHORT_SPREAD = "SHORT_SPREAD"
_MODES = frozenset({_FLAT, _LONG_SPREAD, _SHORT_SPREAD})


@register("strategy", "KalmanPairsStatArbStrategy", interface="event_driven")
class KalmanPairsStatArbStrategy(Strategy):
    """Fade the Kalman-hedged residual of a cointegrated log-price pair."""

    decision_cadence_seconds = 3600
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "symbol_y": HyperParam.string("symbol_y", default="", tunable=False),
            "symbol_x": HyperParam.string("symbol_x", default="", tunable=False),
            "kalman_delta": HyperParam.floating("kalman_delta", default=1e-4, low=1e-9, high=0.5),
            "kalman_obs_noise": HyperParam.floating(
                "kalman_obs_noise",
                default=1e-5,
                low=1e-9,
                high=10.0,
                description=(
                    "Observation VARIANCE R in log-price^2 units (1e-5 == sd ~0.32% per bar). "
                    "Too wide a value flattens the standardized innovation below entry_z."
                ),
            ),
            "signal_mode": HyperParam.categorical(
                "signal_mode",
                default="spread_z",
                choices=_SIGNAL_MODES,
                description=(
                    "spread_z: z-score of the Engle-Granger residual (current hedge on the "
                    "trailing prices). innovation: the filter's standardized innovation."
                ),
            ),
            "z_window": HyperParam.integer("z_window", default=60, low=5, high=5000),
            "min_updates": HyperParam.integer("min_updates", default=60, low=2, high=100000),
            "min_beta": HyperParam.floating("min_beta", default=0.0, low=0.0, high=10.0),
            "require_cointegration": HyperParam.boolean(
                "require_cointegration",
                default=True,
                description=(
                    "Apply the preregistered, uncalibrated residual-ADF heuristic. "
                    "This is not a calibrated 5% Engle-Granger test."
                ),
            ),
            "adf_window": HyperParam.integer("adf_window", default=90, low=12, high=5000),
            "entry_z": HyperParam.floating("entry_z", default=2.0, low=0.1, high=20.0),
            "exit_z": HyperParam.floating("exit_z", default=0.5, low=0.0, high=10.0),
            "stop_z": HyperParam.floating("stop_z", default=4.0, low=0.2, high=50.0),
            "half_life_multiple": HyperParam.floating(
                "half_life_multiple", default=2.0, low=0.0, high=20.0
            ),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=120, low=1, high=100000),
            "exit_on_cointegration_break": HyperParam.boolean(
                "exit_on_cointegration_break", default=False
            ),
            "leg_allocation": HyperParam.floating(
                "leg_allocation", default=0.10, low=0.0, high=1.0, tunable=False
            ),
            "max_leg_allocation": HyperParam.floating(
                "max_leg_allocation", default=0.30, low=0.0, high=1.0, tunable=False
            ),
            "max_order_value": HyperParam.floating(
                "max_order_value", default=500.0, low=0.0, high=1_000_000.0, tunable=False
            ),
        }

    def __init__(self, bars: Any, events: Any, **params: Any) -> None:
        self.bars = bars
        self.events = events
        self.symbol_list = list(getattr(bars, "symbol_list", []) or [])
        resolved = resolve_params_from_schema(self.get_param_schema(), params, keep_unknown=False)

        symbol_y = str(resolved["symbol_y"] or "").strip()
        symbol_x = str(resolved["symbol_x"] or "").strip()
        self.symbol_y = symbol_y or (self.symbol_list[0] if self.symbol_list else "")
        self.symbol_x = symbol_x or (self.symbol_list[1] if len(self.symbol_list) > 1 else "")
        # ponytail: an unresolvable / degenerate pair makes the sleeve inert instead of
        # raising, so a single-symbol universe cannot break a multi-strategy run.
        self.enabled = (
            bool(self.symbol_y) and bool(self.symbol_x) and self.symbol_y != self.symbol_x
        )
        self.pair_id = f"{self.symbol_y}|{self.symbol_x}"

        self.kalman_delta = min(0.5, max(1e-9, float(resolved["kalman_delta"])))
        self.kalman_obs_noise = max(1e-9, float(resolved["kalman_obs_noise"]))
        mode = str(resolved["signal_mode"]).strip().lower()
        self.signal_mode = mode if mode in _SIGNAL_MODES else "spread_z"
        self.z_window = max(5, int(resolved["z_window"]))
        self.min_updates = max(2, int(resolved["min_updates"]))
        self.min_beta = max(0.0, float(resolved["min_beta"]))
        self.require_cointegration = bool(resolved["require_cointegration"])
        self.adf_window = max(12, int(resolved["adf_window"]))
        self.entry_z = max(0.0, float(resolved["entry_z"]))
        self.exit_z = max(0.0, float(resolved["exit_z"]))
        self.stop_z = max(self.entry_z, float(resolved["stop_z"]))
        self.half_life_multiple = max(0.0, float(resolved["half_life_multiple"]))
        self.max_hold_bars = max(1, int(resolved["max_hold_bars"]))
        self.exit_on_cointegration_break = bool(resolved["exit_on_cointegration_break"])
        self.leg_allocation = max(0.0, float(resolved["leg_allocation"]))
        self.max_leg_allocation = max(0.0, float(resolved["max_leg_allocation"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))

        self._kalman: KalmanHedgeState | None = None
        history = max(self.z_window, self.adf_window) + 2
        self._log_ys: deque[float] = deque(maxlen=history)
        self._log_xs: deque[float] = deque(maxlen=history)
        self._paired_history: deque[tuple[str, float, float]] = deque(maxlen=history)
        self._mode = _FLAT
        self._bars_held = 0
        self._last_pair_key = ""
        self._pending_y: tuple[str, float] | None = None
        self._pending_x: tuple[str, float] | None = None
        self._unmatched_leg_drops = 0
        self._emission_failures = 0

    # ------------------------------------------------------------------ state

    def get_state(self) -> dict[str, Any]:
        return {
            "kalman": self._kalman.to_dict() if self._kalman is not None else None,
            "paired_history": [
                [stamp, float(log_y), float(log_x)] for stamp, log_y, log_x in self._paired_history
            ],
            "mode": self._mode,
            "bars_held": int(self._bars_held),
            "last_pair_key": str(self._last_pair_key),
            "pending_y": list(self._pending_y) if self._pending_y is not None else None,
            "pending_x": list(self._pending_x) if self._pending_x is not None else None,
            "unmatched_leg_drops": int(self._unmatched_leg_drops),
            "emission_failures": int(self._emission_failures),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        required = {
            "kalman",
            "paired_history",
            "mode",
            "bars_held",
            "last_pair_key",
            "pending_y",
            "pending_x",
            "unmatched_leg_drops",
            "emission_failures",
        }
        if not required.issubset(state):
            return
        kalman_payload = state["kalman"]
        kalman = None if kalman_payload is None else KalmanHedgeState.from_dict(kalman_payload)
        if kalman_payload is not None and kalman is None:
            return
        if kalman is not None and not all(
            math.isfinite(value)
            for value in (kalman.beta, kalman.alpha, kalman.p00, kalman.p01, kalman.p11)
        ):
            return
        paired_history = _parse_paired_history(state["paired_history"], self._paired_history.maxlen)
        if paired_history is None:
            return
        mode = str(state["mode"]).upper()
        if mode not in _MODES:
            return
        try:
            bars_held = max(0, int(state["bars_held"]))
            unmatched_leg_drops = max(0, int(state["unmatched_leg_drops"]))
            emission_failures = max(0, int(state["emission_failures"]))
        except TypeError, ValueError:
            return
        pending_y = _parse_pending(state["pending_y"])
        pending_x = _parse_pending(state["pending_x"])
        if state["pending_y"] is not None and pending_y is None:
            return
        if state["pending_x"] is not None and pending_x is None:
            return
        if not isinstance(state["last_pair_key"], str):
            return
        last_pair_key = state["last_pair_key"]
        if paired_history and last_pair_key != paired_history[-1][0]:
            return

        self._kalman = kalman
        self._paired_history = deque(paired_history, maxlen=self._paired_history.maxlen)
        self._log_ys = deque((item[1] for item in paired_history), maxlen=self._log_ys.maxlen)
        self._log_xs = deque((item[2] for item in paired_history), maxlen=self._log_xs.maxlen)
        self._mode = mode
        self._bars_held = bars_held
        self._last_pair_key = last_pair_key
        self._pending_y = pending_y
        self._pending_x = pending_x
        self._unmatched_leg_drops = unmatched_leg_drops
        self._emission_failures = emission_failures

    # ----------------------------------------------------------------- events

    def calculate_signals_window(self, event: Any, aggregator: Any = None) -> None:
        _ = aggregator
        for symbol in _event_symbols(event, self.symbol_list):
            snapshot = _window_snapshot(event, symbol)
            if snapshot is None:
                continue
            self._process(symbol, snapshot, time_key(snapshot.time), snapshot.time)

    def calculate_signals(self, event: Any) -> None:
        event_type = str(getattr(event, "type", "")).upper()
        if event_type == "MARKET_WINDOW":
            self.calculate_signals_window(event)
            return
        if event_type != "MARKET":
            return
        snapshot = _market_snapshot(event)
        if snapshot is not None:
            symbol = str(getattr(event, "symbol", ""))
            self._process(symbol, snapshot, time_key(snapshot.time), snapshot.time)

    # ------------------------------------------------------------------- core

    def _process(self, symbol: str, snapshot: _Snapshot, key: str, event_time: Any) -> None:
        if not self.enabled:
            return
        close = safe_float(snapshot.close)
        if close is None or close <= 0.0 or not key:
            return
        if symbol == self.symbol_y:
            self._pending_y = (key, close)
        elif symbol == self.symbol_x:
            self._pending_x = (key, close)
        else:
            return
        pending_y, pending_x = self._pending_y, self._pending_x
        if pending_y is None or pending_x is None:
            return
        if pending_y[0] != pending_x[0]:
            self._unmatched_leg_drops += 1
            if symbol == self.symbol_y:
                self._pending_x = None
            else:
                self._pending_y = None
            return
        if pending_y[0] == self._last_pair_key:
            return
        self._last_pair_key = pending_y[0]
        self._step(event_time, pending_y[1], pending_x[1], pending_y[0])

    def _step(self, event_time: Any, close_y: float, close_x: float, pair_key: str) -> None:
        log_y, log_x = math.log(close_y), math.log(close_x)
        state = kalman_hedge_ratio_step(
            self._kalman,
            log_y,
            log_x,
            delta=self.kalman_delta,
            obs_noise=self.kalman_obs_noise,
            init_var=1.0,
        )
        if state is None:
            return
        self._kalman = state
        self._log_ys.append(log_y)
        self._log_xs.append(log_x)
        self._paired_history.append((pair_key, log_y, log_x))
        z = self._signal_z(state)
        if self._mode != _FLAT:
            self._manage_open(event_time, close_y, close_x, state, z, self._bars_held + 1)
            return
        if z is not None:
            self._maybe_enter(event_time, close_y, close_x, state, z)

    def _residuals(self, window: int) -> list[float]:
        """Engle-Granger residuals: TODAY's hedge applied to the trailing prices.

        ``[log_y_tau - beta_t*log_x_tau - alpha_t]`` over the last ``window``
        aligned prints.  This is the series every gate reads -- the filter's own
        a-posteriori residual is white noise by construction and would make the
        ADF gate and the half-life cap vacuous.
        """
        state = self._kalman
        if state is None:
            return []
        span = max(1, int(window))
        log_ys = list(self._log_ys)[-span:]
        log_xs = list(self._log_xs)[-span:]
        beta, alpha = float(state.beta), float(state.alpha)
        return [log_y - beta * log_x - alpha for log_y, log_x in zip(log_ys, log_xs)]

    def _signal_z(self, state: KalmanHedgeState) -> float | None:
        if self.signal_mode == "spread_z":
            return zscore(self._residuals(self.z_window), window=self.z_window)
        return state.innovation_z

    def _adf_pass(self) -> bool | None:
        """Evaluate the preregistered, uncalibrated residual-ADF heuristic."""
        if len(self._log_ys) < self.adf_window:
            return None
        statistic = adf_t_statistic(self._residuals(self.adf_window), lags=1)
        critical = adf_critical_value("5%")
        if statistic is None or critical is None:
            return None
        return statistic <= critical

    def _half_life_cap(self) -> int | None:
        if self.half_life_multiple <= 0.0:
            return None
        half_life = ar1_half_life(self._residuals(self.adf_window))
        if half_life is None:
            return None
        cap = round(self.half_life_multiple * half_life)
        return max(1, min(self.max_hold_bars, cap))

    def _maybe_enter(
        self,
        event_time: Any,
        close_y: float,
        close_x: float,
        state: KalmanHedgeState,
        z: float,
    ) -> None:
        if state.updates < self.min_updates or state.beta <= self.min_beta:
            return
        if abs(z) < self.entry_z:
            return
        if self.require_cointegration and self._adf_pass() is not True:
            return
        x_allocation = min(self.max_leg_allocation, self.leg_allocation * state.beta)
        if self.leg_allocation <= 0.0 or x_allocation <= 0.0:
            return
        if z >= self.entry_z:
            mode, y_side, x_side = _SHORT_SPREAD, "SHORT", "LONG"
        else:
            mode, y_side, x_side = _LONG_SPREAD, "LONG", "SHORT"
        emissions = (
            ("y", self.symbol_y, y_side, close_y, self.leg_allocation),
            ("x", self.symbol_x, x_side, close_x, x_allocation),
        )
        for leg, symbol, side, price, allocation in emissions:
            emitted = self._emit_leg(
                self.events,
                strategy_id=_STRATEGY_ID,
                symbol=symbol,
                event_time=event_time,
                signal_type=side,
                strength=allocation,
                price=price,
                metadata=_target_metadata(
                    strategy=self.__class__.__name__,
                    target_allocation=allocation,
                    max_order_value=self.max_order_value,
                    beta=float(state.beta),
                    alpha=float(state.alpha),
                    z=float(z),
                    mode=mode,
                    signal_mode=self.signal_mode,
                    cointegration_gate="uncalibrated_residual_adf_heuristic",
                    leg=leg,
                    pair=self.pair_id,
                ),
            )
            if not emitted:
                return
        self._mode = mode
        self._bars_held = 0

    def _manage_open(
        self,
        event_time: Any,
        close_y: float,
        close_x: float,
        state: KalmanHedgeState,
        z: float | None,
        bars_held: int,
    ) -> None:
        reason = ""
        if z is not None:
            flipped = (self._mode == _SHORT_SPREAD and z < 0.0) or (
                self._mode == _LONG_SPREAD and z > 0.0
            )
            if abs(z) <= self.exit_z or flipped:
                reason = "reversion"
            elif abs(z) >= self.stop_z:
                reason = "stop"
        if not reason and self.exit_on_cointegration_break and self._adf_pass() is False:
            reason = "coint_break"
        if not reason:
            cap = self._half_life_cap()
            if cap is not None and bars_held >= cap:
                reason = "half_life_cap"
            elif bars_held >= self.max_hold_bars:
                reason = "max_hold"
        if not reason:
            self._bars_held = bars_held
            return
        # ponytail: EXIT closes the whole leg -- the portfolio has no partial exits,
        # so a scale-out at the half-life cap is collapsed into one full close.
        for leg, symbol, price in (
            ("y", self.symbol_y, close_y),
            ("x", self.symbol_x, close_x),
        ):
            emitted = self._emit_leg(
                self.events,
                strategy_id=_STRATEGY_ID,
                symbol=symbol,
                event_time=event_time,
                signal_type="EXIT",
                price=price,
                metadata={
                    "strategy": self.__class__.__name__,
                    "reason": reason,
                    "beta": float(state.beta),
                    "alpha": float(state.alpha),
                    "z": None if z is None else float(z),
                    "mode": self._mode,
                    "signal_mode": self.signal_mode,
                    "cointegration_gate": "uncalibrated_residual_adf_heuristic",
                    "leg": leg,
                    "pair": self.pair_id,
                    "bars_held": int(bars_held),
                },
            )
            if not emitted:
                return
        self._mode = _FLAT
        self._bars_held = 0

    def _emit_leg(self, *args: Any, **kwargs: Any) -> bool:
        """Queue one leg and record failures without committing pair state."""
        try:
            _emit(*args, **kwargs)
        except Exception:
            self._emission_failures += 1
            return False
        return True


def _parse_pending(payload: Any) -> tuple[str, float] | None:
    if not isinstance(payload, (list, tuple)) or len(payload) != 2:
        return None
    close = safe_float(payload[1])
    key = payload[0]
    return (
        None
        if close is None or close <= 0.0 or not isinstance(key, str) or not key
        else (key, close)
    )


def _parse_paired_history(
    payload: Any, maxlen: int | None
) -> list[tuple[str, float, float]] | None:
    if not isinstance(payload, list):
        return None
    parsed: list[tuple[str, float, float]] = []
    for item in payload:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            return None
        stamp = item[0]
        log_y, log_x = safe_float(item[1]), safe_float(item[2])
        if not isinstance(stamp, str) or not stamp or log_y is None or log_x is None:
            return None
        parsed.append((stamp, log_y, log_x))
    keep = int(maxlen or 0)
    return parsed[-keep:] if keep > 0 else []


__all__ = ["KalmanPairsStatArbStrategy"]

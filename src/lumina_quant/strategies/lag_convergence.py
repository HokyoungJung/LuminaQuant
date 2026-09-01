"""Two-asset lag-convergence strategy using relative momentum spread.

The legacy book assumes relative-momentum mean reversion with **no** spread
stationarity/cointegration test and a fixed 1:1 notional per leg.  Two config-
gated, default-OFF improvements are available (byte-identical when OFF):

* ``require_cointegration`` — before taking a convergence trade, require the
  log-price spread ``log(X) - beta*log(Y)`` (with ``beta`` the trailing OLS
  hedge ratio) to be stationary by an ADF t-statistic gate.  A momentum spread
  can diverge indefinitely when the two legs are **not** cointegrated; this gate
  refuses entries on non-mean-reverting pairs.
* ``beta_neutral_sizing`` — size the two legs by inverse realized volatility
  (risk/beta-neutral) instead of 1:1, so a common market move nets out and the
  position expresses the *relative* view rather than a levered directional bet.

Both flags default OFF; with both OFF the emitted signals (types, order,
metadata and unit ``strength``) are byte-identical to the legacy behavior.
"""

from __future__ import annotations

import math
from collections import deque

from lumina_quant.core.events import SignalEvent
from lumina_quant.indicators.common import safe_float
from lumina_quant.indicators.momentum import momentum_return, momentum_spread
from lumina_quant.indicators.stationarity import adf_t_statistic
from lumina_quant.strategy import Strategy
from lumina_quant.tuning import HyperParam, resolve_params_from_schema


class LagConvergenceStrategy(Strategy):
    """Pair strategy that trades convergence of lagged momentum spread."""

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "symbol_x": HyperParam.string("symbol_x", default="", tunable=False),
            "symbol_y": HyperParam.string("symbol_y", default="", tunable=False),
            "lag_bars": HyperParam.integer(
                "lag_bars",
                default=3,
                low=1,
                high=2048,
                optuna={"type": "int", "low": 1, "high": 12},
                grid=[1, 2, 3, 5, 8],
            ),
            "entry_threshold": HyperParam.floating(
                "entry_threshold",
                default=0.015,
                low=0.001,
                high=1.0,
                optuna={"type": "float", "low": 0.004, "high": 0.05, "step": 0.001},
                grid=[0.008, 0.012, 0.015, 0.02, 0.03],
            ),
            "exit_threshold": HyperParam.floating(
                "exit_threshold",
                default=0.004,
                low=0.0,
                high=1.0,
                optuna={"type": "float", "low": 0.001, "high": 0.02, "step": 0.001},
                grid=[0.002, 0.004, 0.006, 0.01],
            ),
            "stop_threshold": HyperParam.floating(
                "stop_threshold",
                default=0.05,
                low=0.001,
                high=2.0,
                optuna={"type": "float", "low": 0.01, "high": 0.12, "step": 0.002},
                grid=[0.03, 0.05, 0.08],
            ),
            "max_hold_bars": HyperParam.integer(
                "max_hold_bars",
                default=96,
                low=1,
                high=10000,
                optuna={"type": "int", "low": 12, "high": 240},
                grid=[24, 48, 96, 160],
            ),
            "stop_loss_pct": HyperParam.floating(
                "stop_loss_pct",
                default=0.03,
                low=0.001,
                high=0.5,
                optuna={"type": "float", "low": 0.005, "high": 0.08, "step": 0.005},
                grid=[0.01, 0.02, 0.03, 0.04],
            ),
        }

    def __init__(
        self,
        bars,
        events,
        symbol_x=None,
        symbol_y=None,
        lag_bars=3,
        entry_threshold=0.015,
        exit_threshold=0.004,
        stop_threshold=0.05,
        max_hold_bars=96,
        stop_loss_pct=0.03,
        require_cointegration=False,
        beta_neutral_sizing=False,
        coint_window=96,
        coint_max_tstat=-2.0,
        beta_window=96,
    ):
        self.bars = bars
        self.events = events
        self.symbol_list = list(self.bars.symbol_list)
        if len(self.symbol_list) < 2:
            raise ValueError("LagConvergenceStrategy requires at least two symbols.")

        resolved = resolve_params_from_schema(
            self.get_param_schema(),
            {
                "symbol_x": symbol_x,
                "symbol_y": symbol_y,
                "lag_bars": lag_bars,
                "entry_threshold": entry_threshold,
                "exit_threshold": exit_threshold,
                "stop_threshold": stop_threshold,
                "max_hold_bars": max_hold_bars,
                "stop_loss_pct": stop_loss_pct,
            },
            keep_unknown=False,
        )
        symbol_x = resolved["symbol_x"]
        symbol_y = resolved["symbol_y"]
        lag_bars = resolved["lag_bars"]
        entry_threshold = resolved["entry_threshold"]
        exit_threshold = resolved["exit_threshold"]
        stop_threshold = resolved["stop_threshold"]
        max_hold_bars = resolved["max_hold_bars"]
        stop_loss_pct = resolved["stop_loss_pct"]

        self.symbol_x = str(symbol_x) if symbol_x else str(self.symbol_list[0])
        self.symbol_y = str(symbol_y) if symbol_y else str(self.symbol_list[1])
        if self.symbol_x == self.symbol_y:
            raise ValueError("symbol_x and symbol_y must be different.")

        self.lag_bars = max(1, int(lag_bars))
        self.entry_threshold = float(entry_threshold)
        self.exit_threshold = max(0.0, float(exit_threshold))
        self.stop_threshold = max(self.entry_threshold + 1e-9, float(stop_threshold))
        self.max_hold_bars = max(1, int(max_hold_bars))
        self.stop_loss_pct = float(stop_loss_pct)
        # STRATEGY-IMPROVE (strategy-local, config-gated, default OFF => byte-
        # identical to the legacy fixed-1:1 / no-stationarity book).
        self.require_cointegration = bool(require_cointegration)
        self.beta_neutral_sizing = bool(beta_neutral_sizing)
        self.coint_window = max(8, int(coint_window))
        self.coint_max_tstat = float(coint_max_tstat)
        self.beta_window = max(8, int(beta_window))

        history_len = max(16, self.lag_bars + 8)
        # Retain enough history for the opt-in stationarity/sizing windows. When
        # both flags are OFF this is exactly ``max(16, lag_bars + 8)`` as before.
        if self.require_cointegration or self.beta_neutral_sizing:
            history_len = max(history_len, self.coint_window + 2, self.beta_window + 2)
        self._x_history = deque(maxlen=history_len)
        self._y_history = deque(maxlen=history_len)

        self._mode = "OUT"
        self._bars_in_position = 0
        self._last_pair_time_key = ""
        self._last_spread = None

    def get_state(self):
        return {
            "x_history": list(self._x_history),
            "y_history": list(self._y_history),
            "mode": self._mode,
            "bars_in_position": int(self._bars_in_position),
            "last_pair_time_key": str(self._last_pair_time_key),
            "last_spread": self._last_spread,
        }

    def set_state(self, state):
        if not isinstance(state, dict):
            return

        self._x_history.clear()
        self._y_history.clear()

        x_maxlen = int(self._x_history.maxlen) if self._x_history.maxlen is not None else 0
        y_maxlen = int(self._y_history.maxlen) if self._y_history.maxlen is not None else 0

        for value in list(state.get("x_history") or [])[-x_maxlen:]:
            parsed = safe_float(value)
            if parsed is not None and parsed > 0.0:
                self._x_history.append(parsed)

        for value in list(state.get("y_history") or [])[-y_maxlen:]:
            parsed = safe_float(value)
            if parsed is not None and parsed > 0.0:
                self._y_history.append(parsed)

        mode = str(state.get("mode", "OUT")).upper()
        self._mode = mode if mode in {"OUT", "LONG_X_SHORT_Y", "SHORT_X_LONG_Y"} else "OUT"
        try:
            self._bars_in_position = max(0, int(state.get("bars_in_position", 0)))
        except Exception:
            self._bars_in_position = 0
        self._last_pair_time_key = str(state.get("last_pair_time_key", ""))
        self._last_spread = safe_float(state.get("last_spread"))

    def _aligned_pair_timestamp(self):
        tx = self.bars.get_latest_bar_datetime(self.symbol_x)
        ty = self.bars.get_latest_bar_datetime(self.symbol_y)
        if tx is None or ty is None or tx != ty:
            return None
        return tx

    def _resolve_pair_closes(self):
        close_x = safe_float(self.bars.get_latest_bar_value(self.symbol_x, "close"))
        close_y = safe_float(self.bars.get_latest_bar_value(self.symbol_y, "close"))
        if close_x is None or close_y is None or close_x <= 0.0 or close_y <= 0.0:
            return None, None
        return close_x, close_y

    def _emit(self, symbol, event_time, signal_type, metadata, stop_loss=None, strength=1.0):
        self.events.put(
            SignalEvent(
                strategy_id="lag_convergence",
                symbol=symbol,
                datetime=event_time,
                signal_type=signal_type,
                strength=float(strength),
                stop_loss=stop_loss,
                metadata=metadata,
            )
        )

    def _leg_vol(self, hist):
        """Trailing realized log-return volatility over ``beta_window`` returns."""
        vals = [float(value) for value in list(hist)[-(self.beta_window + 1) :] if value > 0.0]
        if len(vals) < self.beta_window + 1:
            return None
        rets = [math.log(vals[idx] / vals[idx - 1]) for idx in range(1, len(vals))]
        if len(rets) < 2:
            return None
        mean_r = sum(rets) / len(rets)
        var = sum((value - mean_r) ** 2 for value in rets) / (len(rets) - 1)
        if var <= 0.0:
            return None
        return math.sqrt(var)

    def _leg_strengths(self):
        """Per-leg ``(strength_x, strength_y)``.

        Legacy (flag OFF) is a fixed 1:1 book: ``(1.0, 1.0)``.  With
        ``beta_neutral_sizing`` ON the legs are sized inversely to their realized
        volatility (risk/beta-neutral), normalized so the dominant leg is 1.0.
        Falls back to 1:1 when either leg's vol is unavailable/degenerate.
        """
        if not self.beta_neutral_sizing:
            return 1.0, 1.0
        vol_x = self._leg_vol(self._x_history)
        vol_y = self._leg_vol(self._y_history)
        if vol_x is None or vol_y is None or vol_x <= 0.0 or vol_y <= 0.0:
            return 1.0, 1.0
        inv_x = 1.0 / vol_x
        inv_y = 1.0 / vol_y
        norm = max(inv_x, inv_y)
        if norm <= 0.0:
            return 1.0, 1.0
        return inv_x / norm, inv_y / norm

    def _hedge_beta(self, window):
        """Trailing OLS cointegrating slope ``beta`` in ``log(X) = a + beta*log(Y)``."""
        n = min(len(self._x_history), len(self._y_history))
        if n < window:
            return None
        xs = [math.log(float(value)) for value in list(self._x_history)[-window:]]
        ys = [math.log(float(value)) for value in list(self._y_history)[-window:]]
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        cov = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(len(xs)))
        var = sum((ys[i] - mean_y) ** 2 for i in range(len(ys)))
        if var <= 1e-12:
            return None
        beta = cov / var
        return beta if math.isfinite(beta) else None

    def _spread_is_stationary(self):
        """ADF stationarity gate on the beta-adjusted log-price spread."""
        window = self.coint_window
        n = min(len(self._x_history), len(self._y_history))
        if n < window:
            return False
        beta = self._hedge_beta(window)
        if beta is None:
            return False
        xs = [math.log(float(value)) for value in list(self._x_history)[-window:]]
        ys = [math.log(float(value)) for value in list(self._y_history)[-window:]]
        spread = [xs[i] - beta * ys[i] for i in range(len(xs))]
        tstat = adf_t_statistic(spread, lags=1)
        return tstat is not None and tstat <= self.coint_max_tstat

    def _emit_entry(self, event_time, spread, close_x, close_y, mode):
        metadata = {
            "strategy": "LagConvergenceStrategy",
            "mode": mode,
            "spread": float(spread),
            "lag_bars": int(self.lag_bars),
            "entry_threshold": float(self.entry_threshold),
        }
        strength_x, strength_y = self._leg_strengths()

        if mode == "LONG_X_SHORT_Y":
            self._emit(
                self.symbol_x,
                event_time,
                "LONG",
                metadata,
                stop_loss=close_x * (1.0 - self.stop_loss_pct),
                strength=strength_x,
            )
            self._emit(
                self.symbol_y,
                event_time,
                "SHORT",
                metadata,
                stop_loss=close_y * (1.0 + self.stop_loss_pct),
                strength=strength_y,
            )
        else:
            self._emit(
                self.symbol_x,
                event_time,
                "SHORT",
                metadata,
                stop_loss=close_x * (1.0 + self.stop_loss_pct),
                strength=strength_x,
            )
            self._emit(
                self.symbol_y,
                event_time,
                "LONG",
                metadata,
                stop_loss=close_y * (1.0 - self.stop_loss_pct),
                strength=strength_y,
            )

    def _emit_exit(self, event_time, spread, reason):
        metadata = {
            "strategy": "LagConvergenceStrategy",
            "mode": self._mode,
            "spread": float(spread),
            "reason": reason,
        }
        self._emit(self.symbol_x, event_time, "EXIT", metadata)
        self._emit(self.symbol_y, event_time, "EXIT", metadata)

    def calculate_signals(self, event):
        if getattr(event, "type", None) != "MARKET":
            return
        if getattr(event, "symbol", None) not in {self.symbol_x, self.symbol_y}:
            return

        pair_time = self._aligned_pair_timestamp()
        if pair_time is None:
            return
        time_key = str(pair_time)
        if time_key == self._last_pair_time_key:
            return
        self._last_pair_time_key = time_key

        close_x, close_y = self._resolve_pair_closes()
        if close_x is None or close_y is None:
            return

        self._x_history.append(close_x)
        self._y_history.append(close_y)
        if len(self._x_history) <= self.lag_bars or len(self._y_history) <= self.lag_bars:
            return

        base_x = self._x_history[-1 - self.lag_bars]
        base_y = self._y_history[-1 - self.lag_bars]
        if base_x <= 0.0 or base_y <= 0.0:
            return

        momentum_x = momentum_return(close_x, base_x)
        momentum_y = momentum_return(close_y, base_y)
        if momentum_x is None or momentum_y is None:
            return
        spread = momentum_spread(momentum_x, momentum_y)
        self._last_spread = float(spread)

        if self._mode == "OUT":
            # Opt-in cointegration/stationarity gate: refuse convergence entries
            # when the beta-adjusted spread is not mean-reverting. Short-circuits
            # to ``True`` when the flag is OFF (byte-identical legacy path).
            entry_ok = (not self.require_cointegration) or self._spread_is_stationary()
            if entry_ok and spread <= -self.entry_threshold:
                self._emit_entry(pair_time, spread, close_x, close_y, "LONG_X_SHORT_Y")
                self._mode = "LONG_X_SHORT_Y"
                self._bars_in_position = 0
            elif entry_ok and spread >= self.entry_threshold:
                self._emit_entry(pair_time, spread, close_x, close_y, "SHORT_X_LONG_Y")
                self._mode = "SHORT_X_LONG_Y"
                self._bars_in_position = 0
            return

        self._bars_in_position += 1
        reason = None
        if abs(spread) <= self.exit_threshold:
            reason = "converged"
        elif abs(spread) >= self.stop_threshold:
            reason = "spread_stop"
        elif self._bars_in_position >= self.max_hold_bars:
            reason = "max_hold"

        if reason is None:
            return

        self._emit_exit(pair_time, spread, reason)
        self._mode = "OUT"
        self._bars_in_position = 0

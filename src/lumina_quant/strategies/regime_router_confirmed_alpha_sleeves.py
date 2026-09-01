"""Vol/cycle-confirmed regime router (crypto basket, OHLCV-only, research sleeve).

THEORY: regime-switching models (Ang & Bekaert 2002) posit that expected
returns, volatility, and correlation all shift discretely across a small
number of latent states, and that conditioning trades on the CURRENT state
(rather than reacting to every raw signal wiggle) is what avoids repeated
whipsaw losses in the transition zones between states.  A pure breadth+
benchmark vote is a noisy point estimate of that latent state: in a choppy
tape it can cross a directional threshold for a bar or two on noise alone.
This sleeve adds a SECOND, independent read of the regime -- conditional
volatility from a GARCH(1,1) fit (``indicators/garch.py``) and, as a
fallback, dominant-cycle phase from a causal periodogram
(``indicators/spectral_cycle.py``) -- and only allows a directional flip
when the two reads CONCUR.  Rising conditional volatility corroborates a
bear regime (vol clusters around drawdowns); falling/normal conditional
volatility corroborates a bull regime (vol tends to compress in orderly
uptrends).  This is the standard qualitative link between the GARCH
volatility state and the return regime used in regime-switching-GARCH
literature following Ang & Bekaert; no separate return forecast is needed
here because the breadth+benchmark vote already supplies the directional
call, and the GARCH/cycle read only supplies the CONFIRMATION.

DISTINCT FROM existing regime sleeves:
- ``BullBearRegimeRotationStrategy`` (bull_bear_regime_rotation.py) uses the
  SAME breadth+benchmark vote but acts on it immediately with no
  confirmation gate -- it is this sleeve's un-confirmed parent, and the
  whole point of this module is to diverge from it on exactly the inputs
  where the vote is unconfirmed (see the non-redundancy test).
- ``HurstRegimeGatedStrategy`` (robust_meta_overlays.py) gates a trend vs a
  mean-reversion CHILD STRATEGY per symbol by a rolling Hurst exponent; it
  is a per-symbol overlay dispatcher, not a basket-wide directional router.
- ``BreadthRegimeTrendTimerStrategy`` (vol_term_breadth_alpha_sleeves.py)
  times TOTAL EXPOSURE (long-only scaling) off breadth; it does not take
  short positions and has no vol/cycle confirmation.
This sleeve's novelty is specifically the CONFIRMATION REQUIREMENT (a
GARCH conditional-vol regime, or a spectral cycle-phase fallback, must
CONCUR with the breadth+benchmark vote before a directional flip is taken)
combined with 3-state hysteresis (bull-long / bear-short / chop-flat, with
separate entry and exit thresholds so a state, once entered, is sticky).

SIGNAL:
1. Base vote (identical shape to the parent): per-symbol momentum
   (``simple_return`` over ``momentum_lookback``) + trend-MA agreement gate
   cross-sectional BREADTH (fraction of eligible symbols with a
   trend-confirmed up/down score past ``signal_threshold``), PLUS a
   benchmark (BTC by default) return over ``benchmark_lookback`` that must
   agree in sign and clear its own threshold.  Base bull vote = up-breadth
   >= ``bull_breadth`` AND benchmark return >= ``benchmark_bull_threshold``.
   Base bear vote = down-breadth >= ``bear_breadth`` AND benchmark return
   <= ``-benchmark_bear_threshold`` (only if ``allow_short``).
2. Confirmation (computed on the benchmark's own trailing closes only):
   - GARCH: a GARCH(1,1) is fit (deterministic variance-targeted grid
     search, ``garch11_fit``) over the trailing ``garch_window`` benchmark
     simple returns, refit every ``garch_refit_bars`` decision bars (or
     immediately if no fit is cached yet).  Each bar the one-step-ahead
     variance forecast (``garch11_next_variance``) is compared against the
     fitted long-run variance ``omega / (1 - alpha - beta)``; the ratio
     >= ``confirm_bear_vol_ratio`` reads as a RISING-vol regime, <=
     ``confirm_bull_vol_ratio`` reads as a FALLING/normal-vol regime,
     otherwise AMBIGUOUS.  Fit failure (insufficient/degenerate history)
     reads as UNAVAILABLE.
   - Spectral fallback: only consulted when the GARCH read is UNAVAILABLE.
     ``dominant_cycle`` over the trailing ``cycle_window`` log benchmark
     closes yields a phase; when its purity clears ``min_cycle_purity`` the
     phase fraction in the rising half ``[0.5, 1.0)`` reads as a
     cycle-confirmed bull, the falling half ``[0.0, 0.5)`` as a
     cycle-confirmed bear.
   - ``bear_confirmed`` = GARCH RISING, OR (GARCH UNAVAILABLE AND cycle
     falling-confirmed).  ``bull_confirmed`` = GARCH FALLING, OR (GARCH
     UNAVAILABLE AND cycle rising-confirmed).  An AMBIGUOUS GARCH read
     confirms neither side and is NOT overridden by the spectral fallback
     (GARCH, once available, is the primary read).
3. Flip rule: the base bull vote enters BULL only if ``bull_confirmed``;
   the base bear vote enters BEAR only if ``bear_confirmed`` and
   ``allow_short``.  An unconfirmed vote does not flip the state -- the
   router stays in (or falls back to) CHOP-flat instead of chasing the
   unconfirmed vote, which is exactly the chop-whipsaw case the unconfirmed
   parent is prone to.
4. Hysteresis (3-state: BULL / BEAR / CHOP): once a directional state is
   entered it persists (no re-confirmation required) while its breadth
   stays above ``exit_breadth`` and the benchmark does not contradict it;
   otherwise the state decays to CHOP.  Entry and exit thresholds are
   deliberately separated (``bull_breadth``/``bear_breadth`` for entry vs
   the looser ``exit_breadth`` for persistence) so a state cannot flap on a
   single noisy bar.
5. Sizing: gross exposure is breadth-scaled (as in the parent) and then
   additionally VOL-SCALED down when the GARCH forecast/long-run-variance
   ratio is elevated (``min(1.0, 1.0 / ratio)``, floored at
   ``min_vol_size_scale``) so a confirmed-but-still-turbulent regime is
   traded at reduced size.

Research sleeve only: OHLCV-only, no-lookahead (all reads are on CLOSED
trailing bars), never-raise, deterministic, and pure Python.  Promotion
still requires cost-realistic walk-forward/shadow validation on the
data-bearing machine; 0% real allocation until sign-off.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

from lumina_quant.core.plugin_registry import register
from lumina_quant.indicators.alpha_features import simple_return
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.indicators.garch import garch11_fit, garch11_next_variance
from lumina_quant.indicators.moving_average import simple_moving_average
from lumina_quant.indicators.spectral_cycle import cycle_phase_fraction, dominant_cycle
from lumina_quant.strategies.adaptive_crypto_alpha_sleeves import (
    _age_cross_positions,
    _emit_rebalance_targets,
    _state_size,
)
from lumina_quant.strategies.external_alpha_sleeves import (
    _Snapshot,
    _event_symbols,
    _market_snapshot,
    _safe_non_negative_int,
    _window_snapshot,
)
from lumina_quant.strategies.robust_alpha_sleeves import (
    _CrossSectionalState,
    _mode,
    _restore_deque,
)
from lumina_quant.strategy import Strategy
from lumina_quant.tuning import HyperParam, resolve_params_from_schema

_REGIMES = {"BULL", "BEAR", "CHOP"}
_VOL_STATES = {"RISING", "FALLING", "AMBIGUOUS", "UNAVAILABLE"}
_CYCLE_STATES = {"RISING", "FALLING", "AMBIGUOUS", "UNAVAILABLE"}


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


def _trailing_simple_returns(closes: list[float], window: int) -> list[float] | None:
    """Return up to ``window`` trailing bar-over-bar simple returns, oldest first."""
    win = max(1, int(window))
    tail = closes[-(win + 1) :]
    if len(tail) < 2:
        return None
    returns: list[float] = []
    for idx in range(1, len(tail)):
        prev = tail[idx - 1]
        if prev == 0.0 or not math.isfinite(prev):
            return None
        returns.append(float(tail[idx] / prev - 1.0))
    if not all(math.isfinite(value) for value in returns):
        return None
    return returns


def _trailing_log_closes(closes: list[float], window: int) -> list[float] | None:
    """Return up to ``window`` trailing log-closes, oldest first."""
    win = max(2, int(window))
    tail = closes[-win:]
    if len(tail) < 2 or any((value is None or value <= 0.0) for value in tail):
        return None
    logs = [math.log(value) for value in tail]
    if not all(math.isfinite(value) for value in logs):
        return None
    return logs


@register("strategy", "RegimeRouterConfirmedRotationStrategy", interface="event_driven")
class RegimeRouterConfirmedRotationStrategy(Strategy):
    """Breadth+benchmark regime router gated by GARCH/cycle confirmation.

    See module docstring for the full theory, signal, and hysteresis spec.
    Distinct from ``BullBearRegimeRotationStrategy`` by requiring vol/cycle
    confirmation before a directional flip; distinct from
    ``HurstRegimeGatedStrategy``/``BreadthRegimeTrendTimerStrategy`` by being
    a directly-traded long/short basket router rather than a child-strategy
    overlay or a long-only exposure timer.
    """

    decision_cadence_seconds = 1800
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False
    strategy_name = "RegimeRouterConfirmedRotationStrategy"
    strategy_id = "regime_router_confirmed_rotation"

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "momentum_lookback": HyperParam.integer(
                "momentum_lookback", default=48, low=3, high=20000
            ),
            "trend_ma_window": HyperParam.integer("trend_ma_window", default=48, low=3, high=20000),
            "signal_threshold": HyperParam.floating(
                "signal_threshold", default=0.015, low=0.0, high=1.0
            ),
            "bull_breadth": HyperParam.floating("bull_breadth", default=0.58, low=0.0, high=1.0),
            "bear_breadth": HyperParam.floating("bear_breadth", default=0.55, low=0.0, high=1.0),
            "exit_breadth": HyperParam.floating("exit_breadth", default=0.42, low=0.0, high=1.0),
            "benchmark_symbol": HyperParam.string(
                "benchmark_symbol", default="BTC/USDT", tunable=False
            ),
            "benchmark_lookback": HyperParam.integer(
                "benchmark_lookback", default=48, low=3, high=20000
            ),
            "benchmark_bull_threshold": HyperParam.floating(
                "benchmark_bull_threshold", default=0.005, low=0.0, high=1.0
            ),
            "benchmark_bear_threshold": HyperParam.floating(
                "benchmark_bear_threshold", default=0.005, low=0.0, high=1.0
            ),
            "garch_window": HyperParam.integer("garch_window", default=64, low=32, high=20000),
            "garch_refit_bars": HyperParam.integer(
                "garch_refit_bars", default=12, low=1, high=5000
            ),
            "confirm_bear_vol_ratio": HyperParam.floating(
                "confirm_bear_vol_ratio", default=1.15, low=1.0, high=10.0
            ),
            "confirm_bull_vol_ratio": HyperParam.floating(
                "confirm_bull_vol_ratio", default=1.00, low=0.0, high=10.0
            ),
            "cycle_window": HyperParam.integer("cycle_window", default=64, low=8, high=20000),
            "cycle_min_period": HyperParam.integer(
                "cycle_min_period", default=6, low=2, high=20000
            ),
            "cycle_max_period": HyperParam.integer(
                "cycle_max_period", default=32, low=2, high=20000
            ),
            "min_cycle_purity": HyperParam.floating(
                "min_cycle_purity", default=0.15, low=0.0, high=1.0
            ),
            "min_vol_size_scale": HyperParam.floating(
                "min_vol_size_scale", default=0.35, low=0.0, high=1.0
            ),
            "max_longs": HyperParam.integer("max_longs", default=8, low=0, high=256),
            "max_shorts": HyperParam.integer("max_shorts", default=6, low=0, high=256),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
            "max_gross": HyperParam.floating("max_gross", default=1.00, low=0.0, high=5.0),
            "rebalance_bars": HyperParam.integer("rebalance_bars", default=3, low=1, high=10080),
            "stop_loss_pct": HyperParam.floating("stop_loss_pct", default=0.10, low=0.0, high=0.50),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=180, low=1, high=200000),
            "min_symbols": HyperParam.integer("min_symbols", default=5, low=2, high=512),
            "target_allocation": HyperParam.floating(
                "target_allocation", default=0.90, low=0.0, high=5.0, tunable=False
            ),
            "max_order_value": HyperParam.floating(
                "max_order_value", default=750.0, low=0.0, high=1_000_000.0, tunable=False
            ),
            "min_price": HyperParam.floating("min_price", default=0.10, low=0.0, high=1_000_000.0),
        }

    def __init__(self, bars: Any, events: Any, **params: Any) -> None:
        self.bars = bars
        self.events = events
        self.symbol_list = list(getattr(self.bars, "symbol_list", []) or [])
        resolved = resolve_params_from_schema(self.get_param_schema(), params, keep_unknown=False)
        self.momentum_lookback = max(1, int(resolved["momentum_lookback"]))
        self.trend_ma_window = max(2, int(resolved["trend_ma_window"]))
        self.signal_threshold = max(0.0, float(resolved["signal_threshold"]))
        self.bull_breadth = max(0.0, min(1.0, float(resolved["bull_breadth"])))
        self.bear_breadth = max(0.0, min(1.0, float(resolved["bear_breadth"])))
        self.exit_breadth = max(0.0, min(1.0, float(resolved["exit_breadth"])))
        self.benchmark_symbol = self._resolve_benchmark(str(resolved["benchmark_symbol"]))
        self.benchmark_lookback = max(1, int(resolved["benchmark_lookback"]))
        self.benchmark_bull_threshold = max(0.0, float(resolved["benchmark_bull_threshold"]))
        self.benchmark_bear_threshold = max(0.0, float(resolved["benchmark_bear_threshold"]))
        self.garch_window = max(32, int(resolved["garch_window"]))
        self.garch_refit_bars = max(1, int(resolved["garch_refit_bars"]))
        self.confirm_bear_vol_ratio = max(1.0, float(resolved["confirm_bear_vol_ratio"]))
        self.confirm_bull_vol_ratio = max(0.0, float(resolved["confirm_bull_vol_ratio"]))
        # An inverted band (bull threshold above bear threshold) would let a
        # single ratio confirm both sides at once; clamp to keep it sane.
        self.confirm_bull_vol_ratio = min(self.confirm_bull_vol_ratio, self.confirm_bear_vol_ratio)
        self.cycle_window = max(8, int(resolved["cycle_window"]))
        self.cycle_min_period = max(2, int(resolved["cycle_min_period"]))
        self.cycle_max_period = max(self.cycle_min_period, int(resolved["cycle_max_period"]))
        self.min_cycle_purity = max(0.0, min(1.0, float(resolved["min_cycle_purity"])))
        self.min_vol_size_scale = max(0.0, min(1.0, float(resolved["min_vol_size_scale"])))
        self.max_longs = max(0, int(resolved["max_longs"]))
        self.max_shorts = max(0, int(resolved["max_shorts"]))
        self.allow_short = bool(resolved["allow_short"])
        self.max_gross = max(0.0, float(resolved["max_gross"]))
        self.rebalance_bars = max(1, int(resolved["rebalance_bars"]))
        self.stop_loss_pct = max(0.0, float(resolved["stop_loss_pct"]))
        self.max_hold_bars = max(1, int(resolved["max_hold_bars"]))
        self.min_symbols = max(2, int(resolved["min_symbols"]))
        self.target_allocation = max(0.0, float(resolved["target_allocation"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        size = _state_size(
            self.momentum_lookback,
            self.trend_ma_window,
            self.benchmark_lookback,
            self.max_hold_bars,
            self.garch_window + 1,
            self.cycle_window,
        )
        self._state = {
            symbol: _CrossSectionalState(deque(maxlen=size), deque(maxlen=size))
            for symbol in self.symbol_list
        }
        self._last_eval_time_key = ""
        self._tick = 0
        self._regime = "CHOP"
        self._last_up_breadth = 0.0
        self._last_down_breadth = 0.0
        self._last_benchmark_return: float | None = None
        self._garch_omega: float | None = None
        self._garch_alpha: float | None = None
        self._garch_beta: float | None = None
        self._last_vol_state = "UNAVAILABLE"
        self._last_vol_ratio: float | None = None
        self._last_cycle_state = "UNAVAILABLE"
        self._last_cycle_purity: float | None = None

    def _resolve_benchmark(self, preferred: str) -> str:
        if preferred in self.symbol_list:
            return preferred
        for candidate in ("BTC/USDT", "BTCUSDT", "ETH/USDT", "ETHUSDT"):
            if candidate in self.symbol_list:
                return candidate
        return self.symbol_list[0] if self.symbol_list else preferred

    def get_state(self) -> dict[str, Any]:
        return {
            "last_eval_time_key": self._last_eval_time_key,
            "tick": int(self._tick),
            "regime": self._regime,
            "last_up_breadth": float(self._last_up_breadth),
            "last_down_breadth": float(self._last_down_breadth),
            "last_benchmark_return": self._last_benchmark_return,
            "garch_omega": self._garch_omega,
            "garch_alpha": self._garch_alpha,
            "garch_beta": self._garch_beta,
            "last_vol_state": self._last_vol_state,
            "last_vol_ratio": self._last_vol_ratio,
            "last_cycle_state": self._last_cycle_state,
            "last_cycle_purity": self._last_cycle_purity,
            "symbol_state": {symbol: _pack_cross(item) for symbol, item in self._state.items()},
        }

    def set_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        self._last_eval_time_key = str(state.get("last_eval_time_key", ""))
        self._tick = _safe_non_negative_int(state.get("tick"))
        raw_regime = str(state.get("regime", "CHOP")).upper()
        self._regime = raw_regime if raw_regime in _REGIMES else "CHOP"
        up = safe_float(state.get("last_up_breadth"))
        down = safe_float(state.get("last_down_breadth"))
        bench = safe_float(state.get("last_benchmark_return"))
        if up is not None:
            self._last_up_breadth = max(0.0, min(1.0, float(up)))
        if down is not None:
            self._last_down_breadth = max(0.0, min(1.0, float(down)))
        self._last_benchmark_return = bench
        omega = safe_float(state.get("garch_omega"))
        alpha = safe_float(state.get("garch_alpha"))
        beta = safe_float(state.get("garch_beta"))
        if omega is not None and alpha is not None and beta is not None and omega > 0.0:
            self._garch_omega, self._garch_alpha, self._garch_beta = omega, alpha, beta
        else:
            self._garch_omega = None
            self._garch_alpha = None
            self._garch_beta = None
        raw_vol_state = str(state.get("last_vol_state", "UNAVAILABLE")).upper()
        self._last_vol_state = raw_vol_state if raw_vol_state in _VOL_STATES else "UNAVAILABLE"
        self._last_vol_ratio = safe_float(state.get("last_vol_ratio"))
        raw_cycle_state = str(state.get("last_cycle_state", "UNAVAILABLE")).upper()
        self._last_cycle_state = (
            raw_cycle_state if raw_cycle_state in _CYCLE_STATES else "UNAVAILABLE"
        )
        self._last_cycle_purity = safe_float(state.get("last_cycle_purity"))
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

    def _symbol_score(
        self, symbol: str, item: _CrossSectionalState
    ) -> tuple[float | None, dict[str, Any]]:
        closes = list(item.closes)
        ret = simple_return(closes, lookback=self.momentum_lookback)
        ma = simple_moving_average(closes, self.trend_ma_window)
        if ret is None or ma is None or not closes:
            return None, {}
        close = closes[-1]
        if close <= self.min_price:
            return None, {}
        ma_gap = (close / ma) - 1.0 if ma > 0.0 else 0.0
        score = float(ret + 0.25 * ma_gap)
        return score, {
            "raw_momentum_return": float(ret),
            "ma_gap": float(ma_gap),
            "symbol_scope": symbol,
        }

    def _benchmark_return(self) -> float | None:
        item = self._state.get(self.benchmark_symbol)
        if item is None:
            return None
        return simple_return(list(item.closes), lookback=self.benchmark_lookback)

    def _breadth_rows(
        self,
    ) -> tuple[
        float,
        float,
        list[tuple[float, str, dict[str, Any]]],
        list[tuple[float, str, dict[str, Any]]],
    ]:
        eligible = 0
        up_count = 0
        down_count = 0
        up_rows: list[tuple[float, str, dict[str, Any]]] = []
        down_rows: list[tuple[float, str, dict[str, Any]]] = []
        for symbol, item in self._state.items():
            score, meta = self._symbol_score(symbol, item)
            if score is None:
                continue
            eligible += 1
            if score >= self.signal_threshold:
                up_count += 1
                up_rows.append((score, symbol, meta))
            elif score <= -self.signal_threshold:
                down_count += 1
                down_rows.append((score, symbol, meta))
        up_breadth = (up_count / eligible) if eligible else 0.0
        down_breadth = (down_count / eligible) if eligible else 0.0
        up_rows.sort(key=lambda row: row[0], reverse=True)
        down_rows.sort(key=lambda row: row[0])
        return float(up_breadth), float(down_breadth), up_rows, down_rows

    def _refresh_garch(self) -> None:
        item = self._state.get(self.benchmark_symbol)
        if item is None:
            return
        returns = _trailing_simple_returns(list(item.closes), self.garch_window)
        if returns is None:
            return
        due = self._garch_omega is None or (self._tick % self.garch_refit_bars == 0)
        if not due:
            return
        fit = garch11_fit(returns)
        if fit is None:
            return
        omega, alpha, beta = fit
        self._garch_omega, self._garch_alpha, self._garch_beta = omega, alpha, beta

    def _vol_state(self) -> tuple[str, float | None]:
        if self._garch_omega is None or self._garch_alpha is None or self._garch_beta is None:
            return "UNAVAILABLE", None
        item = self._state.get(self.benchmark_symbol)
        if item is None:
            return "UNAVAILABLE", None
        returns = _trailing_simple_returns(list(item.closes), self.garch_window)
        if returns is None:
            return "UNAVAILABLE", None
        next_variance = garch11_next_variance(
            returns, omega=self._garch_omega, alpha=self._garch_alpha, beta=self._garch_beta
        )
        if next_variance is None:
            return "UNAVAILABLE", None
        persistence = self._garch_alpha + self._garch_beta
        if persistence >= 1.0:
            return "UNAVAILABLE", None
        long_run_variance = self._garch_omega / (1.0 - persistence)
        if long_run_variance <= 0.0 or not math.isfinite(long_run_variance):
            return "UNAVAILABLE", None
        ratio = next_variance / long_run_variance
        if not math.isfinite(ratio):
            return "UNAVAILABLE", None
        if ratio >= self.confirm_bear_vol_ratio:
            return "RISING", float(ratio)
        if ratio <= self.confirm_bull_vol_ratio:
            return "FALLING", float(ratio)
        return "AMBIGUOUS", float(ratio)

    def _cycle_state(self) -> tuple[str, float | None]:
        item = self._state.get(self.benchmark_symbol)
        if item is None:
            return "UNAVAILABLE", None
        log_closes = _trailing_log_closes(list(item.closes), self.cycle_window)
        if log_closes is None:
            return "UNAVAILABLE", None
        result = dominant_cycle(
            log_closes, min_period=self.cycle_min_period, max_period=self.cycle_max_period
        )
        if result is None:
            return "UNAVAILABLE", None
        _, _, phase, purity = result
        if purity < self.min_cycle_purity:
            return "AMBIGUOUS", float(purity)
        fraction = cycle_phase_fraction(phase)
        if fraction is None:
            return "AMBIGUOUS", float(purity)
        if 0.5 <= fraction < 1.0:
            return "RISING", float(purity)
        return "FALLING", float(purity)

    def _size_scale(self, vol_ratio: float | None) -> float:
        if vol_ratio is None or vol_ratio <= 0.0 or not math.isfinite(vol_ratio):
            return 1.0
        return max(self.min_vol_size_scale, min(1.0, 1.0 / vol_ratio))

    def _classify_regime(
        self,
        up_breadth: float,
        down_breadth: float,
        benchmark_ret: float | None,
        *,
        bull_confirmed: bool,
        bear_confirmed: bool,
    ) -> str:
        bench = 0.0 if benchmark_ret is None else float(benchmark_ret)
        base_bull_vote = up_breadth >= self.bull_breadth and bench >= self.benchmark_bull_threshold
        base_bear_vote = (
            self.allow_short
            and down_breadth >= self.bear_breadth
            and bench <= -self.benchmark_bear_threshold
        )
        if base_bull_vote and bull_confirmed:
            return "BULL"
        if base_bear_vote and bear_confirmed:
            return "BEAR"
        # Hysteresis: an already-entered directional state persists on
        # breadth decay/benchmark agreement alone -- no re-confirmation is
        # required to STAY in a state, only to ENTER one.
        if self._regime == "BULL" and up_breadth > self.exit_breadth and bench >= 0.0:
            return "BULL"
        if self._regime == "BEAR" and down_breadth > self.exit_breadth and bench <= 0.0:
            return "BEAR"
        return "CHOP"

    def _flatten(self, event_time: Any) -> None:
        _emit_rebalance_targets(
            self.events,
            self._state,
            {},
            event_time=event_time,
            strategy_id=self.strategy_id,
            strategy_name=self.strategy_name,
            target_gross_exposure=0.0,
            max_order_value=self.max_order_value,
            stop_loss_pct=self.stop_loss_pct,
            max_hold_bars=self.max_hold_bars,
            threshold=max(self.signal_threshold, 1e-12),
        )

    def _rebalance(self, event_time: Any) -> None:
        if len(self.symbol_list) < self.min_symbols:
            return
        up_breadth, down_breadth, up_rows, down_rows = self._breadth_rows()
        benchmark_ret = self._benchmark_return()
        self._refresh_garch()
        vol_state, vol_ratio = self._vol_state()
        cycle_state, cycle_purity = self._cycle_state()
        bear_confirmed = vol_state == "RISING" or (
            vol_state == "UNAVAILABLE" and cycle_state == "FALLING"
        )
        bull_confirmed = vol_state == "FALLING" or (
            vol_state == "UNAVAILABLE" and cycle_state == "RISING"
        )
        regime = self._classify_regime(
            up_breadth,
            down_breadth,
            benchmark_ret,
            bull_confirmed=bull_confirmed,
            bear_confirmed=bear_confirmed,
        )
        self._last_up_breadth = up_breadth
        self._last_down_breadth = down_breadth
        self._last_benchmark_return = benchmark_ret
        self._last_vol_state = vol_state
        self._last_vol_ratio = vol_ratio
        self._last_cycle_state = cycle_state
        self._last_cycle_purity = cycle_purity
        self._regime = regime

        if regime == "CHOP":
            self._flatten(event_time)
            return
        if self._tick % self.rebalance_bars:
            _age_cross_positions(
                self.events,
                self._state,
                event_time=event_time,
                strategy_id=self.strategy_id,
                strategy_name=self.strategy_name,
                stop_loss_pct=self.stop_loss_pct,
                max_hold_bars=self.max_hold_bars,
            )
            return

        selected_rows = up_rows if regime == "BULL" else down_rows
        if not selected_rows:
            self._flatten(event_time)
            return

        size_scale = self._size_scale(vol_ratio)
        if regime == "BULL":
            targets = {
                symbol: (
                    "LONG",
                    score,
                    {
                        **meta,
                        "regime": regime,
                        "up_breadth": up_breadth,
                        "down_breadth": down_breadth,
                        "benchmark_return": benchmark_ret,
                        "vol_state": vol_state,
                        "vol_ratio": vol_ratio,
                        "cycle_state": cycle_state,
                    },
                )
                for score, symbol, meta in selected_rows[: self.max_longs]
                if score >= self.signal_threshold
            }
            scale = min(1.0, max(0.0, up_breadth))
        else:
            targets = {
                symbol: (
                    "SHORT",
                    score,
                    {
                        **meta,
                        "regime": regime,
                        "up_breadth": up_breadth,
                        "down_breadth": down_breadth,
                        "benchmark_return": benchmark_ret,
                        "vol_state": vol_state,
                        "vol_ratio": vol_ratio,
                        "cycle_state": cycle_state,
                    },
                )
                for score, symbol, meta in selected_rows[: self.max_shorts]
                if score <= -self.signal_threshold
            }
            scale = min(1.0, max(0.0, down_breadth))
        if not targets:
            self._flatten(event_time)
            return
        gross = (
            self.target_allocation
            * min(self.max_gross, 1.0)
            * max(self.exit_breadth, scale)
            * size_scale
        )
        _emit_rebalance_targets(
            self.events,
            self._state,
            targets,
            event_time=event_time,
            strategy_id=self.strategy_id,
            strategy_name=self.strategy_name,
            target_gross_exposure=gross,
            max_order_value=self.max_order_value,
            stop_loss_pct=self.stop_loss_pct,
            max_hold_bars=self.max_hold_bars,
            threshold=max(self.signal_threshold, 1e-12),
        )


# One parameter variant per >=30m timeframe for the W3 candidate-library
# integrator (thin builder consumes this directly; kept in the lane module
# so candidate_library.py only needs a few wiring lines -- see plan W3).
_REGIME_ROUTER_CONFIRMED_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "30m": (
        {
            "variant": "core",
            "momentum_lookback": 48,
            "trend_ma_window": 48,
            "signal_threshold": 0.015,
            "bull_breadth": 0.58,
            "bear_breadth": 0.55,
            "exit_breadth": 0.42,
            "benchmark_lookback": 48,
            "benchmark_bull_threshold": 0.005,
            "benchmark_bear_threshold": 0.005,
            "garch_window": 64,
            "garch_refit_bars": 12,
            "confirm_bear_vol_ratio": 1.15,
            "confirm_bull_vol_ratio": 1.00,
            "cycle_window": 64,
            "cycle_min_period": 6,
            "cycle_max_period": 32,
            "min_cycle_purity": 0.15,
            "min_vol_size_scale": 0.35,
            "max_longs": 8,
            "max_shorts": 6,
            "max_gross": 1.00,
            "rebalance_bars": 3,
            "stop_loss_pct": 0.10,
            "max_hold_bars": 180,
            "target_allocation": 0.90,
        },
    ),
    "1h": (
        {
            "variant": "core",
            "momentum_lookback": 48,
            "trend_ma_window": 48,
            "signal_threshold": 0.02,
            "bull_breadth": 0.58,
            "bear_breadth": 0.55,
            "exit_breadth": 0.42,
            "benchmark_lookback": 48,
            "benchmark_bull_threshold": 0.008,
            "benchmark_bear_threshold": 0.008,
            "garch_window": 64,
            "garch_refit_bars": 12,
            "confirm_bear_vol_ratio": 1.15,
            "confirm_bull_vol_ratio": 1.00,
            "cycle_window": 64,
            "cycle_min_period": 6,
            "cycle_max_period": 32,
            "min_cycle_purity": 0.15,
            "min_vol_size_scale": 0.35,
            "max_longs": 8,
            "max_shorts": 6,
            "max_gross": 1.00,
            "rebalance_bars": 2,
            "stop_loss_pct": 0.12,
            "max_hold_bars": 180,
            "target_allocation": 0.90,
        },
    ),
    "4h": (
        {
            "variant": "core",
            "momentum_lookback": 42,
            "trend_ma_window": 42,
            "signal_threshold": 0.03,
            "bull_breadth": 0.56,
            "bear_breadth": 0.54,
            "exit_breadth": 0.40,
            "benchmark_lookback": 42,
            "benchmark_bull_threshold": 0.015,
            "benchmark_bear_threshold": 0.015,
            "garch_window": 60,
            "garch_refit_bars": 8,
            "confirm_bear_vol_ratio": 1.15,
            "confirm_bull_vol_ratio": 1.00,
            "cycle_window": 60,
            "cycle_min_period": 5,
            "cycle_max_period": 28,
            "min_cycle_purity": 0.15,
            "min_vol_size_scale": 0.35,
            "max_longs": 8,
            "max_shorts": 6,
            "max_gross": 1.00,
            "rebalance_bars": 1,
            "stop_loss_pct": 0.14,
            "max_hold_bars": 120,
            "target_allocation": 0.90,
        },
    ),
    "1d": (
        {
            "variant": "core",
            "momentum_lookback": 30,
            "trend_ma_window": 30,
            "signal_threshold": 0.04,
            "bull_breadth": 0.55,
            "bear_breadth": 0.52,
            "exit_breadth": 0.38,
            "benchmark_lookback": 30,
            "benchmark_bull_threshold": 0.02,
            "benchmark_bear_threshold": 0.02,
            "garch_window": 48,
            "garch_refit_bars": 6,
            "confirm_bear_vol_ratio": 1.15,
            "confirm_bull_vol_ratio": 1.00,
            "cycle_window": 48,
            "cycle_min_period": 4,
            "cycle_max_period": 20,
            "min_cycle_purity": 0.15,
            "min_vol_size_scale": 0.35,
            "max_longs": 8,
            "max_shorts": 6,
            "max_gross": 1.00,
            "rebalance_bars": 1,
            "stop_loss_pct": 0.18,
            "max_hold_bars": 90,
            "target_allocation": 0.90,
        },
    ),
}


__all__ = ["_REGIME_ROUTER_CONFIRMED_SLICE", "RegimeRouterConfirmedRotationStrategy"]

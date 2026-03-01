"""Pair trading strategy using rolling z-score of a hedge-adjusted spread."""

from __future__ import annotations

import logging
import math
from collections import deque
from itertools import islice
from statistics import mean

from lumina_quant.core.events import SignalEvent
from lumina_quant.indicators.common import safe_float
from lumina_quant.indicators.rolling_stats import rolling_beta, rolling_corr, sample_std
from lumina_quant.indicators.vwap import rolling_vwap
from lumina_quant.strategy import Strategy
from lumina_quant.symbols import canonical_symbol
from lumina_quant.tuning import HyperParam, resolve_params_from_schema

LOGGER = logging.getLogger(__name__)


class PairTradingZScoreStrategy(Strategy):
    """Market-neutral pair strategy with rolling z-score entry/exit rules.

    Rules:
    - Enter SHORT_SPREAD when z-score >= entry_z.
    - Enter LONG_SPREAD when z-score <= -entry_z.
    - Exit on mean reversion, stop breach, max hold, or correlation breakdown.

    No-lookahead:
    - Uses only bars available at current timestamp.
    - Fill timing remains controlled by execution handler (next bar open).
    """

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "lookback_window": HyperParam.integer(
                "lookback_window",
                default=96,
                low=10,
                high=20000,
                optuna={"type": "int", "low": 48, "high": 240},
                grid=[72, 96, 144],
            ),
            "hedge_window": HyperParam.integer(
                "hedge_window",
                default=192,
                low=10,
                high=40000,
                optuna={"type": "int", "low": 96, "high": 480},
                grid=[144, 192, 288],
            ),
            "entry_z": HyperParam.floating(
                "entry_z",
                default=2.0,
                low=0.5,
                high=20.0,
                optuna={"type": "float", "low": 1.2, "high": 3.0, "step": 0.1},
                grid=[1.6, 2.0, 2.4],
            ),
            "exit_z": HyperParam.floating(
                "exit_z",
                default=0.35,
                low=0.0,
                high=20.0,
                optuna={"type": "float", "low": 0.1, "high": 1.0, "step": 0.05},
                grid=[0.25, 0.35, 0.5],
            ),
            "stop_z": HyperParam.floating(
                "stop_z",
                default=3.5,
                low=0.6,
                high=50.0,
                optuna={"type": "float", "low": 2.5, "high": 5.0, "step": 0.1},
                grid=[3.0, 3.5, 4.0],
            ),
            "stop_z_min_gap": HyperParam.floating(
                "stop_z_min_gap",
                default=0.1,
                low=0.0,
                high=5.0,
                optuna={"type": "float", "low": 0.0, "high": 1.0, "step": 0.05},
                grid=[0.0, 0.1, 0.2, 0.4],
            ),
            "min_correlation": HyperParam.floating(
                "min_correlation",
                default=0.15,
                low=-1.0,
                high=1.0,
                optuna={"type": "float", "low": -0.2, "high": 0.8, "step": 0.05},
                grid=[0.0, 0.15, 0.3],
            ),
            "max_hold_bars": HyperParam.integer(
                "max_hold_bars",
                default=240,
                low=1,
                high=100000,
                optuna={"type": "int", "low": 24, "high": 480},
                grid=[96, 240, 384],
            ),
            "cooldown_bars": HyperParam.integer(
                "cooldown_bars",
                default=6,
                low=0,
                high=100000,
                optuna={"type": "int", "low": 0, "high": 48},
                grid=[0, 6, 12],
            ),
            "reentry_z_buffer": HyperParam.floating(
                "reentry_z_buffer",
                default=0.2,
                low=0.0,
                high=20.0,
                optuna={"type": "float", "low": 0.0, "high": 0.8, "step": 0.05},
                grid=[0.0, 0.2, 0.35],
            ),
            "min_z_turn": HyperParam.floating(
                "min_z_turn",
                default=0.0,
                low=0.0,
                high=20.0,
                optuna={"type": "float", "low": 0.0, "high": 0.8, "step": 0.05},
                grid=[0.0, 0.05, 0.15],
            ),
            "stop_loss_pct": HyperParam.floating(
                "stop_loss_pct",
                default=0.04,
                low=0.001,
                high=0.5,
                optuna={"type": "float", "low": 0.005, "high": 0.12, "step": 0.005},
                grid=[0.02, 0.04, 0.08],
            ),
            "min_abs_beta": HyperParam.floating(
                "min_abs_beta",
                default=0.02,
                low=0.0,
                high=100.0,
                optuna={"type": "float", "low": 0.0, "high": 0.5, "step": 0.01},
                grid=[0.0, 0.02, 0.1],
            ),
            "max_abs_beta": HyperParam.floating(
                "max_abs_beta",
                default=6.0,
                low=0.0,
                high=100.0,
                optuna={"type": "float", "low": 1.0, "high": 10.0, "step": 0.1},
                grid=[3.0, 6.0, 9.0],
            ),
            "min_volume_window": HyperParam.integer(
                "min_volume_window",
                default=24,
                low=1,
                high=100000,
                optuna={"type": "int", "low": 4, "high": 128},
                grid=[12, 24, 48],
            ),
            "min_volume_ratio": HyperParam.floating(
                "min_volume_ratio",
                default=0.0,
                low=0.0,
                high=10.0,
                optuna={"type": "float", "low": 0.0, "high": 1.2, "step": 0.05},
                grid=[0.0, 0.2, 0.5],
            ),
            "symbol_x": HyperParam.string("symbol_x", default="", tunable=False),
            "symbol_y": HyperParam.string("symbol_y", default="", tunable=False),
            "use_log_price": HyperParam.boolean("use_log_price", default=True, tunable=False),
            "vol_lag_bars": HyperParam.integer("vol_lag_bars", default=0, low=0, tunable=False),
            "min_vol_convergence": HyperParam.floating(
                "min_vol_convergence",
                default=0.0,
                low=0.0,
                tunable=False,
            ),
            "vwap_window": HyperParam.integer("vwap_window", default=0, low=0, tunable=False),
            "atr_window": HyperParam.integer("atr_window", default=0, low=0, tunable=False),
            "atr_max_pct": HyperParam.floating(
                "atr_max_pct",
                default=1.0,
                low=0.001,
                tunable=False,
            ),
            "atr_disable_threshold": HyperParam.floating(
                "atr_disable_threshold",
                default=0.999,
                low=0.0,
                high=1.0,
                tunable=False,
            ),
            "beta_stop_scale_min": HyperParam.floating(
                "beta_stop_scale_min",
                default=0.75,
                low=0.0,
                high=10.0,
            ),
            "beta_stop_scale_max": HyperParam.floating(
                "beta_stop_scale_max",
                default=2.5,
                low=0.0,
                high=20.0,
            ),
            "vol_divergence_floor": HyperParam.floating(
                "vol_divergence_floor",
                default=3.0,
                low=0.0,
                high=20.0,
            ),
            "vol_divergence_multiplier": HyperParam.floating(
                "vol_divergence_multiplier",
                default=3.0,
                low=0.0,
                high=20.0,
            ),
        }

    def __init__(
        self,
        bars,
        events,
        lookback_window=96,
        hedge_window=192,
        entry_z=2.0,
        exit_z=0.35,
        stop_z=3.5,
        stop_z_min_gap=0.1,
        min_correlation=0.15,
        max_hold_bars=240,
        cooldown_bars=6,
        reentry_z_buffer=0.2,
        min_z_turn=0.0,
        stop_loss_pct=0.04,
        min_abs_beta=0.02,
        max_abs_beta=6.0,
        min_volume_window=24,
        min_volume_ratio=0.0,
        symbol_x=None,
        symbol_y=None,
        use_log_price=True,
        vol_lag_bars=0,
        min_vol_convergence=0.0,
        vwap_window=0,
        atr_window=0,
        atr_max_pct=1.0,
        atr_disable_threshold=0.999,
        beta_stop_scale_min=0.75,
        beta_stop_scale_max=2.5,
        vol_divergence_floor=3.0,
        vol_divergence_multiplier=3.0,
    ):
        self.bars = bars
        self.events = events
        self.symbol_list = [canonical_symbol(symbol) for symbol in list(self.bars.symbol_list)]
        self.symbol_list = [symbol for symbol in self.symbol_list if symbol]
        if len(self.symbol_list) < 2:
            raise ValueError("PairTradingZScoreStrategy requires at least two symbols.")

        resolved = resolve_params_from_schema(
            self.get_param_schema(),
            {
                "lookback_window": lookback_window,
                "hedge_window": hedge_window,
                "entry_z": entry_z,
                "exit_z": exit_z,
                "stop_z": stop_z,
                "stop_z_min_gap": stop_z_min_gap,
                "min_correlation": min_correlation,
                "max_hold_bars": max_hold_bars,
                "cooldown_bars": cooldown_bars,
                "reentry_z_buffer": reentry_z_buffer,
                "min_z_turn": min_z_turn,
                "stop_loss_pct": stop_loss_pct,
                "min_abs_beta": min_abs_beta,
                "max_abs_beta": max_abs_beta,
                "min_volume_window": min_volume_window,
                "min_volume_ratio": min_volume_ratio,
                "symbol_x": symbol_x,
                "symbol_y": symbol_y,
                "use_log_price": use_log_price,
                "vol_lag_bars": vol_lag_bars,
                "min_vol_convergence": min_vol_convergence,
                "vwap_window": vwap_window,
                "atr_window": atr_window,
                "atr_max_pct": atr_max_pct,
                "atr_disable_threshold": atr_disable_threshold,
                "beta_stop_scale_min": beta_stop_scale_min,
                "beta_stop_scale_max": beta_stop_scale_max,
                "vol_divergence_floor": vol_divergence_floor,
                "vol_divergence_multiplier": vol_divergence_multiplier,
            },
            keep_unknown=False,
        )

        symbol_x = resolved["symbol_x"]
        symbol_y = resolved["symbol_y"]
        self.symbol_x = canonical_symbol(str(symbol_x)) if symbol_x else str(self.symbol_list[0])
        self.symbol_y = canonical_symbol(str(symbol_y)) if symbol_y else str(self.symbol_list[1])
        if not self.symbol_x or not self.symbol_y:
            raise ValueError("PairTradingZScoreStrategy requires canonical non-empty symbols.")
        if self.symbol_x == self.symbol_y:
            raise ValueError("symbol_x and symbol_y must be different.")

        self.lookback_window = int(resolved["lookback_window"])
        self.hedge_window = max(self.lookback_window, int(resolved["hedge_window"]))
        self.entry_z = float(resolved["entry_z"])
        self.exit_z = float(resolved["exit_z"])
        self.stop_z_min_gap = float(resolved["stop_z_min_gap"])
        self.stop_z = max(self.entry_z + self.stop_z_min_gap, float(resolved["stop_z"]))
        self.min_correlation = max(-1.0, min(1.0, float(resolved["min_correlation"])))
        self.max_hold_bars = int(resolved["max_hold_bars"])
        self.cooldown_bars = int(resolved["cooldown_bars"])
        self.reentry_z_buffer = float(resolved["reentry_z_buffer"])
        self.min_z_turn = float(resolved["min_z_turn"])
        self.stop_loss_pct = float(resolved["stop_loss_pct"])
        self.min_abs_beta = float(resolved["min_abs_beta"])
        self.max_abs_beta = max(self.min_abs_beta + 1e-9, float(resolved["max_abs_beta"]))
        self.min_volume_window = int(resolved["min_volume_window"])
        self.min_volume_ratio = float(resolved["min_volume_ratio"])
        self.use_log_price = bool(resolved["use_log_price"])
        self.vol_lag_bars = int(resolved["vol_lag_bars"])
        self.min_vol_convergence = float(resolved["min_vol_convergence"])
        self.vwap_window = int(resolved["vwap_window"])
        self.atr_window = int(resolved["atr_window"])
        self.atr_max_pct = float(resolved["atr_max_pct"])
        self.atr_disable_threshold = float(resolved["atr_disable_threshold"])
        self.beta_stop_scale_min = float(resolved["beta_stop_scale_min"])
        self.beta_stop_scale_max = max(
            self.beta_stop_scale_min + 1e-9,
            float(resolved["beta_stop_scale_max"]),
        )
        self.vol_divergence_floor = float(resolved["vol_divergence_floor"])
        self.vol_divergence_multiplier = float(resolved["vol_divergence_multiplier"])

        self._x_history = deque(maxlen=self.hedge_window)
        self._y_history = deque(maxlen=self.hedge_window)
        self._spread_history = deque(maxlen=self.lookback_window)
        self._x_volume_history = deque(maxlen=self.min_volume_window)
        self._y_volume_history = deque(maxlen=self.min_volume_window)

        aux_len = max(
            self.lookback_window + self.vol_lag_bars + 8,
            self.vwap_window + 4,
            self.atr_window + 4,
            16,
        )
        self._x_close_history = deque(maxlen=aux_len)
        self._y_close_history = deque(maxlen=aux_len)
        self._x_return_history = deque(maxlen=aux_len)
        self._y_return_history = deque(maxlen=aux_len)
        self._vol_spread_history = deque(maxlen=max(16, self.lookback_window))
        self._x_tr_history = deque(maxlen=max(1, self.atr_window))
        self._y_tr_history = deque(maxlen=max(1, self.atr_window))
        self._prev_x_close_raw = None
        self._prev_y_close_raw = None

        self._mode = "OUT"
        self._bars_in_position = 0
        self._cooldown_left = 0
        self._last_pair_time_key = ""
        self._extreme_side = "NONE"
        self._prev_zscore = None

        self._last_hedge_ratio = 1.0
        self._last_zscore = None
        self._last_correlation = None
        self._last_vol_zscore = None

    def get_state(self):
        return {
            "mode": self._mode,
            "bars_in_position": int(self._bars_in_position),
            "cooldown_left": int(self._cooldown_left),
            "last_pair_time_key": str(self._last_pair_time_key),
            "extreme_side": str(self._extreme_side),
            "prev_zscore": self._prev_zscore,
            "x_history": list(self._x_history),
            "y_history": list(self._y_history),
            "spread_history": list(self._spread_history),
            "x_volume_history": list(self._x_volume_history),
            "y_volume_history": list(self._y_volume_history),
            "last_hedge_ratio": float(self._last_hedge_ratio),
            "last_zscore": self._last_zscore,
            "last_correlation": self._last_correlation,
            "last_vol_zscore": self._last_vol_zscore,
            "x_close_history": list(self._x_close_history),
            "y_close_history": list(self._y_close_history),
            "x_return_history": list(self._x_return_history),
            "y_return_history": list(self._y_return_history),
            "vol_spread_history": list(self._vol_spread_history),
            "x_tr_history": list(self._x_tr_history),
            "y_tr_history": list(self._y_tr_history),
            "prev_x_close_raw": self._prev_x_close_raw,
            "prev_y_close_raw": self._prev_y_close_raw,
        }

    def set_state(self, state):
        if not isinstance(state, dict):
            return

        mode = str(state.get("mode", "OUT")).upper()
        if mode in {"OUT", "LONG_SPREAD", "SHORT_SPREAD"}:
            self._mode = mode

        try:
            self._bars_in_position = max(0, int(state.get("bars_in_position", 0)))
        except Exception:
            self._bars_in_position = 0
        try:
            self._cooldown_left = max(0, int(state.get("cooldown_left", 0)))
        except Exception:
            self._cooldown_left = 0

        self._last_pair_time_key = str(state.get("last_pair_time_key", ""))
        extreme_side = str(state.get("extreme_side", "NONE")).upper()
        self._extreme_side = extreme_side if extreme_side in {"NONE", "HIGH", "LOW"} else "NONE"
        prev_z = state.get("prev_zscore")
        self._prev_zscore = safe_float(prev_z) if prev_z is not None else None

        for key, target in (
            ("x_history", self._x_history),
            ("y_history", self._y_history),
            ("spread_history", self._spread_history),
            ("x_volume_history", self._x_volume_history),
            ("y_volume_history", self._y_volume_history),
        ):
            raw_values = state.get(key)
            if not isinstance(raw_values, list):
                continue
            target.clear()
            keep = int(target.maxlen) if target.maxlen is not None else len(raw_values)
            for value in raw_values[-keep:]:
                parsed = safe_float(value)
                if parsed is not None:
                    target.append(parsed)

        hedge_ratio = safe_float(state.get("last_hedge_ratio"))
        if hedge_ratio is not None:
            self._last_hedge_ratio = hedge_ratio

        zscore = state.get("last_zscore")
        self._last_zscore = safe_float(zscore) if zscore is not None else None

        corr = state.get("last_correlation")
        self._last_correlation = safe_float(corr) if corr is not None else None

        vol_z = state.get("last_vol_zscore")
        self._last_vol_zscore = safe_float(vol_z) if vol_z is not None else None

        for key, target in (
            ("x_close_history", self._x_close_history),
            ("y_close_history", self._y_close_history),
            ("x_return_history", self._x_return_history),
            ("y_return_history", self._y_return_history),
            ("vol_spread_history", self._vol_spread_history),
            ("x_tr_history", self._x_tr_history),
            ("y_tr_history", self._y_tr_history),
        ):
            raw_values = state.get(key)
            if not isinstance(raw_values, list):
                continue
            target.clear()
            keep = int(target.maxlen) if target.maxlen is not None else len(raw_values)
            for value in raw_values[-keep:]:
                parsed = safe_float(value)
                if parsed is not None:
                    target.append(parsed)

        self._prev_x_close_raw = safe_float(state.get("prev_x_close_raw"))
        self._prev_y_close_raw = safe_float(state.get("prev_y_close_raw"))

        if not isinstance(state.get("x_return_history"), list):
            self._rebuild_log_returns(self._x_close_history, self._x_return_history)
        if not isinstance(state.get("y_return_history"), list):
            self._rebuild_log_returns(self._y_close_history, self._y_return_history)

    @staticmethod
    def _rebuild_log_returns(closes, target) -> None:
        target.clear()
        previous = None
        for current in closes:
            if previous is not None and previous > 0.0 and current > 0.0:
                target.append(math.log(current / previous))
            previous = current

    def _aligned_pair_timestamp(self):
        tx = self.bars.get_latest_bar_datetime(self.symbol_x)
        ty = self.bars.get_latest_bar_datetime(self.symbol_y)
        if tx is None or ty is None or tx != ty:
            return None
        return tx

    def _resolve_pair_prices(self):
        px, py = self._resolve_pair_closes()
        if px is None or py is None or px <= 0.0 or py <= 0.0:
            return None, None

        if self.vwap_window > 1:
            vwap_x = rolling_vwap(self._x_close_history, self._x_volume_history, self.vwap_window)
            vwap_y = rolling_vwap(self._y_close_history, self._y_volume_history, self.vwap_window)
            if vwap_x is not None and vwap_x > 0.0:
                px = px / vwap_x
            if vwap_y is not None and vwap_y > 0.0:
                py = py / vwap_y

        if px <= 0.0 or py <= 0.0:
            return None, None
        if self.use_log_price:
            return math.log(px), math.log(py)
        return px, py

    def _resolve_pair_closes(self):
        px = safe_float(self.bars.get_latest_bar_value(self.symbol_x, "close"))
        py = safe_float(self.bars.get_latest_bar_value(self.symbol_y, "close"))
        if px is None or py is None or px <= 0.0 or py <= 0.0:
            return None, None
        return px, py

    def _resolve_pair_volumes(self):
        vx = safe_float(self.bars.get_latest_bar_value(self.symbol_x, "volume"))
        vy = safe_float(self.bars.get_latest_bar_value(self.symbol_y, "volume"))
        if vx is None or vy is None:
            return None, None
        return max(0.0, vx), max(0.0, vy)

    def _rolling_beta(self):
        if len(self._x_history) < self.hedge_window or len(self._y_history) < self.hedge_window:
            return None
        return rolling_beta(self._x_history, self._y_history)

    def _rolling_corr(self):
        if (
            len(self._x_history) < self.lookback_window
            or len(self._y_history) < self.lookback_window
        ):
            return None
        x_values = list(self._x_history)[-self.lookback_window :]
        y_values = list(self._y_history)[-self.lookback_window :]
        return rolling_corr(x_values, y_values)

    def _volume_filter_passed(self):
        if self.min_volume_ratio <= 0.0:
            return True
        if (
            len(self._x_volume_history) < self.min_volume_window
            or len(self._y_volume_history) < self.min_volume_window
        ):
            return True
        avg_x = mean(self._x_volume_history)
        avg_y = mean(self._y_volume_history)
        if avg_x <= 0.0 or avg_y <= 0.0:
            return True
        latest_x = self._x_volume_history[-1]
        latest_y = self._y_volume_history[-1]
        return latest_x >= (avg_x * self.min_volume_ratio) and latest_y >= (
            avg_y * self.min_volume_ratio
        )

    def _atr_filter_passed(self, close_x, close_y):
        if self.atr_window <= 1:
            return True
        if self.atr_max_pct >= self.atr_disable_threshold:
            return True
        if len(self._x_tr_history) < self.atr_window or len(self._y_tr_history) < self.atr_window:
            return True
        if close_x <= 0.0 or close_y <= 0.0:
            return False
        x_start = len(self._x_tr_history) - self.atr_window
        y_start = len(self._y_tr_history) - self.atr_window
        atr_x = mean(islice(self._x_tr_history, x_start, None))
        atr_y = mean(islice(self._y_tr_history, y_start, None))
        return (atr_x / close_x) <= self.atr_max_pct and (atr_y / close_y) <= self.atr_max_pct

    def _vol_spread_zscore(self):
        if self.min_vol_convergence <= 0.0:
            return None

        # Preserve previous warm-up semantics from close-history reconstruction:
        # x requires lookback+1 returns, y requires lookback+lag+1 returns.
        required_x = self.lookback_window + 1
        required_y = self.lookback_window + self.vol_lag_bars + 1
        if len(self._x_return_history) < required_x or len(self._y_return_history) < required_y:
            return None

        x_start = len(self._x_return_history) - self.lookback_window
        x_slice = tuple(islice(self._x_return_history, x_start, None))
        if self.vol_lag_bars > 0:
            y_start = len(self._y_return_history) - (self.lookback_window + self.vol_lag_bars)
            y_end = len(self._y_return_history) - self.vol_lag_bars
            y_slice = tuple(islice(self._y_return_history, y_start, y_end))
        else:
            y_start = len(self._y_return_history) - self.lookback_window
            y_slice = tuple(islice(self._y_return_history, y_start, None))

        std_x = sample_std(x_slice)
        std_y = sample_std(y_slice)
        if std_x is None or std_y is None or std_x <= 1e-12 or std_y <= 1e-12:
            return None

        vol_spread = math.log((std_x + 1e-12) / (std_y + 1e-12))
        self._vol_spread_history.append(vol_spread)

        baseline_window = max(8, min(self.lookback_window, 32))
        if len(self._vol_spread_history) < baseline_window:
            return None

        spread_mean = mean(self._vol_spread_history)
        spread_std = sample_std(self._vol_spread_history)
        if spread_std is None or spread_std <= 1e-12:
            return None
        return (vol_spread - spread_mean) / spread_std

    def _entry_stop_price(self, direction, price, beta):
        beta_scale = max(
            self.beta_stop_scale_min,
            min(self.beta_stop_scale_max, abs(float(beta))),
        )
        stop_pct = min(0.50, self.stop_loss_pct * beta_scale)
        if direction == "LONG":
            return price * (1.0 - stop_pct)
        return price * (1.0 + stop_pct)

    def _emit_signal(
        self,
        symbol,
        event_time,
        signal_type,
        metadata,
        *,
        stop_loss=None,
        take_profit=None,
    ):
        self.events.put(
            SignalEvent(
                strategy_id="pair_zscore",
                symbol=symbol,
                datetime=event_time,
                signal_type=signal_type,
                strength=1.0,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata=metadata,
            )
        )

    def _emit_pair_entry(self, event_time, mode, zscore, beta, corr):
        close_x, close_y = self._resolve_pair_closes()
        if close_x is None or close_y is None:
            return

        metadata = {
            "strategy": "PairTradingZScoreStrategy",
            "pair_mode": mode,
            "zscore": float(zscore),
            "hedge_ratio": float(beta),
            "correlation": float(corr) if corr is not None else None,
            "symbol_x": self.symbol_x,
            "symbol_y": self.symbol_y,
            "stop_loss_pct": float(self.stop_loss_pct),
            "vol_zscore": self._last_vol_zscore,
            "vol_lag_bars": int(self.vol_lag_bars),
            "vwap_window": int(self.vwap_window),
        }

        if mode == "LONG_SPREAD":
            self._emit_signal(
                self.symbol_x,
                event_time,
                "LONG",
                metadata,
                stop_loss=self._entry_stop_price("LONG", close_x, beta),
            )
            self._emit_signal(
                self.symbol_y,
                event_time,
                "SHORT",
                metadata,
                stop_loss=self._entry_stop_price("SHORT", close_y, beta),
            )
        else:
            self._emit_signal(
                self.symbol_x,
                event_time,
                "SHORT",
                metadata,
                stop_loss=self._entry_stop_price("SHORT", close_x, beta),
            )
            self._emit_signal(
                self.symbol_y,
                event_time,
                "LONG",
                metadata,
                stop_loss=self._entry_stop_price("LONG", close_y, beta),
            )

    def _emit_pair_exit(self, event_time, reason, zscore, beta, corr):
        metadata = {
            "strategy": "PairTradingZScoreStrategy",
            "reason": reason,
            "zscore": float(zscore),
            "hedge_ratio": float(beta),
            "correlation": float(corr) if corr is not None else None,
            "symbol_x": self.symbol_x,
            "symbol_y": self.symbol_y,
            "previous_mode": self._mode,
            "vol_zscore": self._last_vol_zscore,
        }
        self._emit_signal(self.symbol_x, event_time, "EXIT", metadata)
        self._emit_signal(self.symbol_y, event_time, "EXIT", metadata)

    def calculate_signals(self, event):
        if getattr(event, "type", None) != "MARKET":
            return
        event_symbol = canonical_symbol(str(getattr(event, "symbol", "")))
        if event_symbol not in {self.symbol_x, self.symbol_y}:
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

        vol_x, vol_y = self._resolve_pair_volumes()
        if vol_x is not None and vol_y is not None:
            self._x_volume_history.append(vol_x)
            self._y_volume_history.append(vol_y)

        if self.atr_window > 1:
            high_x = safe_float(self.bars.get_latest_bar_value(self.symbol_x, "high"))
            low_x = safe_float(self.bars.get_latest_bar_value(self.symbol_x, "low"))
            high_y = safe_float(self.bars.get_latest_bar_value(self.symbol_y, "high"))
            low_y = safe_float(self.bars.get_latest_bar_value(self.symbol_y, "low"))
            if (
                high_x is not None
                and low_x is not None
                and high_y is not None
                and low_y is not None
                and self._prev_x_close_raw is not None
                and self._prev_y_close_raw is not None
            ):
                tr_x = max(
                    high_x - low_x,
                    abs(high_x - self._prev_x_close_raw),
                    abs(low_x - self._prev_x_close_raw),
                )
                tr_y = max(
                    high_y - low_y,
                    abs(high_y - self._prev_y_close_raw),
                    abs(low_y - self._prev_y_close_raw),
                )
                self._x_tr_history.append(max(0.0, tr_x))
                self._y_tr_history.append(max(0.0, tr_y))

        prev_x_close = self._prev_x_close_raw
        prev_y_close = self._prev_y_close_raw
        if prev_x_close is not None and prev_x_close > 0.0 and close_x > 0.0:
            self._x_return_history.append(math.log(close_x / prev_x_close))
        if prev_y_close is not None and prev_y_close > 0.0 and close_y > 0.0:
            self._y_return_history.append(math.log(close_y / prev_y_close))

        self._x_close_history.append(float(close_x))
        self._y_close_history.append(float(close_y))
        self._prev_x_close_raw = float(close_x)
        self._prev_y_close_raw = float(close_y)

        x_value, y_value = self._resolve_pair_prices()
        if x_value is None or y_value is None:
            return

        self._x_history.append(x_value)
        self._y_history.append(y_value)

        beta = self._rolling_beta()
        if beta is None:
            return
        if not (self.min_abs_beta <= abs(beta) <= self.max_abs_beta):
            return

        spread = x_value - (beta * y_value)
        self._spread_history.append(spread)

        if len(self._spread_history) < self.lookback_window:
            return
        spread_mean = mean(self._spread_history)
        spread_std = sample_std(self._spread_history)
        if spread_std is None or spread_std <= 1e-12:
            return

        zscore = (spread - spread_mean) / spread_std
        corr = self._rolling_corr()
        vol_zscore = self._vol_spread_zscore()
        self._last_hedge_ratio = float(beta)
        self._last_zscore = float(zscore)
        self._last_correlation = float(corr) if corr is not None else None
        self._last_vol_zscore = float(vol_zscore) if vol_zscore is not None else None
        prev_z = self._prev_zscore
        self._prev_zscore = float(zscore)

        if self._mode == "OUT" and self._cooldown_left > 0:
            self._cooldown_left -= 1

        if self._mode == "OUT":
            if self._cooldown_left > 0:
                return
            if corr is not None and corr < self.min_correlation:
                return
            if not self._volume_filter_passed():
                return
            if not self._atr_filter_passed(close_x, close_y):
                return
            if self.min_vol_convergence > 0.0:
                if vol_zscore is None:
                    return
                if zscore > 0.0 and vol_zscore <= 0.0:
                    return
                if zscore < 0.0 and vol_zscore >= 0.0:
                    return
                if abs(vol_zscore) < self.min_vol_convergence:
                    return

            if zscore >= self.entry_z:
                self._extreme_side = "HIGH"
                return
            if zscore <= -self.entry_z:
                self._extreme_side = "LOW"
                return

            if self._extreme_side == "HIGH":
                if prev_z is None:
                    turned = True
                else:
                    turned = (prev_z - zscore) >= self.min_z_turn
                if zscore <= (self.entry_z - self.reentry_z_buffer) and turned:
                    self._emit_pair_entry(pair_time, "SHORT_SPREAD", zscore, beta, corr)
                    self._mode = "SHORT_SPREAD"
                    self._bars_in_position = 0
                    self._extreme_side = "NONE"
                    LOGGER.info("Pair entry SHORT_SPREAD z=%.4f beta=%.4f", zscore, beta)
            elif self._extreme_side == "LOW":
                if prev_z is None:
                    turned = True
                else:
                    turned = (zscore - prev_z) >= self.min_z_turn
                if zscore >= (-self.entry_z + self.reentry_z_buffer) and turned:
                    self._emit_pair_entry(pair_time, "LONG_SPREAD", zscore, beta, corr)
                    self._mode = "LONG_SPREAD"
                    self._bars_in_position = 0
                    self._extreme_side = "NONE"
                    LOGGER.info("Pair entry LONG_SPREAD z=%.4f beta=%.4f", zscore, beta)
            else:
                self._extreme_side = "NONE"
            return

        self._bars_in_position += 1
        exit_reason = None
        if abs(zscore) <= self.exit_z:
            exit_reason = "mean_reversion"
        elif (
            self.min_vol_convergence > 0.0
            and vol_zscore is not None
            and abs(vol_zscore) <= (self.min_vol_convergence * 0.5)
        ):
            exit_reason = "vol_convergence"
        elif abs(zscore) >= self.stop_z:
            exit_reason = "z_stop"
        elif (
            self.min_vol_convergence > 0.0
            and vol_zscore is not None
            and abs(vol_zscore)
            >= max(
                self.vol_divergence_floor,
                self.min_vol_convergence * self.vol_divergence_multiplier,
            )
        ):
            exit_reason = "vol_divergence"
        elif self._bars_in_position >= self.max_hold_bars:
            exit_reason = "max_hold"
        elif corr is not None and corr < (self.min_correlation * 0.5):
            exit_reason = "correlation_break"

        if exit_reason is None:
            return

        self._emit_pair_exit(pair_time, exit_reason, zscore, beta, corr)
        LOGGER.info("Pair exit %s z=%.4f beta=%.4f", exit_reason, zscore, beta)
        self._mode = "OUT"
        self._bars_in_position = 0
        self._cooldown_left = self.cooldown_bars
        self._extreme_side = "NONE"

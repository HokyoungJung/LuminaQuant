"""Pair trading strategy using rolling z-score of a hedge-adjusted spread."""

from __future__ import annotations

import logging
import math
from collections import deque
from itertools import islice
from statistics import mean

from lumina_quant.events import SignalEvent
from lumina_quant.indicators import rolling_beta, rolling_corr, rolling_vwap, safe_float, sample_std
from lumina_quant.strategy import Strategy

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

    def __init__(
        self,
        bars,
        events,
        lookback_window=96,
        hedge_window=192,
        entry_z=2.0,
        exit_z=0.35,
        stop_z=3.5,
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
    ):
        self.bars = bars
        self.events = events
        self.symbol_list = list(self.bars.symbol_list)
        if len(self.symbol_list) < 2:
            raise ValueError("PairTradingZScoreStrategy requires at least two symbols.")

        self.symbol_x = str(symbol_x) if symbol_x else str(self.symbol_list[0])
        self.symbol_y = str(symbol_y) if symbol_y else str(self.symbol_list[1])
        if self.symbol_x == self.symbol_y:
            raise ValueError("symbol_x and symbol_y must be different.")

        self.lookback_window = max(10, int(lookback_window))
        self.hedge_window = max(self.lookback_window, int(hedge_window))
        self.entry_z = max(0.5, float(entry_z))
        self.exit_z = max(0.0, float(exit_z))
        self.stop_z = max(self.entry_z + 0.1, float(stop_z))
        self.min_correlation = max(-1.0, min(1.0, float(min_correlation)))
        self.max_hold_bars = max(1, int(max_hold_bars))
        self.cooldown_bars = max(0, int(cooldown_bars))
        self.reentry_z_buffer = max(0.0, float(reentry_z_buffer))
        self.min_z_turn = max(0.0, float(min_z_turn))
        self.stop_loss_pct = min(0.50, max(0.001, float(stop_loss_pct)))
        self.min_abs_beta = max(0.0, float(min_abs_beta))
        self.max_abs_beta = max(self.min_abs_beta + 1e-9, float(max_abs_beta))
        self.min_volume_window = max(1, int(min_volume_window))
        self.min_volume_ratio = max(0.0, float(min_volume_ratio))
        self.use_log_price = bool(use_log_price)
        self.vol_lag_bars = max(0, int(vol_lag_bars))
        self.min_vol_convergence = max(0.0, float(min_vol_convergence))
        self.vwap_window = max(0, int(vwap_window))
        self.atr_window = max(0, int(atr_window))
        self.atr_max_pct = max(0.001, float(atr_max_pct))

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
        if self.atr_max_pct >= 0.999:
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
        beta_scale = max(0.75, min(2.5, abs(float(beta))))
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
            and abs(vol_zscore) >= max(3.0, self.min_vol_convergence * 3.0)
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

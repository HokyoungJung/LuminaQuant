"""Rolling mean-reversion strategy using standard deviation bands."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from lumina_quant.core.events import SignalEvent
from lumina_quant.indicators.common import safe_float
from lumina_quant.indicators.rolling_stats import RollingZScoreWindow
from lumina_quant.strategy import Strategy
from lumina_quant.tuning import HyperParam, resolve_params_from_schema


@dataclass(slots=True)
class _SymbolState:
    zscore_window: RollingZScoreWindow
    raw_history: deque[float] = field(default_factory=deque)
    state: str = "OUT"
    entry_price: float | None = None
    residual_price: float | None = None
    last_time_key: str = ""


class MeanReversionStdStrategy(Strategy):
    """Single-asset z-score mean reversion with optional shorts."""

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "window": HyperParam.integer(
                "window",
                default=64,
                low=8,
                high=8192,
                optuna={"type": "int", "low": 16, "high": 256},
                grid=[24, 48, 64, 96, 128],
            ),
            "entry_z": HyperParam.floating(
                "entry_z",
                default=2.0,
                low=0.2,
                high=20.0,
                optuna={"type": "float", "low": 0.8, "high": 3.5, "step": 0.1},
                grid=[1.2, 1.6, 2.0, 2.4, 2.8],
            ),
            "exit_z": HyperParam.floating(
                "exit_z",
                default=0.5,
                low=0.0,
                high=10.0,
                optuna={"type": "float", "low": 0.1, "high": 1.5, "step": 0.05},
                grid=[0.2, 0.4, 0.6, 0.8],
            ),
            "stop_loss_pct": HyperParam.floating(
                "stop_loss_pct",
                default=0.03,
                low=0.001,
                high=0.5,
                optuna={"type": "float", "low": 0.005, "high": 0.12, "step": 0.005},
                grid=[0.01, 0.02, 0.03, 0.04],
            ),
            "allow_short": HyperParam.boolean(
                "allow_short",
                default=True,
                optuna={"type": "categorical", "choices": [True, False]},
                grid=[True, False],
            ),
            "residualize_btc": HyperParam.boolean(
                "residualize_btc",
                default=False,
                tunable=False,
            ),
            "btc_symbol": HyperParam.string(
                "btc_symbol",
                default="BTC/USDT",
                tunable=False,
            ),
        }

    def __init__(
        self,
        bars,
        events,
        window=64,
        entry_z=2.0,
        exit_z=0.5,
        stop_loss_pct=0.03,
        allow_short=True,
        residualize_btc=False,
        btc_symbol=None,
        min_residual_history_points=3,
        residual_base_price=100.0,
    ):
        self.bars = bars
        self.events = events
        self.symbol_list = list(self.bars.symbol_list)
        resolved = resolve_params_from_schema(
            self.get_param_schema(),
            {
                "window": window,
                "entry_z": entry_z,
                "exit_z": exit_z,
                "stop_loss_pct": stop_loss_pct,
                "allow_short": allow_short,
                "residualize_btc": residualize_btc,
                "btc_symbol": btc_symbol,
            },
            keep_unknown=False,
        )
        self.window = int(resolved["window"])
        self.entry_z = float(resolved["entry_z"])
        self.exit_z = float(resolved["exit_z"])
        self.stop_loss_pct = float(resolved["stop_loss_pct"])
        self.allow_short = bool(resolved["allow_short"])
        self.residualize_btc = bool(resolved["residualize_btc"])
        self.min_residual_history_points = int(min_residual_history_points)
        self.residual_base_price = float(residual_base_price)
        default_btc = "BTC/USDT" if "BTC/USDT" in self.symbol_list else self.symbol_list[0]
        raw_btc_symbol = resolved["btc_symbol"]
        self.btc_symbol = str(raw_btc_symbol) if raw_btc_symbol else default_btc
        if self.btc_symbol not in self.symbol_list:
            self.btc_symbol = default_btc
        history_len = self.window + 1

        self._state = {
            symbol: _SymbolState(
                zscore_window=RollingZScoreWindow(self.window),
                raw_history=deque(maxlen=history_len),
            )
            for symbol in self.symbol_list
        }

    def get_state(self):
        return {
            "symbol_state": {
                symbol: {
                    "prices": list(item.zscore_window.values),
                    "raw_prices": list(item.raw_history),
                    "sum_price": item.zscore_window.sum_value,
                    "sum_squares": item.zscore_window.sum_squares,
                    "state": item.state,
                    "entry_price": item.entry_price,
                    "residual_price": item.residual_price,
                    "last_time_key": item.last_time_key,
                }
                for symbol, item in self._state.items()
            }
        }

    def set_state(self, state):
        if not isinstance(state, dict):
            return
        raw_symbol_state = state.get("symbol_state")
        if not isinstance(raw_symbol_state, dict):
            return

        for symbol, raw in raw_symbol_state.items():
            if symbol not in self._state or not isinstance(raw, dict):
                continue
            item = self._state[symbol]
            prices = []
            for value in list(raw.get("prices") or [])[-self.window :]:
                parsed = safe_float(value)
                if parsed is not None:
                    prices.append(parsed)
            item.zscore_window = RollingZScoreWindow(self.window)
            item.zscore_window.load_state(prices)
            item.raw_history = deque(maxlen=self.window + 1)
            for value in list(raw.get("raw_prices") or [])[-(self.window + 1) :]:
                parsed = safe_float(value)
                if parsed is not None:
                    item.raw_history.append(parsed)

            restored_state = str(raw.get("state", "OUT")).upper()
            item.state = restored_state if restored_state in {"OUT", "LONG", "SHORT"} else "OUT"
            item.entry_price = safe_float(raw.get("entry_price"))
            item.residual_price = safe_float(raw.get("residual_price"))
            item.last_time_key = str(raw.get("last_time_key", ""))

    def _resolve_close(self, symbol, event):
        if getattr(event, "symbol", None) == symbol:
            close_price = safe_float(getattr(event, "close", None))
            if close_price is not None:
                return close_price
        return safe_float(self.bars.get_latest_bar_value(symbol, "close"))

    def _append_price(self, item, close_price):
        item.zscore_window.append(close_price)

    def _append_raw_price(self, item, close_price):
        item.raw_history.append(float(close_price))

    def _resolve_signal_close(self, symbol, close_price):
        if not self.residualize_btc:
            return close_price
        if symbol == self.btc_symbol:
            return None
        item = self._state[symbol]
        btc_item = self._state.get(self.btc_symbol)
        btc_close = safe_float(self.bars.get_latest_bar_value(self.btc_symbol, "close"))
        if btc_item is None or btc_close is None:
            return None

        asset_prices = list(item.raw_history)
        btc_prices = list(btc_item.raw_history)
        history_len = min(len(asset_prices), len(btc_prices), self.window + 1)
        if history_len < self.min_residual_history_points:
            return None
        asset_tail = asset_prices[-history_len:]
        btc_tail = btc_prices[-history_len:]
        asset_rets: list[float] = []
        btc_rets: list[float] = []
        for prev_asset, next_asset, prev_btc, next_btc in zip(
            asset_tail[:-1],
            asset_tail[1:],
            btc_tail[:-1],
            btc_tail[1:],
            strict=True,
        ):
            if prev_asset <= 0.0 or prev_btc <= 0.0:
                continue
            asset_rets.append((float(next_asset) / float(prev_asset)) - 1.0)
            btc_rets.append((float(next_btc) / float(prev_btc)) - 1.0)
        if len(asset_rets) < 2 or len(asset_rets) != len(btc_rets):
            return None
        mean_asset = sum(asset_rets) / float(len(asset_rets))
        mean_btc = sum(btc_rets) / float(len(btc_rets))
        btc_var = sum((value - mean_btc) ** 2 for value in btc_rets)
        if btc_var <= 1e-12:
            beta = 0.0
        else:
            cov = sum(
                (asset - mean_asset) * (btc - mean_btc)
                for asset, btc in zip(asset_rets, btc_rets, strict=True)
            )
            beta = cov / btc_var
        residual_ret = asset_rets[-1] - (beta * btc_rets[-1])
        base_price = (
            item.residual_price if item.residual_price is not None else self.residual_base_price
        )
        return max(1e-9, float(base_price) * (1.0 + float(residual_ret)))

    @staticmethod
    def _emit(events, symbol, event_time, signal_type, metadata, stop_loss=None) -> None:
        events.put(
            SignalEvent(
                strategy_id="mean_reversion_std",
                symbol=symbol,
                datetime=event_time,
                signal_type=signal_type,
                strength=1.0,
                stop_loss=stop_loss,
                metadata=metadata,
            )
        )

    def calculate_signals(self, event):
        if getattr(event, "type", None) != "MARKET":
            return

        symbol = getattr(event, "symbol", None)
        if symbol not in self._state:
            return

        item = self._state[symbol]
        close_price = self._resolve_close(symbol, event)
        if close_price is None:
            return

        event_time = getattr(event, "time", None)
        time_key = "" if event_time is None else str(event_time)
        if time_key and time_key == item.last_time_key:
            return
        if time_key:
            item.last_time_key = time_key

        self._append_raw_price(item, close_price)
        signal_close = self._resolve_signal_close(symbol, close_price)
        if signal_close is None:
            return

        zscore = None
        if len(item.zscore_window.values) >= self.window:
            zscore = item.zscore_window.zscore(signal_close)

        if item.state == "LONG" and item.entry_price is not None:
            stop_hit = close_price <= item.entry_price * (1.0 - self.stop_loss_pct)
            revert_hit = zscore is not None and zscore >= -self.exit_z
            if stop_hit or revert_hit:
                self._emit(
                    self.events,
                    symbol,
                    event_time,
                    "EXIT",
                    {
                        "strategy": "MeanReversionStdStrategy",
                        "reason": "stop_loss" if stop_hit else "mean_reversion",
                        "zscore": zscore,
                        "residualize_btc": bool(self.residualize_btc),
                    },
                )
                item.state = "OUT"
                item.entry_price = None

        elif item.state == "SHORT" and item.entry_price is not None:
            stop_hit = close_price >= item.entry_price * (1.0 + self.stop_loss_pct)
            revert_hit = zscore is not None and zscore <= self.exit_z
            if stop_hit or revert_hit:
                self._emit(
                    self.events,
                    symbol,
                    event_time,
                    "EXIT",
                    {
                        "strategy": "MeanReversionStdStrategy",
                        "reason": "stop_loss" if stop_hit else "mean_reversion",
                        "zscore": zscore,
                        "residualize_btc": bool(self.residualize_btc),
                    },
                )
                item.state = "OUT"
                item.entry_price = None

        if item.state == "OUT" and zscore is not None:
            if zscore <= -self.entry_z:
                self._emit(
                    self.events,
                    symbol,
                    event_time,
                    "LONG",
                    {
                        "strategy": "MeanReversionStdStrategy",
                        "zscore": zscore,
                        "residualize_btc": bool(self.residualize_btc),
                    },
                    stop_loss=close_price * (1.0 - self.stop_loss_pct),
                )
                item.state = "LONG"
                item.entry_price = close_price
            elif self.allow_short and zscore >= self.entry_z:
                self._emit(
                    self.events,
                    symbol,
                    event_time,
                    "SHORT",
                    {
                        "strategy": "MeanReversionStdStrategy",
                        "zscore": zscore,
                        "residualize_btc": bool(self.residualize_btc),
                    },
                    stop_loss=close_price * (1.0 + self.stop_loss_pct),
                )
                item.state = "SHORT"
                item.entry_price = close_price

        self._append_price(item, signal_close)
        if self.residualize_btc:
            item.residual_price = float(signal_close)

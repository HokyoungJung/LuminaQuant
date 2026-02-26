"""Rare-event composite-score strategy.

The strategy consumes only the latest close per event and keeps a bounded
in-memory deque per symbol, avoiding repeated full-history loads.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from lumina_quant.core.events import SignalEvent
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.indicators.rare_event import rare_event_scores_latest
from lumina_quant.strategy import Strategy
from lumina_quant.tuning import HyperParam, resolve_params_from_schema


@dataclass(slots=True)
class _SymbolState:
    closes: deque
    mode: str = "OUT"
    entry_price: float | None = None
    last_time_key: str = ""


class RareEventScoreStrategy(Strategy):
    """Trade on rare directional events using a 0~1 composite rarity score."""

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "history_bars": HyperParam.integer(
                "history_bars",
                default=512,
                low=64,
                high=20000,
                optuna={"type": "int", "low": 256, "high": 2048, "step": 64},
                grid=[256, 512, 1024],
            ),
            "lookbacks": HyperParam.int_tuple(
                "lookbacks",
                default=(1, 2, 3, 4, 5),
                min_value=1,
                max_value=1024,
                tunable=False,
            ),
            "return_factor": HyperParam.floating(
                "return_factor",
                default=1.0,
                low=0.0,
                high=20.0,
                optuna={"type": "float", "low": 0.2, "high": 2.0, "step": 0.1},
                grid=[0.6, 1.0, 1.4],
            ),
            "trend_rolling_window": HyperParam.integer(
                "trend_rolling_window",
                default=20,
                low=5,
                high=4096,
                optuna={"type": "int", "low": 8, "high": 64},
                grid=[12, 20, 32],
            ),
            "local_extremum_window": HyperParam.integer(
                "local_extremum_window",
                default=200,
                low=10,
                high=20000,
                optuna={"type": "int", "low": 48, "high": 400},
                grid=[96, 160, 240],
            ),
            "entry_score": HyperParam.floating(
                "entry_score",
                default=0.18,
                low=0.01,
                high=0.95,
                optuna={"type": "float", "low": 0.03, "high": 0.40, "step": 0.01},
                grid=[0.08, 0.12, 0.18, 0.24],
            ),
            "exit_score": HyperParam.floating(
                "exit_score",
                default=0.55,
                low=0.03,
                high=0.99,
                optuna={"type": "float", "low": 0.35, "high": 0.90, "step": 0.01},
                grid=[0.45, 0.55, 0.65, 0.75],
            ),
            "exit_score_min_gap": HyperParam.floating(
                "exit_score_min_gap",
                default=0.02,
                low=0.0,
                high=0.5,
                optuna={"type": "float", "low": 0.0, "high": 0.2, "step": 0.01},
                grid=[0.0, 0.02, 0.04, 0.08],
            ),
            "entry_streak": HyperParam.integer(
                "entry_streak",
                default=3,
                low=2,
                high=100,
                optuna={"type": "int", "low": 2, "high": 8},
                grid=[2, 3, 4, 5],
            ),
            "stop_loss_pct": HyperParam.floating(
                "stop_loss_pct",
                default=0.03,
                low=0.002,
                high=0.4,
                optuna={"type": "float", "low": 0.005, "high": 0.10, "step": 0.005},
                grid=[0.01, 0.02, 0.03, 0.05],
            ),
            "allow_short": HyperParam.boolean(
                "allow_short",
                default=True,
                optuna={"type": "categorical", "choices": [True, False]},
                grid=[True, False],
            ),
            "diff": HyperParam.boolean(
                "diff",
                default=False,
                optuna={"type": "categorical", "choices": [False, True]},
                grid=[False, True],
            ),
            "history_padding_bars": HyperParam.integer(
                "history_padding_bars",
                default=12,
                low=0,
                high=20000,
                tunable=False,
            ),
        }

    def __init__(
        self,
        bars,
        events,
        history_bars: int = 512,
        lookbacks: tuple[int, ...] = (1, 2, 3, 4, 5),
        return_factor: float = 1.0,
        trend_rolling_window: int = 20,
        local_extremum_window: int = 200,
        entry_score: float = 0.18,
        exit_score: float = 0.55,
        exit_score_min_gap: float = 0.02,
        entry_streak: int = 3,
        stop_loss_pct: float = 0.03,
        allow_short: bool = True,
        diff: bool = False,
        history_padding_bars: int = 12,
    ):
        self.bars = bars
        self.events = events
        self.symbol_list = list(self.bars.symbol_list)
        resolved = resolve_params_from_schema(
            self.get_param_schema(),
            {
                "history_bars": history_bars,
                "lookbacks": lookbacks,
                "return_factor": return_factor,
                "trend_rolling_window": trend_rolling_window,
                "local_extremum_window": local_extremum_window,
                "entry_score": entry_score,
                "exit_score": exit_score,
                "exit_score_min_gap": exit_score_min_gap,
                "entry_streak": entry_streak,
                "stop_loss_pct": stop_loss_pct,
                "allow_short": allow_short,
                "diff": diff,
                "history_padding_bars": history_padding_bars,
            },
            keep_unknown=False,
        )

        self.lookbacks = tuple(int(x) for x in resolved["lookbacks"])
        self.return_factor = float(resolved["return_factor"])
        self.trend_rolling_window = int(resolved["trend_rolling_window"])
        self.local_extremum_window = int(resolved["local_extremum_window"])
        self.entry_score = float(resolved["entry_score"])
        self.exit_score_min_gap = float(resolved["exit_score_min_gap"])
        self.exit_score = max(
            self.entry_score + self.exit_score_min_gap,
            float(resolved["exit_score"]),
        )
        self.entry_streak = int(resolved["entry_streak"])
        self.stop_loss_pct = float(resolved["stop_loss_pct"])
        self.allow_short = bool(resolved["allow_short"])
        self.diff = bool(resolved["diff"])
        self.history_padding_bars = int(resolved["history_padding_bars"])

        base_history = int(resolved["history_bars"])
        min_required = max(
            max(self.lookbacks) + 4,
            self.trend_rolling_window + self.history_padding_bars,
            self.local_extremum_window + 2,
        )
        self._history_bars = max(base_history, min_required)

        self._state = {
            symbol: _SymbolState(closes=deque(maxlen=self._history_bars)) for symbol in self.symbol_list
        }

    def get_state(self):
        return {
            "symbol_state": {
                symbol: {
                    "closes": list(item.closes),
                    "mode": item.mode,
                    "entry_price": item.entry_price,
                    "last_time_key": item.last_time_key,
                }
                for symbol, item in self._state.items()
            }
        }

    def set_state(self, state):
        if not isinstance(state, dict):
            return
        symbol_state = state.get("symbol_state")
        if not isinstance(symbol_state, dict):
            return
        for symbol, raw in symbol_state.items():
            if symbol not in self._state or not isinstance(raw, dict):
                continue
            item = self._state[symbol]
            item.closes.clear()
            for value in list(raw.get("closes") or [])[-item.closes.maxlen :]:
                parsed = safe_float(value)
                if parsed is not None:
                    item.closes.append(parsed)

            restored_mode = str(raw.get("mode", "OUT")).upper()
            item.mode = restored_mode if restored_mode in {"OUT", "LONG", "SHORT"} else "OUT"
            item.entry_price = safe_float(raw.get("entry_price"))
            item.last_time_key = str(raw.get("last_time_key", ""))

    def _resolve_close(self, symbol, event):
        if getattr(event, "symbol", None) == symbol:
            event_time = getattr(event, "time", getattr(event, "datetime", None))
            close = safe_float(getattr(event, "close", None))
            if close is not None:
                return event_time, close
        event_time = self.bars.get_latest_bar_datetime(symbol)
        close = safe_float(self.bars.get_latest_bar_value(symbol, "close"))
        if close is None:
            return None, None
        return event_time, close

    def _emit(self, symbol, event_time, signal_type, *, stop_loss=None, metadata=None):
        self.events.put(
            SignalEvent(
                strategy_id="rare_event_score",
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
        event_time, close = self._resolve_close(symbol, event)
        if close is None:
            return

        bar_key = time_key(event_time)
        if bar_key and bar_key == item.last_time_key:
            return
        item.last_time_key = bar_key

        item.closes.append(close)
        scores = rare_event_scores_latest(
            item.closes,
            lookbacks=self.lookbacks,
            return_factor=self.return_factor,
            trend_rolling_window=self.trend_rolling_window,
            local_extremum_window=self.local_extremum_window,
            diff=self.diff,
            max_points=self._history_bars,
        )
        if scores is None:
            return

        rarity = float(scores.composite_score)
        streak = int(scores.rare_streak_value)
        metadata = {
            "strategy": "RareEventScoreStrategy",
            "rarity_score": rarity,
            "rare_return_score": float(scores.rare_return_score),
            "rare_return_lookback": int(scores.rare_return_lookback),
            "rare_streak_score": float(scores.rare_streak_score),
            "rare_streak_value": streak,
            "trend_break_score": float(scores.trend_break_score),
            "local_extremum_score": float(scores.local_extremum_score),
            "local_extremum_side": int(scores.local_extremum_side),
        }

        if item.mode == "LONG":
            stop_hit = close <= (item.entry_price or close) * (1.0 - self.stop_loss_pct)
            normalize_hit = rarity >= self.exit_score or streak >= -1
            if stop_hit or normalize_hit:
                self._emit(
                    symbol,
                    event_time,
                    "EXIT",
                    metadata={**metadata, "reason": "long_stop" if stop_hit else "long_normalize"},
                )
                item.mode = "OUT"
                item.entry_price = None
            return

        if item.mode == "SHORT":
            stop_hit = close >= (item.entry_price or close) * (1.0 + self.stop_loss_pct)
            normalize_hit = rarity >= self.exit_score or streak <= 1
            if stop_hit or normalize_hit:
                self._emit(
                    symbol,
                    event_time,
                    "EXIT",
                    metadata={**metadata, "reason": "short_stop" if stop_hit else "short_normalize"},
                )
                item.mode = "OUT"
                item.entry_price = None
            return

        if rarity > self.entry_score:
            return

        if streak <= -self.entry_streak:
            stop_loss = close * (1.0 - self.stop_loss_pct)
            self._emit(symbol, event_time, "LONG", stop_loss=stop_loss, metadata=metadata)
            item.mode = "LONG"
            item.entry_price = close
            return

        if self.allow_short and streak >= self.entry_streak:
            stop_loss = close * (1.0 + self.stop_loss_pct)
            self._emit(symbol, event_time, "SHORT", stop_loss=stop_loss, metadata=metadata)
            item.mode = "SHORT"
            item.entry_price = close

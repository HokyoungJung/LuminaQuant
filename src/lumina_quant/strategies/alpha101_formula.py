"""Alpha101 formula strategy with tunable constant overrides.

IMPORTANT SIGNAL SEMANTICS.  The published WorldQuant Alpha101 formulas are
CROSS-SECTIONAL: ``rank()`` ranks each name against the *universe* at a single
timestamp, and the alpha's economic content lives in that cross-section.  The
default single-asset adaptation here replaces ``rank()`` with a TIME-SERIES
self-rank (a symbol ranked against its own history), which is a DIFFERENT,
weaker signal -- NOT the published Alpha101.  Treat single-asset output as a
time-series factor, not as the cross-sectional alpha.

When a real cross-section is available (>= 2 symbols printing the same bar), the
opt-in, config-gated ``cross_sectional_rank`` flag (default OFF => byte-identical)
standardizes each name's raw alpha score across the peer cross-section at that
bar, restoring the cross-sectional economic content.  With the flag OFF the
emitted signals are byte-identical to the legacy time-series self-rank book.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from lumina_quant.core.events import SignalEvent
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.indicators.formulaic_alpha import compute_alpha101
from lumina_quant.strategy_defaults import (
    ALPHA101_ID_UPPER_BOUND,
    ALPHA101_SCORE_STATE_MIN_HISTORY,
    ALPHA101_ZSCORE_CAP,
)
from lumina_quant.strategy import Strategy
from lumina_quant.tuning import HyperParam, resolve_params_from_schema


@dataclass(slots=True)
class _Alpha101SymbolState:
    opens: deque
    highs: deque
    lows: deque
    closes: deque
    volumes: deque
    scores: deque
    mode: str = "OUT"
    entry_price: float | None = None
    last_time_key: str = ""


def _coerce_override_map(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    out: dict[str, float] = {}
    for key, raw in value.items():
        try:
            out[str(key)] = float(raw)
        except Exception:
            continue
    return out


def _latest_zscore(values: deque, *, window: int) -> float | None:
    items = [float(item) for item in list(values) if math.isfinite(float(item))]
    window_i = max(8, int(window))
    if len(items) < window_i:
        return None
    tail = items[-window_i:]
    latest = float(tail[-1])
    hist = tail[:-1]
    if not hist:
        return None
    mean = float(sum(hist) / len(hist))
    variance = float(sum((item - mean) ** 2 for item in hist) / max(1, len(hist) - 1))
    std = math.sqrt(max(0.0, variance))
    if std <= 1e-12:
        delta = latest - mean
        if abs(delta) <= 1e-12:
            return 0.0
        return ALPHA101_ZSCORE_CAP if delta > 0.0 else -ALPHA101_ZSCORE_CAP
    return (latest - mean) / std


class Alpha101FormulaStrategy(Strategy):
    """Apply one Alpha101 formula per symbol and trade factor z-score extremes.

    By default the per-symbol score is standardized against its OWN history
    (time-series self-rank) -- a weaker, DIFFERENT signal than the published
    cross-sectional Alpha101.  Set ``cross_sectional_rank=True`` (default OFF) to
    standardize across the peer cross-section at each bar instead, restoring the
    cross-sectional economic content when >= 2 symbols are available.
    """

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "alpha_id": HyperParam.integer(
                "alpha_id",
                default=101,
                low=1,
                high=101,
                tunable=False,
            ),
            "rank_window": HyperParam.integer(
                "rank_window",
                default=20,
                low=4,
                high=256,
                tunable=False,
            ),
            "history_window": HyperParam.integer(
                "history_window",
                default=160,
                low=32,
                high=2048,
                tunable=False,
            ),
            "score_window": HyperParam.integer(
                "score_window",
                default=48,
                low=8,
                high=1024,
                tunable=False,
            ),
            "entry_z": HyperParam.floating(
                "entry_z",
                default=1.2,
                low=0.1,
                high=10.0,
                tunable=False,
            ),
            "exit_z": HyperParam.floating(
                "exit_z",
                default=0.35,
                low=0.0,
                high=10.0,
                tunable=False,
            ),
            "signal_sign": HyperParam.categorical(
                "signal_sign",
                default=1.0,
                choices=(1.0, -1.0),
                tunable=False,
            ),
            "stop_loss_pct": HyperParam.floating(
                "stop_loss_pct",
                default=0.03,
                low=0.001,
                high=0.5,
                tunable=False,
            ),
            "allow_short": HyperParam.boolean(
                "allow_short",
                default=True,
                tunable=False,
            ),
        }

    def __init__(
        self,
        bars,
        events,
        alpha_id: int = 101,
        rank_window: int = 20,
        history_window: int = 160,
        score_window: int = 48,
        entry_z: float = 1.2,
        exit_z: float = 0.35,
        signal_sign: float = 1.0,
        stop_loss_pct: float = 0.03,
        allow_short: bool = True,
        alpha_param_overrides: Mapping[str, float] | None = None,
        cross_sectional_rank: bool = False,
    ):
        self.bars = bars
        self.events = events
        self.symbol_list = list(self.bars.symbol_list)
        resolved = resolve_params_from_schema(
            self.get_param_schema(),
            {
                "alpha_id": alpha_id,
                "rank_window": rank_window,
                "history_window": history_window,
                "score_window": score_window,
                "entry_z": entry_z,
                "exit_z": exit_z,
                "signal_sign": signal_sign,
                "stop_loss_pct": stop_loss_pct,
                "allow_short": allow_short,
                "alpha_param_overrides": alpha_param_overrides,
            },
            keep_unknown=True,
        )

        self.alpha_id = max(1, min(ALPHA101_ID_UPPER_BOUND, int(resolved["alpha_id"])))
        self.rank_window = max(4, int(resolved["rank_window"]))
        self.history_window = max(32, int(resolved["history_window"]))
        self.score_window = max(8, int(resolved["score_window"]))
        self.entry_z = float(resolved["entry_z"])
        self.exit_z = max(0.0, float(resolved["exit_z"]))
        self.signal_sign = -1.0 if float(resolved["signal_sign"]) < 0.0 else 1.0
        self.stop_loss_pct = float(resolved["stop_loss_pct"])
        self.allow_short = bool(resolved["allow_short"])
        # STRATEGY-IMPROVE (strategy-local, config-gated, default OFF => byte-
        # identical to the legacy time-series self-rank). When ON and a real
        # cross-section is available, standardize the raw alpha score across
        # peers at each bar (true cross-sectional rank).
        self.cross_sectional_rank = bool(cross_sectional_rank)
        # Transient (NOT serialized) cache of each symbol's latest (bar_key,
        # score) used only for the opt-in cross-sectional standardization.
        self._xs_latest: dict[str, tuple[str, float]] = {}
        self.alpha_param_overrides = _coerce_override_map(resolved.get("alpha_param_overrides"))

        history_len = max(self.history_window + 4, self.score_window + 16)
        self._state = {
            symbol: _Alpha101SymbolState(
                opens=deque(maxlen=history_len),
                highs=deque(maxlen=history_len),
                lows=deque(maxlen=history_len),
                closes=deque(maxlen=history_len),
                volumes=deque(maxlen=history_len),
                scores=deque(maxlen=max(self.score_window + 8, ALPHA101_SCORE_STATE_MIN_HISTORY)),
            )
            for symbol in self.symbol_list
        }

    def get_state(self):
        return {
            "symbol_state": {
                symbol: {
                    "opens": list(item.opens),
                    "highs": list(item.highs),
                    "lows": list(item.lows),
                    "closes": list(item.closes),
                    "volumes": list(item.volumes),
                    "scores": list(item.scores),
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
            for field in ("opens", "highs", "lows", "closes", "volumes", "scores"):
                target = getattr(item, field)
                target.clear()
                for value in list(raw.get(field) or [])[-target.maxlen :]:
                    parsed = safe_float(value)
                    if parsed is not None:
                        target.append(parsed)
            restored_mode = str(raw.get("mode", "OUT")).upper()
            item.mode = restored_mode if restored_mode in {"OUT", "LONG", "SHORT"} else "OUT"
            item.entry_price = safe_float(raw.get("entry_price"))
            item.last_time_key = str(raw.get("last_time_key", ""))

    def _resolve_bar(
        self, symbol: str, event
    ) -> tuple[Any, float | None, float | None, float | None, float | None, float | None]:
        if getattr(event, "symbol", None) == symbol:
            event_time = getattr(event, "time", getattr(event, "datetime", None))
            open_price = safe_float(getattr(event, "open", None))
            high_price = safe_float(getattr(event, "high", None))
            low_price = safe_float(getattr(event, "low", None))
            close_price = safe_float(getattr(event, "close", None))
            volume = safe_float(getattr(event, "volume", None))
            if (
                open_price is not None
                and high_price is not None
                and low_price is not None
                and close_price is not None
                and volume is not None
            ):
                return event_time, open_price, high_price, low_price, close_price, volume

        event_time = self.bars.get_latest_bar_datetime(symbol)
        open_price = safe_float(self.bars.get_latest_bar_value(symbol, "open"))
        high_price = safe_float(self.bars.get_latest_bar_value(symbol, "high"))
        low_price = safe_float(self.bars.get_latest_bar_value(symbol, "low"))
        close_price = safe_float(self.bars.get_latest_bar_value(symbol, "close"))
        volume = safe_float(self.bars.get_latest_bar_value(symbol, "volume"))
        return event_time, open_price, high_price, low_price, close_price, volume

    def _compute_score(self, item: _Alpha101SymbolState) -> float | None:
        if len(item.closes) < self.history_window:
            return None
        value = compute_alpha101(
            self.alpha_id,
            opens=list(item.opens),
            highs=list(item.highs),
            lows=list(item.lows),
            closes=list(item.closes),
            volumes=list(item.volumes),
            rank_window=self.rank_window,
            param_overrides=self.alpha_param_overrides,
        )
        if value is None:
            return None
        value_f = float(value)
        return self.signal_sign * value_f if math.isfinite(value_f) else None

    def _cross_sectional_zscore(
        self, item: _Alpha101SymbolState, symbol: str, bar_key: str, score: float
    ) -> float | None:
        """Standardize ``score`` across peers printing the same bar.

        Records this symbol's current-bar score, then z-scores it against every
        symbol whose latest cached score shares ``bar_key`` (a true, causal
        cross-section: only current-bar scores are used).  When fewer than two
        peers have printed for this bar it is genuinely single-asset, so it falls
        back to the time-series self-rank z-score.
        """
        self._xs_latest[symbol] = (str(bar_key), float(score))
        peers = [value for (key, value) in self._xs_latest.values() if bar_key and key == bar_key]
        if len(peers) < 2:
            return _latest_zscore(item.scores, window=self.score_window)
        mean = float(sum(peers) / len(peers))
        variance = float(sum((value - mean) ** 2 for value in peers) / (len(peers) - 1))
        std = math.sqrt(max(0.0, variance))
        if std <= 1e-12:
            delta = float(score) - mean
            if abs(delta) <= 1e-12:
                return 0.0
            return ALPHA101_ZSCORE_CAP if delta > 0.0 else -ALPHA101_ZSCORE_CAP
        z = (float(score) - mean) / std
        return max(-ALPHA101_ZSCORE_CAP, min(ALPHA101_ZSCORE_CAP, z))

    def _emit(
        self,
        symbol: str,
        event_time: Any,
        signal_type: str,
        *,
        zscore: float | None,
        stop_loss: float | None = None,
    ) -> None:
        self.events.put(
            SignalEvent(
                strategy_id="alpha101_formula",
                symbol=symbol,
                datetime=event_time,
                signal_type=signal_type,
                strength=1.0,
                stop_loss=stop_loss,
                metadata={
                    "strategy": "Alpha101FormulaStrategy",
                    "alpha_id": int(self.alpha_id),
                    "rank_window": int(self.rank_window),
                    "signal_sign": float(self.signal_sign),
                    "zscore": None if zscore is None else float(zscore),
                    "alpha_param_override_count": len(self.alpha_param_overrides),
                },
            )
        )

    def calculate_signals(self, event) -> None:
        if getattr(event, "type", None) != "MARKET":
            return
        symbol = getattr(event, "symbol", None)
        if symbol not in self._state:
            return

        item = self._state[symbol]
        event_time, open_price, high_price, low_price, close_price, volume = self._resolve_bar(
            symbol, event
        )
        if (
            open_price is None
            or high_price is None
            or low_price is None
            or close_price is None
            or volume is None
        ):
            return

        bar_key = time_key(event_time)
        if bar_key and bar_key == item.last_time_key:
            return
        item.last_time_key = bar_key

        item.opens.append(open_price)
        item.highs.append(high_price)
        item.lows.append(low_price)
        item.closes.append(close_price)
        item.volumes.append(max(0.0, volume))

        score = self._compute_score(item)
        if score is None:
            return
        item.scores.append(score)
        if self.cross_sectional_rank:
            zscore = self._cross_sectional_zscore(item, symbol, bar_key, score)
        else:
            zscore = _latest_zscore(item.scores, window=self.score_window)
        if zscore is None:
            return

        if item.mode == "LONG":
            stop_hit = close_price <= (item.entry_price or close_price) * (1.0 - self.stop_loss_pct)
            if stop_hit or zscore <= self.exit_z:
                self._emit(symbol, event_time, "EXIT", zscore=zscore)
                item.mode = "OUT"
                item.entry_price = None
            elif self.allow_short and zscore <= -self.entry_z:
                self._emit(symbol, event_time, "EXIT", zscore=zscore)
                self._emit(
                    symbol,
                    event_time,
                    "SHORT",
                    zscore=zscore,
                    stop_loss=close_price * (1.0 + self.stop_loss_pct),
                )
                item.mode = "SHORT"
                item.entry_price = close_price
            return

        if item.mode == "SHORT":
            stop_hit = close_price >= (item.entry_price or close_price) * (1.0 + self.stop_loss_pct)
            if stop_hit or zscore >= -self.exit_z:
                self._emit(symbol, event_time, "EXIT", zscore=zscore)
                item.mode = "OUT"
                item.entry_price = None
            elif zscore >= self.entry_z:
                self._emit(symbol, event_time, "EXIT", zscore=zscore)
                self._emit(
                    symbol,
                    event_time,
                    "LONG",
                    zscore=zscore,
                    stop_loss=close_price * (1.0 - self.stop_loss_pct),
                )
                item.mode = "LONG"
                item.entry_price = close_price
            return

        if zscore >= self.entry_z:
            self._emit(
                symbol,
                event_time,
                "LONG",
                zscore=zscore,
                stop_loss=close_price * (1.0 - self.stop_loss_pct),
            )
            item.mode = "LONG"
            item.entry_price = close_price
        elif self.allow_short and zscore <= -self.entry_z:
            self._emit(
                symbol,
                event_time,
                "SHORT",
                zscore=zscore,
                stop_loss=close_price * (1.0 + self.stop_loss_pct),
            )
            item.mode = "SHORT"
            item.entry_price = close_price

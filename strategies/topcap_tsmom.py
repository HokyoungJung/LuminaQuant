"""Top-cap time-series momentum strategy with long/short rotation."""

from __future__ import annotations

import logging
from collections import deque
from itertools import islice
from statistics import mean

from lumina_quant.events import SignalEvent
from lumina_quant.indicators import momentum_return, safe_float, safe_int, time_key
from lumina_quant.strategy import Strategy
from lumina_quant.tuning import HyperParam, resolve_params_from_schema

LOGGER = logging.getLogger(__name__)


class TopCapTimeSeriesMomentumStrategy(Strategy):
    """Long/short momentum rotation over a top-cap symbol universe.

    Rules:
    - Compute momentum over `lookback_bars` on aligned bars.
    - Rebalance every `rebalance_bars`.
    - Long strongest symbols above `signal_threshold`.
    - Short weakest symbols below `-signal_threshold`.
    - Optional BTC regime gate: risk-on keeps longs only, risk-off keeps shorts only.

    No-lookahead:
    - Uses only bars available at signal timestamp.
    - Orders are still executed by the engine on next bar open.
    """

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "lookback_bars": HyperParam.integer(
                "lookback_bars",
                default=16,
                low=3,
                high=8192,
                tunable=False,
            ),
            "rebalance_bars": HyperParam.integer(
                "rebalance_bars",
                default=16,
                low=1,
                high=8192,
                tunable=False,
            ),
            "signal_threshold": HyperParam.floating(
                "signal_threshold",
                default=0.04,
                low=0.0,
                high=1.0,
                tunable=False,
            ),
            "stop_loss_pct": HyperParam.floating(
                "stop_loss_pct",
                default=0.08,
                low=0.01,
                high=0.5,
                tunable=False,
            ),
            "max_longs": HyperParam.integer(
                "max_longs",
                default=6,
                low=1,
                high=256,
                tunable=False,
            ),
            "max_shorts": HyperParam.integer(
                "max_shorts",
                default=5,
                low=1,
                high=256,
                tunable=False,
            ),
            "min_price": HyperParam.floating(
                "min_price",
                default=0.10,
                low=0.0,
                high=1000000.0,
                tunable=False,
            ),
            "btc_regime_ma": HyperParam.integer(
                "btc_regime_ma",
                default=48,
                low=0,
                high=8192,
                tunable=False,
            ),
            "btc_symbol": HyperParam.string("btc_symbol", default="BTC/USDT", tunable=False),
        }

    def __init__(
        self,
        bars,
        events,
        lookback_bars=16,
        rebalance_bars=16,
        signal_threshold=0.04,
        stop_loss_pct=0.08,
        max_longs=6,
        max_shorts=5,
        min_price=0.10,
        btc_regime_ma=48,
        btc_symbol=None,
    ):
        self.bars = bars
        self.events = events
        self.symbol_list = list(self.bars.symbol_list)
        if not self.symbol_list:
            raise ValueError("TopCapTimeSeriesMomentumStrategy requires at least one symbol.")

        resolved = resolve_params_from_schema(
            self.get_param_schema(),
            {
                "lookback_bars": lookback_bars,
                "rebalance_bars": rebalance_bars,
                "signal_threshold": signal_threshold,
                "stop_loss_pct": stop_loss_pct,
                "max_longs": max_longs,
                "max_shorts": max_shorts,
                "min_price": min_price,
                "btc_regime_ma": btc_regime_ma,
                "btc_symbol": btc_symbol,
            },
            keep_unknown=False,
        )

        self.lookback_bars = int(resolved["lookback_bars"])
        self.rebalance_bars = int(resolved["rebalance_bars"])
        self.signal_threshold = float(resolved["signal_threshold"])
        self.stop_loss_pct = float(resolved["stop_loss_pct"])
        self.max_longs = int(resolved["max_longs"])
        self.max_shorts = int(resolved["max_shorts"])
        self.min_price = float(resolved["min_price"])
        self.btc_regime_ma = int(resolved["btc_regime_ma"])

        default_btc = "BTC/USDT" if "BTC/USDT" in self.symbol_list else self.symbol_list[0]
        btc_symbol = resolved["btc_symbol"]
        self.btc_symbol = str(btc_symbol) if btc_symbol else default_btc
        if self.btc_symbol not in self.symbol_list:
            self.btc_symbol = default_btc

        history_len = max(self.lookback_bars, self.btc_regime_ma) + 2
        self._price_history = {symbol: deque(maxlen=history_len) for symbol in self.symbol_list}

        self._position_state = dict.fromkeys(self.symbol_list, "OUT")
        self._last_symbol_time_key = dict.fromkeys(self.symbol_list, "")
        self._last_eval_time_key = ""
        self._tick = 0

    def get_state(self):
        return {
            "position_state": dict(self._position_state),
            "last_symbol_time_key": dict(self._last_symbol_time_key),
            "last_eval_time_key": str(self._last_eval_time_key),
            "tick": int(self._tick),
            "price_history": {
                symbol: list(history) for symbol, history in self._price_history.items()
            },
        }

    def set_state(self, state):
        if not isinstance(state, dict):
            return

        raw_positions = state.get("position_state")
        if isinstance(raw_positions, dict):
            for symbol, mode in raw_positions.items():
                if symbol in self._position_state and mode in {"OUT", "LONG", "SHORT"}:
                    self._position_state[symbol] = mode

        raw_time_keys = state.get("last_symbol_time_key")
        if isinstance(raw_time_keys, dict):
            for symbol, value in raw_time_keys.items():
                if symbol in self._last_symbol_time_key:
                    self._last_symbol_time_key[symbol] = str(value)

        self._last_eval_time_key = str(state.get("last_eval_time_key", ""))
        self._tick = max(0, safe_int(state.get("tick", 0), 0))

        raw_history = state.get("price_history")
        if not isinstance(raw_history, dict):
            return
        for symbol, values in raw_history.items():
            if symbol not in self._price_history or not isinstance(values, list):
                continue
            target = self._price_history[symbol]
            target.clear()
            keep = int(target.maxlen) if target.maxlen is not None else len(values)
            for value in values[-keep:]:
                parsed = safe_float(value)
                if parsed is not None and parsed > 0.0:
                    target.append(parsed)

    def _resolve_close(self, symbol, event):
        if getattr(event, "symbol", None) == symbol:
            close = safe_float(getattr(event, "close", None))
            if close is not None and close > 0.0:
                return close
        close = safe_float(self.bars.get_latest_bar_value(symbol, "close"))
        if close is None or close <= 0.0:
            return None
        return close

    def _btc_regime(self):
        if self.btc_regime_ma <= 0:
            return "BOTH"
        btc_history = self._price_history.get(self.btc_symbol)
        if btc_history is None or len(btc_history) <= self.btc_regime_ma:
            return "BOTH"
        start = len(btc_history) - self.btc_regime_ma
        avg = mean(islice(btc_history, start, None))
        return "RISK_ON" if btc_history[-1] >= avg else "RISK_OFF"

    def _emit_signal(self, symbol, event_time, signal_type, momentum, regime):
        close_price = safe_float(self.bars.get_latest_bar_value(symbol, "close"))
        stop_loss = None
        if close_price is not None and close_price > 0.0:
            if signal_type == "LONG":
                stop_loss = close_price * (1.0 - self.stop_loss_pct)
            elif signal_type == "SHORT":
                stop_loss = close_price * (1.0 + self.stop_loss_pct)

        metadata = {
            "strategy": "TopCapTimeSeriesMomentumStrategy",
            "momentum": float(momentum) if momentum is not None else None,
            "regime": regime,
            "lookback_bars": int(self.lookback_bars),
            "signal_threshold": float(self.signal_threshold),
        }
        self.events.put(
            SignalEvent(
                strategy_id="topcap_tsmom",
                symbol=symbol,
                datetime=event_time,
                signal_type=signal_type,
                strength=1.0,
                stop_loss=stop_loss,
                metadata=metadata,
            )
        )

    def _build_targets(self, aligned_symbols):
        momentum_rows = []
        for symbol in aligned_symbols:
            history = self._price_history[symbol]
            if len(history) <= self.lookback_bars:
                continue
            latest = history[-1]
            if latest < self.min_price:
                continue
            base = history[-1 - self.lookback_bars]
            if base <= 0.0:
                continue
            momentum = momentum_return(latest, base)
            if momentum is None:
                continue
            momentum_rows.append((momentum, symbol))

        if not momentum_rows:
            return {}, {}, "BOTH"

        momentum_rows.sort(key=lambda row: row[0])
        momentum_map = {symbol: momentum for momentum, symbol in momentum_rows}

        longs = [
            symbol
            for momentum, symbol in reversed(momentum_rows)
            if momentum >= self.signal_threshold
        ][: self.max_longs]
        shorts = [
            symbol for momentum, symbol in momentum_rows if momentum <= -self.signal_threshold
        ][: self.max_shorts]

        regime = self._btc_regime()
        if regime == "RISK_ON":
            shorts = []
        elif regime == "RISK_OFF":
            longs = []

        long_set = set(longs)
        shorts = [symbol for symbol in shorts if symbol not in long_set]

        target = dict.fromkeys(longs, "LONG")
        target.update(dict.fromkeys(shorts, "SHORT"))
        return target, momentum_map, regime

    def calculate_signals(self, event):
        if getattr(event, "type", None) != "MARKET":
            return

        event_symbol = getattr(event, "symbol", None)
        if event_symbol not in self._price_history:
            return

        event_time = self.bars.get_latest_bar_datetime(event_symbol)
        event_time_key = time_key(event_time)
        if not event_time_key:
            return
        if self._last_symbol_time_key.get(event_symbol) == event_time_key:
            return
        self._last_symbol_time_key[event_symbol] = event_time_key

        close_price = self._resolve_close(event_symbol, event)
        if close_price is None:
            return
        self._price_history[event_symbol].append(close_price)

        if event_time_key == self._last_eval_time_key:
            return

        aligned_symbols = []
        for symbol in self.symbol_list:
            if time_key(self.bars.get_latest_bar_datetime(symbol)) != event_time_key:
                continue
            if len(self._price_history[symbol]) <= self.lookback_bars:
                continue
            aligned_symbols.append(symbol)

        minimum = max(4, min(len(self.symbol_list), self.max_longs + self.max_shorts))
        if len(aligned_symbols) < minimum:
            return

        self._last_eval_time_key = event_time_key
        self._tick += 1
        if self._tick % self.rebalance_bars != 0:
            return

        targets, momentum_map, regime = self._build_targets(aligned_symbols)

        for symbol, state in list(self._position_state.items()):
            target_state = targets.get(symbol, "OUT")
            momentum = momentum_map.get(symbol)
            if state == target_state:
                continue
            if state != "OUT" and target_state == "OUT":
                self._emit_signal(symbol, event_time, "EXIT", momentum, regime)
                self._position_state[symbol] = "OUT"
            elif state == "LONG" and target_state == "SHORT":
                self._emit_signal(symbol, event_time, "EXIT", momentum, regime)
                self._emit_signal(symbol, event_time, "SHORT", momentum, regime)
                self._position_state[symbol] = "SHORT"
            elif state == "SHORT" and target_state == "LONG":
                self._emit_signal(symbol, event_time, "EXIT", momentum, regime)
                self._emit_signal(symbol, event_time, "LONG", momentum, regime)
                self._position_state[symbol] = "LONG"

        for symbol, target_state in targets.items():
            if self._position_state.get(symbol) != "OUT":
                continue
            signal_type = "LONG" if target_state == "LONG" else "SHORT"
            self._emit_signal(symbol, event_time, signal_type, momentum_map.get(symbol), regime)
            self._position_state[symbol] = target_state

        LOGGER.info(
            "TopCap rebalance @%s regime=%s longs=%d shorts=%d",
            event_time_key,
            regime,
            sum(1 for state in self._position_state.values() if state == "LONG"),
            sum(1 for state in self._position_state.values() if state == "SHORT"),
        )

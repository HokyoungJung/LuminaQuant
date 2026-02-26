"""Bitcoin buy-and-hold baseline strategy.

This strategy emits a single LONG signal on the configured symbol and then
holds indefinitely. It is intentionally simple and serves as a stable baseline
for comparison and later customization.
"""

from __future__ import annotations

from lumina_quant.core.events import SignalEvent
from lumina_quant.strategy import Strategy
from lumina_quant.tuning import HyperParam, resolve_params_from_schema


class BitcoinBuyHoldStrategy(Strategy):
    """One-shot long entry baseline strategy for BTC buy-and-hold."""

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "symbol": HyperParam.string("symbol", default="BTC/USDT", tunable=False),
            "strength": HyperParam.floating(
                "strength",
                default=1.0,
                low=0.0,
                high=5.0,
                tunable=False,
            ),
        }

    def __init__(self, bars, events, symbol: str = "BTC/USDT", strength: float = 1.0):
        self.bars = bars
        self.events = events
        self.symbol_list = list(getattr(self.bars, "symbol_list", []))
        if not self.symbol_list:
            raise ValueError("BitcoinBuyHoldStrategy requires at least one symbol.")

        resolved = resolve_params_from_schema(
            self.get_param_schema(),
            {
                "symbol": symbol,
                "strength": strength,
            },
            keep_unknown=False,
        )
        symbol = str(resolved["symbol"])
        strength = float(resolved["strength"])

        requested = str(symbol).strip()
        if requested in self.symbol_list:
            self.target_symbol = requested
        elif "BTC/USDT" in self.symbol_list:
            self.target_symbol = "BTC/USDT"
        else:
            self.target_symbol = str(self.symbol_list[0])

        self.strength = float(strength)
        self._entered = False
        self._last_signal_time = None

    def get_state(self) -> dict:
        return {
            "target_symbol": self.target_symbol,
            "strength": self.strength,
            "entered": self._entered,
            "last_signal_time": self._last_signal_time,
        }

    def set_state(self, state: dict) -> None:
        if not isinstance(state, dict):
            return

        target_symbol = str(state.get("target_symbol") or "").strip()
        if target_symbol and target_symbol in self.symbol_list:
            self.target_symbol = target_symbol

        strength = state.get("strength")
        if strength is not None:
            try:
                self.strength = float(strength)
            except Exception:
                pass

        entered = state.get("entered")
        if entered is not None:
            self._entered = bool(entered)

        self._last_signal_time = state.get("last_signal_time")

    def calculate_signals(self, event) -> None:
        if getattr(event, "type", None) != "MARKET":
            return

        event_symbol = getattr(event, "symbol", None)
        if event_symbol != self.target_symbol:
            return

        if self._entered:
            return

        event_time = getattr(event, "time", None)
        if event_time is None:
            event_time = getattr(event, "datetime", None)
        if event_time is not None and event_time == self._last_signal_time:
            return

        self.events.put(
            SignalEvent(
                strategy_id="bitcoin_buy_hold",
                symbol=self.target_symbol,
                datetime=event_time,
                signal_type="LONG",
                strength=self.strength,
                metadata={
                    "strategy": "BitcoinBuyHoldStrategy",
                    "mode": "one_shot_entry",
                },
            )
        )
        self._entered = True
        self._last_signal_time = event_time

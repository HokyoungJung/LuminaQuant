import talib
import numpy as np

from lumina_quant.strategy import Strategy
from lumina_quant.events import SignalEvent


class RsiStrategy(Strategy):
    """
    RSI Strategy:
    - Long when RSI < 30 (Oversold)
    - Exit when RSI > 70 (Overbought) or > 50 (Mean Reversion)
    """

    def __init__(self, bars, events, rsi_period=14, oversold=30, overbought=70):
        self.bars = bars
        self.events = events
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.symbol_list = self.bars.symbol_list
        self.bought = {s: "OUT" for s in self.symbol_list}

    def get_state(self):
        return {"bought": dict(self.bought)}

    def set_state(self, state):
        if "bought" in state and isinstance(state["bought"], dict):
            self.bought = state["bought"]

    def calculate_signals(self, event):
        if event.type == "MARKET":
            for s in self.symbol_list:
                # RSI needs at least period + 1 data points
                bars = self.bars.get_latest_bars_values(
                    s, "close", N=self.rsi_period + 5
                )

                if len(bars) > self.rsi_period:
                    closes = np.array(bars)

                    # Calculate RSI
                    rsi_values = talib.RSI(closes, timeperiod=self.rsi_period)
                    current_rsi = rsi_values[-1]

                    # Trading Logic
                    if current_rsi < self.oversold and self.bought[s] == "OUT":
                        print(
                            f"LONG Signal: {s} | RSI: {current_rsi:.2f} < {self.oversold}"
                        )
                        signal = SignalEvent(1, s, event.time, "LONG", 1.0)
                        self.events.put(signal)
                        self.bought[s] = "LONG"

                    elif current_rsi > self.overbought and self.bought[s] == "LONG":
                        print(
                            f"EXIT Signal: {s} | RSI: {current_rsi:.2f} > {self.overbought}"
                        )
                        signal = SignalEvent(1, s, event.time, "EXIT", 1.0)
                        self.events.put(signal)
                        self.bought[s] = "OUT"

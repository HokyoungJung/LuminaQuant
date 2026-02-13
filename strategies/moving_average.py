import talib
import numpy as np

from lumina_quant.strategy import Strategy
from lumina_quant.events import SignalEvent


class MovingAverageCrossStrategy(Strategy):
    """
    Carries out a basic Moving Average Crossover strategy with a short/long simple weighted moving average.
    """

    def __init__(self, bars, events, short_window=10, long_window=30):
        self.bars = bars
        self.events = events
        self.short_window = short_window
        self.long_window = long_window
        self.symbol_list = self.bars.symbol_list
        self.bought = {s: "OUT" for s in self.symbol_list}

    def get_state(self):
        return {"bought": dict(self.bought)}

    def set_state(self, state):
        if "bought" in state:
            self.bought = state["bought"]

    def calculate_signals(self, event):
        if event.type == "MARKET":
            for s in self.symbol_list:
                bars = self.bars.get_latest_bars_values(
                    s, "close", N=self.long_window + 1
                )  # Get enough data

                if len(bars) > self.long_window:
                    closes = np.array(bars)

                    # TA-LIB usage
                    ma_short = talib.SMA(closes, timeperiod=self.short_window)
                    ma_long = talib.SMA(closes, timeperiod=self.long_window)

                    symbol = s
                    curr_short = ma_short[-1]
                    curr_long = ma_long[-1]

                    # Trading logic
                    if curr_short > curr_long and self.bought[s] == "OUT":
                        print(
                            f"LONG Signal: {s} | Short: {curr_short} > Long: {curr_long}"
                        )
                        sig_dir = "LONG"
                        signal = SignalEvent(1, symbol, event.time, sig_dir, 1.0)
                        self.events.put(signal)
                        self.bought[s] = "LONG"

                    elif curr_short < curr_long and self.bought[s] == "LONG":
                        print(
                            f"EXIT Signal: {s} | Short: {curr_short} < Long: {curr_long}"
                        )
                        sig_dir = "EXIT"
                        signal = SignalEvent(1, symbol, event.time, sig_dir, 1.0)
                        self.events.put(signal)
                        self.bought[s] = "OUT"

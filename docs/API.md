# Developer API Reference

This guide explains how to extend LuminaQuant by creating custom strategies and components.

## 1. Creating a Strategy

Strategies inherit from the `Strategy` abstract base class.

### Class Structure

```python
from lumina_quant.strategy import Strategy
from lumina_quant.events import SignalEvent

class MyStrategy(Strategy):
    def __init__(self, bars, events, my_param=10):
        """
        bars: DataHandler instance (gives access to historical data)
        events: EventQueue (put SignalEvents here)
        **kwargs: Any parameters defined in config.yaml
        """
        self.bars = bars
        self.events = events
        self.my_param = my_param
        self.symbol_list = self.bars.symbol_list

    def calculate_signals(self, event):
        """
        Called on every 'MARKET' event (new bar).
        """
        if event.type == "MARKET":
            for s in self.symbol_list:
                # 1. Get Data
                # get_latest_bars_values(symbol, "close", N) returns list of floats
                closes = self.bars.get_latest_bars_values(s, "close", N=self.my_param)
                
                # 2. Logic
                if len(closes) < self.my_param:
                    continue
                
                # 3. Generate Signal
                if closes[-1] > closes[0]:
                    # SignalEvent(strategy_id, symbol, datetime, signal_type, strength)
                    # signal_type: "LONG", "SHORT", "EXIT"
                    signal = SignalEvent(1, s, event.time, "LONG", 1.0)
                    self.events.put(signal)

    def get_state(self):
        return {}

    def set_state(self, state):
        _ = state
```

## 2. Data Handler API

The `DataHandler` provides methods to access market data during backtests and live trading.

- `get_latest_bar(symbol)`: Returns the full latest OHLCV tuple.
- `get_latest_bars(symbol, N=1)`: Returns list of last N tuples.
- `get_latest_bar_value(symbol, val_type)`: Returns a single float value (e.g. "close", "high").
- `get_latest_bars_values(symbol, val_type, N=1)`: Returns list of floats.

## 3. Exchange Interface (`ExchangeInterface`)

If you want to add a new exchange driver, implement this interface.

```python
class ExchangeInterface(ABC):
    @abstractmethod
    def connect(self): pass

    @abstractmethod
    def get_balance(self, currency: str) -> float: pass

    @abstractmethod
    def get_all_positions(self) -> Dict[str, float]: pass

    @abstractmethod
    def execute_order(self, ...): pass
    
    @abstractmethod
    def fetch_open_orders(self, symbol=None): pass
    
    @abstractmethod
    def cancel_order(self, order_id, symbol=None): pass
```

See `lumina_quant/exchanges/` for `CCXTExchange` and `MT5Exchange` implementations.

## 4. Runtime Config Layer

- New typed API: `lumina_quant.configuration.load_runtime_config`.
- Env override prefix: `LQ_` (for nested keys: `LQ__LIVE__EXCHANGE__LEVERAGE=3`).
- Backward compatibility: legacy keys still work through `lumina_quant.config`.

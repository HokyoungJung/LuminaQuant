# Developer API Reference

This guide explains how to extend LuminaQuant by creating custom strategies and components.

## 1. Creating a Strategy

Strategies inherit from the `Strategy` abstract base class.

### Class Structure

```python
from lumina_quant.strategy import Strategy
from lumina_quant.core.events import SignalEvent


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

### Optional richer strategy contract

Strategies can now declare the canonical inputs they require without depending on provider-specific raw payloads.

```python
class MyContextStrategy(Strategy):
    required_inputs = ("market_window",)
    required_features = ("feature_points",)
    preferred_contract = "context"  # market_event | market_window | context

    def calculate_signals_context(self, context):
        event = context.event
        aggregator = context.aggregator
        feature_lookup = context.feature_lookup
        _ = (event, aggregator, feature_lookup)
```

Dispatch order is:
1. `calculate_signals_context(context)` when `preferred_contract == "context"`
2. otherwise `calculate_signals_window(event, aggregator)`
3. otherwise legacy `calculate_signals(event)`

## 2. Data Handler API

The `DataHandler` provides methods to access market data during backtests and live trading.

- `get_latest_bar(symbol)`: Returns the full latest OHLCV tuple.
- `get_latest_bars(symbol, N=1)`: Returns list of last N tuples.
- `get_latest_bar_value(symbol, val_type)`: Returns a single float value (e.g. "close", "high").
- `get_latest_bars_values(symbol, val_type, N=1)`: Returns list of floats.

External data can be supplied through canonical OHLCV CSV/parquet roots (backtest/optimize) or canonical `MARKET_WINDOW` / 1s OHLCV external live adapters. See `docs/EXTERNAL_DATA.md`.

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

See `lumina_quant/exchanges/` for `BinanceFuturesExchange`, `MT5Exchange`, and the Phase 1 `PolymarketExchange` implementation.

## 4. Runtime Config Layer

- New typed API: `lumina_quant.configuration.load_runtime_config`.
- Env override prefix: `LQ_` (for nested keys: `LQ__LIVE__EXCHANGE__LEVERAGE=3`).
- Backward compatibility: legacy keys still work through `lumina_quant.config`.

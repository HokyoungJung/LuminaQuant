# Exchange Setup & Configuration

LuminaQuant supports multiple exchanges through a unified interface. Currently supported drivers are **CCXT** (for Crypto) and **MetaTrader 5** (for Forex/Stocks).

## 1. Binance (via CCXT)

LuminaQuant relies on the `ccxt` library to connect to over 100+ crypto exchanges, with first-class support for Binance.

### Prerequisites
- Python `ccxt` package (installed via `uv sync`).
- A Binance Account (Real or Testnet).
- API Key and Secret Key.

### Configuration (`config.yaml`)

```yaml
live:
  mode: "paper"  # paper|real
  exchange:
    driver: "ccxt"
    name: "binance"
    market_type: "future"   # spot|future
    position_mode: "HEDGE"  # ONEWAY|HEDGE
    margin_mode: "isolated" # isolated|cross
    leverage: 3
```

### Environment Variables (`.env`)
Create a `.env` file in the project root:

```ini
BINANCE_API_KEY=your_actual_api_key
BINANCE_SECRET_KEY=your_actual_secret_key
```

### Advanced Usage (Parameters)
When executing orders in your strategy, you can pass CCXT-specific parameters via the `params` dictionary.

```python
# Example: Sending a Time-In-Force 'GTC' Limit Order
self.execution_handler.execute_order(
    OrderEvent(
        symbol="BTC/USDT",
        order_type="LMT",
        quantity=0.1,
        headers={"price": 50000}, # Config dependent
        direction="BUY"
    ),
    params={"timeInForce": "GTC"} # Passed directly to ccxt.create_order
)
```

---

## 2. MetaTrader 5 (MT5)

Direct integration with the MetaTrader 5 terminal allows trading Forex, CFDs, Stocks, and Futures.

### Prerequisites
1.  **OS**: Windows (MT5 Python API is Windows-only).
2.  **Software**: MetaTrader 5 Terminal installed and running.
3.  **Account**: Logged into a Demo or Real account in the terminal.
4.  **Settings**:
    *   Go to **Tools -> Options -> Expert Advisors**.
    *   Check ✅ **Allow algorithmic trading**.
    *   Check ✅ **Allow DLL imports** (sometimes required depending on setup).

### Configuration (`config.yaml`)

```yaml
live:
  exchange:
    driver: "mt5"
    name: "metatrader" # This name is ignored by the mt5 driver
  
  # Default Order Settings
  mt5_magic: 234000      # Unique ID for this bot's orders
  mt5_deviation: 20      # Max slippage deviation in points
  mt5_comment: "LuminaBot"
```

### Advanced Usage (Parameters)
You can override global settings per-order using `params`.

```python
# Example: Sending an order with a custom Magic Number
signal = SignalEvent(...)
# In Execution Handler or via custom order creation:
exchange.execute_order(
    symbol="EURUSD",
    type="market",
    side="buy",
    quantity=0.1,
    params={
        "magic": 999999,        # Override config magic
        "deviation": 50,        # Override deviation
        "comment": "SuperStrat" # Custom comment
    }
)
```

### Troubleshooting MT5
- **"Not Connected"**: Ensure the MT5 Terminal is open and you are logged in.
- **"IPC Error"**: Sometimes happens if MT5 is run as Administrator but Python is not, or vice versa. Run both with same permissions.
- **Symbol Not Found**: Ensure the symbol (e.g., "EURUSD") is visible in the **Market Watch** window in MT5.

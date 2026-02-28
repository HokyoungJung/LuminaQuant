# Trading Manual: Concrete Operations

This guide provides concrete instructions and code snippets for common trading operations in LuminaQuant.

## 0. Live Safety Quick Checklist (Ops)

Before running in `live.mode=real`:

1. Keep `live.require_real_enable_flag: true` in `config.yaml`.
2. Start with explicit arming:
   - `LUMINA_ENABLE_LIVE_REAL=true uv run python run_live.py --enable-live-real`
3. Prefer graceful shutdown with stop-file:
   - launch with `--stop-file /tmp/lq.stop`
   - stop by `touch /tmp/lq.stop`
4. Keep emergency fallback documented for your host (systemd stop / process kill).

## 1. Buying and Selling

In LuminaQuant, you do not "place orders" directly in a loop. Instead, your **Strategy** generates **Signals**, which the system converts into Orders.

### The Signal Flow
1.  **Strategy** (`calculate_signals`) detects a condition.
2.  **Strategy** emits a `SignalEvent` ("LONG", "SHORT", or "EXIT").
3.  **Portfolio** receives Signal -> Checks Risk -> Generates `OrderEvent`.
4.  **ExecutionHandler** receives Order -> Sends to Exchange (Binance/MT5).

### Code Example: Basic Buy/Sell
Inside your strategy's `calculate_signals(self, event)` method:

```python
# LONG (Buy)
# signal_type="LONG" -> System buys positive quantity
self.events.put(SignalEvent(
    strategy_id=1,
    symbol="BTC/USDT",
    datetime=event.time,
    signal_type="LONG",
    strength=1.0
))

# SHORT (Sell)
# signal_type="SHORT" -> System sells (opens short)
self.events.put(SignalEvent(
    strategy_id=1,
    symbol="BTC/USDT",
    datetime=event.time,
    signal_type="SHORT",
    strength=1.0
))

# EXIT (Close Position)
# signal_type="EXIT" -> System calculates current qty and flattens it
self.events.put(SignalEvent(
    strategy_id=1,
    symbol="BTC/USDT",
    datetime=event.time,
    signal_type="EXIT",
    strength=1.0
))
```

---

## 2. Take Profit (TP) & Stop Loss (SL)

There are two ways to implement TP/SL: **Hard** (Exchange-side) and **Soft** (Strategy-side).

### Method A: Hard TP/SL (Exchange Side)
*Best for reliability. The order sits on the exchange.*

#### MetaTrader 5 (MT5)
You must modify `live_execution.py` or use a custom execution logic to pass `params` into the order.
Currently, `LiveExecutionHandler` does not automatically attach `params` from `SignalEvent`.

**Recommended Approach**:
In your strategy, when you want to send an order with TP/SL, interacting directly with the `exchange` object is possible but breaks the backtest abstraction.
**Better**: The `SignalEvent` class doesn't carry TP/SL data by default. You would typically execute "Hard" TP/SL by customizing the `Portfolio` or `ExecutionHandler`.

**However**, if you interact with the exchange manually in your strategy (Advanced):

```python
# Advanced: Direct Exchange Call (Live Only)
if self.mode == "LIVE":
    self.exchange.execute_order(
        symbol="EURUSD",
        type="market",
        side="buy",
        quantity=0.1,
        params={
            "sl": 1.0500,  # Stop Loss Price
            "tp": 1.0700   # Take Profit Price
        }
    )
```

#### Binance (CCXT)
Similar to MT5, pass `stopLoss` or `takeProfit` if supported, or send separate OCO orders.
```python
params = {
    "stopLoss": { "triggerPrice": 49000 },
    "takeProfit": { "triggerPrice": 55000 }
}
```

### Method B: Soft TP/SL (Strategy Side)
*Universal method (works in Backtest & Live).*
You track the price in your strategy and send an `EXIT` signal when limits are hit.

```python
class MyStrategy(Strategy):
    def __init__(self, ...):
        self.entry_price = {}

    def calculate_signals(self, event):
        if event.type == "MARKET":
            price = self.bars.get_latest_bar_value(s, "close")
            
            # Check Long Exit
            if self.bought[s] == "LONG":
                entry = self.entry_price[s]
                
                # Stop Loss (e.g., 2% drop)
                if price < entry * 0.98:
                    self.events.put(SignalEvent(..., signal_type="EXIT"))
                
                # Take Profit (e.g., 5% gain)
                elif price > entry * 1.05:
                    self.events.put(SignalEvent(..., signal_type="EXIT"))
```

---

## 3. Trailing Stop

Trailing stops are best implemented as **Soft Stops** in the strategy to ensure consistency between Backtest and Live.

### Implementation Guide
1.  Track `highest_price` since entry (for Long).
2.  Update `stop_price` dynamically: `stop_price = highest_price * (1 - trail_percent)`.
3.  If current price < `stop_price`, trigger EXIT.

```python
class TrailingStopStrategy(Strategy):
    def __init__(self, ..., trail_pct=0.02):
        self.trail_pct = trail_pct
        self.high_water_mark = {} # Track highest price since entry

    def calculate_signals(self, event):
        price = ...
        
        # Logic for LONG position
        if self.bought[s] == "LONG":
            # 1. Update High Water Mark
            if price > self.high_water_mark[s]:
                self.high_water_mark[s] = price
            
            # 2. Calculate Trailing Stop Price
            stop_price = self.high_water_mark[s] * (1 - self.trail_pct)
            
            # 3. Check Breach
            if price < stop_price:
                self.events.put(SignalEvent(..., signal_type="EXIT"))
```

---

## 4. Leverage Settings

### Binance (Futures)
Leverage is usually an account setting, but can be set via API.
LuminaQuant's `CCXTExchange` does not auto-set leverage on connect.

**How to set it:**
Add this to your strategy's `__init__` or `live_trader.py`:

```python
# Inside Strategy or Setup
if self.exchange.name == "binance":
    # Set 10x leverage for BTC/USDT
    self.exchange.exchange.set_leverage(10, "BTC/USDT")
```

### MetaTrader 5
Leverage is determined by your **Broker Account Settings**. You cannot change it via API in most cases. You must login to your broker portal to change account leverage (e.g. 1:100, 1:500).

---

## 5. Order Types

By default, LuminaQuant uses **Market Orders** for immediate execution.
To use **Limit Orders**, you must modify the `ExecutionHandler` or extending `SignalEvent` to carry a `price`.

**Current Default**:
- Signal `LONG`/`SHORT` -> `OrderEvent(type='MKT')` -> `execute_order(type='market')`.

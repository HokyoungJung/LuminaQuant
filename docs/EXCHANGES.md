# Exchange Setup & Configuration

LuminaQuant supports multiple exchange drivers through a unified interface.

For crypto, the production-critical path is now **native Binance USDⓈ-M Futures only**.
CCXT is not required for Binance live trading, user streams, or historical market-data ingestion.

## 1. Binance USDⓈ-M Futures (native)

### Supported scope
- native Futures REST market data
- native Futures aggTrade historical ingestion
- native Futures websocket aggTrade live market data
- native Futures order placement / cancel / query
- native Futures balances / positions / open orders
- native Futures user data stream
- native Futures leverage / margin / position-mode controls

### Configuration (`config.yaml`)

```yaml
live:
  mode: "paper"  # paper|real
  market_data_source: "binance_futures"  # committed|binance_futures|external|polymarket_live
  order_state_source: "user_stream"   # polling|user_stream
  exchange:
    driver: "binance_futures"
    name: "binance"
    market_type: "future"
    position_mode: "HEDGE"   # ONEWAY|HEDGE
    margin_mode: "isolated"  # isolated|cross
    leverage: 3
```

### Environment variables (`.env`)

```ini
BINANCE_API_KEY=your_actual_api_key
BINANCE_SECRET_KEY=your_actual_secret_key
```

### Notes
- `live.exchange.market_type` must remain `future` for Binance.
- Canonical raw market data is **aggTrades**.
- 1s bars are derived from raw aggTrades.
- Higher timeframes are resampled from real lower-timeframe data.
- Final validation is real-data-only and latest-anchored; see `docs/FINAL_VALIDATION.md`.

---

## 2. MetaTrader 5 (MT5)

Direct integration with the MetaTrader 5 terminal allows trading Forex, CFDs, Stocks, and Futures.

### Prerequisites
1. **OS**: Windows (MT5 Python API is Windows-only).
2. **Software**: MetaTrader 5 Terminal installed and running.
3. **Account**: Logged into a Demo or Real account in the terminal.
4. **Settings**:
   - Tools -> Options -> Expert Advisors
   - Allow algorithmic trading
   - Allow DLL imports (if required by your setup)

### Configuration (`config.yaml`)

```yaml
live:
  exchange:
    driver: "mt5"
    name: "metatrader"

  mt5_magic: 234000
  mt5_deviation: 20
```

---

## 3. Polymarket (Phase 1)

Current support is scoped to:
- market-data ingestion
- signal generation
- paper/shadow execution lanes
- experimental real order placement/cancel/open-order polling when `allow_real_execution: true`

### Configuration (`config.yaml`)

```yaml
live:
  market_data_source: polymarket_live
  exchange:
    driver: "polymarket"
    name: "polymarket"
```

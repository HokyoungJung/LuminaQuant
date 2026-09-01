# Multiasset exchange expansion coverage inventory

Generated: `2026-05-07T11:03:16.099763Z`
Market root: `data/market_parquet`
Safe OOS end for BTC/ETH/SOL raw-first replay/backtest: `2026-05-06`

## Raw-first OHLCV coverage

| symbol | train | val | oos | first OOS missing |
|---|---:|---:|---:|---|
| `BTC/USDT` | 365/365 | 59/59 | 67/67 | `` |
| `ETH/USDT` | 365/365 | 59/59 | 67/67 | `` |
| `SOL/USDT` | 365/365 | 59/59 | 67/67 | `` |
| `BNB/USDT` | 365/365 | 59/59 | 67/67 | `` |
| `TRX/USDT` | 365/365 | 59/59 | 67/67 | `` |
| `XRP/USDT` | 0/365 | 0/59 | 0/67 | `2026-03-01,2026-03-02,2026-03-03` |
| `ADA/USDT` | 0/365 | 0/59 | 0/67 | `2026-03-01,2026-03-02,2026-03-03` |
| `DOGE/USDT` | 0/365 | 0/59 | 0/67 | `2026-03-01,2026-03-02,2026-03-03` |
| `AVAX/USDT` | 0/365 | 0/59 | 0/67 | `2026-03-01,2026-03-02,2026-03-03` |
| `TON/USDT` | 0/365 | 0/59 | 0/67 | `2026-03-01,2026-03-02,2026-03-03` |
| `XAU/USDT` | 0/365 | 0/59 | 0/67 | `2026-03-01,2026-03-02,2026-03-03` |
| `XAG/USDT` | 0/365 | 0/59 | 0/67 | `2026-03-01,2026-03-02,2026-03-03` |
| `XPT/USDT` | 0/365 | 0/59 | 0/67 | `2026-03-01,2026-03-02,2026-03-03` |
| `XPD/USDT` | 0/365 | 0/59 | 0/67 | `2026-03-01,2026-03-02,2026-03-03` |

## Feature inventory

### `binance`

| symbol | rows | funding | mark | index | OI | taker-flow | liquidation | last timestamp |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `ADAUSDT` | 621383 | 1295 | 620785 | 985 | 222 | 0 | 0 | `2026-03-28T16:24:00+00:00` |
| `AVAXUSDT` | 621383 | 1295 | 620785 | 985 | 222 | 0 | 0 | `2026-03-28T16:24:00+00:00` |
| `BNBUSDT` | 707732 | 1473 | 707039 | 87840 | 17568 | 0 | 0 | `2026-05-06T23:59:00+00:00` |
| `BTCUSDT` | 2703191 | 2712 | 1301202 | 86401 | 17280 | 2101668 | 0 | `2026-05-06T23:59:00+00:00` |
| `DOGEUSDT` | 621383 | 1295 | 620785 | 985 | 222 | 0 | 0 | `2026-03-28T16:24:00+00:00` |
| `ETHUSDT` | 2109455 | 1473 | 707039 | 87840 | 17568 | 2102583 | 0 | `2026-05-06T23:59:00+00:00` |
| `SOLUSDT` | 2107358 | 1473 | 707039 | 87840 | 17568 | 2099436 | 0 | `2026-05-06T23:59:00+00:00` |
| `TONUSDT` | 621991 | 2589 | 620785 | 985 | 197 | 0 | 0 | `2026-03-28T16:24:00+00:00` |
| `TRXUSDT` | 707732 | 1473 | 707039 | 87840 | 17568 | 0 | 0 | `2026-05-06T23:59:00+00:00` |
| `XAGUSDT` | 86631 | 361 | 86286 | 86007 | 8201 | 0 | 0 | `2026-03-28T16:24:00+00:00` |
| `XAUUSDT` | 125766 | 523 | 125260 | 125002 | 8000 | 0 | 0 | `2026-03-28T16:24:00+00:00` |
| `XPDUSDT` | 53078 | 220 | 52873 | 52873 | 2176 | 0 | 0 | `2026-03-28T16:24:00+00:00` |
| `XPTUSDT` | 53093 | 220 | 52888 | 52888 | 2191 | 0 | 0 | `2026-03-28T16:24:00+00:00` |
| `XRPUSDT` | 621383 | 1295 | 620785 | 985 | 222 | 0 | 0 | `2026-03-28T16:24:00+00:00` |

### `hyperliquid`

| symbol | rows | funding | mark | index | OI | taker-flow | liquidation | last timestamp |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `BTCUSDT` | 11737 | 11737 | 1 | 1 | 1 | 0 | 0 | `2026-05-07T09:58:46.992000+00:00` |
| `ETHUSDT` | 11737 | 11737 | 1 | 1 | 1 | 0 | 0 | `2026-05-07T09:58:46.992000+00:00` |
| `SOLUSDT` | 11737 | 11737 | 1 | 1 | 1 | 0 | 0 | `2026-05-07T09:58:46.992000+00:00` |

## Decisions

- BTC/ETH/SOL Binance raw-first train/val are usable; current-tail OOS must stop at the safe complete date if the latest configured OOS day is missing.
- Hyperliquid and Tickmill are feature/regime sources only. Direct trading is blocked until spread/swap/funding/fill/session/lot-size models and raw-first evidence exist.
- Tickmill/MT5 coverage remains blocked when the MT5 bridge is not configured; the Tickmill collector report records this separately.

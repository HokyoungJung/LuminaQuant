# Multiasset exchange expansion coverage inventory

Generated: `2026-05-07T09:59:24.318291Z`
Market root: `data/market_parquet`
Safe OOS end for BTC/ETH/SOL raw-first replay/backtest: `2026-05-04`

## Raw-first OHLCV coverage

| symbol | train | val | oos | first OOS missing |
|---|---:|---:|---:|---|
| `BTC/USDT` | 365/365 | 59/59 | 65/66 | `2026-05-05` |
| `ETH/USDT` | 365/365 | 59/59 | 65/66 | `2026-05-05` |
| `SOL/USDT` | 365/365 | 59/59 | 65/66 | `2026-05-05` |
| `XAU/USDT` | 0/365 | 0/59 | 0/66 | `2026-03-01,2026-03-02,2026-03-03` |
| `XAG/USDT` | 0/365 | 0/59 | 0/66 | `2026-03-01,2026-03-02,2026-03-03` |
| `XPT/USDT` | 0/365 | 0/59 | 0/66 | `2026-03-01,2026-03-02,2026-03-03` |
| `XPD/USDT` | 0/365 | 0/59 | 0/66 | `2026-03-01,2026-03-02,2026-03-03` |

## Feature inventory

### `binance`

| symbol | rows | funding | mark | index | OI | taker-flow | liquidation | last timestamp |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `BTCUSDT` | 2700563 | 2707 | 1298577 | 83776 | 16755 | 2101668 | 0 | `2026-05-05T04:14:00+00:00` |
| `ETHUSDT` | 2106827 | 1468 | 704414 | 85215 | 17043 | 2102583 | 0 | `2026-05-05T04:14:00+00:00` |
| `SOLUSDT` | 2104730 | 1468 | 704414 | 85215 | 17043 | 2099436 | 0 | `2026-05-05T04:14:00+00:00` |
| `XAGUSDT` | 86631 | 361 | 86286 | 86007 | 8201 | 0 | 0 | `2026-03-28T16:24:00+00:00` |
| `XAUUSDT` | 125766 | 523 | 125260 | 125002 | 8000 | 0 | 0 | `2026-03-28T16:24:00+00:00` |
| `XPDUSDT` | 53078 | 220 | 52873 | 52873 | 2176 | 0 | 0 | `2026-03-28T16:24:00+00:00` |
| `XPTUSDT` | 53093 | 220 | 52888 | 52888 | 2191 | 0 | 0 | `2026-03-28T16:24:00+00:00` |

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

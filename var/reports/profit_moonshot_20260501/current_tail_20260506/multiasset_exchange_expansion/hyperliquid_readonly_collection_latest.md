# Hyperliquid read-only collection

Generated: `2026-05-07T09:59:19.246817Z`
Endpoint: `https://api.hyperliquid.xyz/info`

## External-doc anchors verified at runtime

- Hyperliquid public `/info` supports `metaAndAssetCtxs`, `fundingHistory`, `predictedFundings`, and related perp context requests.
- Hyperliquid public `/info` also documents `candleSnapshot`, with only the most recent 5000 candles available.
- Current official base perp fee tier lists taker 0.045% and maker 0.015%; this report does **not** authorize direct Hyperliquid trading.

## Collection summary

| symbol | funding rows | first funding | last funding | current mark | current OI | candle train/val/oos rows |
|---|---:|---|---|---:|---:|---|
| `BTC/USDT` | 11736 | `2025-01-01T00:00:00.054000+00:00` | `2026-05-04T23:00:00.055000+00:00` | 80900.0 | 31497.60588 | 1968/1416/1560 |
| `ETH/USDT` | 11736 | `2025-01-01T00:00:00.054000+00:00` | `2026-05-04T23:00:00.055000+00:00` | 2327.0 | 524647.7849999991 | 1968/1416/1560 |
| `SOL/USDT` | 11736 | `2025-01-01T00:00:00.054000+00:00` | `2026-05-04T23:00:00.055000+00:00` | 89.393 | 3799441.04 | 1968/1416/1560 |

## Replay eligibility

- Funding history is usable as a read-only confirmation feature when split coverage is present.
- Historical OI is **not** replay-eligible from `metaAndAssetCtxs`; that endpoint provides current context only.
- Historical mark/candle context is not raw-first and train coverage is partial because the official candle endpoint is capped at recent candles; it is report/context only unless a raw-first/listing-aware source is added.
- Direct trading remains blocked: no Hyperliquid fill/funding/liquidation/orderbook parity model was promoted.

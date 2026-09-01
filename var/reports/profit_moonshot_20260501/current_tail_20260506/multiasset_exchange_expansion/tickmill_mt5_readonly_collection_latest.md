# Tickmill/MT5 read-only collection

Generated: `2026-05-07T09:59:19.932244Z`
Status: `blocked`

## External-doc anchors verified at runtime

- Tickmill instruments page lists Forex, cryptocurrencies, commodities, stock indices, stocks/ETFs, and MetaTrader 4/5 platform availability.
- Tickmill spreads/swaps page defines spread as bid/ask difference, requires MT4/MT5 symbol properties for trading hours, and states swaps are applied overnight with Wednesday triple-swap handling.

## Blocker

- `LQ_MT5_BRIDGE_PYTHON / LQ__LIVE__MT5_BRIDGE_PYTHON is not configured.`
- Tickmill macro filters were not replay-eligible because no MT5 read-only OHLCV/properties were available in this session.
- Direct Tickmill trading remains blocked: spread, swap, session, lot-size, and fill assumptions are not modeled from terminal properties.

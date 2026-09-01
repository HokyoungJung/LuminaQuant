# Profit Moonshot — multiasset exchange alpha execution report

Generated: `2026-05-07T10:01:28.645017Z`
Decision: **no new successful alpha promoted**.

## Gate

- Success requires OOS return `> +0.8284%`, OOS MDD `< 0.1778%`, Sharpe `> 1.0`, liquidations `0`, separated train/val/OOS raw-first evidence, and RSS `< 8GB`.
- Sub-1 Sharpe remains shadow-only; no candidate below Sharpe 1.0 is labelled success.
- Hyperliquid/Tickmill direct trading remains blocked until spread/swap/funding/fill/session/lot-size models and raw-first evidence exist.

## Process / sequence

- Started from `private/main` on branch `private-main`; duplicate backtest/refresh/collector process check found `0` relevant processes.
- Executed required order: Hyperliquid read-only → Tickmill/MT5 read-only → stateful replay → no live-equivalent backtest because there were `0` replay survivors.

## Raw-first / feature coverage inventory

- Safe complete crypto OOS end: `2026-05-04` (BTC/ETH/SOL OOS through `2026-05-05` is missing `2026-05-05`).
- BTC/ETH/SOL Binance raw-first train and validation are complete; XAU/XAG/XPT/XPD direct metals remain raw-first blocked for train/val/OOS.

| symbol | train raw-first | val raw-first | OOS raw-first | first OOS missing |
|---|---:|---:|---:|---|
| `BTC/USDT` | 365/365 | 59/59 | 65/66 | `2026-05-05` |
| `ETH/USDT` | 365/365 | 59/59 | 65/66 | `2026-05-05` |
| `SOL/USDT` | 365/365 | 59/59 | 65/66 | `2026-05-05` |
| `XAU/USDT` | 0/365 | 0/59 | 0/66 | `2026-03-01,2026-03-02,2026-03-03` |
| `XAG/USDT` | 0/365 | 0/59 | 0/66 | `2026-03-01,2026-03-02,2026-03-03` |
| `XPT/USDT` | 0/365 | 0/59 | 0/66 | `2026-03-01,2026-03-02,2026-03-03` |
| `XPD/USDT` | 0/365 | 0/59 | 0/66 | `2026-03-01,2026-03-02,2026-03-03` |

## Hyperliquid read-only

- Upserted `35211` `exchange=hyperliquid` feature rows from public `/info` only.
- Funding history covers BTC/ETH/SOL train `8760`, val `1416`, OOS `1560` hourly rows each (`2025-01-01` through `2026-05-04`).
- `metaAndAssetCtxs` supplied current funding/mark/oracle/OI snapshots only; historical OI/mark context is therefore not replay-eligible. Official candle snapshots are context only, not repo raw-first trade evidence.

| symbol | funding rows | first | last | candle train/val/oos rows |
|---|---:|---|---|---|
| `BTC/USDT` | 11736 | `2025-01-01T00:00:00.054000+00:00` | `2026-05-04T23:00:00.055000+00:00` | 1968/1416/1560 |
| `ETH/USDT` | 11736 | `2025-01-01T00:00:00.054000+00:00` | `2026-05-04T23:00:00.055000+00:00` | 1968/1416/1560 |
| `SOL/USDT` | 11736 | `2025-01-01T00:00:00.054000+00:00` | `2026-05-04T23:00:00.055000+00:00` | 1968/1416/1560 |

## Tickmill/MT5 read-only

- Status: `blocked`.
- Blocker: `LQ_MT5_BRIDGE_PYTHON / LQ__LIVE__MT5_BRIDGE_PYTHON is not configured.`
- Macro filters were not replay-eligible because no MT5 read-only terminal/bridge data or symbol properties were available in this session.

## Stateful replay results

- Specs evaluated: `38`
- Replay survivors: `0`
- Success candidates: `0`
- Replay max RSS: `6710.816 MiB` (< 8192 MiB).

| rank | spec | train ret | val ret | OOS ret | OOS Sharpe | OOS MDD | OOS trips | decision |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | `hl_funding_divergence_funding_guard_50ppm` | -0.0641% | -0.3573% | +0.2064% | 0.460760 | +0.1175% | 22 | reject: train/val pre-gate negative |
| 2 | `hl_funding_divergence_funding_guard_75ppm` | -0.0367% | -0.3573% | +0.2064% | 0.460760 | +0.1175% | 22 | reject: train/val pre-gate negative |
| 3 | `hl_funding_divergence_funding_guard_100ppm` | -0.0792% | -0.3573% | +0.2064% | 0.460760 | +0.1175% | 22 | reject: train/val pre-gate negative |
| 4 | `hl_funding_divergence_funding_guard_150ppm` | -0.0792% | -0.3573% | +0.2064% | 0.460760 | +0.1175% | 22 | reject: train/val pre-gate negative |
| 5 | `hl_funding_divergence_funding_guard_250ppm` | -0.0792% | -0.3573% | +0.2064% | 0.460760 | +0.1175% | 22 | reject: train/val pre-gate negative |
| 6 | `hl_funding_divergence_base_50ppm` | -0.0525% | -0.2336% | +0.1530% | 0.338964 | +0.1264% | 22 | reject: train/val pre-gate negative |
| 7 | `hl_funding_divergence_base_75ppm` | -0.1794% | -0.2336% | +0.1530% | 0.338964 | +0.1264% | 22 | reject: train/val pre-gate negative |
| 8 | `hl_funding_divergence_base_100ppm` | -0.2217% | -0.2336% | +0.1530% | 0.338964 | +0.1264% | 22 | reject: train/val pre-gate negative |
| 9 | `hl_funding_divergence_base_150ppm` | -0.2217% | -0.2336% | +0.1530% | 0.338964 | +0.1264% | 22 | reject: train/val pre-gate negative |
| 10 | `hl_funding_divergence_base_250ppm` | -0.2217% | -0.2336% | +0.1530% | 0.338964 | +0.1264% | 22 | reject: train/val pre-gate negative |
| 11 | `replay_base_12h_threshold_100bp` | -0.3704% | -0.0889% | +0.1196% | 0.251137 | +0.1607% | 26 | reject: train/val pre-gate negative |
| 12 | `hl_funding_abs_cap_base_50ppm` | -0.2962% | -0.0889% | +0.1196% | 0.251137 | +0.1607% | 26 | reject: train/val pre-gate negative |
| 13 | `hl_funding_abs_cap_base_75ppm` | -0.3289% | -0.0889% | +0.1196% | 0.251137 | +0.1607% | 26 | reject: train/val pre-gate negative |
| 14 | `hl_funding_abs_cap_base_100ppm` | -0.3704% | -0.0889% | +0.1196% | 0.251137 | +0.1607% | 26 | reject: train/val pre-gate negative |
| 15 | `hl_funding_abs_cap_base_150ppm` | -0.3704% | -0.0889% | +0.1196% | 0.251137 | +0.1607% | 26 | reject: train/val pre-gate negative |

## Live-equivalent raw-first backtest

- Not run: no replay survivor earned a one-mode full backtest slot. This preserves the plan requirement to discard vector/context-only candidates before full engine validation.

## Final decision

- Keep incumbent `profit_moonshot_hourly_shock_reversion_eth_12h_mode` as OOS-return best (`+0.8284%`, MDD `0.2819%`, Sharpe `0.100651`).
- Keep `profit_moonshot_hourly_shock_reversion_eth_12h_funding_guard_mode` as Sharpe/MDD shadow (`+0.7206%`, MDD `0.1778%`, Sharpe `0.111225`).
- Hyperliquid funding divergence improved replay OOS shape in a few variants but failed train/validation gates; Hyperliquid OI/mark and Tickmill macro lanes are blocked by missing replay-eligible read-only history.

## Verification

- `uv run ruff check <changed files>`: passed.
- `uv run python -m py_compile ...` for changed Python files: passed.
- `uv run pytest tests/test_hyperliquid_readonly.py tests/test_mt5_exchange.py tests/test_mt5_exchange_bridge.py -q`: `8 passed`.
- Stateful replay completed with max RSS `6710.816 MiB`.

## Source links

- Hyperliquid perps info endpoint: https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint/perpetuals
- Hyperliquid info endpoint / candle snapshot: https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint
- Hyperliquid fees: https://hyperliquid.gitbook.io/hyperliquid-docs/trading/fees
- Tickmill instruments: https://www.tickmill.com/trading-instruments/
- Tickmill spreads/swaps: https://www.tickmill.com/conditions/spreads-swaps

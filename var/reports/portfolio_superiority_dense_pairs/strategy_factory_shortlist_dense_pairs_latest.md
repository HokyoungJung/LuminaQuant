# Strategy Factory Portfolio Shortlist

- Generated: 2026-04-22T12:24:10.229724+00:00
- Mode: oos
- Manifest: `var/reports/portfolio_superiority_dense_pairs/dense_pairs_manifest.json`
- Research report: `var/reports/portfolio_superiority_dense_pairs/strategy_factory_report_dense_pairs_latest.json`
- Candidate count: 6

## Family / timeframe mix

- Family:
  - market_neutral: 6
- Timeframe:
  - 1h: 6

## Top candidates

| # | Name | Timeframe | Family | Score | Symbols |
|---:|---|---|---|---:|---:|
| 1 | pair_spread_1h_core_bnbusdt_trxusdt_2.6_0.70 | 1h | market_neutral | -499998.9329 | 2 |
| 2 | pair_spread_1h_core_bnbusdt_xauusdt_2.2_0.55 | 1h | market_neutral | -499999.7197 | 2 |
| 3 | pair_spread_1h_core_btcusdt_xauusdt_2.2_0.55 | 1h | market_neutral | -500000.4867 | 2 |
| 4 | pair_spread_1h_core_xauusdt_xagusdt_2.2_0.55 | 1h | market_neutral | -1500000.0000 | 2 |
| 5 | pair_spread_1h_core_btcusdt_xagusdt_2.6_0.70 | 1h | market_neutral | -1500000.0000 | 2 |
| 6 | pair_spread_1h_exec_tightstop_tp_xptusdt_xpdusdt_2.2_0.55 | 1h | market_neutral | -1500000.0000 | 2 |

## Usage

```bash
uv run python scripts/run_research_pipeline.py --backend parquet-postgres
```

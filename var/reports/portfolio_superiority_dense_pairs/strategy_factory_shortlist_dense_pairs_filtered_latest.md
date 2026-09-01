# Strategy Factory Portfolio Shortlist

- Generated: 2026-04-22T12:24:13.091640+00:00
- Mode: oos
- Manifest: `var/reports/portfolio_superiority_dense_pairs/dense_pairs_filtered_manifest.json`
- Research report: `var/reports/portfolio_superiority_dense_pairs/strategy_factory_report_dense_pairs_filtered_latest.json`
- Candidate count: 1

## Family / timeframe mix

- Family:
  - market_neutral: 1
- Timeframe:
  - 1h: 1

## Top candidates

| # | Name | Timeframe | Family | Score | Symbols |
|---:|---|---|---|---:|---:|
| 1 | pair_spread_1h_state_vwap_bnbusdt_trxusdt_2.2_0.55 | 1h | market_neutral | -499999.4003 | 2 |

## Usage

```bash
uv run python scripts/run_research_pipeline.py --backend parquet-postgres
```

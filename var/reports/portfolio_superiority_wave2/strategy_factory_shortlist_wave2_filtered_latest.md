# Strategy Factory Portfolio Shortlist

- Generated: 2026-04-21T14:04:39.171486+00:00
- Mode: oos
- Manifest: `var/reports/portfolio_superiority_wave2/wave2_filtered_manifest.json`
- Research report: `var/reports/portfolio_superiority_wave2/strategy_factory_report_wave2_filtered_latest.json`
- Candidate count: 1

## Family / timeframe mix

- Family:
  - market_neutral: 1
- Timeframe:
  - 1h: 1

## Top candidates

| # | Name | Timeframe | Family | Score | Symbols |
|---:|---|---|---|---:|---:|
| 1 | pair_spread_1h_exec_tightstop_tp_bnbusdt_trxusdt_2.2_0.55 | 1h | market_neutral | 9.4168 | 2 |

## Usage

```bash
uv run python scripts/run_research_pipeline.py --backend parquet-postgres
```

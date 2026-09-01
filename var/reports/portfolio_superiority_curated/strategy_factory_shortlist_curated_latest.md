# Strategy Factory Portfolio Shortlist

- Generated: 2026-04-21T13:02:51.869778+00:00
- Mode: oos
- Manifest: `var/reports/portfolio_superiority_curated/curated_manifest.json`
- Research report: `var/reports/portfolio_superiority_curated/strategy_factory_report_curated_latest.json`
- Candidate count: 5

## Family / timeframe mix

- Family:
  - carry: 1
  - market_neutral: 3
  - trend: 1
- Timeframe:
  - 1h: 3
  - 30m: 2

## Top candidates

| # | Name | Timeframe | Family | Score | Symbols |
|---:|---|---|---|---:|---:|
| 1 | carry_trend_factor_rotation_1h_balanced_lo_24_8_0.200 | 1h | carry | -500000.4920 | 5 |
| 2 | composite_trend_stable_30m_stable_ls_crashguard_ls_0.75_0.45_0.20_0.82 | 30m | trend | -1500000.0000 | 9 |
| 3 | mean_reversion_std_30m_guarded_lo_72_2.20 | 30m | market_neutral | -1500000.0000 | 9 |
| 4 | pair_spread_1h_exec_tightstop_tp_xauusdt_xagusdt_2.2_0.55 | 1h | market_neutral | -1500000.0000 | 2 |
| 5 | pair_spread_1h_exec_tightstop_tp_xptusdt_xpdusdt_2.2_0.55 | 1h | market_neutral | -1500000.0000 | 2 |

## Usage

```bash
uv run python scripts/run_research_pipeline.py --backend parquet-postgres
```

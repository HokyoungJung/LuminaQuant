# Strategy Factory Portfolio Shortlist

- Generated: 2026-04-21T14:04:39.155017+00:00
- Mode: oos
- Manifest: `var/reports/portfolio_superiority_wave2/wave2_manifest.json`
- Research report: `var/reports/portfolio_superiority_wave2/strategy_factory_report_wave2_latest.json`
- Candidate count: 5

## Family / timeframe mix

- Family:
  - carry: 1
  - cross_sectional: 1
  - market_neutral: 3
- Timeframe:
  - 1d: 1
  - 1h: 4

## Top candidates

| # | Name | Timeframe | Family | Score | Symbols |
|---:|---|---|---|---:|---:|
| 1 | pair_spread_1h_exec_tightstop_tp_bnbusdt_trxusdt_2.2_0.55 | 1h | market_neutral | 9.4168 | 2 |
| 2 | last_day_liquidity_regime_1d_guarded_lo_1_1_0.006 | 1d | cross_sectional | -500000.4167 | 5 |
| 3 | perp_crowding_carry_1h_0.25_0.08 | 1h | carry | -500001.4272 | 5 |
| 4 | pair_spread_1h_exec_tightstop_tp_xauusdt_xagusdt_2.2_0.55 | 1h | market_neutral | -1500000.0000 | 2 |
| 5 | pair_spread_1h_exec_tightstop_tp_btcusdt_xauusdt_2.2_0.55 | 1h | market_neutral | -500000.0422 | 2 |

## Usage

```bash
uv run python scripts/run_research_pipeline.py --backend parquet-postgres
```

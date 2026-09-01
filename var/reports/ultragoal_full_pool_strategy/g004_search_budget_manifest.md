# G004 Frozen Candidate and Portfolio Search Budget

- Goal: `G004` Freeze candidate and portfolio search budgets
- Generated: `2026-07-03T14:00:35.611885+00:00`
- Budget hash: `b35dd9a26b2880f6b5e2c8c091fedcd6956871940573ca9d9fc0405f8ec0ac50`
- Candidate set hash: `01ca7a5c04b490b5472a62b49d0fcc7d432f0e2045c0e6fae9b1bfcb079a0564`
- Current HEAD: `42ee1a7e` (`42ee1a7eaadabb9d6e9bd4fb3f57d08fbe50dcca`)
- Saved-artifact commit provided at resume: `e4e4f282`
- Candidate count: `1466` across `20` families
- Historical eligible symbols after quarantine exclusion: `BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, TRXUSDT, XRPUSDT, DOGEUSDT, ADAUSDT, AVAXUSDT`
- G002 historical-ready excluded by quarantine: `TONUSDT`
- Feature-dependent eligible symbols: `BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, TRXUSDT, XRPUSDT, DOGEUSDT, ADAUSDT, AVAXUSDT`
- Quarantined symbols excluded: `119`
- Effective trials upper bound: `24794` = candidates `1466` + portfolio grid `23328`

## Selection policy

Locked OOS is report-only. It cannot alter formulas, params, thresholds, shortlist membership, tie-breaks, portfolio weights, or promotion narrative. Train/validation only drives G005-G007 selection under this frozen budget.

## Cost and risk grids

- Cost stress bps: `[10, 15, 20, 30]`
- Gross caps: `[1.0, 1.5, 2.0]`
- MDD guards: `[0.15, 0.2, 0.3]`
- Construction modes: `equal_weight_ranked_survivors, inverse_volatility_ranked_survivors, quality_gated_erc, quality_gated_hrp`

## Family caps

- `breakout`: `162`
- `carry`: `81`
- `cross_sectional`: `77`
- `deep_research_leaf`: `9`
- `event_alpha`: `18`
- `flow`: `36`
- `formulaic_alpha`: `6`
- `intraday_alpha`: `18`
- `market_neutral`: `209`
- `mean_reversion`: `138`
- `micro`: `3`
- `momentum`: `135`
- `profit_moonshot_breakout`: `2`
- `profit_moonshot_cross_sectional`: `3`
- `profit_moonshot_reversion`: `2`
- `profit_reboot_cross_sectional`: `4`
- `profit_reboot_mean_reversion`: `3`
- `profit_reboot_pair_carry`: `3`
- `seasonality`: `63`
- `trend`: `494`

## Verification

- Verification report: `/home/hoky/Quants-agent/LuminaQuant/var/reports/ultragoal_full_pool_strategy/g004_verification_test_report.json`
- Verification status: `passed` (16/16 assertions passed)

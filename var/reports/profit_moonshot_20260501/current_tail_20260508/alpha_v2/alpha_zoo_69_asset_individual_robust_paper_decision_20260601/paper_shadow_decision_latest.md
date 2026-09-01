# 69-Asset Individual-Robust Paper/Shadow Decision

- decision: `paper_shadow_selected`
- candidate: `individual_robust:hybrid_v3_5`
- ready_for_paper_shadow: `true`
- ready_for_real: `false`
- real_money_execution: `false`
- walkforward OOS comp: `17.78%`
- walkforward OOS pos: `5/10`
- min monthly OOS: `-7.28%`
- max OOS MDD: `15.00%`
- monitored universe: `69` assets
- selected exposure symbols: `13`

## Gate checks

- **PASS** `candidate_family_is_individual_robust`: actual `individual_robust`, expected `individual_robust`
- **PASS** `fold_count`: actual `10`, expected `>= 8`
- **PASS** `compounded_oos_return`: actual `0.17780152462562038`, expected `>= 0.0`
- **PASS** `min_monthly_oos_return`: actual `-0.07281058857782507`, expected `>= -0.08`
- **PASS** `max_oos_mdd`: actual `0.14995273722571856`, expected `<= 0.16`
- **PASS** `positive_validation_folds`: actual `10/10`, expected `>= 100%`
- **PASS** `ready_for_paper_folds`: actual `7/10`, expected `>= 7`
- **PASS** `latest_oos_return`: actual `0.0005307853747047453`, expected `>= -0.02`
- **PASS** `latest_validation_return`: actual `0.3144140223450156`, expected `<= 0.35`

## Top selected symbol exposure

- `BTCUSDT`: `69.55%` gross notional fraction
- `PLTRUSDT`: `66.24%` gross notional fraction
- `TRXUSDT`: `55.76%` gross notional fraction
- `ETHUSDT`: `26.94%` gross notional fraction
- `CRCLUSDT`: `20.40%` gross notional fraction
- `BNBUSDT`: `17.61%` gross notional fraction
- `COINUSDT`: `17.30%` gross notional fraction
- `MSTRUSDT`: `16.24%` gross notional fraction
- `XRPUSDT`: `14.02%` gross notional fraction
- `HOODUSDT`: `11.44%` gross notional fraction
- `AMZNUSDT`: `8.28%` gross notional fraction
- `DOGEUSDT`: `5.01%` gross notional fraction
- `TONUSDT`: `3.30%` gross notional fraction

## Stop rules

- quarantine if monthly OOS return <= -8%
- quarantine if live/paper drawdown exceeds 15%
- quarantine if two consecutive forward months are negative
- quarantine if validation spike guard repeatedly flags the selected hybrid
- real money remains blocked until separate exchange-fill telemetry review

This artifact is paper/shadow only and is not a real-money approval.

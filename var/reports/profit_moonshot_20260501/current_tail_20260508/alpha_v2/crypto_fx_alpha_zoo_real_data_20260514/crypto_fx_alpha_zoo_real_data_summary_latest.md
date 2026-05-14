# Crypto/FX Alpha Zoo real-data return/MDD-diagnostic policy summary — 2026-05-14

- Strategy: `CryptoFxAlphaZooStateStrategy`
- Deployable success: `True`
- Reason: strict zero-liquidation lane passed OOS return, MDD, Sharpe/Sortino/smart Sortino/Calmar gates; return/MDD is diagnostic-only
- Strict rejection reasons: ``

## Selection and calibration provenance

- Selection inputs: `train, validation`
- uses_locked_oos_for_selection: `False`
- Locked-OOS role: `gate/report only after candidate freeze`
- Current-base/calendar tuple: `hypothesis_reference_only`, not a selection or promotion target
- Candidate ledger records: `67259`; train+validation `45311`; locked-OOS `21948`
- Edge calibration records: `45311`; locked-OOS calibration records `0`

## Strict zero-liquidation lane

- Candidate count: `6`
- Deployable candidate count: `5`
- Highest zero-liquidation integer: `6.0`
- Selected strict candidate live status: `deployable_success_true`
- OOS return: `41.0967%` vs current-base `6.4281%`
- OOS return/MDD: `3.007073` vs current-base `6.916878`
- OOS MDD: `13.6667%`
- Sharpe/Sortino/smart Sortino/Calmar: `2.143209` / `2.841936` / `2.500237` / `3.007073`
- Strict safety: `{'strict_safe': True, 'liquidation_count': 0, 'minimum_margin_buffer': 9049.12596153846}`

## Diagnostic nonfatal 5x/6x lane

- `5.0x`: OOS return `33.4851%`, return/MDD `2.911176`, total liquidations `0`, min buffer `9207.604968`, promotion_allowed `False`
- `6.0x`: OOS return `41.0967%`, return/MDD `3.007073`, total liquidations `0`, min buffer `9049.125962`, promotion_allowed `False`

## Paper-forward diagnostics (non-promotional)

- Candidate/leverage: `alpha_zoo_conservative_exit` / `6.0x`
- Trade-return cost model: `allocation_fraction * leverage * (gross_return - round_trip_slippage_bps/10000 - funding_bps_per_day/10000*holding_days)`
- locked-OOS by regime: neutral: 41.0967% (540)
- locked-OOS by symbol: SOL/USDT: 19.2672% (126), BNB/USDT: 10.3167% (128), TRX/USDT: 4.9566% (139), ETH/USDT: 1.4843% (132), BTC/USDT: 0.6807% (15)
- locked-OOS by side: SHORT: 26.3040% (259), LONG: 11.7120% (281)
- locked-OOS by factor family: crypto_residual_momentum: 30.1822% (184), crypto_residual_reversal: 8.4241% (237), volume_vwap_pressure: -0.0370% (119)
- locked-OOS by exit reason: score_exit: 41.0936% (526), take_profit: 16.1976% (4), end_of_sample: -0.0788% (2), stop_loss: -13.8700% (8)
- locked-OOS slippage_sensitivity: 0bps: 41.0967%, 2.5bps: 30.1241%, 5bps: 20.0034%, 10bps: 2.0585%, 20bps: -26.1930%
- locked-OOS funding_cost_sensitivity: 0bps: 41.0967%, 1bps: 40.4210%, 2bps: 39.7486%, 5bps: 37.7505%, 10bps: 34.4835%

## Memory

- peak_rss_mib: `626.726562`
- pass_under_8gb: `True`

## Artifacts

- screen: `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/crypto_fx_alpha_zoo_real_data_20260514/crypto_fx_alpha_zoo_screen_latest.json`
- ledger: `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/crypto_fx_alpha_zoo_real_data_20260514/candidate_outcome_ledger_latest.jsonl`
- calibration: `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/crypto_fx_alpha_zoo_real_data_20260514/edge_calibration_latest.json`
- replay: `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/crypto_fx_alpha_zoo_real_data_20260514/crypto_fx_alpha_zoo_state_replay_latest.json`
- summary: `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/crypto_fx_alpha_zoo_real_data_20260514/crypto_fx_alpha_zoo_real_data_summary_latest.json`
- summary_md: `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/crypto_fx_alpha_zoo_real_data_20260514/crypto_fx_alpha_zoo_real_data_summary_latest.md`

## Research history/source ledger

- regenerated: `False`
- reason: No new external source class or global chronology/source-ledger change; reused existing current-tail cache and 20260512 lagged FRED external-state artifact, added only session-scoped Alpha Zoo return/MDD-diagnostic artifacts.

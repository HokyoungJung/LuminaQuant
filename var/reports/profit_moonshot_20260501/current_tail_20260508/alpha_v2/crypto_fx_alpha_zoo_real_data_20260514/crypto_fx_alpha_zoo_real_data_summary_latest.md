# Crypto/FX Alpha Zoo real-data strict-policy summary — 2026-05-14

- Strategy: `CryptoFxAlphaZooStateStrategy`
- Deployable success: `False`
- Reason: no strict zero-liquidation Alpha Zoo replay row passed all hard gates including OOS return/MDD versus the current-base reference
- Strict rejection reasons: `oos_return_mdd_beats_current_base`

## Selection and calibration provenance

- Selection inputs: `train, validation`
- uses_locked_oos_for_selection: `False`
- Locked-OOS role: `gate/report only after candidate freeze`
- Current-base/calendar tuple: `hypothesis_reference_only`, not a selection or promotion target
- Candidate ledger records: `67259`; train+validation `45311`; locked-OOS `21948`
- Edge calibration records: `45311`; locked-OOS calibration records `0`

## Strict zero-liquidation lane

- Candidate count: `6`
- Deployable candidate count: `0`
- Highest zero-liquidation integer: `6.0`
- Selected strict candidate live status: `no_live_promotion_return_mdd_gate_failed`
- OOS return: `41.0967%` vs current-base `6.4281%`
- OOS return/MDD: `3.007073` vs current-base `6.916878`
- OOS MDD: `13.6667%`
- Sharpe/Sortino/smart Sortino/Calmar: `2.143209` / `2.841936` / `2.500237` / `3.007073`
- Strict safety: `{'strict_safe': True, 'liquidation_count': 0, 'minimum_margin_buffer': 9049.12596153846}`

## Diagnostic nonfatal 5x/6x lane

- `5.0x`: OOS return `33.4851%`, return/MDD `2.911176`, total liquidations `0`, min buffer `9207.604968`, promotion_allowed `False`
- `6.0x`: OOS return `41.0967%`, return/MDD `3.007073`, total liquidations `0`, min buffer `9049.125962`, promotion_allowed `False`

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
- reason: No new external source class or global chronology/source-ledger change; reused existing current-tail cache and 20260512 lagged FRED external-state artifact, added only session-scoped Alpha Zoo strict-policy artifacts.

# Optuna Hybrid Paper/Testnet Handoff

- selected profile: `hybrid_v3_5_optuna_three_profile_blend` (`v3_5`)
- optimizer: Optuna TPESampler, `240` trials per version
- paper/testnet only: `true`; ready_for_real: `false`; real_money_execution: `false`
- train / validation / locked-OOS report-only return: `611.5025%` / `138.3170%` / `20.8319%`
- validation / locked-OOS MDD: `18.9796%` / `10.5735%`
- RPT bps train/validation/OOS: `83.39` / `79.17` / `25.29`
- average train+validation weights: `{"aggressive_mdd30_gross10_shadow": 0.781093636345445, "balanced_mdd12_gross5": 0.10913956140451388, "growth_mdd20_gross8": 0.10976680225004104}`
- final active weights after exposure dampening: `{"aggressive_mdd30_gross10_shadow": 0.5726993181554131, "balanced_mdd12_gross5": 0.07983098667432496, "growth_mdd20_gross8": 0.08067156944866473}`

## Monitoring requirements

1. Paper/testnet only; no real-money routing.
2. Record realized BBO spread, fees, slippage, rejects/timeouts, and all-in round-trip cost.
3. Record replay/live notional parity per source sleeve and hybrid allocation.
4. Record liquidation-inclusive MDD, liquidation count, account wipeout count, and margin buffer.
5. Compare realized return-per-turnover against the primary `10bps` threshold.
6. Keep locked-OOS report-only; do not use it for tuning or selection.

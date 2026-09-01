# Crypto/FX Alpha Zoo real-data result — 2026-05-13

> Superseded on 2026-05-14 by the return/MDD-diagnostic policy in
> `docs/session_handoff_20260514_crypto_fx_alpha_zoo_real_data_strict_policy.md`.
> The 2026-05-13 metrics remain historical; the current Alpha Zoo successor keeps
> return/MDD diagnostic-only and records the strict 6x row as deployable under
> the latest operator correction.

## Decision

- Strategy: `CryptoFxAlphaZooStateStrategy` with real current-tail crypto data, lagged FRED state context, triple-barrier candidate ledger, train/validation-only edge calibration, and narrow formulaic replay grid.
- Result superseded: under the restored 2026-05-14 strict policy this same strict `6.0x` row is `deployable_success=False` because OOS return/MDD `3.007073` does not beat the current-base reference `6.916878`.
- Promotion policy correction: OOS return and OOS return/MDD are both hard deployability gates for this Alpha Zoo lane.
- Best frozen TV-selected Alpha Zoo replay: `alpha_zoo_conservative_exit` (`state_distilled_external_risk_filter_seed`).
- Current-base/calendar tuple remains `hypothesis_reference_only`; it was not a selection or promotion target.

## Real-data factor screen and ledger

- Source: `/home/hoky/Quants-agent/LuminaQuant/var/cache/profit_moonshot_fresh_start/joined_panel_de62df511cec53df6ad39521.parquet`.
- Rows/factors: `58845` rows, `63` factors, `20` selected cards.
- Symbols: `BNB/USDT, BTC/USDT, ETH/USDT, SOL/USDT, TRX/USDT`.
- Direct FX OHLCV trading: blocked; lagged FRED state is regime context only.
- Candidate ledger: `45160` rows; train+validation `30494`; locked-OOS `14666`.
- Factor/card validity: `calendar_primary=False`, `uses_locked_oos_for_selection=False`, `strategy_validity.pass=True`.

## Edge calibration provenance

- Policy: `physical_train_validation_record_filter_before_bucket_estimation`.
- Input rows: `45160`; calibration rows: `30494`.
- Locked-OOS input rows: `14666`; locked-OOS calibration rows: `0`; excluded locked-OOS rows: `14666`.
- Calibrated edge keys: `12`.

## Train/validation-only replay selection

- Grid profile: `narrow_train_validation_formulaic` with `9` candidates.
- Selection inputs: `['train', 'validation']`; `uses_locked_oos_for_selection=False`.
- Locked-OOS metrics were hidden during grid selection and opened only after candidate freeze.

## Locked-OOS gate/report results

- Unlevered selected OOS: return `6.1108%`, return/MDD `2.553222`.
- Strict highest zero-liquidation integer: `6.0x`.
- Strict `6.0x` OOS: return `41.0967%`, MDD `13.6667%`, return/MDD `3.007073`, Sharpe `2.143209`, Sortino `2.841936`, smart Sortino `2.500237`.
- Liquidations: `0`; min margin buffer `9049.125962`; strict safe `True`.
- Restored strict gate result: OOS return beats current-base, OOS MDD is within 25%, Sharpe/Sortino/smart Sortino/Calmar are positive, but return/MDD fails the hard reference hurdle (`3.007073` vs reference `6.916878`).

## Diagnostic nonfatal 5x/6x lane

- `5.0x`: OOS return `33.4851%`, MDD `11.5023%`, return/MDD `2.911176`, total liquidations `0`, min buffer `9207.604968`, promotion_allowed `False` in the diagnostic lane.
- `6.0x`: OOS return `41.0967%`, MDD `13.6667%`, return/MDD `3.007073`, total liquidations `0`, min buffer `9049.125962`; not promoted under the restored strict return/MDD gate, and the diagnostic lane remains non-promotional.

## State-distilled/reference comparison

- Prior valid strict state-distilled external-risk reference remains deployable-false vs current-base: `fresh_state_distilled_ext_both_lb168_fast72_z075_ret180_h168_tp600_fl0_xr125` at `4.0x`, OOS return `2.4852%`, return/MDD `0.981244`, zero-liquidation gates pass.
- New Alpha Zoo strict 6x improves OOS return versus that reference and current-base reference, but the lower current-base return/MDD comparison is a hard promotion failure under the restored 2026-05-14 policy.

## Memory and source-ledger note

- Peak stage RSS: `512.711` MiB (< 8192 MiB).
- Research history/source ledger regeneration: not required — No new external source class or global chronology/source-ledger change; reused existing current-tail cache and 20260512 lagged FRED external-state artifact, added only session-scoped Alpha Zoo artifacts.

## Artifacts

- Screen: `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/crypto_fx_alpha_zoo_real_data_20260513/crypto_fx_alpha_zoo_screen_latest.json`
- Candidate ledger: `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/crypto_fx_alpha_zoo_real_data_20260513/candidate_outcome_ledger_latest.jsonl`
- Edge calibration: `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/crypto_fx_alpha_zoo_real_data_20260513/edge_calibration_latest.json`
- Replay: `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/crypto_fx_alpha_zoo_real_data_20260513/crypto_fx_alpha_zoo_state_replay_latest.json`
- Summary JSON: `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/crypto_fx_alpha_zoo_real_data_20260513/crypto_fx_alpha_zoo_real_data_summary_latest.json`

## Next handoff

Continue from the 2026-05-14 return/MDD-diagnostic successor artifacts, not this superseded deployability note. Treat the strict `6.0x` Alpha Zoo row as the current deployable real-data Alpha Zoo candidate under the latest operator correction. Preserve the non-calendar, train/validation-frozen, locked-OOS gate/report-only, strict zero-liquidation, positive-buffer, OOS MDD <=25%, positive Sharpe/Sortino/smart Sortino/Calmar, and OOS-return gate requirements; keep return/MDD diagnostic-only.

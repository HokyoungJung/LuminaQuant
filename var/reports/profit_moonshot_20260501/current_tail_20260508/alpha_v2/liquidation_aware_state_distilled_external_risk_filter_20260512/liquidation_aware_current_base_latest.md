# Profit moonshot liquidation-aware current-base validation

- generated_at_utc: `2026-05-12T13:29:31.535858Z`
- decision outcome: `no_live_promotion_strategy_validity_failed`
- deployable improvement: `False`
- reselected deployable: `False`
- memory peak RSS: `279.250 MiB`

## Margin model

- mode: `cross`
- maintenance margin rate: `1.0000%`
- stress/funding/fee reserve: `1.9100%`
- Binance docs references recorded in JSON under `source_references`.

## Current base reference replay

- leverage: `2.342733x`
- strategy_validity: `False`
- strategy rejection reasons: `calendar_entry_rule_unsupported, calendar_fixed_month_alpha, calendar_month_entry_rule, calendar_primary_alpha_unsupported, fixed_asset_calendar_target`
- oos: return `+6.4281%`, MDD `0.9293%`, liq `0`, min buffer `9924.1436`, min ratio `187.2044`

## Forced 5x replay

- deployable_success: `False`
- strategy_validity: `False`
- train/validation score: `19.244936`
- OOS return delta vs current-base replay: `+7.6297%`
- OOS return/MDD delta vs current-base replay: `+0.261162`
- train: return `+60.5997%`, MDD `16.2149%`, liq `0`, min buffer `9053.8861`, min ratio `38.4080`
- validation: return `+45.6166%`, MDD `14.0994%`, liq `1`, min buffer `8415.8111`, min ratio `37.1851`
- oos: return `+14.0578%`, MDD `1.9584%`, liq `0`, min buffer `9837.8835`, min ratio `88.9061`

## Selected by train/validation safety

- leverage: `0.000000x`
- locked-OOS used for selection: `False`




## Promotion lanes

- strict lane promotion eligible: `False`
- strict lane rule: `train/validation/OOS liquidation_count must be zero, every split minimum margin buffer must be positive, strategy-validity and train/validation performance gates must pass, then locked-OOS is gate/report-only.`
- diagnostic nonfatal lane promotion eligible: `False`
- diagnostic nonfatal candidates reported: `146`

## Re-selected by train/validation retune

- candidate: `fresh_state_distilled_ext_both_lb168_fast72_z075_ret180_h168_tp600_fl0_xr125`
- source: `candidate_csv_top_train_val_002`
- leverage: `4.000000x`
- deployable_success: `False`
- locked-OOS used for selection: `False`
- train: return `+30.9030%`, MDD `10.2437%`, liq `0`, min buffer `9740.5206`, min ratio `172.3157`
- validation: return `+12.4704%`, MDD `2.5167%`, liq `0`, min buffer `9959.0876`, min ratio `204.1767`
- oos: return `+2.4852%`, MDD `2.5328%`, liq `0`, min buffer `9875.2252`, min ratio `208.3752`

## Best deployable retune candidate

- candidate: `None`
- leverage: `0.000000x`

## Promoted candidate

- candidate: `None`
- source: `None`
- leverage: `0.000000x`




## Decision

- `Calendar-primary current-base tuple is rejected by strategy-validity gates; no live promotion.`

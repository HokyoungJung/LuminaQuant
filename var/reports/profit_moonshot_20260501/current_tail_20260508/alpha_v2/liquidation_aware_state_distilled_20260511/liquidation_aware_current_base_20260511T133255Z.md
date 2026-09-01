# Profit moonshot liquidation-aware current-base validation

- generated_at_utc: `2026-05-11T13:32:55.904696Z`
- decision outcome: `no_live_promotion_strategy_validity_failed`
- deployable improvement: `False`
- reselected deployable: `False`
- memory peak RSS: `266.723 MiB`

## Margin model

- mode: `cross`
- maintenance margin rate: `1.0000%`
- stress/funding/fee reserve: `1.9100%`
- Binance docs references recorded in JSON under `source_references`.

## Current base reference replay

- leverage: `2.342733x`
- strategy_validity: `False`
- strategy rejection reasons: `calendar_fixed_month_alpha, calendar_primary_alpha_unsupported, fixed_asset_calendar_target`
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

- leverage: `4.000000x`
- locked-OOS used for selection: `False`
- train: return `+45.2696%`, MDD `12.9403%`, liq `0`, min buffer `9271.0056`, min ratio `49.6583`
- validation: return `+36.2244%`, MDD `11.2897%`, liq `0`, min buffer `8732.2927`, min ratio `47.8938`
- oos: return `+11.1605%`, MDD `1.5743%`, liq `0`, min buffer `9870.2888`, min ratio `110.6396`

## Re-selected by train/validation retune

- candidate: `fresh_state_distilled_both_lb168_fast72_z075_ret180_h168_ls590_ss100_tp600`
- source: `candidate_csv_top_train_val_000`
- leverage: `4.000000x`
- deployable_success: `False`
- locked-OOS used for selection: `False`
- train: return `+32.9431%`, MDD `9.4768%`, liq `0`, min buffer `9740.5206`, min ratio `177.5034`
- validation: return `+11.6925%`, MDD `3.1028%`, liq `0`, min buffer `9959.0876`, min ratio `204.1618`
- oos: return `+2.4722%`, MDD `2.5328%`, liq `0`, min buffer `9875.3540`, min ratio `208.3866`

## Best deployable retune candidate

- candidate: `None`
- leverage: `0.000000x`

## Promoted candidate

- candidate: `None`
- source: `None`
- leverage: `0.000000x`




## Decision

- `Calendar-primary current-base tuple is rejected by strategy-validity gates; no live promotion.`

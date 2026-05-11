# Profit moonshot liquidation-aware current-base validation

- generated_at_utc: `2026-05-11T12:18:15.543146Z`
- decision outcome: `no_live_promotion_strategy_validity_failed`
- deployable improvement: `False`
- reselected deployable: `False`
- memory peak RSS: `257.195 MiB`

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

- candidate: `fresh_pair_resid_revert_spread_lb24_z150_h72_sc10_st100_tp240_asiaus`
- source: `candidate_csv_top_train_val_002`
- leverage: `1.000000x`
- deployable_success: `False`
- locked-OOS used for selection: `False`
- train: return `+0.0423%`, MDD `0.2462%`, liq `0`, min buffer `9981.9880`, min ratio `4234.7034`
- validation: return `+0.0646%`, MDD `0.0885%`, liq `0`, min buffer `9996.0828`, min ratio `4291.4941`
- oos: return `+0.0241%`, MDD `0.0810%`, liq `0`, min buffer `9993.5388`, min ratio `4338.7711`

## Best deployable retune candidate

- candidate: `None`
- leverage: `0.000000x`

## Promoted candidate

- candidate: `None`
- source: `None`
- leverage: `0.000000x`




## Decision

- `Calendar-primary current-base tuple is rejected by strategy-validity gates; no live promotion.`

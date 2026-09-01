# Profit moonshot liquidation-aware current-base validation

- generated_at_utc: `2026-05-10T06:13:29.573111Z`
- decision outcome: `liquidation_tolerant_reselected_deployable`
- deployable improvement: `True`
- reselected deployable: `True`
- memory peak RSS: `261.203 MiB`

## Margin model

- mode: `cross`
- maintenance margin rate: `1.0000%`
- stress/funding/fee reserve: `1.9100%`
- Binance docs references recorded in JSON under `source_references`.

## Current base reference replay

- leverage: `2.342733x`
- oos: return `+6.4281%`, MDD `0.9293%`, liq `0`, min buffer `9924.1436`, min ratio `187.2044`

## Forced 5x replay

- deployable_success: `True`
- train/validation score: `19.244936`
- OOS return delta vs current-base replay: `+7.6297%`
- OOS return/MDD delta vs current-base replay: `+0.261162`
- train: return `+60.5997%`, MDD `16.2149%`, liq `0`, min buffer `9053.8861`, min ratio `38.4080`
- validation: return `+45.6166%`, MDD `14.0994%`, liq `1`, min buffer `8415.8111`, min ratio `37.1851`
- oos: return `+14.0578%`, MDD `1.9584%`, liq `0`, min buffer `9837.8835`, min ratio `88.9061`

## Selected by train/validation safety

- leverage: `5.000000x`
- locked-OOS used for selection: `False`
- train: return `+60.5997%`, MDD `16.2149%`, liq `0`, min buffer `9053.8861`, min ratio `38.4080`
- validation: return `+45.6166%`, MDD `14.0994%`, liq `1`, min buffer `8415.8111`, min ratio `37.1851`
- oos: return `+14.0578%`, MDD `1.9584%`, liq `0`, min buffer `9837.8835`, min ratio `88.9061`

## Re-selected by train/validation retune

- candidate: `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls620_ss120_tp600__fr`
- source: `integer_audit_selected_by_train_val_stability`
- leverage: `4.000000x`
- deployable_success: `False`
- locked-OOS used for selection: `False`
- train: return `+56.4233%`, MDD `10.7528%`, liq `0`, min buffer `9336.1926`, min ratio `65.6891`
- validation: return `+34.6289%`, MDD `7.6602%`, liq `0`, min buffer `9281.1832`, min ratio `73.4209`
- oos: return `+8.2151%`, MDD `5.3796%`, liq `0`, min buffer `9713.6305`, min ratio `98.0395`

## Best deployable retune candidate

- candidate: `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fr`
- leverage: `5.000000x`

## Promoted candidate

- candidate: `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fr`
- source: `integer_audit_diagnostic_quarantine_04`
- leverage: `5.000000x`
- train: return `+61.6855%`, MDD `15.1068%`, liq `0`, min buffer `9144.6285`, min ratio `41.1213`
- validation: return `+42.6032%`, MDD `13.2668%`, liq `0`, min buffer `8514.0666`, min ratio `41.7885`
- oos: return `+14.6634%`, MDD `1.9646%`, liq `0`, min buffer `9833.7810`, min ratio `93.8848`

## Decision

- `A train/validation-ranked liquidation-tolerant re-selection passed all report-only OOS gates.`

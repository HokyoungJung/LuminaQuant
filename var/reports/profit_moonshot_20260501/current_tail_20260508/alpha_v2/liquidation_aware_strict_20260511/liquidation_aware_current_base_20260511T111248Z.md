# Profit moonshot liquidation-aware current-base validation

- generated_at_utc: `2026-05-11T11:12:48.763678Z`
- decision outcome: `alternate_integer_leverage_deployable`
- deployable improvement: `True`
- reselected deployable: `False`
- memory peak RSS: `257.164 MiB`

## Margin model

- mode: `cross`
- maintenance margin rate: `1.0000%`
- stress/funding/fee reserve: `1.9100%`
- Binance docs references recorded in JSON under `source_references`.

## Current base reference replay

- leverage: `2.342733x`
- oos: return `+6.4281%`, MDD `0.9293%`, liq `0`, min buffer `9924.1436`, min ratio `187.2044`

## Forced 5x replay

- deployable_success: `False`
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

## Retune candidates

- disabled for strict current-base-only replay (`retune_audit_limit=0`, `retune_csv_limit=0`).

## Promoted candidate

- candidate: `current_base_tuple`
- source: `current_base_tuple`
- leverage: `4.000000x`
- train: return `+45.2696%`, MDD `12.9403%`, liq `0`, min buffer `9271.0056`, min ratio `49.6583`
- validation: return `+36.2244%`, MDD `11.2897%`, liq `0`, min buffer `8732.2927`, min ratio `47.8938`
- oos: return `+11.1605%`, MDD `1.5743%`, liq `0`, min buffer `9870.2888`, min ratio `110.6396`

## Decision

- `Forced current-base 5x is unsafe or underqualified; an alternate current-base integer leverage passes strict gates.`

# Profit moonshot liquidation-aware current-base validation

- generated_at_utc: `2026-05-10T03:56:51.552556Z`
- decision outcome: `current_base_retained_liquidation_or_performance_gate_failed`
- deployable improvement: `False`
- memory peak RSS: `251.195 MiB`

## Margin model

- mode: `cross`
- maintenance margin rate: `1.0000%`
- stress/funding/fee reserve: `1.9100%`
- Binance docs references recorded in JSON under `source_references`.

## Current base reference replay

- leverage: `2.342733x`
- oos: return `+0.0695%`, MDD `0.2354%`, liq `0`, min buffer `9988.0839`, min ratio `1805.0191`

## Forced 5x replay

- deployable_success: `False`
- train/validation score: `3.444930`
- OOS return delta vs current-base replay: `+0.0737%`
- OOS return/MDD delta vs current-base replay: `-0.015007`
- train: return `-3.1762%`, MDD `3.3778%`, liq `0`, min buffer `9668.5633`, min ratio `822.7869`
- validation: return `+0.5877%`, MDD `0.3149%`, liq `1`, min buffer `9967.2991`, min ratio `832.0585`
- oos: return `+0.1432%`, MDD `0.5111%`, liq `0`, min buffer `9974.5842`, min ratio `846.8586`

## Selected by train/validation safety

- leverage: `3.000000x`
- locked-OOS used for selection: `False`

## Decision

- `Forced current-base 5x is not deployable under the liquidation-aware gate; retain current base unless a train/validation-safe integer row is explicitly accepted.`

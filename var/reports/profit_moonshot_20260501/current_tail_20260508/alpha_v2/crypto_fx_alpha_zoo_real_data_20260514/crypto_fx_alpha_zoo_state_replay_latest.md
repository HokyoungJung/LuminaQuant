# Crypto/FX Alpha Zoo state real-data replay

Generated: `2026-05-14T13:05:00.064720Z`
Rows: `58845`
Signals: `5684`
Trades: `2843`
Deployable success: `False`
Reason: no strict zero-liquidation Alpha Zoo replay row passed all hard gates including OOS return/MDD versus the current-base reference

## Provenance

- selection inputs: `train, validation`
- uses_locked_oos_for_selection: `False`
- locked-OOS role: `gate/report only after candidate freeze`
- current-base/calendar tuple: `hypothesis_reference_only`, not selection/promotion target
- promotion policy: OOS return and return/MDD must both beat the current-base reference; Sharpe/Sortino/smart Sortino/Calmar must be positive

## Locked-OOS report-only comparison

- candidate OOS return: `6.1108%`
- candidate OOS return/MDD: `2.553222`
- current-base reference OOS return: `6.4281%`
- current-base reference return/MDD: `6.916878`

## Strict zero-liquidation lane

- strict candidate count: `6`
- deployable candidate count: `0`

## Diagnostic nonfatal 5x/6x lane

- `5.0x`: total_liquidations `0`, min_margin_buffer `9207.6050`, promotion_allowed `False`
- `6.0x`: total_liquidations `0`, min_margin_buffer `9049.1260`, promotion_allowed `False`

## Memory

- peak_rss_mib: `280.879`

# Crypto/FX Alpha Zoo state real-data replay

Generated: `2026-05-13T13:22:39.542545Z`
Rows: `58845`
Signals: `5684`
Trades: `2843`
Deployable success: `True`
Reason: strict zero-liquidation lane passed revised OOS return, MDD, Sharpe/Sortino/Calmar gates; return/MDD is diagnostic-only

## Provenance

- selection inputs: `train, validation`
- uses_locked_oos_for_selection: `False`
- locked-OOS role: `gate/report only after candidate freeze`
- current-base/calendar tuple: `hypothesis_reference_only`, not selection/promotion target
- promotion policy: return/MDD is `diagnostic_report_only`; Sharpe/Sortino/smart Sortino/Calmar and MDD cap carry the risk-adjusted gate

## Locked-OOS report-only comparison

- candidate OOS return: `6.1108%`
- candidate OOS return/MDD: `2.553222`
- current-base reference OOS return: `6.4281%`
- current-base reference return/MDD: `6.916878`

## Strict zero-liquidation lane

- strict candidate count: `6`
- deployable candidate count: `5`

## Diagnostic nonfatal 5x/6x lane

- `5.0x`: total_liquidations `0`, min_margin_buffer `9207.6050`, promotion_allowed `False`
- `6.0x`: total_liquidations `0`, min_margin_buffer `9049.1260`, promotion_allowed `False`

## Memory

- peak_rss_mib: `283.059`

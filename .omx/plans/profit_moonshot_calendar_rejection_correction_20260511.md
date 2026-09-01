# Profit moonshot calendar rejection correction — 2026-05-11

- Corrected prior liquidation-only conclusion: current-base calendar tuple is invalid for live promotion.
- Added strategy-validity gate to liquidation-aware replay/promotion path.
- Calendar-primary sleeves now cannot produce `deployable_success=true` even with strong OOS metrics or non-wipeout liquidations.
- New artifact: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/liquidation_aware_strategy_valid_20260511/liquidation_aware_current_base_latest.json`.
- Decision: `no_live_promotion_strategy_validity_failed`; non-calendar retune evaluated 30 integer results, 0 deployable.

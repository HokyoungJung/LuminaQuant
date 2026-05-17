# Alpha Zoo latest-data March-validation high-leverage replay

Generated: `2026-05-17T11:59:58.855315Z`
Data source: `/home/hoky/Quants-agent/LuminaQuant/var/cache/profit_moonshot_fresh_start/joined_panel_76f825ffea81c04f2fe41fbf.parquet`
Memory peak: `736.2 MiB`

## Split contract

- train: `2025-01-01T00:00:00Z` .. `2025-12-31T23:00:00Z`; actual `2025-01-01T00:00:00Z` .. `2025-12-31T23:00:00Z`, rows `None`
- validation: `2026-01-01T00:00:00Z` .. `2026-03-31T23:00:00Z`; actual `2026-01-01T00:00:00Z` .. `2026-03-31T23:00:00Z`, rows `None`
- locked_oos: `2026-04-01T00:00:00Z` .. `2026-05-17T10:00:00Z`; actual `2026-04-01T00:00:00Z` .. `2026-05-17T10:00:00Z`, rows `None`

## Selection provenance

- selection inputs: `['train', 'validation']`
- uses locked-OOS for selection: `False`
- policy: freeze all train/validation-feasible strategy/leverage/allocation candidates by validation-primary train+validation score; locked-OOS can only gate candidates after freeze, not alter scores
- high-leverage lane: isolated margin; liquidation loses the per-position allocation, not the full account.

## Top train/validation candidate

- candidate: `alpha_zoo_conservative_exit_carry_forward_old_split_selected` rank `1`
- leverage/allocation: `9.0x` / `12.50%`
- train return/MDD: `62.54%` / `57.24%`
- validation return/MDD: `71.13%` / `24.33%`
- locked-OOS return/MDD: `-0.60%` / `21.96%`
- locked-OOS Sharpe/Sortino/smart/Calmar: `0.048` / `0.070` / `0.057` / `-0.027`
- liquidations OOS/total account wipeout: `0` / `0`
- live_promotion_possible: `False`; rejections: `['locked_oos_calmar_non_positive', 'locked_oos_return_not_above_current_base_reference']`

## Live-promoted after locked-OOS gate

- candidate: `alpha_zoo_fast_residual` rank `29`
- leverage/allocation: `7.0x` / `15.00%`
- train return/MDD: `1.49%` / `59.94%`
- validation return/MDD: `44.95%` / `13.78%`
- locked-OOS return/MDD: `30.54%` / `11.30%`
- locked-OOS Sharpe/Sortino/smart/Calmar: `1.815` / `2.319` / `2.083` / `2.702`
- liquidations OOS/total account wipeout: `0` / `0`
- live_promotion_possible: `True`; rejections: `[]`

## Artifacts

- latest_json: `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/validation_to_20260331_latest_data_20260517/alpha_zoo_validation_march_high_leverage_latest.json`
- timestamped_json: `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/validation_to_20260331_latest_data_20260517/alpha_zoo_validation_march_high_leverage_20260517T115958Z.json`
- latest_markdown: `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/validation_to_20260331_latest_data_20260517/alpha_zoo_validation_march_high_leverage_latest.md`
- timestamped_markdown: `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/validation_to_20260331_latest_data_20260517/alpha_zoo_validation_march_high_leverage_20260517T115958Z.md`
- candidate_csv: `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/validation_to_20260331_latest_data_20260517/alpha_zoo_validation_march_high_leverage_candidates_latest.csv`

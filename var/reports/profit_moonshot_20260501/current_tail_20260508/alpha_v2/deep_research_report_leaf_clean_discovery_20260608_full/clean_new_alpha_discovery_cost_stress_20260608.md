# Deep Research Report Leaf Clean Discovery Cost Stress (2026-06-08)

- Source summary: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/deep_research_report_leaf_clean_discovery_20260608_full/clean_new_alpha_discovery_summary_20260608.json`
- Search hash: `d92dce2b046441bcf1a7a7ebfa5499844b418a52d6d2416fefad491458967312`
- Selection policy: `rank_train_validation_only_then_attach_locked_oos_report_gate`

## Decision

**FAIL real-money/shadow promotion gate.** Full 10-fold locked-OOS report-only return is low and unstable; cost stress further reduces edge.

## Base aggregate at 10bps

- `annualized_oos_return_approx`: `0.02057000894614114`
- `compounded_oos_return`: `0.017112522890639692`
- `fold_count`: `10`
- `latest_oos_return`: `-0.0037428423110466014`
- `max_oos_mdd`: `0.1185018980371666`
- `min_oos_return`: `-0.07155834360446589`
- `monthly_equity_mdd`: `0.14874230589143944`
- `monthly_sharpe_approx`: `0.19528739997729416`
- `positive_oos_folds`: `5`
- `profit_factor`: `1.1659941187978784`
- `profit_factor_unbounded`: `False`

## Cost stress aggregate

- 10bps: compounded `0.017113`, annualized approx `0.020570`, positive folds `5/10`, min fold `-0.071558`
- 15bps: compounded `0.004570`, annualized approx `0.005487`, positive folds `5/10`, min fold `-0.073694`
- 20bps: compounded `-0.007818`, annualized approx `-0.009375`, positive folds `5/10`, min fold `-0.075826`
- 30bps: compounded `-0.032143`, annualized approx `-0.038446`, positive folds `5/10`, min fold `-0.080074`

## Governance blockers

- annualized_oos_return_far_below_100pct_target
- latest_oos_fold_negative
- positive_oos_folds_only_5_of_10
- monthly_sharpe_approx_0p195
- cost_stress_degrades_low_edge
- feature_backed_report_families_blocked_by_train_validation_coverage_and_zero_locked_oos_feature_coverage

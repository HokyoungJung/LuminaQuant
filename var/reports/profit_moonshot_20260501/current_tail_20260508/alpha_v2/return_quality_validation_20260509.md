# Profit Moonshot Under-8GB Mission Validation

- Status: `running`
- Passed: `False`
- Result path: `.omx/specs/autoresearch-profit-moonshot-alpha-v2/result.json`
- Budget bytes: `8589934592`

## Checks
- FAIL `declared_pass_status` — result is not yet status=passed with passed=true
- PASS `passing_candidate_artifact_exists` — var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/passing_candidate_latest.json
- FAIL `candidate_lifecycle_label` — missing research_success_candidate/live_equivalent label
- FAIL `candidate_return_quality_contract` — {"current_champion_oos_return": 0.012181, "locked_oos_calmar": 42.825182534712944, "locked_oos_max_drawdown": 0.001774102912063881, "locked_oos_monthlyized_return": 0.006121025535450242, "locked_oos_sharpe": 5.477385370503527, "locked_oos_smart_sortino": 2.0593603686568986, "locked_oos_sortino": 6.776899962963284, "locked_oos_total_return": 0.013397126166723838, "maximum_acceptable_oos_mdd": 0.25, "minimum_oos_calmar": 1.0, "minimum_oos_sharpe": 2.0, "minimum_oos_smart_sortino": 3.0, "minimum_oos_sortino": 3.0, "minimum_stable_monthly_return": 0.02, "train_monthlyized_return": 0.0044737445278342225, "val_monthlyized_return": 0.02044010271922625}
- PASS `rss_under_8gib_evidence` — [{"exists": true, "path": "var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/portfolio_target_budget_v1/_memory_guard/profit_moonshot_fresh_portfolio_tuning_memory_latest.json", "peak_rss_bytes": 771981312, "under_8gib": true}, {"exists": true, "path": "var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/portfolio_target_budget_v1/_memory_guard/profit_moonshot_fresh_portfolio_tuning_rss_latest.jsonl", "peak_rss_bytes": 771981312, "under_8gib": true}]
- PASS `local_tests_evidence` — 10 test evidence item(s)
- PASS `ci_success_evidence` — 4 CI evidence item(s)
- PASS `git_push_evidence_if_source_changed` — source unchanged or push evidence present

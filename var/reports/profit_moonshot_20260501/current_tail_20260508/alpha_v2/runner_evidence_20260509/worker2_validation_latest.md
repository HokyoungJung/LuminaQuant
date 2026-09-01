# Profit Moonshot Under-8GB Mission Validation

- Status: `passed`
- Passed: `True`
- Result path: `.omx/specs/autoresearch-profit-moonshot-alpha-v2/result.json`
- Budget bytes: `8589934592`

## Checks
- PASS `declared_pass_status` — result has passed=true and status=passed
- PASS `passing_candidate_artifact_exists` — var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/passing_candidate_latest.json
- PASS `candidate_lifecycle_label` — research/live-equivalent success label present
- PASS `candidate_return_quality_contract` — {"current_champion_oos_return": 0.012181, "locked_oos_calmar": 53.73501485217365, "locked_oos_max_drawdown": 0.008197728267604966, "locked_oos_monthlyized_return": 0.030883445250837083, "locked_oos_sharpe": 5.653697867183353, "locked_oos_smart_sortino": 7.153592515941036, "locked_oos_sortino": 7.396117977603192, "locked_oos_total_return": 0.06858164444753312, "maximum_acceptable_oos_mdd": 0.25, "minimum_oos_calmar": 1.0, "minimum_oos_sharpe": 2.0, "minimum_oos_smart_sortino": 3.0, "minimum_oos_sortino": 3.0, "minimum_stable_monthly_return": 0.02, "minimum_train_calmar": 1.0, "minimum_train_sharpe": 1.5, "minimum_train_sortino": 1.5, "minimum_val_calmar": 3.0, "minimum_val_sharpe": 3.0, "minimum_val_sortino": 3.0, "train_calmar": 3.8842110332761934, "train_monthlyized_return": 0.020000000000000018, "train_sharpe": 1.7212556285918195, "train_sortino": 1.51511166991183, "val_calmar": 32.1417458698937, "val_monthlyized_return": 0.09848998131232078, "val_sharpe": 4.09640969448007, "val_sortino": 4.885875261022447}
- PASS `rss_under_8gib_evidence` — [{"exists": true, "path": "var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/portfolio_monthly_budget_v1/_memory_guard/profit_moonshot_fresh_portfolio_tuning_memory_latest.json", "peak_rss_bytes": 955871232, "under_8gib": true}, {"exists": true, "path": "var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/portfolio_monthly_budget_v1/_memory_guard/profit_moonshot_fresh_portfolio_tuning_rss_latest.jsonl", "peak_rss_bytes": 955871232, "under_8gib": true}]
- PASS `local_tests_evidence` — 10 test evidence item(s)
- PASS `ci_success_evidence` — 4 CI evidence item(s)
- PASS `git_push_evidence_if_source_changed` — source unchanged or push evidence present

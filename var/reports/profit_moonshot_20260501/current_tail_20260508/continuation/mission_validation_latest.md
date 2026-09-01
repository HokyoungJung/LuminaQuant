# Profit Moonshot Under-8GB Mission Validation

- Status: `passed`
- Passed: `True`
- Result path: `.omx/specs/autoresearch-profit-moonshot-pass-under-8gb/result.json`
- Budget bytes: `8589934592`

## Checks
- PASS `declared_pass_status` — result has passed=true and status=passed
- PASS `passing_candidate_artifact_exists` — var/reports/profit_moonshot_20260501/current_tail_20260507/external_overhaul/passing_candidate_latest.json
- PASS `candidate_lifecycle_label` — research/live-equivalent success label present
- PASS `rss_under_8gib_evidence` — [{"exists": true, "path": "var/reports/profit_moonshot_20260501/current_tail_20260507/external_overhaul/logs/fresh_start_overhaul_replay_perf2.time.log", "peak_rss_bytes": 295038976, "under_8gib": true}, {"exists": true, "path": "var/reports/profit_moonshot_20260501/current_tail_20260507/external_overhaul/logs/fresh_portfolio_tuning_perf4.time.log", "peak_rss_bytes": 388091904, "under_8gib": true}, {"exists": true, "path": "var/reports/profit_moonshot_20260501/current_tail_20260507/external_overhaul/_memory_guard/profit_moonshot_fresh_start_replay_memory_latest.json", "peak_rss_bytes": 295038976, "under_8gib": true}, {"exists": true, "path": "var/reports/profit_moonshot_20260501/current_tail_20260507/external_overhaul/_memory_guard/profit_moonshot_fresh_portfolio_tuning_memory_latest.json", "peak_rss_bytes": 382947328, "under_8gib": true}, {"exists": true, "path": "var/reports/profit_moonshot_20260501/current_tail_20260507/external_overhaul/_memory_guard/profit_moonshot_fresh_start_replay_rss_latest.jsonl", "peak_rss_bytes": 2707296256, "under_8gib": true}, {"exists": true, "path": "var/reports/profit_moonshot_20260501/current_tail_20260507/external_overhaul/_memory_guard/profit_moonshot_fresh_portfolio_tuning_rss_latest.jsonl", "peak_rss_bytes": 2653155328, "under_8gib": true}, {"exists": true, "path": "var/reports/profit_moonshot_20260501/current_tail_20260507/external_overhaul/logs/calendar_optuna_64.time.log", "peak_rss_bytes": 264884224, "under_8gib": true}]
- PASS `local_tests_evidence` — 3 test evidence item(s)
- PASS `ci_success_evidence` — 2 CI evidence item(s)
- PASS `git_push_evidence_if_source_changed` — source unchanged or push evidence present

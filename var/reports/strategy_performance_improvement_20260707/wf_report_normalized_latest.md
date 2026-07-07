# Strategy performance-improvement WF evidence report

- generated_at_utc: `2026-07-07T12:26:24.138445Z`
- source_status: `loaded`
- source_json_path: `var/reports/strategy_performance_improvement_20260707/full_universe_walkforward/full_universe_walkforward_summary_latest.json`
- command_log_path: `var/reports/strategy_performance_improvement_20260707/command_log.md`
- worker_count: `2`
- fold_count: `0`
- fold_candidate_row_count: `0`
- peak_rss_mb: `190.85`
- peak_rss_source: `runner_peak_rss_mib_or_blocker_peak_rss_mb`
- native_backend: `numba` (pyo3_available=True)
- requested/loaded/missing symbols: `110` / `0` / `110`
- latest_data_utc: `None`
- leak_checks_pass: `False`
- full_universe_claim_status: `blocked_not_claimed_missing_direct_1m_bars`

## Interpretation

This wrapper is an evidence normalizer. It does not run or promote the expensive walk-forward by itself.
If `source_status` is `missing`, treat this artifact as a concrete blocker report rather than full-universe success evidence.

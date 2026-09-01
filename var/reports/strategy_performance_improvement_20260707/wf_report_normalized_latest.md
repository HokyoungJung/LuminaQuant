# Strategy performance-improvement WF evidence report

- generated_at_utc: `2026-07-08T11:48:04.880345Z`
- source_status: `loaded`
- source_json_path: `var/reports/strategy_performance_improvement_20260707/full_universe_walkforward/full_universe_walkforward_summary_latest.json`
- command_log_path: `var/reports/strategy_performance_improvement_20260707/command_log.md`
- worker_count: `2`
- fold_count: `11`
- fold_candidate_row_count: `1733`
- peak_rss_mb: `1834.56640625`
- peak_rss_source: `runner_peak_rss_mib_or_blocker_peak_rss_mb`
- native_backend: `numba` (pyo3_available=True)
- requested/loaded/missing symbols: `110` / `110` / `0`
- latest_data_utc: `2026-07-04T06:30:00`
- leak_checks_pass: `True`
- full_universe_claim_status: `claimed_loaded_all_requested_symbols_completed_walkforward`

## Interpretation

This wrapper is an evidence normalizer. It does not run or promote the expensive walk-forward by itself.
If `source_status` is `missing`, treat this artifact as a concrete blocker report rather than full-universe success evidence.

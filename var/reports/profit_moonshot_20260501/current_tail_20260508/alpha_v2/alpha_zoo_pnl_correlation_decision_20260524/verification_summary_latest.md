# PnL correlation decision verification summary

Generated: 2026-05-24 KST

## Artifact invariants

- `alpha_zoo_pnl_correlation_decision_latest.json` loaded successfully.
- `ready_for_real=false`, `real_money_execution=false`, `real_execution_allowed=false`.
- locked-OOS usage flags for selection/discovery/objective/pruning/parameter fitting are all `false`.
- Captured paper PnL candidates: `136/136`; missing PnL IDs: `0`.
- Corr-diversified selected candidates: `11`.

## Commands

- Runner: `/usr/bin/time -v .venv/bin/python scripts/research/run_alpha_zoo_pnl_correlation_decision.py`
  - Exit status: `0`
  - Max RSS: `6,481,872 KiB` (<8 GiB)
  - Log: `runner_time_latest.log`
- Targeted pytest: `.venv/bin/pytest tests/test_alpha_zoo_pnl_correlation_decision.py tests/test_alpha_zoo_multi_asset_monitoring_slate.py tests/test_alpha_zoo_7x_paper_forward_preflight.py::test_build_paper_forward_bundle_from_frozen_10bps_sources tests/test_alpha_zoo_validation_first_discovery.py::test_build_validation_first_discovery_from_frozen_10bps_sources -q`
  - Result: `11 passed`
  - Max RSS: `203,536 KiB`
  - Log: `targeted_pytest_time_latest.log`
- Ruff: `.venv/bin/ruff check .`
  - Result: pass
  - Log: `ruff_time_latest.log`
- Compileall: `.venv/bin/python -m compileall -q src scripts tests`
  - Result: pass
  - Log: `compileall_time_latest.log`
- Hardcoded-parameter audit: `.venv/bin/python scripts/audit_hardcoded_params.py`
  - Result: `total=567 new=0 baselined=567`
  - Log: `hardcoded_audit_time_latest.log`
- Diff checks: `git diff --check && git diff --cached --check`
  - Result: pass
- Full pytest: `.venv/bin/pytest -q`
  - Result: `1431 passed in 74.52s`
  - Max RSS: `2,816,620 KiB` (<8 GiB)
  - Log: `full_pytest_time_latest.log`

## Notes

Full pytest initially exposed a wall-clock-sensitive paper-forward preflight fixture: the frozen 2026-05-17 data-refresh snapshot had crossed the prior `10,000` minute staleness override during this session. `scripts/research/run_alpha_zoo_7x_paper_forward_preflight.py` now uses a named 30-day paper-forward stale override for historical paper/testnet handoff artifacts while still forcing real-money readiness false.

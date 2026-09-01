# Session Handoff — Profit Moonshot External Overhaul (2026-05-07)

Goal: keep full-session memory under 8 GiB, pass the current profit gate, then commit/push and require remote CI green.

Current state:
- Team execution finished; all team tasks completed and the team runtime was shut down.
- Performance goal state: `.omx/goals/performance/profit-moonshot-backtest-perf-under-8gb/`.
- Mission state: `.omx/specs/autoresearch-profit-moonshot-pass-under-8gb/result.json`.
- Mission validator: `scripts/research/validate_profit_moonshot_pass_under_8gb.py`.
- Validator report: `var/reports/profit_moonshot_20260501/current_tail_20260507/external_overhaul/mission_validation_latest.md`.

Passing artifacts now exist:
- Replay: `fresh_start_overhaul_replay_latest.json` — 4,941 specs, 63 replay survivors/success candidates, peak RSS 281.371 MiB, `/usr/bin/time` max RSS 288,124 KB.
- Portfolio tuning: `fresh_portfolio_tuning_latest.json` — 2 portfolio success candidates, peak RSS 370.113 MiB, `/usr/bin/time` max RSS 378,996 KB.
- Optuna tuning: `calendar_optuna_latest.json` — 64 train/validation-objective TPE trials, 5 locked-OOS success candidates, peak RSS 252.613 MiB, `/usr/bin/time` max RSS 258,676 KB.
- Passing candidate bundle: `passing_candidate_latest.json`.

Best observed locked-OOS examples:
- Replay/Optuna TRX calendar: OOS return about +1.007%, OOS MDD 0.1532%, OOS Sharpe 7.95, 15 OOS round trips.
- Portfolio success: OOS return about +0.8789%, OOS MDD 0.1760%, OOS Sharpe 5.62.

Latest local verification:
- `uv run --extra dev pytest -q tests/test_profit_moonshot_fresh_start_replay.py tests/test_profit_moonshot_fresh_portfolio_tuning.py tests/test_optuna_tune_profit_moonshot_calendar.py tests/test_collect_hyperliquid_readonly_smoke.py tests/test_profit_moonshot_pass_under_8gb_validator.py tests/test_portfolio_followup_memory_guard.py tests/test_hyperliquid_readonly.py tests/test_screen_profit_moonshot_external_alpha.py` → 35 passed.
- `uv run --extra dev ruff check scripts/research/replay_profit_moonshot_fresh_start.py scripts/research/tune_profit_moonshot_fresh_portfolio.py scripts/research/optuna_tune_profit_moonshot_calendar.py scripts/research/validate_profit_moonshot_pass_under_8gb.py tests/test_profit_moonshot_fresh_start_replay.py tests/test_profit_moonshot_fresh_portfolio_tuning.py tests/test_optuna_tune_profit_moonshot_calendar.py tests/test_collect_hyperliquid_readonly_smoke.py tests/test_profit_moonshot_pass_under_8gb_validator.py` → All checks passed.
- `uv run python -m py_compile scripts/research/replay_profit_moonshot_fresh_start.py scripts/research/tune_profit_moonshot_fresh_portfolio.py scripts/research/optuna_tune_profit_moonshot_calendar.py scripts/research/validate_profit_moonshot_pass_under_8gb.py` → ok.

Remaining hard gates before completion:
1. Commit all intended changes with Lore protocol and OmX co-author trailer.
2. Push `private-main` to `private main`.
3. Capture remote CI green evidence.
4. Update `.omx/specs/autoresearch-profit-moonshot-pass-under-8gb/result.json` to `status: "passed"`, `passed: true`, with `git_evidence` and `ci_evidence`.
5. Re-run `uv run python scripts/research/validate_profit_moonshot_pass_under_8gb.py`; it must exit 0.
6. Only then mark the Codex goal complete and complete the OMX performance goal.

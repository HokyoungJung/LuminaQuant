# Standard live-refit verification summary — 2026-05-28

Scope: refreshed-data Alpha Zoo Optuna hybrid standard live refit with latest 8 complete weeks as validation and final refit on train+validation.

## Data and runner evidence

- Watch symbols: `BTCUSDT`, `ETHUSDT`, `SOLUSDT`, `BNBUSDT`, `TRXUSDT`.
- Common committed 1s data end: `2026-05-28T10:59:59Z`.
- Standard split: train `2025-01-01T00:00:00Z` → `2026-04-02T10:00:00Z`; validation `2026-04-02T11:00:00Z` → `2026-05-28T10:00:00Z`; locked-OOS/test set disabled for live final refit.
- Optuna trials: 240 per hybrid version; selected `hybrid_v3_5_optuna_three_profile_blend`.
- Final refit inputs: `train`, `validation`.
- Runner RSS: `/usr/bin/time` max RSS `6,324,184 KiB`; artifact peak `6175.96 MiB` (< 8 GiB).
- Data refresh RSS: artifact peak about `5.82 GiB`; `/usr/bin/time` max RSS `6,045,100 KiB` (< 8 GiB).
- WAL compaction RSS: `/usr/bin/time` max RSS `1,565,284 KiB` (< 8 GiB).

## Selected final-refit metrics

| Metric | Value |
| --- | ---: |
| Train return | `+3447.4699%` |
| Validation return | `+38.0717%` |
| Train MDD | `40.4164%` |
| Validation MDD | `7.4789%` |
| Train RPT proxy | `368.22bps` |
| Validation RPT proxy | `36.77bps` |
| Train trades | `4167` |
| Validation trades | `447` |
| Gross notional fraction | `4.6902x` |

## Local verification

- `uv run ruff check <changed files>` → passed.
- `uv run pytest tests/test_alpha_zoo_live_training_policy.py tests/test_alpha_zoo_integer_leverage_optuna_hybrid_decision.py tests/test_native_hybrid_optuna_backend.py -q` → `12 passed`, max RSS `176,400 KiB`.
- `uv run python -m compileall -q src scripts tests` → passed, max RSS `38,696 KiB`.
- `uv run ruff format --check .` → passed after formatting three files, max RSS `76,176 KiB`.
- `uv run ruff check .` → passed, max RSS `38,056 KiB`.
- `uv run python scripts/check_architecture.py` → passed.
- `uv run python scripts/verify_docs.py` → passed (`117 markdown files checked`).
- `uv run python scripts/audit_hardcoded_params.py` → `total=567 new=0 baselined=567`.
- `git diff --check` → passed.
- `uv run pytest -q` → `1506 passed in 89.76s`, max RSS `2,736,260 KiB`.

## Safety

The generated research/live-facing artifacts remain paper/testnet-only: `ready_for_real=false`, `real_money_execution=false`, and `real_execution_allowed=false`.

# G008 final reverification summary

- Status: **PASS_WITH_G008_WATCH_AND_HIDDEN_GOAL_BLOCKER**
- Deployment: **no real money / no small sleeve / 0% allocation or paper-control only**
- Code-reviewer: **APPROVE**
- Architect: **WATCH / no safety FAIL**
- Hidden Codex goal: wrong completed legacy latency objective; formal checkpoint remains blocked.

## Verification

- compileall: pass
- Ruff check: pass
- Ruff format check: pass
- Pytest core: `34 passed in 0.76s`
- Pytest stream/BBO: `6 passed in 0.04s`
- `git diff --check`: pass
- Artifact assertions: pass; all final candidates remain `real_money_allowed=false` and `small_sleeve_allowed=false`.
- Latest BBO post-update verification: JSON/artifact assertions pass; Ruff/format/git diff check pass; `PYTHONPATH=. ...` subset `12 passed in 0.53s` after no-PYTHONPATH collection retry.
- Historical BBO ingest adapter: `scripts/import_binance_book_ticker_history.py` added for explicitly approved external BBO history files only; `PYTHONPATH=. uv run pytest -q tests/test_import_binance_book_ticker_history.py` -> `3 passed in 0.08s`; no deployment decision changed.
- Official public archive BBO backfill: `scripts/backfill_binance_public_book_ticker_history.py` added for pre-manifested Binance USD-M daily `bookTicker` ZIPs; Ruff/format pass; import+backfill tests `8 passed in 0.16s`; sample official URL returned `HTTP/2 200` / `application/zip`; no deployment decision changed.
- Actual official smoke backfill + rerun: `2024-03-30` BTC/ETH/SOL/BNB/TRX archives persisted `1,440` cadence-sampled rows; support inventory reached BTC 1029 / ETH 1043 / SOL 1034 / BNB 1004 / TRX 988 BBO rows; cap=500 clean rerun generated `2026-06-07T11:08:08.338686Z` and stayed `-0.24%` OOS comp / `-0.57%` annualized / no promotion.



## Latest BBO accumulation recheck

- Source generated: `2026-06-07T10:32:06.564484Z`.
- Inferred candidate cap `500`, folds `5`, candidate rows `2500`.
- Result unchanged: OOS comp `-0.24%`, annualized `-0.57%`, monthly equity MDD `8.72%`, hit `3/5`.
- Decision impact: no promotion; real-money and small-sleeve remain disallowed.

## Remaining blockers

1. `.omx/ultragoal` blocked checkpoint is intentionally non-terminal: ledger has G008 `goal_blocked`, while `goals.json` keeps G008 `in_progress`.
2. Fresh `get_goal` is the old completed latency objective, not the active aggregate.
3. Architect verdict remains WATCH, not CLEAR; caveats are annotated and do not change the no-live-deployment conclusion.
- Latest cap=500 BBO freeze verification: JSON/artifact assertions pass; `git diff --check` pass; `PYTHONPATH=. uv run pytest -q tests/test_alpha_zoo_clean_new_alpha_discovery.py tests/test_strategy_support_inventory.py tests/test_collect_binance_book_ticker_feature_points.py` -> `12 passed in 0.54s`.

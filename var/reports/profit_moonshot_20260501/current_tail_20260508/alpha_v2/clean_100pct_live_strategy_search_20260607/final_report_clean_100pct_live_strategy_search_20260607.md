# Clean 100%+ live strategy search — final report

- Generated UTC: `2026-06-07T07:11:42Z`
- Manifest: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/clean_100pct_live_strategy_search_20260607/immutable_manifest_clean_100pct_live_strategy_search_20260607.json` (`sha256 2ace861549f3c72182e5ac18ec87fd44f1d53b72f102eef1c42725f4bbece9ea`)
- Ultragoal: `.omx/ultragoal G007-final-strict-label-assignment-and-re` (final report construction/provenance)
- Independent review reconciliation: `.omx/ultragoal G008-resolve-final-independent-review-and` (code-reviewer APPROVE; architect WATCH/no safety FAIL)

## 결론

**현 시점 실전 투입 가능한 연 100%+ clean 검증 후보는 찾지 못했다.**

- `real_money_candidate`: **없음**
- `small_sleeve_candidate`: **없음**
- 100%+ historical/shadow label: **있음**, 하지만 `shadow_freeze_only`라 실전 승격 금지
- 허용 가능한 사용: paper/control 또는 shadow freeze 관찰뿐

## 후보 라벨

| Candidate | Final label | Ann approx | OOS comp | Max OOS MDD | Real money | Reason |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `clean_input_meta_selector` | `shadow_freeze_only` | 110.46% | 85.91% | 19.29% | no | 110%+ annualized historical report label exists, but selector grid ranking used historical locked-OOS context; requires fresh-forward and paper telemetry. |
| `relaxed_efficiency_hybrid_v3_5_69_asset_historical_incumbent` | `paper_control` | 209.00% | 156.03% | 19.75% | no | Prior source-artifact historical OOS control; locked-OOS/cost fixed-blend aggregation for the same id also reports 160.90% ann / 122.36% comp. Neither lineage is promotable from existing OOS artifacts, and fresh-forward/paper-fill telemetry plus current 10/15/20bps verifier rows are missing. |
| `strict_no_leak_best_single_10bps` | `paper_control` | n/a | 54.56% | 30.63% | no | Best stricter no-leak control remains below 100% target and high drawdown/tail cost stress blocks live use. |
| `dynamic_conviction_switch_85_symbol_baseline` | `paper_control` | 42.57% | 34.39% | 27.69% | no | Best clean-mechanics 85-symbol baseline is useful as paper/control, but below 100% and high MDD/sparse folds block promotion. |
| `clean_new_alpha_discovery_full` | `rejected` | 3.01% | 2.51% | 8.77% | no | Train/validation first artifact reports only 3.01% annualized OOS and is clean_promotion_eligible=false. |
| `clean_new_alpha_discovery_feature_bounded` | `rejected` | -0.57% | -0.24% | 8.32% | no | Feature-bounded variant reports -0.57% annualized OOS and is not a promotion candidate. |

## G008 independent review reconciliation

- `code-reviewer` lane: **APPROVE** for the code-reviewer scope; 0 CRITICAL/HIGH/MEDIUM, 1 LOW provenance note. Artifact: `final_independent_review_20260607/code_reviewer_final_report.md`.
- `architect` lane: **WATCH / no safety FAIL**. It supports the no-real-money/no-small-sleeve conclusion, but requires provenance and metric-lineage annotation before calling this fully cleanly closed. Artifact: `final_independent_review_20260607/architect_final_report.md`.
- Reconciliation: G007 remains the final-report construction/provenance goal; G008 supplies the independent-review evidence and checkpoint reconciliation. Formal ultragoal completion is still constrained by the hidden Codex `get_goal` snapshot pointing to an older completed latency objective.
- Metric lineage note: `relaxed_efficiency_hybrid_v3_5_69_asset_historical_incumbent` has two historical support metrics under the same candidate id: source artifact 209.00% ann / 156.03% comp, and locked-OOS/cost fixed-blend aggregation 160.90% ann / 122.36% comp. Both remain `paper_control`, not live expected return.

## Hard gate 결과

1. `no_nested_oos_mining`: 기존 OOS 결과는 promotion 근거가 아니라 contamination map/control로만 사용했다.
2. `execution_cost_gate`: 10/15/20bps + paper fill telemetry + capacity proxy가 실전 요구조건인데, 현재 후보들은 이를 충족하지 못한다.
3. `theory_plausibility_gate`: trend/momentum, lagged volatility scaling, cost-aware implementation은 이론적으로 가능하지만 OOS-inspired selector를 정당화하지 않는다.

## 실전 도입 기대 성과

현재 evidence로는 연 100%+ 기대성과를 실전 기대값으로 제시하면 안 된다. 가장 정직한 deployment expectation은 **0% allocation / paper-control only**이며, 실거래 전 fresh-forward shadow와 paper fill telemetry가 필요하다.

## 다음 clean 경로

1. 이 manifest 또는 successor manifest를 먼저 고정한다.
2. train/validation-only Optuna/hybrid search를 실행한다.
3. 선택 후에만 locked OOS를 report-only로 attach한다.
4. 10/15/20bps, turnover/RPT, capacity/liquidity, paper fill telemetry를 통과한 뒤에야 `small_sleeve_candidate` 검토가 가능하다.

## External sources

- Time Series Momentum — https://www.aqr.com/insights/research/journal-article/time-series-momentum
- Volatility Managed Portfolios — https://www.nber.org/papers/w22208
- Trading Costs — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3229719
- Backtest overfitting / Pseudo-Mathematics — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2308659

## Latest BBO accumulation recheck

- Source: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_clean_new_alpha_discovery_20260607_feature_bounded/clean_new_alpha_discovery_latest.json` generated `2026-06-07T11:08:08.338686Z` after the official one-day BBO smoke backfill.
- Inferred candidate cap: `500`; folds: `5`; candidate rows: `2500`.
- Result: OOS comp `-0.24%`, annualized `-0.57%`, monthly equity MDD `8.72%`, Sharpe `0.04`, hit `3/5`.
- BBO rows now observed after official one-day smoke backfill plus forward sidecar: BTCUSDT 1029, ETHUSDT 1043, SOLUSDT 1034, BNBUSDT 1004, TRXUSDT 988.
- Impact: no promotion flag changed; `clean_new_alpha_discovery_feature_bounded` remains `rejected` and `real_money_execution=false`.
- Follow-up implementation: `scripts/import_binance_book_ticker_history.py` can now ingest explicitly approved external Binance BBO history files into feature points (`csv`, `jsonl`/`ndjson`, `parquet`) and is covered by `tests/test_import_binance_book_ticker_history.py`. This is plumbing only: it does not approve a data vendor, does not change the cap=500 OOS result above, and cannot unlock real-money without a new clean walk-forward rerun on approved history.
- Public archive follow-up: `scripts/backfill_binance_public_book_ticker_history.py` can now pull official Binance USD-M daily `bookTicker` ZIPs from `data.binance.vision`, normalize the official `transaction_time`/`best_*` column shape, optionally cadence-sample rows, and persist feature points. This is still data plumbing only: symbol/date scope must be pre-manifested, and it does not change the no-real-money/no-small-sleeve conclusion without a fresh clean walk-forward plus paper fill/cost telemetry.
- Actual smoke run: imported official `2024-03-30` BTC/ETH/SOL/BNB/TRX `bookTicker` archives, cadence-sampled each symbol-day to `288` five-minute rows (`1,440` persisted rows total), then reran the cap=500 BBO-aware clean discovery. The aggregate stayed OOS comp `-0.24%`, annualized `-0.57%`, monthly equity MDD `8.72%`, Sharpe `0.04`, hit `3/5`; promotion remains `false`.
- Coverage audit: the official BTCUSDT daily `bookTicker` prefix currently shows latest listing `2024-03-30`, and a `2025-12-01..2025-12-07` core5 probe returned `35/35` missing archives. Therefore official public-data BBO proves ingestion but does not cover the current `2025-01-01..2026-05-31` train/validation fold windows.

## Verification

- JSON/manifest/report assertions: pass.
- Ruff targeted: pass; Ruff format check: pass.
- Pytest targeted: core suite `34 passed in 0.76s`; stream/BBO suite `6 passed in 0.04s`.
- Quality gate artifact: `final_quality_gate_20260607.json`.
- G008 code-reviewer lane: APPROVE; architect lane: WATCH/no safety FAIL; both reports are stored in `final_independent_review_20260607/`.
- Latest BBO accumulation recheck: pass/no promotion; feature-bounded result remains -0.24% OOS comp / -0.57% annualized after cap=500 rerun.
- Formal ultragoal checkpoint is still not fully closed: G008 blocked checkpoint is intentionally non-terminal: ledger records `goal_blocked` while `.omx/ultragoal/goals.json` keeps G008 `in_progress`, and hidden Codex `get_goal` points to an older completed latency objective. This blocks merge-ready/final-checkpoint approval, but not the research conclusion above.
- Latest BBO post-update verification: JSON/artifact assertions pass; Ruff/format/git diff check pass; `PYTHONPATH=. uv run pytest -q tests/test_alpha_zoo_clean_new_alpha_discovery.py tests/test_strategy_support_inventory.py tests/test_collect_binance_book_ticker_feature_points.py` -> `12 passed in 0.53s` after no-PYTHONPATH collection retry.
- Code-reviewer re-review after latest BBO accumulation: no no-live/no-small-sleeve blocker; one LOW stale dirty-workspace caveat in `final_quality_gate_20260607.json` was fixed and de-hashed so the caveat no longer goes stale when follow-up commits are added.
- Latest cap=500 BBO freeze verification: JSON/artifact assertions pass; `git diff --check` pass; `PYTHONPATH=. uv run pytest -q tests/test_alpha_zoo_clean_new_alpha_discovery.py tests/test_strategy_support_inventory.py tests/test_collect_binance_book_ticker_feature_points.py` -> `12 passed in 0.54s`.
- Historical BBO ingest adapter verification: `PYTHONPATH=. uv run pytest -q tests/test_import_binance_book_ticker_history.py` -> `3 passed in 0.08s`; existing BBO/alpha focused suite stayed `12 passed in 0.48s`.
- Public Binance bookTicker archive backfill verification: Ruff check pass; Ruff format `4 files already formatted`; `PYTHONPATH=. uv run pytest -q tests/test_import_binance_book_ticker_history.py tests/test_backfill_binance_public_book_ticker_history.py` -> `8 passed in 0.16s`; `curl -I` on the official BTCUSDT 2024-03-30 archive returned `HTTP/2 200` / `application/zip`.
- Post-smoke rerun verification: final focused check passed (`4 files already formatted`, Ruff all checks passed, `22 passed in 0.66s`, docs verification `119 markdown files checked`); clean result remained rejected/no-promotion.

# Resume prompt — 2026-07-04 LuminaQuant walk-forward correction

Use this prompt in a fresh GJC session:

```text
Repo/worktree: /home/hoky/Quants-agent/LuminaQuant
Branch: private-main

Pull the latest private-main first. Do not overwrite unrelated local changes. Continue from the saved 2026-07-04 walk-forward correction artifacts; do not restart planning unless the artifacts are missing or verification fails.

Context:
- The previous fixed-split/selected-3-alpha report was corrected because alpha selection must be walk-forward.
- Data refresh basis: Binance fapi 1m universe, 128 symbols, 2025-01-01T00:00:00Z through 2026-07-04T07:15:59.999Z.
- Selection method: monthly expanding train + previous 2-month validation only.
- Locked OOS is diagnostic/report-only after fold selection is frozen; OOS is not used for rank/status/research-selected selection.
- Safety constraints remain hard: no live/paper/testnet/real-money/orders; TONUSDT excluded.

Read first:
- docs/research_note/research_note.md, section "2026-07-04 KST — 최신 데이터 전수 walk-forward 정정 리포트"
- var/reports/latest_alpha_refresh_20260704_full_walkforward/full_all_strategy_walkforward_selection_latest.md
- var/reports/latest_alpha_refresh_20260704_full_walkforward/full_all_strategy_walkforward_selection_verification.json
- var/reports/latest_alpha_refresh_20260704_full_walkforward/full_all_strategy_walkforward_selection_cleanup.json
- var/reports/latest_alpha_refresh_20260704_full_walkforward/full_walkforward_consolidated_run_summary.json

Key saved results:
- 16/16 consolidated shard reports present; missing 0; run failure_count 0.
- Per fold accounting: evaluated 1400 + timeout-filtered 4 = accounted 1404.
- Timeout-filtered fail-closed 1h Alpha101 ids: 6afebe39638237ca, c49978799aff2168, 82a31b3aa1d93bb0, b730cdad557b46e3.
- Train/validation repeated-selection research-selected candidates: 22.
- Mean research-selected validation return/sharpe: +10.24% / 2.621.
- Mean locked-OOS diagnostic return/sharpe: -1.98% / -4.289.
- Only 1/22 research-selected candidates has mean locked-OOS diagnostic return and sharpe both positive.
- Deployment conclusion: no_execution_promotion / research_only_no_execution.

Top research-selected candidates by train+validation rank:
1. 19d07d85cab54789 alpha101_formula_4h_a011_a011_flow_swing_dir — folds 3, validation +10.68% / 2.944, OOS diagnostic +3.07% / -2.576.
2. 632d6e864ee01bd8 pair_spread_4h_fast_cycle_btcusdt_bnbusdt_1.6_0.35 — folds 2, validation +10.06% / 6.158, OOS diagnostic -0.24% / -20.896.
3. a27ef46e91376b7c pair_spread_4h_fast_cycle_btcusdt_bnbusdt_1.8_0.45 — folds 2, validation +9.61% / 5.322, OOS diagnostic -0.14% / -22.303.
4. 1f1fd241c12f0bc2 pair_spread_4h_balanced_btcusdt_bnbusdt_1.6_0.35 — folds 2, validation +9.72% / 5.409, OOS diagnostic +0.28% / 0.720.
5. 914ff36cba1555ea pair_spread_4h_balanced_btcusdt_bnbusdt_2.0_0.50 — folds 2, validation +9.02% / 5.039, OOS diagnostic -0.54% / -23.974.

Verification already run before handoff:
- /home/hoky/Quants-agent/LuminaQuant/.venv/bin/python -m ruff check .
- /home/hoky/Quants-agent/LuminaQuant/.venv/bin/python -m pytest -q
- npm run typecheck in apps/dashboard_web
- npm run lint in apps/dashboard_web
- npm test in apps/dashboard_web

If asked to continue research, start from the saved report and verification files above. Treat all selected candidates as research-only until fresh-forward/shadow/live-readiness gates are separately satisfied. Do not enable execution.
```

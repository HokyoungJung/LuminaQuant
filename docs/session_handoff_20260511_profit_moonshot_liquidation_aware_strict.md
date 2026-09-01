# Profit moonshot liquidation-aware strict result — 2026-05-11

## Decision

Forced integer `5x` on the **current-base sleeve tuple** is **not deployable** under the strict liquidation-aware gate.

Deployable improvement found inside the allowed current-base integer grid: **promote current-base tuple `4x` for this validation artifact**, because it is the highest train/validation-selected integer leverage with zero liquidations, positive margin buffers, OOS MDD under 25%, and OOS return/return-MDD improvement versus the current-base `2.342733x` replay.

Outcome: `alternate_integer_leverage_deployable`. Forced 5x deployable: `False`. Selected integer deployable: `True`. Retuned/reselected deployable: `False`.

## Blocking evidence for forced 5x

- Forced `5x` validation liquidation count: `1`.
- Event: `2026-02-05T20:00:00Z` `BTC/USDT` `LONG` in `fresh_pair_resid_revert_spread_lb24_z150_h120_sc10_st100_tp400_all`.
- Reason: `intrabar_low_breached_liquidation_threshold`; entry `76515.975090`, liquidation threshold `62674.235196`, adverse trigger `62233.300000`.
- Account wipeout: `False`; event drawdown `0.0802%`; equity-loss fraction `0.0802%`.
- Strict promotion rule is zero-liquidation only, so `liquidation_count > 0` blocks promoted success even though margin buffers remain positive.

## Baseline preservation

- Pushed green handoff head preserved: `77f10d54174628c24f1a6bbba34a74505a2a40b5`.
- Code/performance baseline preserved: `02f4520cf906f48089b8852c2651a0f1e4bd0c1c`.
- This run adds strict liquidation/margin replay evidence and does not overwrite the baseline artifacts.

## Selection boundary

- Integer grid evaluated: `[1, 2, 3, 4, 5, 6]`, plus current-base leverage `2.3427334297703024x`.
- Candidate seeds: current-base tuple only; retune audit/CSV seeds disabled (`retune_audit_limit=0`, `retune_csv_limit=0`).
- Selection policy: train/validation only.
- Locked-OOS: report-only/gate-only; `uses_locked_oos_for_selection=false`.
- Liquidation tolerance defaults: total `0`, per split `0`; tolerance metadata is diagnostic only and cannot permit promotion.
- Highest zero-liquidation integer: `4.0x`.

## Split metrics — current-base 2.342733x replay

| Split | Return | MDD | Liquidations | Min margin buffer | Min margin ratio | Sharpe | Sortino | Smart Sortino | Calmar |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | +24.5533% | 7.4996% | 0 | 9605.2221 | 86.0745 | 1.5009 | 1.4484 | — | 3.2744 |
| validation | +20.1842% | 6.6189% | 0 | 9256.9424 | 85.8239 | 3.8793 | 4.7141 | — | 32.0477 |
| oos | +6.4281% | 0.9293% | 0 | 9924.1436 | 187.2044 | 5.2024 | 6.7957 | 6.5431 | 43.9983 |

## Split metrics — forced current-base 5x

| Split | Return | MDD | Liquidations | Min margin buffer | Min margin ratio | Sharpe | Sortino | Smart Sortino | Calmar |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | +60.5997% | 16.2149% | 0 | 9053.8861 | 38.4080 | 1.5900 | 1.5433 | — | 3.7378 |
| validation | +45.6166% | 14.0994% | 1 | 8415.8111 | 37.1851 | 3.8887 | 4.7217 | — | 65.5527 |
| oos | +14.0578% | 1.9584% | 0 | 9837.8835 | 88.9061 | 5.2184 | 6.8398 | 6.3039 | 54.2369 |

## Split metrics — promoted current-base 4x strict integer

| Split | Return | MDD | Liquidations | Min margin buffer | Min margin ratio | Sharpe | Sortino | Smart Sortino | Calmar |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | +45.2696% | 12.9403% | 0 | 9271.0056 | 49.6583 | 1.5410 | 1.4904 | — | 3.4988 |
| validation | +36.2244% | 11.2897% | 0 | 8732.2927 | 47.8938 | 3.9371 | 4.8549 | — | 51.1856 |
| oos | +11.1605% | 1.5743% | 0 | 9870.2888 | 110.6396 | 5.2196 | 6.8329 | 6.4026 | 50.1789 |

## Performance gates for promoted 4x

- OOS return: `+11.1605%` vs current-base `+6.4281%`; delta `+4.7324%`.
- OOS MDD: `1.5743%` <= `25.0000%`.
- OOS return/MDD: `7.089245` vs current-base `6.916878`; delta `0.172367`.
- OOS Sharpe/Sortino/smart Sortino/Calmar: `5.2196` / `6.8329` / `6.4026` / `50.1789`.
- Liquidations: train/validation/OOS `0/0/0`; all minimum margin buffers >0.

## Margin model

Conservative Binance USDⓈ-M perpetual-style scalar model:
- margin mode: `cross`
- maintenance margin rate: `1.0000%`
- taker fee: `0.1000%`
- slippage: `0.0500%`
- funding reserve per 8h: `0.0100%`
- stress buffer: `0.2500%`
- liquidation fee reserve: `0.5000%`
- total liquidation reserve: `1.9100%`
- Official Binance references are recorded in the JSON artifact under `source_references`.

## Memory and artifacts

- Strict replay exit status: `0`.
- `/usr/bin/time -v` max RSS: `263336 KiB`.
- Artifact memory peak: `257.1641 MiB`; under 8 GiB: `True`.
- Latest JSON: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/liquidation_aware_strict_20260511/liquidation_aware_current_base_latest.json`
- Latest Markdown: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/liquidation_aware_strict_20260511/liquidation_aware_current_base_latest.md`
- Timestamped JSON/Markdown: `liquidation_aware_current_base_20260511T111248Z.json` / matching `.md`.
- Replay timing log: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/liquidation_aware_strict_20260511/strict_replay_20260511_time.log`

## Verification evidence

- Regression-first evidence: `test_retune_seed_defaults_keep_replay_current_base_only` failed before the retune-disable fix and passed after patching.
- Targeted tests: `uv run --extra dev pytest -q tests/test_profit_moonshot_liquidation_aware_validation.py tests/test_profit_moonshot_fresh_portfolio_tuning.py tests/test_profit_moonshot_pass_under_8gb_validator.py` → `40 passed in 0.11s`, max RSS `174188 KiB`.
- Strict replay: `/usr/bin/time -v uv run --extra dev python scripts/research/run_profit_moonshot_liquidation_aware_validation.py --output-dir var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/liquidation_aware_strict_20260511` → exit `0`, max RSS `263336 KiB`.
- Full pytest after Ralph/deslop: `/usr/bin/time -v uv run --extra dev pytest -q` → `1260 passed in 387.21s (0:06:27)`, max RSS `2739276 KiB`.
- Local quality gates after full test: `uv run --extra dev ruff check .`, `python3 -m compileall -q src scripts tests`, and `git diff --check` passed.
- Architect Ralph verification: `CLEAR`, no required fixes before commit.
- Push and GitHub Actions `ci` / `private-ci`: pending final Lore commit.

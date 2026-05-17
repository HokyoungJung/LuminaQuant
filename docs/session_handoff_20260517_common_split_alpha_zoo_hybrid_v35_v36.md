# Session handoff — Common-split Alpha Zoo vs Hybrid v3.5/v3.6 (2026-05-17)

## Baseline and objective

- Preserved baseline: `private/main@80a557c133930f51748ec20c4e582aa0d6f678de`.
- Goal: rerun existing best `CryptoFxAlphaZooStateStrategy / alpha_zoo_conservative_exit / strict 6x` and fixed-input hybrid v3.5/v3.6 Optuna candidates on the same common split.
- Old Alpha Zoo split is **historical only** and not a common-split selection/promotion basis.

## Common split authority

| split | start | end | unique timestamps | rows |
| --- | --- | --- | ---: | ---: |
| train | `2025-01-01T00:00:00Z` | `2025-12-31T23:00:00Z` | 8760 | 43800 |
| validation | `2026-01-01T00:00:00Z` | `2026-02-28T23:00:00Z` | 1416 | 7080 |
| locked-OOS | `2026-03-01T00:00:00Z` | `2026-05-06T23:00:00Z` | 1593 | 7965 |

## Implementation

Changed/added code:

- `scripts/research/run_common_split_alpha_zoo_hybrid_v35_v36.py`
  - Applies exact common split labels and filters outside-window timestamps.
  - Adds split-bounded forward-return labels so train/validation labels cannot look into later splits.
  - Rebuilds Alpha screen/calibration/replay on common split.
  - Separates old-split carry-forward replay from common-split reselected replay.
  - Calls fixed-input hybrid v3.5/v3.6 runner with common-split Alpha replay/calibration.
  - Emits manifest, split periods/hash, provenance, strict lane, diagnostic lane, contamination audit, and memory summary.
  - Adds `--reuse-stage-artifacts` for fast top-level report reassembly from existing stage outputs.
- `scripts/research/replay_crypto_fx_alpha_zoo_state.py`
  - Persists full `trade_split_periods` for replay payloads so candidate active min/max timestamps are not derived from truncated preview trades.
- `scripts/research/run_profit_moonshot_hybrid_v35_v36_fixed_inputs.py`
  - Applies a `common_split_manifest.split_contract` from the Alpha replay payload when reconstructing the A0 Alpha stream, preventing fallback to fractional splits.
- `tests/test_common_split_alpha_zoo_hybrid_v35_v36.py`
  - Locks exact split boundaries, split-bounded label behavior, locked-OOS poison resistance, calibration locked-OOS exclusion, timestamp hash semantics, common Alpha stream split handling, strict-row fallback, and hybrid active-window reporting.

## Artifact paths

Main report directory:

`var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/common_split_alpha_zoo_hybrid_v35_v36_20260517/`

Key artifacts:

- `common_split_alpha_zoo_hybrid_v35_v36_latest.json`
- `common_split_alpha_zoo_hybrid_v35_v36_latest.md`
- `alpha_zoo_common_split/crypto_fx_alpha_zoo_screen_common_split_latest.json`
- `alpha_zoo_common_split/edge_calibration_common_split_latest.json`
- `alpha_zoo_common_split/crypto_fx_alpha_zoo_state_replay_carry_forward_common_split_latest.json`
- `alpha_zoo_common_split/crypto_fx_alpha_zoo_state_replay_reselected_common_split_latest.json`
- `hybrid_v35_v36_common_split/hybrid_v35_v36_fixed_inputs_common_split_latest.json`
- `hybrid_v35_v36_common_split/hybrid_v35_v36_fixed_inputs_common_split_latest.md`

## Selection provenance and locked-OOS audit

- Alpha screen/factor-card selection: train+validation only; `uses_locked_oos_for_selection=false`.
- Edge calibration: train+validation records only; `locked_oos_calibration_record_count=0`.
- Alpha carry-forward: old selected params replayed on common split and labeled `common_split_old_split_selected_carry_forward`.
- Alpha reselected: default grid reselected on train+validation, selected `alpha_zoo_conservative_exit`.
- Hybrid Optuna: fixed input universe `A0 + P0 + E0 + S1 + S2 + S3 + S4`, `n_trials=80`, `seed=42`; selection policy says locked-OOS not used for objective, pruning, or selection.
- Audit result: `violation=false`, `violation_reasons=[]`.

## Candidate results

### Alpha Zoo historical old split — reference only

| split | period | return | MDD | Sharpe | Sortino | smart Sortino | Calmar | trades | liq | min buffer | deployable |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| train | `2025-01-01T00:00:00Z` → `2025-10-22T04:00:00Z` | +68.8842% | 29.5651% | 1.569139 | 1.919776 | 1.481707 | 2.329914 | 1779 | 0 | 9049.125962 | false |
| validation | `2025-10-22T05:00:00Z` → `2026-01-28T06:00:00Z` | +30.1195% | 9.5595% | 1.552041 | 2.095744 | 1.912882 | 3.150734 | 524 | 0 | 9527.695928 | false |
| locked-OOS | `2026-01-28T07:00:00Z` → `2026-05-06T23:00:00Z` | +41.0967% | 13.6667% | 2.143209 | 2.841936 | 2.500237 | 3.007073 | 540 | 0 | 9572.449083 | false |

Reason: `historical_old_split_only_not_common_split_selection`.

### Alpha Zoo common-split carry-forward and reselected strict 6x

Carry-forward old-selected and common-split reselected both select/retain `alpha_zoo_conservative_exit` and produce the same strict 6x row.

| split | period | active trade period | return | MDD | Sharpe | Sortino | smart Sortino | Calmar | trades | liq | min buffer | deployable |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| train | `2025-01-01T00:00:00Z` → `2025-12-31T23:00:00Z` | `2025-01-02T00:00:00` → `2025-12-31T15:00:00` | +114.4617% | 29.5651% | 2.062010 | 2.566305 | 1.980707 | 3.871508 | 2143 | 0 | 9049.125962 | true |
| validation | `2026-01-01T00:00:00Z` → `2026-02-28T23:00:00Z` | `2026-01-01T11:00:00` → `2026-02-28T10:00:00` | +19.9681% | 13.6667% | 1.251761 | 1.556901 | 1.369707 | 1.461080 | 335 | 0 | 9572.449083 | true |
| locked-OOS | `2026-03-01T00:00:00Z` → `2026-05-06T23:00:00Z` | `2026-03-01T01:00:00` → `2026-05-06T23:00:00` | +20.5127% | 6.7884% | 1.772136 | 2.578776 | 2.414847 | 3.021741 | 365 | 0 | 9643.447509 | true |

### Fixed-input hybrid v3.5/v3.6 Optuna

| candidate | split | period | return | MDD | Sharpe | Sortino | smart Sortino | Calmar | active hours | liq | min buffer | deployable | rejection reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| v3.5 | train | `2025-01-01T00:00:00Z` → `2025-12-31T23:00:00Z` | +47.7257% | 11.0421% | 2.705685 | 2.932093 | 2.640523 | 4.322148 | 7514 | not_replayed | not_replayed | false | dedicated integrated margin replay required |
| v3.5 | validation | `2026-01-01T00:00:00Z` → `2026-02-28T23:00:00Z` | +13.3102% | 2.2622% | 5.390597 | 7.610005 | 7.441656 | 51.558258 | 1302 | not_replayed | not_replayed | false | same |
| v3.5 | locked-OOS | `2026-03-01T00:00:00Z` → `2026-05-06T23:00:00Z` | +8.5233% | 1.7654% | 5.259028 | 7.316663 | 7.189734 | 32.173151 | 1467 | not_replayed | not_replayed | false | same |
| v3.6 | train | `2025-01-01T00:00:00Z` → `2025-12-31T23:00:00Z` | +49.5204% | 7.6947% | 2.897597 | 2.999204 | 2.784914 | 6.435678 | 7514 | not_replayed | not_replayed | false | dedicated integrated margin replay required |
| v3.6 | validation | `2026-01-01T00:00:00Z` → `2026-02-28T23:00:00Z` | +12.4946% | 1.5354% | 7.002337 | 8.680826 | 8.549560 | 69.800312 | 1302 | not_replayed | not_replayed | false | same |
| v3.6 | locked-OOS | `2026-03-01T00:00:00Z` → `2026-05-06T23:00:00Z` | +7.7916% | 1.7491% | 4.859674 | 5.991026 | 5.888040 | 29.199963 | 1467 | not_replayed | not_replayed | false | same |

## Strict zero-liquidation integer leverage lane

| leverage | deployable | strict_safe | locked-OOS return | locked-OOS MDD | liq | min buffer |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1x | false | true | +3.2428% | 1.1567% | 0 | 9841.520994 |
| 2x | true | true | +6.5558% | 2.3032% | 0 | 9683.041987 |
| 3x | true | true | +9.9391% | 3.4396% | 0 | 9524.562981 |
| 4x | true | true | +13.3930% | 4.5659% | 0 | 9366.083974 |
| 5x | true | true | +16.9175% | 5.6821% | 0 | 9207.604968 |
| 6x | true | true | +20.5127% | 6.7884% | 0 | 9049.125962 |

## Diagnostic nonfatal 5x/6x lane

Separate diagnostic lane only; not live promotion. 5x and 6x both have `promotion_allowed=false`, total liquidation `0`, and min buffers `9207.604968` / `9049.125962` respectively.

## Decision

- Best common-split strict candidate: `crypto_fx_alpha_zoo_state_calibrated` / selected strategy `alpha_zoo_conservative_exit`, strict `6x`.
- Existing old-split best remains historical-only; common-split old-selected carry-forward and reselected Alpha Zoo are equivalent on this run.
- Hybrid v3.5/v3.6 are not live-promotable because liquidation count/min margin buffer are not available without dedicated integrated margin replay.
- Peak RSS: `769.1015625 MiB`, under 8 GiB.
- Research history/source ledger: not regenerated; no new global source family or chronology/source-ledger change.

## Verification status

Fresh post-deslop local verification passed on 2026-05-17 UTC. Log:

`var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/common_split_alpha_zoo_hybrid_v35_v36_20260517/local_verification_common_split_post_deslop_20260517T054855Z.log`

Commands/results:

- `uv run --extra dev pytest tests/test_crypto_fx_alpha_zoo.py tests/test_triple_barrier_labeler.py tests/test_edge_calibration.py tests/test_crypto_fx_alpha_zoo_state_strategy.py -q` → `23 passed`
- `uv run --extra dev pytest tests/test_profit_moonshot_fresh_start_replay.py tests/test_profit_moonshot_liquidation_aware_validation.py tests/test_profit_moonshot_live_final_selection.py tests/test_profit_moonshot_pass_under_8gb_validator.py -q` → `74 passed`
- `uv run --extra dev pytest -q` → `1319 passed`
- `uv run --extra dev ruff check .` → passed
- `uv run --extra dev python -m compileall -q src scripts tests` → passed
- `git diff --check` → passed
- `git diff --cached --check` → passed

## Addendum — 2026-05-17 KST integrated margin replay for mixed fixed-input hybrids

Follow-up question: the fixed-input hybrid rows were blocked only because the mixed A0/P0/E0/S1/S2/S3/S4 allocator lacked liquidation-count/min-buffer evidence. This is now patched.

Implementation delta:

- `scripts/research/run_profit_moonshot_hybrid_v35_v36_fixed_inputs.py`
  - Adds `mixed_allocator_integrated_margin_replay` using frozen post-selection allocator weights and fixed stream gross-notional fractions.
  - Attaches split-level `liquidation_count`, `minimum_margin_buffer`, `margin_replay_available`, and active-hour evidence to `hybrid_v3_5_optuna` and `hybrid_v3_6_optuna`.
  - Keeps margin replay out of Optuna objective/pruning/selection; locked-OOS remains post-freeze gate/report-only.
- `scripts/research/run_common_split_alpha_zoo_hybrid_v35_v36.py`
  - Propagates hybrid live-promotion status from stage artifacts instead of hardcoding `False`.
- `tests/test_profit_moonshot_hybrid_v35_v36_fixed_inputs.py`
  - Adds regression coverage for account-level liquidation pressure and safe live-policy promotion after integrated margin replay.

Updated hybrid common-split status:

| candidate | split | return | MDD | Sharpe | Sortino | smart Sortino | Calmar | active hours | liq | min buffer | deployable |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| v3.5 | train | +47.7257% | 11.0421% | 2.705685 | 2.932093 | 2.640523 | 4.322148 | 7514 | 0 | 9932.438663 | true |
| v3.5 | validation | +13.3102% | 2.2622% | 5.390597 | 7.610005 | 7.441656 | 51.558258 | 1302 | 0 | 14594.054033 | true |
| v3.5 | locked-OOS | +8.5233% | 1.7654% | 5.259028 | 7.316663 | 7.189734 | 32.173151 | 1467 | 0 | 16587.499982 | true |
| v3.6 | train | +49.5204% | 7.6947% | 2.897597 | 2.999204 | 2.784914 | 6.435678 | 7514 | 0 | 9847.514685 | true |
| v3.6 | validation | +12.4946% | 1.5354% | 7.002337 | 8.680826 | 8.549560 | 69.800312 | 1302 | 0 | 14690.924128 | true |
| v3.6 | locked-OOS | +7.7916% | 1.7491% | 4.859674 | 5.991026 | 5.888040 | 29.199963 | 1467 | 0 | 16664.270300 | true |

Decision update:

- Hybrid v3.5/v3.6 are now live-promotion-capable under strict integrated margin evidence (`total_liquidation_count=0`, minimum margin buffer positive, no rejection reasons).
- They are still lower-return than Alpha Zoo strict 6x on the same locked-OOS (`+20.5127%` Alpha Zoo vs `+8.5233%` v3.5 / `+7.7916%` v3.6). Alpha Zoo remains the common-split performance leader.
- Locked-OOS contamination audit remains clean (`violation=false`); margin replay is post-freeze gate/report-only.
- Peak RSS in the top-level report remains `769.1015625 MiB` (<8 GiB).
- Research history/source ledger not regenerated; no new global source family or chronology/source-ledger change.

Updated report artifacts:

- `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/common_split_alpha_zoo_hybrid_v35_v36_20260517/common_split_alpha_zoo_hybrid_v35_v36_latest.json`
- `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/common_split_alpha_zoo_hybrid_v35_v36_20260517/common_split_alpha_zoo_hybrid_v35_v36_latest.md`
- `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/common_split_alpha_zoo_hybrid_v35_v36_20260517/hybrid_v35_v36_common_split/hybrid_v35_v36_fixed_inputs_common_split_latest.json`
- `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/common_split_alpha_zoo_hybrid_v35_v36_20260517/hybrid_v35_v36_common_split/hybrid_v35_v36_fixed_inputs_common_split_latest.md`

Verification after integrated margin addendum passed on 2026-05-17 UTC. Log: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/common_split_alpha_zoo_hybrid_v35_v36_20260517/local_verification_integrated_margin_20260517T071937Z.log`.

- Alpha Zoo targeted suite: `23 passed`
- Moonshot validation suite: `74 passed`
- Full pytest: `1321 passed`
- `ruff check .`, `compileall`, `git diff --check`, and `git diff --cached --check`: passed

## Addendum — 2026-05-17 KST Alpha Zoo strict 6x live wiring and live-equivalent tests

Follow-up objective: make the common-split #1 strategy (`CryptoFxAlphaZooStateStrategy` / `alpha_zoo_conservative_exit` / strict `6x`) runnable through the live decision path while preserving the replay contract as closely as possible.

Implementation delta:

- `src/lumina_quant/live_selection.py`
  - Maps `crypto_fx_alpha_zoo*`, `alpha_zoo*`, and `profit_moonshot_alpha_zoo*` live references to `CryptoFxAlphaZooStateStrategy`.
  - Preserves live-decision runtime overrides: `symbols`, `strategy_timeframe`, `strategy_params`/`calibrated_edges`, `leverage`, exchange overrides, `target_allocation`, `window_seconds`, `ingest_window_seconds`, and `decision_cadence_seconds`.
- `src/lumina_quant/cli/live.py`
  - Applies strategy-class decision overrides before `LiveConfig.validate()` and before `LiveTrader` construction.
  - Keeps stale live-selection artifacts out of the path when a strategy-class decision is present.
- `src/lumina_quant/configuration/validate.py`
  - Raises the live exchange leverage cap from `3x` to `6x`, matching the strict zero-liquidation integer grid cap used by the common-split Alpha Zoo winner.
- `src/lumina_quant/strategies/crypto_fx_alpha_zoo_state.py`
  - Accepts `decision_cadence_seconds` as a non-tunable runtime parameter.
  - Aggregates MARKET_WINDOW rows into OHLCV before state update so a `3600s` live window can match the hourly replay contract instead of consuming only the final 1s row.
- New decision artifact:
  - `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/common_split_alpha_zoo_hybrid_v35_v36_20260517/live_alpha_zoo_strict_6x_decision_latest.json`
  - `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/common_split_alpha_zoo_hybrid_v35_v36_20260517/live_alpha_zoo_strict_6x_decision_latest.md`

Live decision contract:

- `decision=selected_live_mode`, `target_kind=strategy_class`, `strategy_name=CryptoFxAlphaZooStateStrategy`.
- Symbols: `BTC/USDT`, `ETH/USDT`, `SOL/USDT`, `BNB/USDT`, `TRX/USDT`.
- Runtime: `strategy_timeframe=1h`, `window_seconds=3600`, `ingest_window_seconds=3600`, `decision_cadence_seconds=3600`, isolated `6x`, `target_allocation=0.10`.
- Strategy params include the common-split reselected `alpha_zoo_conservative_exit` params plus train+validation calibrated edges; locked-OOS remains gate/report-only.

Verification added:

- Live selection inference and decision override preservation for Alpha Zoo.
- Live CLI propagation of Alpha Zoo decision params, calibrated edges, symbols, 1h/3600s window/cadence, target allocation, and 6x leverage into `LiveConfig`/`LiveTrader`.
- Runtime validation accepts live `6x` and rejects `>6x`.
- `CryptoFxAlphaZooStateStrategy` MARKET_WINDOW path matches MARKET_BATCH path for hourly Alpha Zoo decisions.
- Decision artifact maps to the live runtime and retains the strict 6x locked-OOS replay evidence (`+20.512682%`, MDD `6.788365%`, liquidation `0`, min buffer positive).

Preflight evidence:

- With the new decision artifact and a non-stale-refresh test horizon, `scripts/ops/live_readiness_preflight.py` reports `paper_run_allowed`; `decision_runtime_compatible=true` and `decision_allows_live_start=true`.

Operator command for paper/live review:

```bash
uv run lq live --transport poll --decision-file var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/common_split_alpha_zoo_hybrid_v35_v36_20260517/live_alpha_zoo_strict_6x_decision_latest.json
```

Caveat: this wiring preserves the replay contract through committed `MARKET_WINDOW` 3600s OHLCV aggregation. Real-mode execution still requires normal live preflight, credentials, `LUMINA_ENABLE_LIVE_REAL`, fresh committed data, and operator review; live fills/slippage/funding can differ from replay.

### Final live-wiring verification — 2026-05-17T09:10Z

Additional live-applicability hardening after the first wiring pass:

- Live decision exchange overrides now also synchronize derived live config fields used by the exchange/trader bootstrap: `EXCHANGE_ID`, `MARKET_TYPE`, `POSITION_MODE`, and `MARGIN_MODE`, not only `EXCHANGE`/`LEVERAGE`; unknown strategy-class decisions fail closed instead of falling back to a default strategy.
- The Alpha Zoo strict 6x decision artifact remains the live-equivalent entry point and still targets isolated Binance futures `6x`, `1h` strategy cadence, `3600s` committed MARKET_WINDOW ingestion, five-symbol crypto universe, and train+validation calibrated edges.
- Preflight with the decision artifact returned `paper_run_allowed`, `decision_runtime_compatible=true`, `decision_allows_live_start=true`, `ready_for_paper=true`, `ready_for_real=false` (real mode remains credentials/operator-gated).

Fresh local verification passed after the live hardening:

- `uv run --extra dev pytest tests/test_live_selection_infer.py tests/test_live_fail_fast_missing_committed_data.py tests/test_live_config_source_validation.py tests/test_crypto_fx_alpha_zoo_state_strategy.py tests/test_common_split_alpha_zoo_hybrid_v35_v36.py -q` → `46 passed`.
- `uv run --extra dev pytest tests/test_live_readiness_ops_scripts.py tests/test_live_binance_market_window_aggregation.py tests/test_market_window_emission_parity_live_vs_backtest.py tests/test_live_trader_config_snapshot.py -q` → `11 passed`.
- `uv run --extra dev pytest tests/test_crypto_fx_alpha_zoo.py tests/test_triple_barrier_labeler.py tests/test_edge_calibration.py tests/test_crypto_fx_alpha_zoo_state_strategy.py -q` → `24 passed`.
- `uv run --extra dev pytest tests/test_profit_moonshot_fresh_start_replay.py tests/test_profit_moonshot_liquidation_aware_validation.py tests/test_profit_moonshot_live_final_selection.py tests/test_profit_moonshot_pass_under_8gb_validator.py -q` → `74 passed`.
- `uv run --extra dev pytest -q` → `1328 passed in 288.32s`.
- `uv run --extra dev ruff check .`, `uv run --extra dev python -m compileall -q src scripts tests`, `git diff --check`, and `git diff --cached --check` passed.

Live-equivalent caveat remains: this validates artifact-to-live runtime wiring and strategy input parity for committed 3600s MARKET_WINDOW bars. It does not assert identical real exchange fills, fees, slippage, funding, or latency.

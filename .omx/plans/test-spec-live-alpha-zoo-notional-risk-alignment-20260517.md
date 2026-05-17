# Test Spec — Live Alpha Zoo notional/risk alignment

Date: 2026-05-17 KST
Linked PRD: `.omx/plans/prd-live-alpha-zoo-notional-risk-alignment-20260517.md`

## Test objectives

1. Prove live sizing and replay sizing use the same exposure semantics for the selected Alpha Zoo lane.
2. Prove legacy fixed `$5000` max-order behavior cannot silently truncate a percentage/equity-scaled futures lane unless explicitly configured as an emergency cap.
3. Prove liquidation-inclusive replay records account-level loss and MDD for isolated high-leverage candidates.
4. Preserve paper/real readiness gates and real-mode fail-fast behavior.

## Unit tests

### Live decision parsing

- File: `tests/test_live_selection_infer.py`
- Add/update cases for `live_alpha_zoo_fast_residual_7x_isolated_decision_latest.json` shape:
  - `strategy_name == CryptoFxAlphaZooStateStrategy`
  - symbols normalized
  - `exchange.margin_mode == isolated`
  - `exchange.leverage == 7`
  - `target_allocation == 0.15`
  - explicit sizing mode equals `isolated_margin_fraction` or equivalent chosen name.

### Portfolio sizing service

- File: new or existing portfolio sizing tests.
- Fixture: equity `$10,000`, price `$100`, leverage `7`, allocation `0.15`, stop distance from strategy `2.5%`.
- Expected in isolated-margin mode: target notional about `$10,500` before exchange rounding/liquidity caps.
- Expected in notional-fraction mode: target notional about `$1,500`.
- Backward compatibility: omitted sizing mode keeps notional-fraction behavior.

### Risk manager caps

- File: new or existing risk-manager tests.
- Fixed `$5000` default should not block the isolated-margin lane when the lane uses equity-scaled caps and no explicit absolute emergency cap is set.
- Explicit absolute emergency cap should still block orders above that value.
- Symbol/total caps should be expressed as equity-scaled notional caps or clearly documented margin caps, with tests for both pass and block cases.

### Real-mode protective order behavior

- File: `tests/test_live_execution_state_machine.py`
- Existing fail-fast test must remain green: real mode with stop_loss/take_profit and no explicit exchange protective params raises.
- Paper mode can still log/warn without blocking.

## Integration tests

### CLI decision artifact propagation

- File: `tests/test_live_fail_fast_missing_committed_data.py` or a dedicated live decision test.
- Assert `LiveConfig` snapshot after CLI decision load contains:
  - 1h timeframe/window/cadence
  - futures market type
  - isolated margin
  - 7x leverage
  - target allocation 0.15
  - selected sizing mode
  - equity-scaled risk caps if specified in the artifact.

### Paper-equivalent smoke

- Add a dry-run or test helper that constructs a live portfolio/order from a deterministic Alpha Zoo signal and asserts order notional matches replay expected notional within rounding tolerance.
- No real exchange calls; use mocks.

## Replay artifact tests

- Add a test around the high-leverage runner ensuring:
  - selection score reads train+validation only
  - locked-OOS fields do not affect rank/tie-break before gate
  - isolated liquidation events, if injected in a synthetic trade path, reduce account equity and contribute to MDD
  - total account wipeout count rejects promotion
  - strict zero-liquidation lane remains separate.

## Cost diagnostics tests

- Assert report includes no-cost and cost-stressed metrics.
- Assert slippage grid includes at least `1, 3, 5, 10, 20 bps` or documents an intentional superset.
- Assert funding grid includes at least `1, 2, 5, 10, 20 bps/day` or documents an intentional superset.
- Assert headline promotion explicitly states whether it is no-cost-only or survives a declared cost threshold.

## Required local verification

```sh
uv run --extra dev pytest tests/test_live_selection_infer.py tests/test_live_fail_fast_missing_committed_data.py tests/test_live_execution_state_machine.py tests/test_crypto_fx_alpha_zoo_state_strategy.py -q
uv run --extra dev pytest tests/test_profit_moonshot_fresh_start_replay.py tests/test_profit_moonshot_liquidation_aware_validation.py tests/test_profit_moonshot_live_final_selection.py tests/test_profit_moonshot_pass_under_8gb_validator.py -q
uv run --extra dev pytest -q
uv run --extra dev ruff check .
uv run --extra dev python -m compileall -q src scripts tests
git diff --check
git diff --cached --check
```

## Completion evidence required in final implementation report

- Commit hash and pushed branch.
- New artifact directory paths.
- Split min/max timestamps.
- Selected candidate, sizing mode, leverage, allocation, notional/equity, margin/equity.
- No-cost and cost-stressed OOS metrics.
- Liquidation count, total account wipeout count, minimum margin buffer or isolated-equivalent account-risk metric.
- Paper-equivalent live sizing parity evidence.
- Preflight status (`ready_for_paper`, `ready_for_real`).
- CI/private-CI links if pushed.

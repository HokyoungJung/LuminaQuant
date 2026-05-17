# PRD — Live Alpha Zoo notional/risk alignment and liquidation-inclusive performance tuning

Date: 2026-05-17 KST
Owner: next LuminaQuant research/live execution session
Status: ready for `$ralplan $team $ralph` handoff

## Requirements summary

The next session must make the current leading Alpha Zoo live candidate executable under a sizing and risk contract that matches the research replay assumptions, then retune for maximum realistic performance while keeping drawdown roughly in the observed range and never allowing total account wipeout.

Current research anchor:

- Candidate: `CryptoFxAlphaZooStateStrategy` / `alpha_zoo_fast_residual`
- Current latest-data split: train `2025-01-01T00:00:00Z..2025-12-31T23:00:00Z`, validation `2026-01-01T00:00:00Z..2026-03-31T23:00:00Z`, locked-OOS `2026-04-01T00:00:00Z..2026-05-17T10:00:00Z`
- Current high-leverage isolated replay winner: `7x`, `allocation_fraction=0.15`, locked-OOS return `+30.53573988518672%`, MDD `11.302719903692077%`, Sharpe `1.8153544967585846`, Sortino `2.3185908095190877`, smart Sortino `2.083139398143474`, Calmar `2.7016275856939624`, trade count `391`, liquidation count `0`, account wipeout count `0`.
- Strict fallback: same params at `6x`, `allocation_fraction=0.10`, locked-OOS return `+16.77825536078088%`, MDD `6.59514326586287%`, liquidation `0`, minimum margin buffer `9150.924759759895`.
- Artifacts:
  - `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/validation_to_20260331_latest_data_20260517/alpha_zoo_validation_march_high_leverage_latest.json`
  - `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/validation_to_20260331_latest_data_20260517/live_alpha_zoo_fast_residual_7x_isolated_decision_latest.json`
  - `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/validation_to_20260331_latest_data_20260517/live_readiness_preflight_7x_latest.json`

Key mismatch to resolve:

- Research replay currently models per-trade account return as `allocation_fraction * leverage * gross_return` in `scripts/research/replay_crypto_fx_alpha_zoo_state.py`.
- Live sizing currently treats `target_allocation` as a notional cap in `src/lumina_quant/services/portfolio.py`; `target_allocation=0.15` therefore tends to cap notional near `15%` of equity, not the replay's `0.15 * 7 = 105%` notional exposure.
- `config.yaml` / runtime schema still contain a static `max_order_value: 5000.0`, which is a legacy fixed-dollar guardrail and does not scale with equity. It should be replaced or subordinated by equity-scaled/leverage-aware caps for this lane.

## Non-negotiable constraints

1. No real-money order execution in this implementation session. Real-mode startup remains blocked unless the user separately authorizes credentialed production execution.
2. No total account wipeout in any promoted candidate. If isolated liquidation is allowed in high-performance lanes, liquidation losses must be included in account equity and MDD.
3. Locked-OOS remains post-freeze gate/report-only. Objective, pruning, ranking, tie-break, allocation/leverage tuning, and cost calibration must use train+validation only.
4. No calendar/month/day/hour entry rule.
5. Return/MDD is diagnostic/report-only, not a hard promotion gate. MDD, liquidation/account-wipeout, and risk-adjusted metrics are hard gates.
6. Memory peak must remain below 8 GiB.
7. Preserve strict zero-liquidation lane separately from any isolated high-performance lane.
8. Every artifact must record actual min/max timestamps per split and selection provenance.

## Target decision

Implement an explicit sizing contract so the live runtime and replay agree on one of these modes:

- `notional_fraction`: `target_allocation` means `notional / equity`.
- `isolated_margin_fraction`: `target_allocation` means `isolated_margin / equity`, so target notional is `equity * target_allocation * leverage`.

For the current 7x/0.15 candidate, the intended high-performance interpretation is `isolated_margin_fraction`: target notional approximately `105%` of equity, with isolated margin approximately `15%` of equity. Binance Futures supports symbol-level initial leverage and margin type changes (`/fapi/v1/leverage`, `/fapi/v1/marginType`), and Hedge mode orders require `positionSide=LONG/SHORT`; local exchange code already has bootstrap and order-param plumbing, but must be verified in paper/testnet.

## Acceptance criteria

### Sizing/risk contract

- Live decision artifacts include an explicit sizing mode, e.g. `sizing_mode: isolated_margin_fraction` or `target_allocation_mode: isolated_margin_fraction`.
- `PortfolioSizingService.risk_based_quantity` and `RiskManager.check_order` can enforce equity-scaled notional caps without relying on a fixed `$5000` default for this lane.
- A $10,000 equity, `target_allocation=0.15`, `leverage=7`, isolated-margin-mode entry produces an intended notional close to `$10,500`, subject only to exchange min/step and configured liquidity/cost caps.
- Backward compatibility is preserved for existing strategies/configs that assume `target_allocation` is a notional cap.

### Replay/live parity

- Add or update tests proving that the selected live decision artifact maps to `CryptoFxAlphaZooStateStrategy`, 1h cadence, isolated futures, `7x`, and the same sizing mode used by replay.
- Add a paper-equivalent sizing test that compares live generated order quantity/notional with replay expected notional for a fixed equity/price/leverage/allocation fixture.
- Existing live fail-closed behavior for unsupported strategy decisions and real-mode protective order gaps remains intact.

### Optimization / risk gates

- Retune leverage/allocation with latest data using train+validation only. Suggested search grid:
  - leverage `1x..20x` initially; widen only if exchange/account caps and MDD remain sane.
  - allocation fractions `0.03, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20`.
  - sizing modes: strict notional, isolated-margin high-performance.
- Locked-OOS report must include liquidation-inclusive returns/MDD. If an isolated liquidation occurs, account return should be capped by isolated allocation loss and included in cumulative equity/MDD; total account wipeout must be zero.
- Hard promotion gate for high-performance live candidate:
  - `total_account_wipeout_count == 0`
  - locked-OOS MDD target near current 7x result: prefer `<= 12%`; tolerate only with explicit rationale up to an absolute hard cap `<= 25%`.
  - locked-OOS return above current-base reference and above strict 6x fallback after cost diagnostics.
  - Sharpe/Sortino/smart Sortino/Calmar positive and qualitatively acceptable.
  - memory <8 GiB.
- Strict zero-liquidation lane remains reported independently with `liquidation_count == 0` and positive minimum margin buffer.

### Cost realism

- Headline report must distinguish no-cost replay from cost-stressed replay.
- Add cost grid at least: round-trip slippage/fees `1, 3, 5, 10, 20 bps`; funding drag `1, 2, 5, 10, 20 bps/day`.
- Promotion should state whether performance survives a realistic fee/slippage threshold. Current 7x/0.15 sensitivity is fragile: locked-OOS flips negative around ~6.77 bps average gross edge per trade; at 5 bps it falls to about `+6.3199%`, at 10 bps about `-13.4130%`.
- The next session may optimize for maker/passive execution only if the live execution path actually supports the assumed order type and fill model; otherwise use taker/market cost assumptions.

### Operational readiness

- Real-mode remains blocked unless all are true: `live.mode=real`, `testnet=false`, `LUMINA_ENABLE_LIVE_REAL=true`, real Binance Futures credentials present, Postgres/audit store reachable, exchange-side isolated/leverage verified per symbol, real preflight returns `ready_for_real=true`, and user explicitly authorizes real-money start.
- Paper/testnet smoke is required before any real-mode proposal.
- Stop/take-profit handling must either map explicit exchange protective params in real mode or intentionally omit strategy stop/take fields with documented compensating risk controls. Current real-mode fail-fast must not be bypassed silently.

## Implementation steps

1. **Map current sizing path and add contract fields**
   - Files likely touched: `src/lumina_quant/live_selection.py`, `src/lumina_quant/cli/live.py`, `src/lumina_quant/configuration/schema.py`, `src/lumina_quant/configuration/runtime_access.py`, `src/lumina_quant/services/portfolio.py`, `src/lumina_quant/risk_manager.py`.
   - Add explicit sizing mode fields from decision artifact to `LiveConfig` snapshot.
   - Preserve existing notional-cap behavior as the default.

2. **Replace static max-order behavior for this lane**
   - Add equity-scaled cap fields such as `max_order_notional_pct`, `max_symbol_notional_pct`, `max_total_notional_pct`, or a clearly named equivalent.
   - Keep absolute `max_order_value` as an optional emergency ceiling only when explicitly set; do not let a stale default `$5000` silently truncate a percentage-based futures lane.

3. **Add replay/live sizing parity tests**
   - Add unit tests for fixed equity/price/leverage/allocation.
   - Update existing live decision tests for the 7x isolated decision to assert sizing mode and notional target behavior.

4. **Retune latest-data high-performance lane**
   - Extend `scripts/research/run_alpha_zoo_validation_march_high_leverage.py` or add a follow-up runner under `scripts/research/`.
   - Produce artifacts under a new directory, e.g. `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/live_notional_risk_aligned_alpha_zoo_20260518/`.
   - Include no-cost and cost-stressed tables; include strict lane and high-performance isolated lane separately.

5. **Paper-equivalent live smoke**
   - Generate a dry-run/paper artifact showing the live runtime would size the top candidate at the same intended notional as replay, after rounding and risk caps.
   - Do not place real orders.

6. **Docs and handoff**
   - Update `docs/research_note_profit_moonshot_alpha_zoo_real_data_20260512.md`.
   - Update `.omx/notepad.md`.
   - Add a session handoff with final artifact paths and exact next command/prompt.
   - State explicitly whether global research history/source ledger changed; likely no if only same-source latest-tail artifacts are reused.

## RALPLAN-DR summary

### Principles

1. Live and replay must use the same exposure contract before comparing performance.
2. Maximize return only inside hard account-survival and drawdown gates.
3. Isolated liquidation can be modeled as a per-position loss, but it must hit equity/MDD.
4. Cost realism is not optional for a short-holding high-turnover strategy.
5. Real-money execution remains a separate authorization and preflight problem.

### Decision drivers

1. Maintain as much of the current 7x high-performance return as possible.
2. Keep locked-OOS MDD roughly near the current 11.30% result and below 25% hard cap.
3. Avoid silent live truncation from legacy static caps or mismatched target-allocation semantics.

### Viable options

- **Option A — Align live to replay with `isolated_margin_fraction`**
  - Pros: preserves intended 7x/0.15 = 105% notional economics; best chance to retain current performance.
  - Cons: requires risk-cap rewrite, stronger tests, and careful paper smoke.
- **Option B — Align replay to current live notional cap**
  - Pros: smallest live-code change and safest operationally.
  - Cons: likely collapses return roughly toward 1/7 of the current 7x result; does not satisfy “maximize performance” unless re-optimized.
- **Option C — Dual-mode explicit contract**
  - Pros: backward compatible and makes future strategy artifacts unambiguous.
  - Cons: more implementation surface; requires tests for both modes.

Favored path: **Option C with Option A as the selected lane for Alpha Zoo high-performance candidate**, because it preserves existing behavior while allowing the intended isolated-margin strategy to run without hidden truncation.

## ADR

### Decision

Add an explicit live/replay sizing-mode contract and retune Alpha Zoo under liquidation-inclusive, equity-scaled risk caps.

### Drivers

- Current artifact performance assumes `allocation_fraction * leverage` notional exposure.
- Current live sizing and static `$5000` cap can materially under-size the intended trade.
- User wants maximum performance while maintaining MDD and preventing total account liquidation.

### Alternatives considered

- Leave live sizing as-is: rejected because it produces different economics than the research result.
- Raise `max_order_value` only: rejected because fixed dollar caps are not equity-scaled and remain wrong for both small and large accounts.
- Use cross-margin: rejected for this lane because current replay and user intent are isolated-margin based.

### Consequences

- More explicit runtime config and decision artifacts.
- Additional parity tests become required for live promotion.
- Cost-stressed metrics may demote the current top candidate despite no-cost replay strength.

### Follow-ups

- Add exchange-side margin/leverage verification to real preflight if not already present.
- Add maker/passive execution research only after order-type/fill assumptions are implemented.
- Consider volatility/liquidity throttles only if they are trained/tuned without locked-OOS leakage.

## Verification commands

Run at minimum after implementation:

```sh
uv run --extra dev pytest tests/test_live_selection_infer.py tests/test_live_fail_fast_missing_committed_data.py tests/test_live_execution_state_machine.py tests/test_crypto_fx_alpha_zoo_state_strategy.py -q
uv run --extra dev pytest tests/test_profit_moonshot_fresh_start_replay.py tests/test_profit_moonshot_liquidation_aware_validation.py tests/test_profit_moonshot_live_final_selection.py tests/test_profit_moonshot_pass_under_8gb_validator.py -q
uv run --extra dev pytest -q
uv run --extra dev ruff check .
uv run --extra dev python -m compileall -q src scripts tests
git diff --check
git diff --cached --check
```

If pushing implementation, confirm GitHub Actions `ci` and `private-ci` green.

## Available-agent staffing guidance

- `explore`: map current sizing/risk/preflight code before edits.
- `executor`: implement config/sizing/risk/replay changes.
- `test-engineer`: own parity, risk-cap, and replay artifact tests.
- `verifier`: validate artifacts, leakage gates, MDD/liquidation accounting, and CI evidence.
- `critic` or `code-reviewer`: review live real-money safety boundaries before any real-mode handoff.

Recommended next-session mode: `$ralplan $team $ralph` if coordinating code/replay/docs lanes; otherwise `$ralph` with this PRD and test spec if one owner is doing the implementation.

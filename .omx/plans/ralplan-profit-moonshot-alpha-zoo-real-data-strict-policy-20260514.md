# RALPLAN-DR — Profit Moonshot Alpha Zoo Return/MDD-Diagnostic Policy Amendment (2026-05-14)

## Scope

This file supersedes the earlier same-day strict return/MDD gate plan. Latest operator clarification: `return/MDD` is **not** required as a strict deployability gate. Preserve the non-calendar, train/validation-only, locked-OOS gate/report-only, strict zero-liquidation policy while treating return/MDD as diagnostic/report-only.

## Principles

1. **No invalid live promotion:** calendar/month/day/hour/current-base tuples remain hypothesis references only.
2. **Train/validation ownership:** selection and edge calibration use only train+validation; locked-OOS opens after candidate freeze for gate/report only.
3. **Strict safety deployment:** deployable success requires zero liquidation, positive buffers, OOS MDD <=25%, OOS return beating the current-base reference, and positive Sharpe/Sortino/smart Sortino/Calmar.
4. **Return/MDD diagnostic:** OOS return/MDD is reported against the invalid current-base reference but is not a promotion hurdle.
5. **Lane separation:** strict zero-liquidation lane may promote; diagnostic nonfatal 5x/6x lane can never promote.

## ADR

- **Decision:** Apply return/MDD-diagnostic policy and regenerate replay/summary/docs from existing real-data screen, triple-barrier ledger, and train/validation calibration artifacts.
- **Drivers:** latest operator clarification; avoid blocking a zero-liquidation real-data Alpha Zoo candidate on a diagnostic ratio.
- **Rejected:** Treat return/MDD as a strict hard gate | contradicted the latest operator correction.
- **Consequences:** current `6.0x` Alpha Zoo state row is deployable under the corrected policy while still reporting return/MDD `3.007073` vs current-base `6.916878` as diagnostic-only.

## Acceptance Criteria

- No calendar/month/day/hour entry rules.
- Selected/calibrated from train+validation only.
- Locked-OOS used only after candidate freeze.
- Strict lane liquidation count `0` and minimum margin buffer `>0`.
- OOS MDD `<=25%`.
- OOS return `> 0.06428110030664325`.
- Sharpe, Sortino, smart Sortino, and Calmar strictly positive (`>0.0`).
- Return/MDD comparison appears in diagnostics/reporting only and is absent from strict `performance_gates`.
- Peak RSS `<8192 MiB`.
- Diagnostic 5x/6x lane reports liquidation count, event drawdown, equity loss, recovery, and `promotion_allowed=false`.

## Verification Path

```bash
uv run --extra dev pytest tests/test_crypto_fx_alpha_zoo.py tests/test_triple_barrier_labeler.py tests/test_edge_calibration.py tests/test_crypto_fx_alpha_zoo_state_strategy.py -q
uv run --extra dev pytest tests/test_profit_moonshot_fresh_start_replay.py tests/test_profit_moonshot_liquidation_aware_validation.py tests/test_profit_moonshot_live_final_selection.py tests/test_profit_moonshot_pass_under_8gb_validator.py -q
uv run --extra dev pytest -q
uv run --extra dev ruff check .
uv run --extra dev python -m compileall -q src scripts tests
git diff --check
git diff --cached --check
```

## Stop Rule

Stop when regenerated artifacts prove either a deployable strict zero-liquidation candidate under the return/MDD-diagnostic policy, or an explicit fail-closed no-promotion result for another strict gate.

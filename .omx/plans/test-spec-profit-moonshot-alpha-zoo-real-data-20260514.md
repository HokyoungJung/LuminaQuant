# Test Spec — Profit Moonshot Alpha Zoo Strict Real-Data Policy (2026-05-14)

## Regression requirements

- `CryptoFxAlphaZooStateStrategy` replay must include `oos_return_mdd_beats_current_base` in strict `performance_gates`.
- A strict row cannot promote when return/MDD fails the current-base reference, even if zero-liquidation, positive-buffer, OOS return, MDD, Sharpe, Sortino, smart Sortino, and Calmar gates pass.
- Aggregate summary generation must fail closed unless replay policy reports `return_mdd_hurdle_required=true` and `return_mdd_role=strict_promotion_gate`.
- Screen and replay real-data commands must use `--strict-real-data` and explicit output paths.
- Summary JSON/MD must derive `deployable_success` from replay strict gates, not hand-enter it.

## Required validation commands

```bash
uv run --extra dev pytest tests/test_crypto_fx_alpha_zoo.py tests/test_triple_barrier_labeler.py tests/test_edge_calibration.py tests/test_crypto_fx_alpha_zoo_state_strategy.py -q
uv run --extra dev pytest tests/test_profit_moonshot_fresh_start_replay.py tests/test_profit_moonshot_liquidation_aware_validation.py tests/test_profit_moonshot_live_final_selection.py tests/test_profit_moonshot_pass_under_8gb_validator.py -q
uv run --extra dev pytest -q
uv run --extra dev ruff check .
uv run --extra dev python -m compileall -q src scripts tests
git diff --check
git diff --cached --check
```

## Artifact assertions

- `locked_oos_calibration_record_count=0`.
- `uses_locked_oos_for_selection=false`.
- `current_base_calendar_tuple_role=hypothesis_reference_only`.
- Diagnostic 5x/6x lane has `promotion_allowed=false`.
- Peak RSS `<8192 MiB`.
- Regenerated 20260514 Alpha Zoo artifacts contain no stale relaxed-policy promotion claim.

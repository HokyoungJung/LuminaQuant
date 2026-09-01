# Test spec — Profit moonshot live final selection 2026-05-10

## Goal
Prove the final live-selection pipeline is data-current, leakage-safe, liquidation/margin-aware, memory-safe under 8 GiB, and produces a canonical final recommendation artifact before commit/push/CI signoff.

## Behavior locks before implementation

### 1. Data cutoff freshness
- Test/helper must reject downstream final-selection artifacts that use the stale default `2026-05-06` when refresh metadata proves a later complete OOS date exists.
- Test/helper must derive `latest_complete_oos_end_date` from the minimum `after_ohlcv_max_utc` across required symbols.
- If the minimum timestamp is intraday and not complete through `23:59:00Z`, the derived date must be the previous UTC date.

### 2. Locked-OOS selection firewall
- A poisoned OOS metric must not alter train/validation ranking.
- Final comparison rows must have:
  - `selection_policy.locked_oos == "report_only_gate_only"`,
  - `selection_policy.uses_locked_oos_for_selection == false`,
  - `selection_policy.selection_inputs == ["train", "validation"]` or equivalent.
- Any row using OOS for selection must be marked non-deployable and rejected by the final gate.

### 3. Liquidation and margin gates
Existing tests in `tests/test_profit_moonshot_liquidation_aware_validation.py` must still prove:
- intrabar adverse high/low crossing the threshold triggers liquidation,
- split summaries record liquidation count, minimum margin buffer, and minimum margin ratio,
- liquidation count and margin-buffer failures block promotion unless explicit tiny tolerance is safe.

Additional final-selection checks must prove:
- liquidations > allowed count reject the row,
- margin buffer <= 0 rejects the row,
- account_wipeout=true rejects the row,
- event drawdown/equity loss above tolerance rejects the row,
- zero-liquidation rows outrank comparable tiny-liquidation rows.

### 4. Candidate-derived vs benchmark-only hybrid labeling
- A row with `kind == "candidate_hybrid"` must have `candidate_derived=true`, `benchmark_only=false`, and a source artifact proving profit-moonshot candidate streams/rows fed the hybrid path.
- A legacy hybrid row must have `kind == "legacy_hybrid_benchmark"`, `candidate_derived=false`, and `benchmark_only=true`.
- Benchmark-only rows may appear in the report but cannot be selected as the final live candidate unless separately promoted by an explicit non-candidate deployability path.

### 5. Final decision artifact schema
A canonical final decision JSON must include:
- `generated_at_utc`,
- `data_cutoff.latest_complete_oos_end_date`,
- `selection_policy`,
- `rows[]`,
- `winner` or `recommendation=no_live_promotion`,
- `metrics_explanation`,
- `memory_ledger`,
- `verification`,
- `source_artifacts`,
- `rejected_alternatives`.

Each row must include train/validation/OOS metrics, liquidation/margin metrics, gate status, source path, kind, candidate-derived/benchmark-only flags, and memory references.

### 6. Memory evidence
- Tests/helpers must parse `/usr/bin/time -v` logs or memory guard outputs and fail if any heavy command max RSS >= 8 GiB.
- Full pipeline must not schedule multiple heavy jobs concurrently.

## Targeted tests to run
Run after code/report changes:
```bash
.venv/bin/pytest tests/test_profit_moonshot_liquidation_aware_validation.py -q
.venv/bin/pytest tests/test_profit_moonshot_fresh_portfolio_tuning.py tests/test_profit_moonshot_pass_under_8gb_validator.py -q
.venv/bin/pytest tests/test_hybrid_online_portfolio.py -q
```

Mandatory final-selection comparison/report tests:
```bash
.venv/bin/pytest tests/test_profit_moonshot_live_final_selection.py -q
```
The test file must cover stale cutoff rejection, latest-complete-date derivation, no-OOS ranking, frozen train/validation score parity, benchmark-only labels, candidate_hybrid exclusion without adapter evidence, gate rejection, metrics explanation, and memory ledger parsing.

## Full verification commands
Run sequentially, never concurrently with heavy backtests:
```bash
/usr/bin/time -v .venv/bin/pytest -q
.venv/bin/ruff check .
.venv/bin/python -m compileall -q src scripts tests
git diff --check
```

## Heavy run evidence commands
Each heavy command must be wrapped and logs saved under `var/reports/profit_moonshot_20260501/live_final_selection_20260510/runner_evidence/`:
```bash
/usr/bin/time -v .venv/bin/python scripts/research/refresh_final_portfolio_validation_data.py ...
/usr/bin/time -v .venv/bin/python scripts/research/tune_profit_moonshot_fresh_portfolio.py --oos-end-date <latest_complete_oos_end_date> ...
/usr/bin/time -v .venv/bin/python scripts/research/run_profit_moonshot_liquidation_aware_validation.py --oos-end-date <latest_complete_oos_end_date> ...
/usr/bin/time -v .venv/bin/python scripts/research/write_profit_moonshot_live_final_selection.py \
  --refresh-json var/reports/profit_moonshot_20260501/live_final_selection_20260510/data_refresh/data_refresh_latest.json \
  --candidate-portfolio-json var/reports/profit_moonshot_20260501/live_final_selection_20260510/candidate_portfolio/fresh_portfolio_tuning_latest.json \
  --liquidation-json var/reports/profit_moonshot_20260501/live_final_selection_20260510/liquidation_validation/liquidation_aware_current_base_latest.json \
  --legacy-hybrid-json var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped/portfolio_hybrid_online_current/hybrid_online_portfolio_latest.json \
  --output-dir var/reports/profit_moonshot_20260501/live_final_selection_20260510/final_decision
```

## Final ranking formula
Use the frozen train/validation-only stability score from `tune_profit_moonshot_fresh_portfolio.py::_train_val_stability_score_from_components`:
`35*min(train_monthlyized_return,0.06) + 45*min(validation_monthlyized_return,0.12) + 0.40*train_sharpe + 0.60*validation_sharpe + 0.35*train_sortino + 0.55*validation_sortino + 0.20*min(train_calmar,20) + 0.30*min(validation_calmar,60) - 35*train_max_drawdown - 45*validation_max_drawdown - 0.15*max(0, leverage-current_base_leverage) - 0.25*max(0, sleeve_count-current_base_sleeve_count)`.
Locked-OOS must not be referenced by the ranking function.

## Required row source artifacts
- current_base: `--liquidation-json.current_base` or equivalent replay section.
- direct_candidate: `--liquidation-json.reselected_candidate`, `forced_5x`, or `candidate_results`.
- candidate_portfolio/multi_portfolio: `--candidate-portfolio-json` selected train/validation rows.
- legacy_hybrid_benchmark: `--legacy-hybrid-json` only, always benchmark-only.
- candidate_hybrid: omitted unless adapter evidence exists in the same branch.
- cash: synthetic.

## Pass/fail gates
Pass only if:
- train/validation/OOS liquidation evidence is within strict/tiny-tolerance policy,
- every split minimum margin buffer > 0,
- OOS MDD <= 25%,
- OOS return and return/MDD improve over current-base replay,
- Sharpe/Sortino/smart Sortino/Calmar are reported and acceptable,
- locked-OOS is report/gate-only,
- max RSS for every heavy command is <8 GiB,
- targeted tests and full verification pass,
- Lore commit pushed to `private/main`,
- GitHub Actions `ci` and `private-ci` green.

Fail with documented `no_live_promotion` if no candidate-derived row clears the gates. Do not weaken gates to force a promotion.

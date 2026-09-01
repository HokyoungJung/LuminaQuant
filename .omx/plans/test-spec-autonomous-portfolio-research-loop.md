# Test Spec — Autonomous Portfolio Research Loop

## Behavior Lock
- Heavy experiment execution stays single-lane under an explicit <8 GiB total-memory contract, and no heavy lane may silently fall back to ambient system-resolved memory.
- Every experiment writes a ledger row with `keep` / `discard` / `crash`.
- Strategy/portfolio promotion still requires leakage-safe validation plus final locked-OOS decision.
- No dead-end experiment is silently kept as a production candidate.

## Cleanup Plan
1. Add the experiment ledger and logging tests first.
2. Add audit / methodology / portfolio tests as each new lane is introduced.
3. Delete or isolate dead experiment glue after each discarded branch.
4. Keep the production decision path clean while research lanes evolve.

## Unit Tests
- experiment ledger writer/reader
- crash classification and logging helpers
- any new selection/objective helpers
- explicit budget propagation for dynamic/overlay/optimizer lanes
- explicit budget propagation in `scripts/research/run_causal_dynamic_portfolio.py` and `scripts/research/run_causal_overlay_portfolio.py`
- any cap-relaxation or memory-budget guard helpers

## Integration Tests
- focused exact-window research batch emits summary/details plus ledger row while reusing the existing exact-window registry/lock
- bundle -> portfolio search -> comparison -> decision -> ledger update
- dynamic/overlay/optimizer lanes prove explicit 8 GiB budget injection in emitted memory artifacts (`memory_policy.explicit_budget_bytes == 8 * 1024**3`)
- crash scenario records `crash` without corrupting the ledger/index

## Workflow / End-to-End Checks
- one full autonomous cycle from idea -> implementation -> experiment -> decision -> ledger
- team reaches terminal state before shutdown
- Ralph completes with fresh verification evidence and architect approval

## Verification Commands
```bash
uv run pytest -q tests/test_exact_window_suite.py tests/test_run_research_candidates_script.py tests/test_run_portfolio_optimization_script.py
uv run pytest -q tests/test_build_portfolio_exact_window_freeze.py tests/test_search_portfolio_four_sleeve_anchored.py tests/test_write_portfolio_max_performance_decision.py tests/test_exact_window_runtime.py tests/test_causal_dynamic_portfolio.py tests/test_causal_overlay_portfolio.py
uv run ruff check src/lumina_quant scripts tests
uv run python -m compileall src scripts tests
```

## Acceptance Evidence
- stack audit artifact
- experiment ledger with at least one `keep`, one `discard`, and any `crash` rows if encountered
- proof that the exact-window registry remains the heavy-run source of truth and the ledger is only an index
- memory evidence for heavy runs
- proof that dynamic/overlay memory artifacts emit explicit 8 GiB budget metadata
- fresh decision artifact showing whether the incumbent was kept or replaced
- team startup/terminal/shutdown evidence
- architect sign-off for the milestone

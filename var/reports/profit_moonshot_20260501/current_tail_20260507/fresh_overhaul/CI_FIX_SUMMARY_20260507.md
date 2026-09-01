# CI fix summary — 2026-05-07

## Root causes fixed

1. `ci` quality job failed at Ruff because `ruff check .` linted generated artifacts under `var/reports/...`.
   - Fix: add `var` to `tool.ruff.extend-exclude` while keeping `src`, `apps`, `scripts`, and `tests` linted.
2. Hardcoded parameter audit would fail after Ruff because the baseline was stale.
   - Fix: refresh `.github/hardcoded_params_baseline.json` to current strategy state (`567` signatures); future new literals still fail.
3. Full pytest exposed stale unit fixture for profit reboot selection.
   - Fix: update `tests/unit/test_profit_reboot_selection.py` to include the current OOS gate contract.

## Local CI evidence

- Ruff: `All checks passed!`
- Architecture gates: live data, market-window parity, native Binance all passed.
- `scripts/check_architecture.py`: passed.
- `scripts/audit_hardcoded_params.py`: `total=567 new=0 baselined=567`.
- Docs verification: `70 markdown files checked`.
- Raw-first CI subset: `76 passed`.
- Dashboard lint/test/typecheck/build: passed.
- Bytecode sanity: passed.
- Full pytest: `1168 passed, 1262 warnings`.
- Benchmark 8GB gate: peak RSS `184.45MiB < 7372.80MiB`.
- GPU contract: runtime smoke passed; GPU contract tests `24 passed`.

## Alpha/tuning evidence

- Fresh-start individual replay: `1219` specs, `0` survivors/success candidates, peak RSS `2547.137 MiB`.
- Fresh multi-sleeve tuning: `25194` portfolio specs, `0` success candidates, peak RSS `2670.785 MiB`.
- Do not promote the fresh candidates; keep searching with feature diagnostics/backfills before live-equivalent promotion.

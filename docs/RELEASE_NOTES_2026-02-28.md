# Release Notes — 2026-02-28

## Scope

This release focuses on onboarding correctness, reproducible minimum-run behavior, and repo governance hygiene.

## Included commits

- `322288fe8db594eb978cb8eefebea8774ff32f33`
  - README/README_KR clone/path consistency fixes
  - Backtest audit persistence fallback when Postgres DSN is absent
- `228e80c11291d96983c769934bb13467a0bcca1b`
  - Docs identity/runbook consistency updates
  - README quickstart path regression test
- `38854ff681aaef1614d4a3221f789354266104dd`
  - Added no-infra minimum run script + smoke test
  - Added CONTRIBUTING/SECURITY and GitHub templates

## Validation summary

Validated on commit `38854ff681aaef1614d4a3221f789354266104dd`:

- `uv run ruff check .`
- `uv run python scripts/check_architecture.py`
- `uv run python scripts/audit_hardcoded_params.py`
- `uv run pytest -q` (CI parity subset + onboarding smoke tests)
- `uv run python scripts/benchmark_backtest.py --iters 1 --warmup 0`
- `uv run python scripts/minimum_viable_run.py --days 45`

## CI status (both remotes)

- Quants-agent CI: https://github.com/hoky1227/Quants-agent/actions/runs/22518761539
- Quants-agent cross-platform-ci: https://github.com/hoky1227/Quants-agent/actions/runs/22518761548
- LuminaQuant CI: https://github.com/HokyoungJung/LuminaQuant/actions/runs/22518761915
- LuminaQuant cross-platform-ci: https://github.com/HokyoungJung/LuminaQuant/actions/runs/22518761913

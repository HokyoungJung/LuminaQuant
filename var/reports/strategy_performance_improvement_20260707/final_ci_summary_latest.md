# Final CI summary — Strategy Performance Improvement

Generated: `2026-07-07T12:27:37.929129Z`

## Decision

Local CI-equivalent and dashboard gates passed after the final golden/report cleanup. The full-universe walk-forward performance claim remains **not claimed** because `data/market_parquet` has no direct 1m-derived bars.

## Checks

- `uv sync --extra optimize --extra dev --extra live --extra dashboard`: **PASS**
- `uv run python scripts/build_native_backends.py`: **PASS**
- `bash scripts/ci/architecture_gate_live_data.sh`: **PASS**
- `bash scripts/ci/architecture_gate_market_window_contract.sh`: **PASS**
- `uv run pytest -q <CI raw-first subset>`: **PASS** — 78 passed
- `uv run python scripts/check_architecture.py`: **PASS**
- `uv run python scripts/audit_hardcoded_params.py`: **PASS** — total=1135, new=0
- `uv run python scripts/verify_docs.py`: **PASS** — 140 markdown files verified
- `timeout 900s uv run pytest -q`: **PASS** — 3571 passed, 20 skipped, 3 xfailed in 241.52s
- `npm install --no-fund --no-audit`: **PASS**
- `npm run lint`: **PASS**
- `npm run test`: **PASS** — 60 tests
- `npm run typecheck`: **PASS**
- `npm run build`: **PASS**
- `uv run pytest -q --cov=... --cov-branch --cov-report=term-missing`: **PASS** — 3571 passed, 20 skipped, 3 xfailed; total coverage 79%
- `coverage report --include financial core modules`: **PASS** — 83% >= 70% gate
- `coverage report --include live/exchanges/core modules`: **PASS** — 70% >= 65% gate
- `uv run python generate_data.py`: **PASS**
- `uv run python scripts/benchmark_backtest.py --iters 1 --warmup 0 --output <tmp>/luminaquant-final-ci-20260707/benchmarks/ci_smoke.json`: **PASS** — median 0.05218s, 12149 bars/sec
- `uv run python scripts/verify_8gb_baseline.py --benchmark <tmp>/luminaquant-final-ci-20260707/benchmarks/ci_smoke.json --skip-dmesg ...`: **PASS** — peak_rss 254.19 MiB < 7372.80 MiB; disk_total 25.040 GiB <= 30 GiB
- `bash scripts/ci/architecture_gate_binance_native.sh`: **PASS**
- `uv run --extra dev ruff format --check . && uv run --extra dev ruff check . && clean-env uv run --extra dev pytest -q <report+strategy+engine targeted suites>`: **PASS** — 70 passed in 27.85s

## Claims

- Strategy candidate status: research/shadow-only; no clean/live promotion.
- Full-universe WF status: blocked/not claimed due missing direct 1m bars.
- Native/Rust status: native backend build/gates passed; no untested Rust hot-path changes were introduced for this candidate.

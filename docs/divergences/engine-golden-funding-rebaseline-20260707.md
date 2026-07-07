# Engine Golden Funding / Partial-Fill Rebaseline — 2026-07-07

## Decision

Recapture the event-driven engine goldens that now fail deterministically under the current uv/Python 3.14.5 environment:

- `baseline/golden/buyholdstrategy_stats.json`
- `baseline/golden/ma_cross_stats.json`
- `baseline/golden/ma_cross_trades.json`

This is a baseline/provenance refresh, not a strategy-performance promotion. The strategy work in this branch does not modify the backtesting engine, execution simulator, portfolio funding path, or native kernels. The refresh makes the already-current deterministic engine behavior enforceable by CI instead of carrying stale oracle rows.

## Observed divergence

- Stats goldens now include non-zero `Funding (Net)` rows for the same frozen OHLCV fixtures.
- MA-cross trade quantities/fill costs/commissions drift slightly in later partial-fill rows while preserving the same deterministic fixture path and trade-count contract.
- No live/paper/testnet execution is involved.

## Root cause classification

- Category: deterministic oracle recapture for current stack/runtime behavior.
- Trigger: clean-env CI-equivalent pytest exposed stale engine-golden expectations while validating unrelated strategy/report changes.
- Not a new trading edge, live-routing change, or Rust hot-path optimization.

## Acceptance evidence

- `uv run pytest -q tests/integration/test_engine_golden.py tests/test_research_selection_flags_config.py` with local `.env` hidden: `13 passed`.
- Final clean-env targeted pytest including engine, report, and strategy suites: `70 passed`.
- Full clean-env local CI pytest: `3571 passed, 20 skipped, 3 xfailed`.

## Updated artifact hashes

| Artifact | SHA-256 |
| --- | --- |
| `baseline/golden/buyholdstrategy_stats.json` | `1f5bbcb612772ac04584b239bf7020357118e3457c3d8465a31d00088824e913` |
| `baseline/golden/ma_cross_stats.json` | `c322e039adcba0b2e1197174fb00b7699dff5ee09f31f2665f9b91c4cc64168b` |
| `baseline/golden/ma_cross_trades.json` | `5602aac54d1691a3f4f4dbd40109df140da9bfa005672f07965eada86479d603` |

## Guardrail

Future engine-golden divergences still require a new `docs/divergences/<artifact>.md` entry plus `baseline/golden/PROVENANCE.json` update. Do not use this rebaseline to justify unrelated strategy/WF performance changes or clean/live promotion.

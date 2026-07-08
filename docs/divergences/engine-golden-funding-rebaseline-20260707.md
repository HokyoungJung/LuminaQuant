# Engine Golden Data-Dict Isolation Rebaseline — 2026-07-08

## Decision

Supersede the 2026-07-07 engine-golden recapture and recapture the event-driven engine goldens to CI/no-sidecar values:

- `baseline/golden/buyholdstrategy_stats.json`
- `baseline/golden/ma_cross_stats.json`
- `baseline/golden/ma_cross_trades.json`

Preloaded `data_dict` backtests now treat supplied OHLCV frames as complete inputs. Ambient local `data/market_parquet` feature stores are not consulted unless the caller explicitly passes `feature_db_path` or `feature_exchange`.

This is a baseline/provenance correction, not a strategy-performance promotion. No live, paper, testnet, order-routing, or native-kernel behavior is promoted by this recapture.

## Observed divergence

- Remote CI, which has no local feature sidecar store, produced `Funding (Net)=0.0000` for the frozen synthetic OHLCV engine goldens.
- The 2026-07-07 local recapture included non-zero funding values from an ambient local feature store, so the committed goldens were machine-dependent.
- MA-cross trade quantities/fill costs/commissions return to the CI/no-sidecar deterministic values while preserving the same deterministic fixture path and trade-count contract.
- No live/paper/testnet execution is involved.

## Root cause classification

- Category: deterministic isolation fix for preloaded fixture/data_dict backtests.
- Trigger: remote CI failed `tests/integration/test_engine_golden.py` after local recapture passed with a hidden `.env`, exposing local sidecar feature-store contamination.
- Fix: `HistoricCSVDataHandler` disables default sidecar feature lookup for preloaded `data_dict` inputs unless feature lookup is explicitly configured.
- Not a new trading edge, live-routing change, or Rust hot-path optimization.

## Acceptance evidence

- `uv run pytest -q tests/integration/test_engine_golden.py tests/test_historic_data_feature_support.py` with local `.env` hidden: `7 passed`.
- `uv run pytest -q` with local `.env` hidden: `3575 passed, 20 skipped, 3 xfailed`.
- `uv run ruff check .`: passed.
- `uv run ruff format --check .`: `1033 files already formatted`.

## Updated artifact hashes

| Artifact | SHA-256 |
| --- | --- |
| `baseline/golden/buyholdstrategy_stats.json` | `a4543e116cc571508379972f312f93a4a9439ab01ad27bdc8ce943dc30cf55a1` |
| `baseline/golden/ma_cross_stats.json` | `c1778993530a7881bb230e0218308cbb3eff4810bf26f75c7d7be94fd5fea4fd` |
| `baseline/golden/ma_cross_trades.json` | `37908ffad55086857bc86c98a0497fbeba0f126f5cf0e6a033defa363c06bb65` |

## Guardrail

Future engine-golden divergences still require a `docs/divergences/<artifact>.md` entry plus `baseline/golden/PROVENANCE.json` update. Do not use this rebaseline to justify unrelated strategy/WF performance changes or clean/live promotion.

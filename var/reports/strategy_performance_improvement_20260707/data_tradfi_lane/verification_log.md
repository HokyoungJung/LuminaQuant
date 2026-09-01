# Data/TradFi lane verification log (sanitized)

Raw pytest rootdir/output lines were omitted to avoid committing absolute worker paths. Bounded evidence is preserved below and in `tradfi_data_coverage_summary_latest.md`.

- `uv run --extra dev pytest tests/test_research_universe.py tests/research/test_tradfi_fetcher_gate.py tests/test_collect_binance_1m_research_universe.py -q`: PASS (`18 passed`)
- `uv run --extra dev ruff check scripts/collect_binance_1m_research_universe.py src/lumina_quant/research_universe.py tests/test_collect_binance_1m_research_universe.py tests/test_research_universe.py tests/research/test_tradfi_fetcher_gate.py`: PASS
- `uv run --extra dev python -m compileall -q scripts/collect_binance_1m_research_universe.py src/lumina_quant/research_universe.py`: PASS

No market-data downloads or writes were performed; collector report shows `dry_run=true`, `fetched_rows=0`, and `upserted_rows=0`.

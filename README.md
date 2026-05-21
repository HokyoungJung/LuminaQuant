# LuminaQuant Public Pipeline

This repository is a sanitized public sample of the LuminaQuant testing pipeline.
It intentionally contains only:

- a deterministic sample OHLCV dataset,
- one educational moving-average sample strategy,
- a lightweight backtest runner,
- a paper-live replay runner,
- a small CLI and CI test suite.

It does **not** contain proprietary strategies, production data, data collection code,
research notes, exchange connectors, credentials, optimization artifacts, or deployment
configuration. The paper-live runner is a local simulator only; it cannot place orders.

## Documentation

- [Public scope](docs/PUBLIC_SCOPE.md)
- [Pipeline usage](docs/PIPELINE_USAGE.md)
- [Korean public scope](docs/PUBLIC_SCOPE_KR.md)
- [Korean pipeline usage](docs/PIPELINE_USAGE_KR.md)

## Quick start

```bash
python -m pip install -e '.[dev]'
lq-public backtest --data sample_data/sample_ohlcv.csv
lq-public paper-live --data sample_data/sample_ohlcv.csv
pytest
ruff check .
```


## Generic metrics and optimization

The public optimizer is intentionally generic. It only searches the sample
moving-average windows over local sample data and scores candidates with
`total_return - 2 * max_drawdown - 0.0001 * trade_count`.

```bash
lq-public optimize --data sample_data/sample_ohlcv.csv --fast-grid 2,3,4 --slow-grid 6,8,10
```

## Optional Rust kernel

A source-checkout Rust backtest kernel is included under
`native/rust_backtest_kernel/`. It mirrors the sample Python backtest summary
and is checked in CI.

```bash
lq-public backtest --engine rust --data sample_data/sample_ohlcv.csv
cargo test --manifest-path native/rust_backtest_kernel/Cargo.toml
```

## Commands

### Backtest

```bash
lq-public backtest --data sample_data/sample_ohlcv.csv --fast-window 3 --slow-window 8
```

### Paper-live replay

```bash
lq-public paper-live --data sample_data/sample_ohlcv.csv --fast-window 3 --slow-window 8
```

Both commands print JSON summaries and use only local sample data.

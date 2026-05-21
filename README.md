# LuminaQuant Public Pipeline

This repository is a sanitized public sample of the LuminaQuant testing pipeline.
It intentionally contains only:

- a deterministic sample OHLCV dataset,
- one educational moving-average sample strategy at `src/lumina_quant/sample_strategy.py`,
- a lightweight backtest runner,
- a paper-live replay runner,
- generic metrics, scoring, and sample-only tuning,
- a public sample TOML configuration file,
- a small CLI and CI test suite.

It does **not** contain proprietary strategies, production data, data collection code,
research notes, exchange connectors, credentials, private optimization artifacts, or deployment
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

The public optimizer is intentionally generic and defaults to Optuna. It can
optimize any importable strategy class whose constructor parameters are defined
in the public TOML search space. The sample objective is
`total_return - 2 * max_drawdown - 0.0001 * trade_count`.

```bash
lq-public optimize --data sample_data/sample_ohlcv.csv --n-trials 16
lq-public optimize --method grid --data sample_data/sample_ohlcv.csv --fast-grid 2,3,4 --slow-grid 6,8,10
lq-public optimize --config sample_configs/public_sample_pipeline.toml
```

## Public sample config

`sample_configs/public_sample_pipeline.toml` holds only sample-data paths,
window ranges, optimization method, Optuna trial count/seed, starting cash, and fee settings. CLI arguments override config
values, so the same public pipeline can be smoke-tested without private files.

```bash
lq-public backtest --config sample_configs/public_sample_pipeline.toml
lq-public paper-live --config sample_configs/public_sample_pipeline.toml
lq-public optimize --config sample_configs/public_sample_pipeline.toml --fast-grid 2,3 --slow-grid 6,8
```

## Bring your own data and strategy

A user can plug in a local CSV and an importable strategy class without editing
the pipeline. The strategy class must implement `on_bar(bar)` and return a
`lumina_quant.models.Signal`. Constructor values can come from CLI
`--strategy-param key=value` overrides or from TOML `strategy_params`; Optuna
search ranges live under `[optimization.search_space.<param>]`.

```bash
lq-public backtest --data your_ohlcv.csv --strategy your_pkg.module:YourStrategy --strategy-param lookback=20
lq-public optimize --config sample_configs/public_sample_pipeline.toml
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

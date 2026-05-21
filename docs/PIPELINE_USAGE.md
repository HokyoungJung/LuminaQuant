# Pipeline Usage

## Backtest

```bash
lq-public backtest --data sample_data/sample_ohlcv.csv --fast-window 3 --slow-window 8
```

The backtest uses only local sample bars, a long/flat sample strategy, fixed
initial cash, and a simple fee model.

## Paper-live replay

```bash
lq-public paper-live --data sample_data/sample_ohlcv.csv --fast-window 3 --slow-window 8
```

Paper-live replay walks through the same sample bars as if they arrived over
time. It returns a safety block showing that real order routing is disabled.

## Optimize sample parameters

```bash
lq-public optimize --data sample_data/sample_ohlcv.csv --fast-grid 2,3,4 --slow-grid 6,8,10
```

The optimizer is a generic grid search over sample moving-average windows. It
does not include proprietary research objectives or production parameters.

## Rust kernel check

```bash
lq-public backtest --engine rust --data sample_data/sample_ohlcv.csv
cargo test --manifest-path native/rust_backtest_kernel/Cargo.toml
```

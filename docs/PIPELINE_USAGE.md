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

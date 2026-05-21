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
lq-public optimize --data sample_data/sample_ohlcv.csv --n-trials 16
lq-public optimize --method grid --data sample_data/sample_ohlcv.csv --fast-grid 2,3,4 --slow-grid 6,8,10
lq-public optimize --config sample_configs/public_sample_pipeline.toml
```

The optimizer defaults to Optuna and can also run deterministic grid search.
It optimizes public TOML search spaces only and does not include proprietary
research objectives or production parameters.

## Public sample config

```bash
lq-public backtest --config sample_configs/public_sample_pipeline.toml
lq-public backtest --config sample_configs/public_sample_pipeline.toml --engine python
lq-public paper-live --config sample_configs/public_sample_pipeline.toml
lq-public optimize --config sample_configs/public_sample_pipeline.toml --fast-grid 2,3 --slow-grid 6,8
```

The TOML config is intentionally sample-only. It includes a strategy class
path, strategy constructor params, and an Optuna search space. CLI flags
override the config, which lets CI verify default settings and override
behavior without private inputs.

## External strategy contract

```python
from lumina_quant.models import Bar, Signal, TargetPosition

class YourStrategy:
    def __init__(self, lookback: int = 20) -> None:
        self.lookback = lookback

    def on_bar(self, bar: Bar) -> Signal:
        return Signal(bar.timestamp, TargetPosition.FLAT, "example")
```

```bash
lq-public backtest --data your_ohlcv.csv --strategy your_pkg.module:YourStrategy --strategy-param lookback=20
```

## Rust kernel check

```bash
lq-public backtest --engine rust --data sample_data/sample_ohlcv.csv
cargo test --manifest-path native/rust_backtest_kernel/Cargo.toml
```

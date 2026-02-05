[Korean Version (한국어 가이드)](README_KR.md)

# Quants Agent - Event-Driven Quantitative Trading System

An advanced, event-driven quantitative trading pipeline designed for robust backtesting and live trading on Binance. Built with a modular architecture to ensure consistency between simulation and real-world execution.

## Features

- **Event-Driven Architecture**: Uses a centralized event loop (`MarkteEvent`, `SignalEvent`, `OrderEvent`, `FillEvent`) to mimic real-world trading mechanics.
- **Accurate Backtesting**:
  - Supports **Trailing Stops**, Stop Loss, and Take Profit.
  - Simulates fill latencies and commissions.
- **Live Trading**:
  - Integration with **Binance** via `ccxt`.
  - **Robust State Management**: Syncs positions with exchange on startup.
  - **Partial Fills**: Handles liquidity constraints by queuing remaining order quantities.
- **Optimization**:
  - Built-in parameter tuning using **Optuna** (Bayesian Optimization) or Grid Search.
  - Walk-Forward Analysis (Train/Validation/Test split).

## Project Structure

```
quants-agent/
├── quants_agent/           # Core Package
│   ├── backtest.py         # Backtesting Engine
│   ├── live_trader.py      # Live Trading Engine
│   ├── ...
├── generate_data.py        # Synthetic Data Generator
├── run_backtest.py         # Backtest Entry Point
├── optimize.py             # Strategy Optimizer (Optuna)
├── run_live.py             # Live Trading Entry Point
└── README.md
```

## Setup

1.  **Install Dependencies**
    ```bash
    uv sync
    # OR
    pip install .
    ```

2.  **Configuration**
    Copy `.env.example` to `.env` and configure your keys:
    ```bash
    cp .env.example .env
    ```
    Edit `.env`:
    ```ini
    BINANCE_API_KEY=your_key
    BINANCE_SECRET_KEY=your_secret
    IS_TESTNET=True  # Set to False for real money
    ```

## Usage

### 1. Data Generation
Generate synthetic data for testing (defaults to 1000 days):
```bash
uv run generate_data.py
```

### 2. Backtesting
Run the standard backtest simulation:
```bash
uv run run_backtest.py
```
Results (Sharpe Ratio, Max Drawdown) will be printed to the console, and an `equity.csv` curve will be saved.

### 3. Optimization
Find the best parameters for your strategy using Optuna:
```bash
uv run optimize.py
```
This will run Training, Validation, and Test phases, saving the best parameters to `best_optimized_parameters/`.

### 4. Live Trading
To run the live trader (requires `.env` with API keys):
```bash
uv run run_live.py
```
*Note: Ensure you have sufficient USDT in your account (or Testnet account).*

## Data Analysis
Performance metrics include:
- **Total Return**
- **Sharpe Ratio**
- **Max Drawdown**
- **Drawdown Duration**

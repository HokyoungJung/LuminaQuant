# Performance Metrics

LuminaQuant calculates a comprehensive set of financial metrics to evaluate strategy performance against a benchmark.

## Core Metrics

### Total Return
The absolute percentage gain or loss of the portfolio over the entire backtest period.
$$ \text{Total Return} = \frac{\text{Final Equity} - \text{Initial Equity}}{\text{Initial Equity}} $$

### Benchmark Return
The return of a "Buy & Hold" strategy using the first symbol in your symbol list. Useful for comparing your active strategy against passive investing.

### CAGR (Compound Annual Growth Rate)
The mean annual growth rate of an investment over a specified period of time longer than one year. It smooths out the volatility of periodic returns.
$$ \text{CAGR} = \left( \frac{\text{Ending Value}}{\text{Beginning Value}} \right)^{\frac{1}{n}} - 1 $$
*where $n$ is the number of years.*

## Risk Metrics

### Annualized Volatility
The standard deviation of daily returns, annualized. It represents the risk or variability of the strategy's returns.
$$ \sigma_{ann} = \sigma_{daily} \times \sqrt{\text{Periods Per Year}} $$
*(Periods: 252 for daily, 8760 for hourly, etc.)*

### Max Drawdown (MDD)
The maximum observed loss from a peak to a trough of a portfolio, before a new peak is attained. It indicates the worst possible downside risk over the period.

### Drawdown Duration
The longest amount of time (in bars/days) the strategy spent recovering from a peak to a new high.

## Risk-Adjusted Returns

### Sharpe Ratio
Measures the performance of an investment compared to a risk-free asset, after adjusting for its risk. Higher is better.
$$ \text{Sharpe} = \frac{R_p - R_f}{\sigma_p} $$
*Note: We assume Risk-Free Rate ($R_f$) is 0 for simplicity in crypto/forex context.*

### Sortino Ratio
Similar to the Sharpe ratio but differentiates harmful volatility from total overall volatility by using the asset's standard deviation of **negative** portfolio returns (downside deviation).
$$ \text{Sortino} = \frac{R_p - R_f}{\sigma_d} $$

### Calmar Ratio
A comparison of the average annual compounded rate of return and the maximum drawdown risk of commodity trading advisors and hedge funds.
$$ \text{Calmar} = \frac{\text{CAGR}}{\text{Max Drawdown}} $$

## Comparative Metrics (vs Benchmark)

### Alpha ($\alpha$)
A measure of the active return on an investment, the performance of that investment compared to a suitable market index. Positive alpha indicates the strategy has "beaten the market".

### Beta ($\beta$)
A measure of the volatility—or systematic risk—of a security or portfolio compared to the market as a whole.
*   $\beta = 1$: Moves with the market.
*   $\beta > 1$: More volatile than the market.
*   $\beta < 1$: Less volatile than the market.
*   $\beta < 0$: Inverse correlation.

### Information Ratio (IR)
A measurement of portfolio returns beyond the returns of a benchmark, usually an index, compared to the volatility of those returns.
$$ \text{IR} = \frac{\text{Active Return}}{\text{Tracking Error}} $$

## Trading Stats

### Daily Win Rate
The percentage of days (or periods) where the portfolio return was positive.

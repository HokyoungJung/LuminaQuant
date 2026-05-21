use std::collections::VecDeque;

#[derive(Debug, Clone, PartialEq)]
pub struct Bar {
    pub timestamp: String,
    pub close: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BacktestSummary {
    pub initial_cash: f64,
    pub final_equity: f64,
    pub total_return: f64,
    pub max_drawdown: f64,
    pub trade_count: usize,
}

fn max_drawdown(values: &[f64]) -> f64 {
    let mut peak = values[0];
    let mut worst = 0.0;
    for value in values {
        peak = peak.max(*value);
        if peak > 0.0 {
            let drawdown = (peak - value) / peak;
            if drawdown > worst {
                worst = drawdown;
            }
        }
    }
    worst
}

pub fn parse_csv(text: &str) -> Result<Vec<Bar>, String> {
    let mut bars = Vec::new();
    for (line_number, line) in text.lines().enumerate() {
        if line_number == 0 || line.trim().is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() != 6 {
            return Err(format!(
                "line {} has {} columns",
                line_number + 1,
                parts.len()
            ));
        }
        let close = parts[4]
            .parse::<f64>()
            .map_err(|err| format!("line {} close parse error: {err}", line_number + 1))?;
        bars.push(Bar {
            timestamp: parts[0].to_string(),
            close,
        });
    }
    if bars.is_empty() {
        return Err("no bars loaded".to_string());
    }
    Ok(bars)
}

pub fn run_backtest(
    bars: &[Bar],
    fast_window: usize,
    slow_window: usize,
    initial_cash: f64,
    fee_bps: f64,
) -> Result<BacktestSummary, String> {
    if fast_window == 0 {
        return Err("fast_window must be >= 1".to_string());
    }
    if slow_window <= fast_window {
        return Err("slow_window must be greater than fast_window".to_string());
    }
    let fee_rate = fee_bps / 10_000.0;
    let mut closes: VecDeque<f64> = VecDeque::with_capacity(slow_window);
    let mut cash = initial_cash;
    let mut quantity = 0.0;
    let mut trade_count = 0usize;
    let mut equity_values = Vec::with_capacity(bars.len());

    for bar in bars {
        if closes.len() == slow_window {
            closes.pop_front();
        }
        closes.push_back(bar.close);
        if closes.len() == slow_window {
            let fast_sum: f64 = closes.iter().skip(slow_window - fast_window).sum();
            let slow_sum: f64 = closes.iter().sum();
            let target_long = (fast_sum / fast_window as f64) > (slow_sum / slow_window as f64);
            if target_long && quantity == 0.0 {
                let fee = cash * fee_rate;
                quantity = ((cash - fee) / bar.close).max(0.0);
                cash = 0.0;
                trade_count += 1;
            } else if !target_long && quantity > 0.0 {
                let gross_value = quantity * bar.close;
                let fee = gross_value * fee_rate;
                cash = gross_value - fee;
                quantity = 0.0;
                trade_count += 1;
            }
        }
        equity_values.push(cash + quantity * bar.close);
    }
    let final_equity = *equity_values
        .last()
        .ok_or_else(|| "backtest requires at least one bar".to_string())?;
    Ok(BacktestSummary {
        initial_cash,
        final_equity,
        total_return: (final_equity / initial_cash) - 1.0,
        max_drawdown: max_drawdown(&equity_values),
        trade_count,
    })
}

impl BacktestSummary {
    pub fn to_json(&self) -> String {
        format!(
            "{{\"initial_cash\":{},\"final_equity\":{},\"total_return\":{},\"max_drawdown\":{},\"trade_count\":{}}}",
            self.initial_cash, self.final_equity, self.total_return, self.max_drawdown, self.trade_count
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_csv, run_backtest};

    #[test]
    fn sample_csv_backtest_is_deterministic() {
        let bars = parse_csv(include_str!("../../../sample_data/sample_ohlcv.csv")).unwrap();
        let result = run_backtest(&bars, 3, 8, 10_000.0, 1.0).unwrap();
        assert_eq!(result.trade_count, 7);
        assert!((result.final_equity - 9512.214127269342).abs() < 1e-9);
        assert!((result.max_drawdown - 0.05896695149575966).abs() < 1e-12);
    }
}

# Production guarded current default + wave2 pair overlay frontier

- generated_at: `2026-04-22T10:47:07.142302+00:00`
- requested pair weights: `10% / 15% / 20% / 25% / 30%`
- residual cash: allowed in 5% steps

## Base OOS metrics
- return: `+0.3654%`
- sharpe: `1.6080`
- max_drawdown: `0.3184%`

## Best requested-range row by OOS Sharpe
- weights: base `90.0%` / pair `10.0%` / cash `0.0%`
- OOS return `+0.3424%` / Sharpe `1.6145` / MaxDD `0.3232%`
- train return `+3.2751%` / val return `+7.5959%`

## Best requested-range row by OOS return
- weights: base `90.0%` / pair `10.0%` / cash `0.0%`
- OOS return `+0.3424%` / Sharpe `1.6145` / MaxDD `0.3232%`

## Diagnostic micro overlay (2.5%~10%)
- best diagnostic weights: base `82.5%` / pair `5.0%` / cash `12.5%`
- OOS return `+0.3085%` / Sharpe `1.6262` / MaxDD `0.2811%`

- requested rows clearing standalone OOS bar: `15`
- requested rows beating base on both return+sharpe: `0`

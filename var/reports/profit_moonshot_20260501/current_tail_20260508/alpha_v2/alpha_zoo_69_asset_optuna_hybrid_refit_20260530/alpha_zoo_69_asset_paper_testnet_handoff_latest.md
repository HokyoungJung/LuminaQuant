# 69-asset Alpha Zoo challenger paper/testnet handoff

Status: backtest-gated paper/testnet challenger only. Real-money remains blocked.

## Decision

- ready_for_paper: `True`
- ready_for_real: `False`
- real_money_execution: `False`
- real_execution_allowed: `False`
- backtest_gate: `{'strict_backtest_gate_pass': True, 'relaxed_backtest_gate_pass': True, 'ready_for_paper': True, 'ready_for_real': False, 'real_money_execution': False, 'real_execution_allowed': False, 'rejection_reasons': []}`

## Backtest evidence

- train return: `19.4962%`
- validation return: `18.5503%`
- validation MDD: `1.5113%`
- train/validation RPT proxy: `30.47952168166796` / `237.95139403615903` bps
- component trade events train/validation: `3121` / `359`
- liquidation/account wipeout train/validation: `0`/`0` and `0`/`0`

## Concentration evidence

- total weighted notional fraction: `0.32501398462185427`
- top symbol: `TONUSDT` at `33.18%`
- top asset group: `crypto_core` at `100.00%`
- effective symbol count: `4.99`
- validation long/short exposure share: `0.38775553391611917` / `0.612244466083881`
- concentration flags: `['top_asset_group_share_above_70pct']`

## Required before any live/real transition

- Build or wire a dedicated live/paper adapter for this 69-asset rule set before exchange-connected paper execution.
- Run only paper/testnet with limit-first order settings and hard real-money vetoes.
- Collect 2-4 weeks of realized fill/BBO/spread/fee/slippage telemetry and compare all-in round-trip cost against the 10bps replay assumption.
- Verify intended-vs-actual notional parity, partial/cancel/timeout rates, protective-order attach/reconciliation, liquidation-distance/margin buffers, and ongoing asset/direction concentration.
- Do not set `ready_for_real=true` or `real_money_execution=true` from this artifact alone.

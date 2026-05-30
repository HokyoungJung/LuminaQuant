# 69-asset Alpha Zoo challenger paper/testnet handoff

Status: backtest-gated paper/testnet challenger only. Real-money remains blocked.

## Decision

- ready_for_paper: `True`
- ready_for_real: `False`
- real_money_execution: `False`
- real_execution_allowed: `False`
- backtest_gate: `{'strict_backtest_gate_pass': True, 'relaxed_backtest_gate_pass': True, 'ready_for_paper': True, 'ready_for_real': False, 'real_money_execution': False, 'real_execution_allowed': False, 'rejection_reasons': []}`

## Backtest evidence

- train return: `9.0960%`
- validation return: `8.5640%`
- validation MDD: `0.6934%`
- train/validation RPT proxy: `10.960323416727334` / `31.837053294540155` bps
- component trade events train/validation: `9613` / `3789`
- liquidation/account wipeout train/validation: `0`/`0` and `0`/`0`

## Concentration evidence

- total weighted notional fraction: `0.26098143962555787`
- top symbol: `TONUSDT` at `16.27%`
- top asset group: `crypto_core` at `49.91%`
- effective symbol count: `11.59`
- validation long/short exposure share: `0.5206173221349235` / `0.4793826778650765`
- concentration flags: `[]`

## Required before any live/real transition

- Build or wire a dedicated live/paper adapter for this 69-asset rule set before exchange-connected paper execution.
- Run only paper/testnet with limit-first order settings and hard real-money vetoes.
- Collect 2-4 weeks of realized fill/BBO/spread/fee/slippage telemetry and compare all-in round-trip cost against the 10bps replay assumption.
- Verify intended-vs-actual notional parity, partial/cancel/timeout rates, protective-order attach/reconciliation, liquidation-distance/margin buffers, and ongoing asset/direction concentration.
- Do not set `ready_for_real=true` or `real_money_execution=true` from this artifact alone.

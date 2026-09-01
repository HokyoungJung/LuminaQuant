# Portfolio Superiority Wave 2 Survival Filter

- Generated: 2026-04-21T14:04:39.175190+00:00
- Input candidates: 14
- Kept candidates: 1

## Rules
- train trade_count > 0
- val total_return > 0
- val sharpe > 0
- oos total_return > 0
- oos sharpe > 0
- oos max_drawdown <= 15%

## Kept candidates
- pair_spread_1h_exec_tightstop_tp_bnbusdt_trxusdt_2.2_0.55 | val_ret=0.0161 | val_sharpe=3.1334 | oos_ret=0.0818 | oos_sharpe=2.7614 | oos_mdd=0.0452

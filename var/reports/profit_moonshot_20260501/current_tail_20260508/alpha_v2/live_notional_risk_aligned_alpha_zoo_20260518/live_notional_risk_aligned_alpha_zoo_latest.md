# Live notional/risk aligned Alpha Zoo report

Generated: `2026-05-18T11:26:42.257560Z`

## Selected live/replay sizing contract

- sizing mode: `isolated_margin_fraction`
- leverage/allocation: `7.0x` / `15.00%`
- notional/equity: `105.00%`
- isolated margin/equity: `15.00%`
- locked-OOS return/MDD: `30.5357%` / `11.3027%`

## Cost sensitivity — locked-OOS

| Scenario | Return | MDD | Sharpe |
| --- | ---: | ---: | ---: |
| fee/slippage `1 bps` | `25.2882%` | `11.9349%` | `1.5474` |
| fee/slippage `3 bps` | `15.4160%` | `13.1860%` | `1.0115` |
| fee/slippage `5 bps` | `6.3199%` | `15.4731%` | `0.4755` |
| fee/slippage `10 bps` | `-13.4130%` | `24.2149%` | `-0.8643` |
| fee/slippage `20 bps` | `-42.5899%` | `44.8361%` | `-3.5439` |
| funding `1 bps/day` | `29.9911%` | `11.3645%` | `1.7886` |
| funding `2 bps/day` | `29.4486%` | `11.4263%` | `1.7617` |
| funding `5 bps/day` | `27.8349%` | `11.6112%` | `1.6812` |
| funding `10 bps/day` | `25.1897%` | `11.9187%` | `1.5466` |
| funding `20 bps/day` | `20.0619%` | `12.5305%` | `1.2760` |

## Paper-equivalent sizing parity

- parity passed: `True`
- risk check passed: `True`

## Preflight

- ready_for_paper: `True`
- ready_for_real: `False`
- recommended_action: `paper_run_allowed`

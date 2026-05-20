# Alpha Zoo 10bps paper/testnet preflight

Generated: `2026-05-20T11:24:22.597982Z`

Real-money execution is disabled: `ready_for_real=false`, `real_money_execution=false`.

| Role | Model | Leverage | Allocation | Train return | Validation return | Locked-OOS return | Paper | Real |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| active | `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_7p0x_0p2alloc` | 7 | 0.200 | 45.6916% | 0.4724% | 1.8382% | `True` | `False` |
| balanced | `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_6p0x_0p175alloc` | 6 | 0.175 | 36.0268% | 0.5942% | 1.5464% | `True` | `False` |

## Monitoring

- Realized fee/slippage/all-in round-trip bps must be compared against the locked 10bps research assumption.
- Active and balanced rows share the same timestamp/symbol/side grouping requirements.
- Isolated liquidation losses are included in equity, drawdown, and account-wipeout checks.

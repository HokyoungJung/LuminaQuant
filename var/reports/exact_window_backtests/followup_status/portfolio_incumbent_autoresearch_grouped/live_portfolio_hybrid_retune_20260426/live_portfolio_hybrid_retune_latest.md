# Live Portfolio HYBRID Retune

- generated_at: `2026-04-26T10:33:41.527202Z`
- tuning universe: committed live portfolio modes only
- selection: validation-primary; OOS report-only; cash efficiency not scored
- best dynamic retune score: `67.23`
- deployable validation-primary static mode: `retuned_live_portfolio_hybrid_mode`, score `71.85`
- best live-deployable after retune: `retuned_live_portfolio_hybrid_mode` (score `71.85`)

## Final allocation

- date: `static_validation_primary_retune`
- cash_weight: `0.00%`
- `aggressive_realized_mode`: `100.00%`

## Live-implementable ranking

| Rank | Candidate | Live? | Score | Train ret/Sh/MDD | Val ret/Sh/MDD | OOS ret/Sh/MDD | Caveat |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | `retuned_live_portfolio_hybrid_mode` | yes | 71.85 | +5.6047%/0.6323/+7.0617% | +9.6581%/4.1656/+1.4277% | +0.1973%/0.5173/+0.7291% | committed live mode using the validation-primary static allocation over live portfolio sleeves |
| 2 | `aggressive_realized_mode` | yes | 71.85 | +5.6047%/0.6323/+7.0617% | +9.6581%/4.1656/+1.4277% | +0.1973%/0.5173/+0.7291% |  |
| 3 | `core_mode` | yes | 70.36 | +4.3264%/0.5227/+7.2086% | +9.1892%/4.1125/+1.3614% | +0.0887%/0.2939/+0.5734% |  |
| 4 | `balanced_overlay_mode` | yes | 69.12 | +3.9764%/0.5293/+6.5394% | +8.3078%/4.1120/+1.2258% | +0.1091%/0.3961/+0.5162% |  |
| 5 | `defensive_overlay_mode` | yes | 66.59 | +3.2558%/0.5475/+5.1916% | +6.5582%/4.1097/+0.9543% | +0.1496%/0.6821/+0.4016% |  |
| 6 | `strict_autoresearch_practical_mode` | yes | 64.99 | +1.9247%/0.3377/+5.4233% | +6.2862%/4.0037/+0.9237% | +0.1973%/1.0123/+0.2548% |  |
| 7 | `production_guarded_state_vwap_pair_mode` | yes | 60.56 | +0.0594%/0.0363/+4.6120% | +5.3235%/3.8978/+0.9004% | +0.4617%/2.8223/+0.2194% |  |
| 8 | `legacy_no_highvol_hybrid_mode` | yes | 53.35 | -4.4080%/-0.8092/+7.5445% | +6.0034%/3.7018/+1.3517% | +0.5959%/1.4957/+0.6722% |  |
| 9 | `pair_tactical_mode` | yes | 47.74 | +0.5244%/0.4982/+0.8228% | +0.5731%/3.0741/+0.0184% | +0.2892%/2.6752/+0.0000% |  |
| 10 | `risk_off_mode` | yes | 11.80 | +0.0000%/0.0000/+0.0000% | +0.0000%/0.0000/+0.0000% | +0.0000%/0.0000/+0.0000% |  |
| 11 | `retuned_live_portfolio_hybrid_dynamic_backtest` | no | 67.23 | +10.6906%/1.5816/+3.2210% | +7.1606%/3.5354/+1.4272% | -0.1704%/-0.5858/+0.5239% | dynamic allocation path; live mode below uses the saved final allocation |

## Caveats

- The retune only uses committed live portfolio modes as sleeves.
- The dynamic allocator path is still research/backtest; the deployable mode uses the validation-primary static allocation.
- OOS is report-only and was not used for tuning or health priors.
- Paper/canary execution is still required before capital promotion.

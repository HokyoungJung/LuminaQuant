# Legacy Metric No-HighVol HYBRID Live Materialization

- generated_at: `2026-04-26T10:32:54.466165Z`
- selection remains validation-primary; OOS is report-only.
- cash efficiency is not scored.
- MDD up to 25% remains eligible and MDD enters only through bounded headroom.

## Result

- Original research #1: `legacy_metric_no_highvol_baseline_raw_score` (metric-only score `95.41`).
- Stream-backed materialized dynamic proxy: score `59.22`, val return `+5.6836%`, val Sharpe `3.0943`.
- Deployable final-allocation live mode: `legacy_no_highvol_hybrid_mode`, score `53.35`, val return `+6.0034%`, val Sharpe `3.7018`.
- Best live-deployable candidate in this comparison: `aggressive_realized_mode` (score `71.85`).
- Best research/backtest candidate shown here: `retuned_full_universe_hybrid_online` (score `85.83`).

## Why the original research #1 cannot be promoted verbatim

The original row has no saved daily stream and was marked `combinable=false`, so there was no strategy artifact for live execution.  This materialization rebuilds it over saved sleeves, then compares the dynamic research path against the actually deployable artifact mode.

## Backtest/live parity check

- dynamic_vs_final_allocation val_return_gap: `-0.3198%`
- dynamic_vs_final_allocation train_return_gap: `+14.0937%`
- dynamic_vs_final_allocation oos_return_gap: `-0.9527%`
A non-zero gap is expected because the live `ArtifactPortfolioModeStrategy` consumes the latest saved final allocation, while the research dynamic allocator changes allocations through history. The report therefore does not label the dynamic path as live-deployable.

## Final allocation for `legacy_no_highvol_hybrid_mode`

- date: `2026-04-20`
- cash_weight: `+36.6197%`
- `state_vwap_pair`: `+38.5032%`
- `wave2_pair`: `+24.8771%`

## Live-implementable comparison

| Rank | Candidate | Live? | Score | Train ret/Sh/MDD | Val ret/Sh/MDD | OOS ret/Sh/MDD | Caveat |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | `aggressive_realized_mode` | yes | 71.85 | +5.6047%/0.6323/+7.0617% | +9.6581%/4.1656/+1.4277% | +0.1973%/0.5173/+0.7291% |  |
| 2 | `core_mode` | yes | 70.36 | +4.3264%/0.5227/+7.2086% | +9.1892%/4.1125/+1.3614% | +0.0887%/0.2939/+0.5734% |  |
| 3 | `balanced_overlay_mode` | yes | 69.12 | +3.9764%/0.5293/+6.5394% | +8.3078%/4.1120/+1.2258% | +0.1091%/0.3961/+0.5162% |  |
| 4 | `defensive_overlay_mode` | yes | 66.59 | +3.2558%/0.5475/+5.1916% | +6.5582%/4.1097/+0.9543% | +0.1496%/0.6821/+0.4016% |  |
| 5 | `strict_autoresearch_practical_mode` | yes | 64.99 | +1.9247%/0.3377/+5.4233% | +6.2862%/4.0037/+0.9237% | +0.1973%/1.0123/+0.2548% |  |
| 6 | `production_guarded_state_vwap_pair_mode` | yes | 60.56 | +0.0594%/0.0363/+4.6120% | +5.3235%/3.8978/+0.9004% | +0.4617%/2.8223/+0.2194% |  |
| 7 | `legacy_no_highvol_hybrid_mode` | yes | 53.35 | -4.4080%/-0.8092/+7.5445% | +6.0034%/3.7018/+1.3517% | +0.5959%/1.4957/+0.6722% | deployable through ArtifactPortfolioModeStrategy using the materialized final allocation; historical dynamic allocator score is reported separately |
| 8 | `pair_tactical_mode` | yes | 47.74 | +0.5244%/0.4982/+0.8228% | +0.5731%/3.0741/+0.0184% | +0.2892%/2.6752/+0.0000% |  |
| 9 | `risk_off_mode` | yes | 11.80 | +0.0000%/0.0000/+0.0000% | +0.0000%/0.0000/+0.0000% | +0.0000%/0.0000/+0.0000% |  |
| 10 | `hybrid_guarded_mode` | yes | 11.80 | +0.0000%/0.0000/+0.0000% | +0.0000%/0.0000/+0.0000% | +0.0000%/0.0000/+0.0000% | current live HYBRID mode uses the latest saved final allocation, not the dynamic allocator path |
| 11 | `retuned_full_universe_hybrid_online` | no | 85.83 | +63.3325%/1.8584/+10.7906% | +23.8229%/3.7181/+4.1257% | +0.0667%/0.1785/+0.7287% | stream-backed in research report, but no committed live portfolio mode artifact yet |
| 12 | `full_universe_v35_train_learned_high_vol_hybrid` | no | 73.77 | +16.7674%/1.0888/+14.7687% | +16.4975%/3.6331/+3.6767% | +0.3731%/1.0072/+0.7090% | stream-backed in research report, but no committed live portfolio mode artifact yet |
| 13 | `full_universe_v36_rolling_dynamic_high_vol_hybrid` | no | 70.43 | +9.5171%/0.6521/+16.9432% | +16.4587%/3.5836/+3.8950% | +0.3197%/0.8910/+0.7126% | stream-backed in research report, but no committed live portfolio mode artifact yet |
| 14 | `legacy_metric_no_highvol_materialized_dynamic_stream_backtest` | no | 59.22 | +9.6856%/1.2873/+4.2449% | +5.6836%/3.0943/+1.1771% | -0.3567%/-3.5249/+0.3670% | stream-backed reconstruction of the metric-only winner; live strategy mode uses its final allocation unless/until a stateful live HYBRID allocator is implemented |

## Explicit caveats

- This is not investment advice and is not a live-trading authorization.
- The original research #1 was metric-only: no daily stream, non-combinable, and not directly executable.
- The restored stream-backed dynamic proxy materially differs from the original saved metrics, so the old score is not portable verbatim.
- The current live portfolio mode architecture executes saved final allocations; it does not yet run the HYBRID allocator statefully inside live trading.
- A stateful live HYBRID allocator would need portfolio/fill/PnL state integration before it can claim exact parity with the dynamic backtest path.
- Exchange fees, funding, partial fills, latency, order-size limits, and slippage still require paper/canary validation.

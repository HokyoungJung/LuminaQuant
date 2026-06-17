# Best historical strategy improvement

- historical best selected model: `codex_lagged_leaf_router_grid:h4_avg1_tr-0.02_tmdd0.50_val0.00_vmdd0.25_lagged_plus_val025_exact_unscaled`
- added risk-trimmed variant: `codex_lagged_leaf_router_grid:h4_avg1_tr-0.02_tmdd0.50_val0.00_vmdd0.25_lagged_plus_val025_fallback_mdd20_cap2`
- scope: strict-core warmup fallback only; main lagged leaf selection remains unchanged.
- status: shadow/paper research only; `ready_for_real=false` remains correct.

## Results

| Source | Universe | Candidate | Comp OOS | Ann approx | Max OOS MDD | Monthly MDD | Sharpe | PF | Return/MDD | Hit |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `historical_85_preregistered` | 85 | `baseline` | 197.37% | 269.80% | 27.69% | 4.50% | 2.12 | 30.04 | 7.13 | 5/10 |
| `historical_85_preregistered` | 85 | `fallback_mdd20_cap2` | 172.51% | 233.01% | 20.26% | 4.50% | 2.02 | 27.56 | 8.52 | 5/10 |
| `expanded_110_latest_tail_full` | 110 | `baseline` | 79.42% | 101.67% | 27.69% | 5.09% | 1.96 | 13.78 | 2.87 | 4/10 |
| `expanded_110_latest_tail_full` | 110 | `fallback_mdd20_cap2` | 64.42% | 81.61% | 18.46% | 5.09% | 2.11 | 11.60 | 3.49 | 4/10 |

## Context

- Frozen `clean_input_meta_selector` on the expanded 110-asset latest-tail artifact only reaches `9.75%` comp / Sharpe `0.50`, so the actual historical high-water model to improve is the lagged leaf router, not TopCap or the clean-input meta selector.
- The new variant does not maximize headline return. It improves risk efficiency by capping the strict-core warmup fallback from validation-MDD30/cap3 to validation-MDD20/cap2 while preserving the same lagged leaf choices after warmup.
- On the historical 85-symbol replay, comp drops from 197.37% to 172.51% but max OOS MDD drops from 27.69% to 20.26%, improving return/max-MDD from 7.13 to 8.52.
- On the expanded 110-symbol replay, comp drops from 79.42% to 64.42% but max OOS MDD drops from 27.69% to 18.46%, improving return/max-MDD from 2.87 to 3.49.

## Fold returns

### historical_85_preregistered

| Fold | Baseline return | Trimmed return | Baseline MDD | Trimmed MDD |
| --- | ---: | ---: | ---: | ---: |
| `2025-09` | 0.00% | 0.00% | 0.00% | 0.00% |
| `2025-10` | 0.00% | 0.00% | 0.00% | 0.00% |
| `2025-11` | 33.48% | 22.32% | 27.69% | 18.46% |
| `2025-12` | 0.00% | 0.00% | 0.00% | 0.00% |
| `2026-01` | 16.03% | 16.03% | 9.34% | 9.34% |
| `2026-02` | 8.44% | 8.44% | 11.81% | 11.81% |
| `2026-03` | -0.05% | -0.05% | 3.93% | 3.93% |
| `2026-04` | -4.46% | -4.46% | 10.06% | 10.06% |
| `2026-05` | 12.51% | 12.51% | 20.26% | 20.26% |
| `2026-06` | 64.80% | 64.80% | 6.69% | 6.69% |

### expanded_110_latest_tail_full

| Fold | Baseline return | Trimmed return | Baseline MDD | Trimmed MDD |
| --- | ---: | ---: | ---: | ---: |
| `2025-09` | 0.00% | 0.00% | 0.00% | 0.00% |
| `2025-10` | 0.00% | 0.00% | 0.00% | 0.00% |
| `2025-11` | 33.48% | 22.32% | 27.69% | 18.46% |
| `2025-12` | 0.00% | 0.00% | 0.00% | 0.00% |
| `2026-01` | 16.03% | 16.03% | 9.34% | 9.34% |
| `2026-02` | 8.44% | 8.44% | 11.81% | 11.81% |
| `2026-03` | -0.05% | -0.05% | 3.93% | 3.93% |
| `2026-04` | -4.46% | -4.46% | 10.06% | 10.06% |
| `2026-05` | -0.62% | -0.62% | 14.37% | 14.37% |
| `2026-06` | 12.56% | 12.56% | 11.36% | 11.36% |


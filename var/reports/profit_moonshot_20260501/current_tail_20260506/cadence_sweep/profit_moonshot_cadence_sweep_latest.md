# Profit moonshot leverage/cadence sweep

- generated_at: `2026-05-06T13:17:10.755652Z`
- mode_count: `45`
- candidate_count: `174`
- validation_screened: `174`
- full_survivors_tested: `1`
- max_rss_mib: `4693.50`
- cost model: event-driven `SimulatedExecutionHandler` with taker_fee=`0.0004`, slippage=`0.0005`, spread=`0.0002`.
- exposure policy: cadence-only overrides; gross/target allocation and max order caps are not increased.
- bankruptcy/account-breach policy: report-facing return/MDD are capped at `-100%`/`100%` when equity reaches non-positive territory; raw arithmetic is preserved in JSON/CSV diagnostic fields. `*` marks such capped rows.

## Full survivor results

| candidate | gate | train ret | train MDD | val ret | OOS ret | OOS Sharpe | OOS MDD | breach | reasons |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `profit_moonshot_adaptive_momentum_boost_mode__cadence_1b` | `False` | -100.0000%* | +100.0000%* | +12.5743% | -0.9619% | 0.012010 | +33.0023% | `train` | `train_total_return_not_positive;oos_return_not_above_0.8284pct_incumbent;oos_mdd_not_below_funding_guard_shadow;oos_sharpe_not_above_1.0_success_target;train_equity_breach_observed;train_liquidation_observed` |

## Top validation cadence screen

| candidate | val ret | val Sharpe | val MDD | trades | native cadences |
| --- | ---: | ---: | ---: | ---: | --- |
| `profit_moonshot_adaptive_momentum_boost_mode__cadence_1b` | +0.0913% | 0.107492 | 0.2825% | 16 | `[72]` |
| `profit_moonshot_adaptive_momentum_140_mode__cadence_1b` | +0.0855% | 0.107591 | 0.2645% | 16 | `[72]` |
| `profit_moonshot_adaptive_momentum_governed_mode__cadence_1b` | +0.0855% | 0.107591 | 0.2645% | 16 | `[96]` |
| `profit_moonshot_adaptive_momentum_130_mode__cadence_1b` | +0.0798% | 0.107725 | 0.2464% | 16 | `[72]` |
| `profit_moonshot_momentum_hybrid_return_mode__cadence_1b` | +0.0783% | 0.106119 | 0.2453% | 46 | `[72, 96]` |
| `profit_moonshot_adaptive_momentum_120_mode__cadence_1b` | +0.0727% | 0.107028 | 0.2258% | 16 | `[72]` |
| `profit_moonshot_momentum_hybrid_core_mode__cadence_1b` | +0.0708% | 0.106890 | 0.2203% | 47 | `[72, 96]` |
| `profit_moonshot_momentum_hybrid_safe_mode__cadence_1b` | +0.0690% | 0.107626 | 0.2130% | 48 | `[72, 96]` |
| `profit_moonshot_adaptive_momentum_vol_target_132_mode__cadence_1b` | +0.0617% | 0.111182 | 0.1852% | 16 | `[72]` |
| `profit_moonshot_adaptive_momentum_mode__cadence_1b` | +0.0598% | 0.106665 | 0.1859% | 16 | `[72]` |
| `profit_reboot_adaptive_momentum_mode__cadence_1b` | +0.0598% | 0.106665 | 0.1859% | 16 | `[72]` |
| `profit_moonshot_adaptive_momentum_vol_target_mode__cadence_1b` | +0.0591% | 0.111154 | 0.1774% | 16 | `[72]` |
| `profit_moonshot_derivatives_taker_flow_mode__native` | +0.0379% | 0.839457 | 0.0079% | 9 | `[15, 180]` |
| `profit_moonshot_derivatives_taker_flow_mode__cadence_180b` | +0.0379% | 0.839457 | 0.0079% | 9 | `[15, 180]` |
| `derivatives_flow_squeeze_mode__cadence_180b` | +0.0346% | 0.398163 | 0.0249% | 14 | `[15, 45, 180]` |
| `profit_moonshot_adaptive_momentum_boost_mode__cadence_15b` | +0.0297% | 0.133969 | 0.0750% | 4 | `[72]` |
| `profit_moonshot_adaptive_momentum_140_mode__cadence_15b` | +0.0278% | 0.134077 | 0.0703% | 4 | `[72]` |
| `profit_moonshot_adaptive_momentum_governed_mode__cadence_15b` | +0.0278% | 0.134077 | 0.0703% | 4 | `[96]` |
| `profit_moonshot_adaptive_momentum_130_mode__cadence_15b` | +0.0260% | 0.134202 | 0.0656% | 4 | `[72]` |
| `profit_moonshot_momentum_hybrid_return_mode__cadence_15b` | +0.0241% | 0.131294 | 0.0620% | 10 | `[72, 96]` |

## Bottleneck notes

- total_wall_seconds: `3.01`
- checkpointed_run_wall_seconds: `1943.70`
- data_cache: `{'entries': 0, 'hits': 0, 'misses': 0, 'loads': 0, 'load_seconds': 0.0, 'freeze_seconds': 0.0, 'max_entries': 5}`
- native_backends: `{'raw_first': {'requested_backend': 'auto', 'resolved_backend': 'rust', 'description': 'rust:/home/hoky/Quants-agent/LuminaQuant/native/rust_rawfirst/target/release/liblumina_rawfirst.so', 'native_library_path': '/home/hoky/Quants-agent/LuminaQuant/native/rust_rawfirst/target/release/liblumina_rawfirst.so', 'native_load_error': None, 'auto_fallback_warning_count': 0, 'auto_fallback_warning_reasons': []}}`
- slowest_runs: `[{'candidate_id': 'profit_moonshot_adaptive_momentum_boost_mode__cadence_1b', 'split': 'train', 'wall_seconds': 1352.3215640349954, 'load_seconds': 879.2898622159992, 'engine_seconds': 473.0317018189962}, {'candidate_id': 'profit_moonshot_adaptive_momentum_boost_mode__cadence_1b', 'split': 'oos', 'wall_seconds': 221.13955763299964, 'load_seconds': 145.12601053800063, 'engine_seconds': 76.01354709499901}, {'candidate_id': 'profit_moonshot_adaptive_momentum_boost_mode__cadence_1b', 'split': 'val', 'wall_seconds': 198.92284824000126, 'load_seconds': 130.88347068800067, 'engine_seconds': 68.03937755200059}, {'candidate_id': 'derivatives_flow_squeeze_mode__cadence_1b', 'split': 'val_screen', 'wall_seconds': 6.815394327999456, 'load_seconds': 2.6726000214694068e-05, 'engine_seconds': 6.815367601999242}, {'candidate_id': 'profit_moonshot_balanced_mode__cadence_1b', 'split': 'val_screen', 'wall_seconds': 6.548423193999952, 'load_seconds': 2.335699991817819e-05, 'engine_seconds': 6.548399837000034}, {'candidate_id': 'profit_moonshot_ensemble_mode__cadence_1b', 'split': 'val_screen', 'wall_seconds': 6.387139057999775, 'load_seconds': 2.389900055277394e-05, 'engine_seconds': 6.387115158999222}, {'candidate_id': 'derivatives_flow_squeeze_mode__native', 'split': 'val_screen', 'wall_seconds': 5.225522873000045, 'load_seconds': 3.0358832239999174, 'engine_seconds': 2.189639649000128}, {'candidate_id': 'profit_moonshot_reversion_mode__cadence_1b', 'split': 'val_screen', 'wall_seconds': 4.887023586999931, 'load_seconds': 2.5058000574063044e-05, 'engine_seconds': 4.8869985289993565}]`
- exactness note: optimized runs reuse frozen raw-first OHLCV rows but still execute the same live-equivalent strategy, portfolio, fill, fee, spread, slippage, and partial-fill engine path.

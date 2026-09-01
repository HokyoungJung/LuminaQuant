# Hybrid Online Portfolio

- generated_at: `2026-05-10T06:14:43.454729Z`
- memory_log: `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/live_final_selection_20260510/hybrid_final/_memory_guard/hybrid_online_portfolio_rss_latest.jsonl`
- peak_rss_mib: `71.56`
- current_market_state: favored_group=`mixed`, trend=`bullish`, breadth=`broad`, volatility=`calm`, pair_liquidity=`strong`

## Deterministic config

```json
{
  "default_boost": 0.2335751227788591,
  "disagreement_cash_scale": 0.856739550695035,
  "disagreement_threshold": 0.22293874953018045,
  "diversified_weight_cap": 0.8757254013166067,
  "lookback_days": 13,
  "min_positive_score": 0.14655939990705838,
  "mixed_health_floor": 0.5897144969917639,
  "negative_health_floor": 0.09432330421001382,
  "pair_pbo_penalty_scale": 1.8475128599670687,
  "pair_score_boost": 0.00015054198515658535,
  "pair_sparsity_penalty_scale": 2.732355465712022,
  "pair_weight_cap": 0.17534288208655102,
  "score_temperature": 0.9070667959024572,
  "sticky_default_bonus": 0.09538352712472187,
  "switch_margin": 0.09865009243733545,
  "use_current_health_priors": true,
  "variant": "fixed_default",
  "warmup_days": null,
  "warmup_ratio": 0.6
}
```

## Split windows

```json
{
  "oos_end_inclusive": "2026-05-09",
  "oos_start": "2026-03-01",
  "pre_oos_days": 424,
  "train_end_inclusive": "2025-12-31",
  "train_start": "2025-01-01",
  "val_end_inclusive": "2026-02-28",
  "val_start": "2026-01-01"
}
```

- warmup_days: `255`
- lookback_days: `13`
- online_start: `2025-09-13`

## Readiness

- beats_cash_refreshed: `True`
- beats_pair_tactical_refreshed: `False`
- beats_balanced_refreshed: `True`
- beats_production_guarded_refreshed: `False`
- max_rss_under_8gib: `True`
- pair_cap_respected: `True`
- recommended_stage: `guarded_candidate`

- refreshed/latest-tail primary approval: hybrid beats cash? `True`
- pair tactical comparator is benchmark-only? `True`

## Historical saved baseline scoreboard

| Sleeve | Kind | OOS return | Sharpe | Max DD |
| --- | --- | ---: | ---: | ---: |
| `hybrid_online_portfolio` | hybrid | +0.7163% | 2.2582 | 0.5672% |
| `risk_off_cash` | active | +0.0000% | 0.0000 | 0.0000% |
| `soft_three_way_regime` | active | +0.2088% | 1.5273 | 0.6156% |
| `pair_tactical_mode` | active | +3.2580% | 5.8913 | 0.0777% |
| `balanced_overlay_80_20` | active | +0.8132% | 2.6442 | 0.4926% |
| `three_way_regime` | benchmark | +1.8307% | 6.9723 | 0.3896% |
| `static_blend_76_24` | benchmark | +0.2813% | 2.0856 | 0.5544% |
| `incumbent_only` | benchmark | -0.3154% | -2.5718 | 0.7291% |

## Refreshed latest-tail scoreboard

| Sleeve | Kind | OOS return | Sharpe | Max DD |
| --- | --- | ---: | ---: | ---: |
| `hybrid_online_portfolio` | hybrid | +0.1618% | 0.7763 | 0.4897% |
| `risk_off_cash` | active | +0.0000% | 0.0000 | 0.0000% |
| `soft_three_way_regime` | active | +0.0887% | 0.7423 | 0.5734% |
| `balanced_overlay_80_20` | active | +0.1091% | 0.4828 | 0.5162% |
| `pair_tactical_mode` | active | +0.2892% | 3.2765 | 0.0000% |
| `production_guarded_portfolio` | active | +0.3654% | 1.6080 | 0.3184% |
| `three_way_regime` | benchmark | +0.1973% | 1.3087 | 0.7291% |
| `static_blend_76_24` | benchmark | +0.4307% | 3.0424 | 0.5107% |
| `incumbent_only` | benchmark | -0.3154% | -2.5718 | 0.7291% |

## Refreshed latest-tail final allocation

- date: `2026-04-03`
- cash_weight: `100.00%`

## Refreshed latest-tail hybrid metrics
- oos_return: `+0.1618%`
- oos_sharpe: `0.7763`
- oos_max_dd: `0.4897%`

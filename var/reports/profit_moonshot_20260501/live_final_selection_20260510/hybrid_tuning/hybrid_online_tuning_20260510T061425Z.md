# Hybrid online portfolio tuning

- generated_at: `2026-05-10T06:14:25.562239Z`
- evaluated_configs: `10`
- objective_profile: `locked_train_val`
- objective_policy: `train_val_only_locked_oos_report`
- locked_oos_label: `locked_oos_report_only`

## Best config

```json
{
  "config": {
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
  },
  "config_name": "baseline",
  "objective": 65.68096814777863,
  "objective_policy": {
    "locked_oos_label": "locked_oos_report_only",
    "objective_policy": "train_val_only_locked_oos_report",
    "objective_profile": "locked_train_val",
    "oos_is_objective_input": false,
    "selection_label": "train_val_validation_only"
  },
  "objective_profile": "locked_train_val",
  "readiness": {
    "beats_cash_refreshed": true,
    "beats_pair_tactical_refreshed": false,
    "pair_cap_respected": true
  },
  "scenarios": {
    "historical_saved_baseline": {
      "split_metrics": {
        "oos": {
          "cagr": 0.07963160763960753,
          "calmar": 14.040364456997649,
          "max_drawdown": 0.005671619699296304,
          "sharpe": 2.2581613861496113,
          "sortino": 3.079683386862322,
          "total_return": 0.007162724672818532,
          "volatility": 0.034183861408638855
        },
        "train": {
          "cagr": 0.05886934928817711,
          "calmar": 1.1236650010605196,
          "max_drawdown": 0.052390480465811406,
          "sharpe": 0.8269557285328427,
          "sortino": 1.3493768683039162,
          "total_return": 0.05886934928817711,
          "volatility": 0.07231036939884057
        },
        "val": {
          "cagr": 0.5390521349295081,
          "calmar": 42.13783484169822,
          "max_drawdown": 0.012792592143250792,
          "sharpe": 3.491136959531074,
          "sortino": 9.482069170812208,
          "total_return": 0.07218159223943044,
          "volatility": 0.12575889937426746
        }
      }
    },
    "refreshed_latest_tail": {
      "split_metrics": {
        "oos": {
          "cagr": 0.017506928297035262,
          "calmar": 3.575246760792684,
          "max_drawdown": 0.004896704890141268,
          "sharpe": 0.7763172405928882,
          "sortino": 0.8998624362864446,
          "total_return": 0.001617979335523323,
          "volatility": 0.022677674055641042
        },
        "train": {
          "cagr": 0.06964170970228722,
          "calmar": 1.394125428183764,
          "max_drawdown": 0.049953690173354715,
          "sharpe": 0.8957917111788695,
          "sortino": 1.5082663105733576,
          "total_return": 0.06964170970228722,
          "volatility": 0.0785758083482393
        },
        "val": {
          "cagr": 0.49167518325202186,
          "calmar": 38.66709022525689,
          "max_drawdown": 0.012715598209944057,
          "sharpe": 3.305313347521935,
          "sortino": 8.940229070146314,
          "total_return": 0.06677633838335151,
          "volatility": 0.12326980391207122
        }
      }
    }
  }
}
```

## Leaderboard

| Config | Objective | Refresh OOS | Refresh Sharpe | Refresh Val | Refresh Train | Beats cash | Beats pair |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `baseline` | 65.6810 | +0.1618% | 0.7763 | +6.6776% | +6.9642% | `True` | `False` |
| `responsive_pair` | 64.8198 | +0.1783% | 0.8446 | +7.0155% | +5.1854% | `True` | `False` |
| `long_memory` | 60.6760 | +0.0041% | 2.8412 | +4.4213% | +7.3505% | `True` | `False` |
| `soft_health` | 58.6653 | +0.0652% | 0.3510 | +4.8267% | +5.9870% | `True` | `False` |
| `pair_penalty_relaxed` | 58.1258 | +0.0828% | 0.4259 | +4.6258% | +6.0442% | `True` | `False` |
| `pair_penalty_strict` | 57.9076 | +0.0828% | 0.4259 | +4.5847% | +6.0289% | `True` | `False` |
| `strict_health` | 57.9057 | +0.0828% | 0.4259 | +4.5845% | +6.0289% | `True` | `False` |
| `balanced_sticky` | 57.5380 | +0.0919% | 0.4598 | +4.4312% | +6.0442% | `True` | `False` |
| `high_conviction_switch` | 57.1399 | +0.0900% | 0.4479 | +4.2814% | +5.9866% | `True` | `False` |
| `conservative_cash_bias` | -934.2265 | -0.1366% | -3.1365 | +5.3911% | +7.0418% | `False` | `False` |

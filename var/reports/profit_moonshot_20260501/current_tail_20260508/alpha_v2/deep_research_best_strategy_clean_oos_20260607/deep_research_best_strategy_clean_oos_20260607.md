# Deep Research Best Strategy — clean OOS walk-forward conclusion (2026-06-07 KST)

## Executive decision

**결론: 지금 real-money 실전 투입 승인 후보는 없다.** 수익을 최대화해야 한다는 목표를 반영해도, hard gates(`no_nested_oos_mining`, `execution_cost_gate`, `theory_plausibility_gate`)를 동시에 적용하면 현재 최선은 다음 2-track 운영이다.

1. **최고 수익 freeze/shadow 후보:** `clean_input_meta_selector` — OOS comp **85.91%**, ann approx **110.46%**, max OOS MDD **19.29%**, hit **5/10**. 하지만 selector grid ranking이 historical locked-OOS를 사용했으므로 label은 **`shadow-freeze-only`**로 고정한다. Fresh-forward shadow 전에는 paper/real 승격 금지.
2. **현재 clean-control/paper 후보:** `strict_no_leak_best_single_10bps` — 10bps에서 total return **54.56%**, MDD **30.63%**, Sharpe **1.26**, PF **1.21**, hit **6/10**. 20bps stress도 **27.10%**로 양수지만 MDD가 **43.63%**까지 커져 real-sleeve는 차단한다.
3. **85-symbol clean baseline:** `dynamic_conviction_switch:t0.85_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd30_scaled` — OOS comp **34.39%**, ann approx **42.57%**, max bar MDD **27.69%**, hit **3/10**. 기계적으로 clean하지만 sparse positive folds와 27%대 MDD 때문에 paper baseline/monitor로만 사용한다.

## Why this is the best practical strategy now

- Backtest-overfitting/PBO와 DSR 문헌은 많은 후보/파라미터를 본 뒤 최고 Sharpe/return을 고르는 경우 성과 인플레이션을 명시적으로 보정해야 함을 경고한다: [PBO/CSCV](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253), [Deflated Sharpe Ratio](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551), [factor multiple testing](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2249314).
- 이론적으로 허용 가능한 core family는 own-past-return trend/time-series momentum, volatility sizing/risk timing, cost-aware execution이다. Time-series momentum은 asset 자기 과거수익 기반 trend effect로 설명 가능하고, volatility management는 alpha/Sharpe 개선 가능성이 있지만 real-time OOS에서는 더 보수적으로 해석해야 한다: [AQR time-series momentum](https://www.aqr.com/insights/research/journal-article/time-series-momentum), [Moreira-Muir NBER](https://www.nber.org/papers/w22208), [real-time volatility-managed caveat](https://www.sciencedirect.com/science/article/abs/pii/S0304405X2030132X).
- Execution cost gate는 fixed 10bps proxy만으로 끝나지 않는다. 실제 비용은 거래 타입/시장/규모/시간에 따라 달라지고, impact-risk tradeoff를 최적화해야 한다: [Frazzini-Israel-Moskowitz trading costs](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3229719), [Almgren-Chriss optimal execution](https://docslib.org/doc/1384720/optimal-execution-of-portfolio-transactions).
- Optuna는 train/validation 내부 objective에만 사용하고, `n_trials`/timeout/seed/frozen manifest를 기록한다. Chronological split은 미래 데이터가 train으로 들어가지 않는 expanding/rolling 형태여야 한다: [Optuna Study.optimize](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html), [scikit-learn TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).

## Candidate gate table

| Track | Evidence class | OOS / stress metrics | Clean/OOS gate | Execution gate | Decision |
| --- | --- | --- | --- | --- | --- |
| `clean_input_meta_selector` | `shadow-freeze-only` | comp 85.91%, ann 110.46%, max OOS MDD 19.29%, monthly MDD 6.32%, Sharpe 1.28, PF 6.26, hit 5/10, min -6.10%, latest 64.80% | Fold choice does not use locked OOS, but selector-grid ranking does. Blockers: `post_oos_selector_grid_ranking_uses_historical_locked_oos, fresh_forward_required_before_promotion` | 10bps base only; no live fill telemetry; no 15bps artifact | **Freeze + shadow only** |
| `lagged_shadow_leaf_router cap150` | shadow-freeze-only | comp 61.40%, ann 77.62%, max bar MDD 29.13%, Sharpe 1.61, PF 8.55, hit 4/10, latest -3.34% | no current-fold OOS selection, non-nested; but post-OOS family, fresh-forward required | 10bps proxy only | **Shadow only** |
| `strict_no_leak_best_single` | deployable-paper / no-real-sleeve | 10bps return 54.56%, MDD 30.63%, Sharpe 1.26, Sortino 1.79; 20bps return 27.10%, MDD 43.63% | clean train+validation then OOS report-only; only 10 eligible symbols | Stress MDD too high; 15bps/paper fills missing | **Paper only** |
| `dynamic_conviction_switch:t0.85_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd30_scaled` | paper-baseline / monitor | comp 34.39%, ann 42.57%, max bar MDD 27.69%, Sharpe 1.12, hit 3/10, train mean 12.68%, val mean 10.34% | clean, non-nested, OOS-free selection | 10bps proxy only | **Paper baseline** |
| `clean_new_alpha_discovery_full` | reject / diagnostic-only | comp 2.51%, ann 3.01%, max OOS MDD 8.77%, monthly MDD 10.28%, Sharpe 0.24, PF 1.20, hit 5/10 | 5-family pre-registered diagnostic, but continuous full-period signal slicing can carry position state across split boundaries; `clean_promotion_eligible=false` | not relevant | **Reject same-window promotion; use only as future fresh-forward hypothesis source** |

## Strategy implementation blueprint

### Frozen shadow strategy

- Freeze artifact: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_clean_meta_selector_research_20260607/clean_meta_selector_freeze_manifest_latest.json`
- Freeze SHA256: `bd26dcd5116337647d9c6f1ce20ed4710a387184f0f64d0cffce02cb6c21c43a`
- Source selector report: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_clean_meta_selector_research_20260607/clean_meta_selector_research_latest.json`
- Policy: monthly day-1 UTC refit/replay, expanding train + previous 2M validation, next month locked OOS report-only.
- During shadow, do **not** change selector formula, grid, family set, tie-breakers, or thresholds based on subsequent historical OOS reruns.

### Paper-control strategy

- Run `strict_no_leak_best_single` as paper/control with 10bps and 20bps diagnostics.
- Reject real allocation while any of these holds: max drawdown >30%, 20bps stress MDD >30%, fold concentration remains crypto-only/10-symbol, or fill telemetry is absent.
- Paper telemetry must record realized spread, slippage, fee, funding, partial fill, reject/reconcile gaps, and per-symbol turnover.

### TradFi expansion

- Keep the 85-symbol monitor universe. Current 85-symbol artifact loaded **85/85** symbols with latest UTC `2026-06-06T08:30:00`.
- TradFi/commodity/stock perps stay monitor/backfill until each asset has sufficient train + 2M validation history. Do not force validation-only or latest-only inclusion.
- Add new assets only through the same fold-local feature-support gate; never by selecting symbols because they helped a locked OOS month.

## Promotion gates

1. **No nested / no OOS mining:** zero `uses_locked_oos_for_selection`, zero nested hybrid material, zero self-feed, zero lag violations. Any selector designed/ranked after historical OOS review is capped at `shadow-freeze-only` until fresh-forward evidence arrives.
2. **Execution-cost gate:** 10bps base must stay positive; 15bps stress must remain positive before any small real sleeve; 20bps diagnostic must not reveal drawdown/tail collapse. Paper fills: mean all-in round-trip ≤10bps, p95 ≤15bps, no unexplained reconciliation gaps.
3. **Theory plausibility gate:** allowed families are trend/own-past-return, volatility/risk management, cost/liquidity-aware allocation, and pre-registered OHLCV/funding/OI/BBO features. Hardcoded month/symbol/rule hacks are rejected.
4. **Clean fresh-forward:** minimum 1–2 new monthly folds for shadow sanity; prefer 4 monthly folds before real-sleeve discussion. Freeze hash must remain unchanged.
5. **Risk cap:** max bar MDD >30% blocks real sleeve; 20–30% demotes to paper/small-shadow; <20% preferred. Latest partial-month spikes are not promotion evidence.

## Final ranking

1. **Highest return to freeze:** `clean_input_meta_selector` — best headline return, but explicitly **shadow-freeze-only**.
2. **Best clean paper-control:** `strict_no_leak_best_single_10bps` — clean/theory-plausible and still positive under 20bps, but drawdown/cost/concentration block real capital.
3. **Best broad-universe clean baseline:** `dynamic_conviction_switch:t0.85_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd30_scaled` — clean mechanics and 85-symbol monitor integration, but sparse hit-rate and high MDD keep it paper-only.
4. **Rejected as current alpha:** 5-family new-alpha diagnostic — return is too low and the artifact is explicitly blocked by `continuous_position_state_across_split_boundaries`; do not promote or tune these same families further on the same OOS.

## Artifacts

```json
{
  "augmented_85": {
    "path": "var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_85_asset_non_nested_augmented_selectors_latest_20260606/alpha_zoo_85_asset_non_nested_augmented_selectors_latest_20260606.json",
    "sha256": "cd7f4ced043cf4067685d6b324c2e9ba1e20c484591eb799fef8399a04acab6d"
  },
  "clean_meta": {
    "path": "var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_clean_meta_selector_research_20260607/clean_meta_selector_research_latest.json",
    "sha256": "0371c1d0578fa1481e41a8f05a06aa948c21a646e286ea23f45d8b28626a2a18"
  },
  "clean_meta_freeze_manifest": {
    "path": "var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_clean_meta_selector_research_20260607/clean_meta_selector_freeze_manifest_latest.json",
    "sha256": "bd26dcd5116337647d9c6f1ce20ed4710a387184f0f64d0cffce02cb6c21c43a"
  },
  "strict_no_leak": {
    "path": "var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/deep_research_strict_no_leak_20260606/strict_no_leak_selector_latest.json",
    "sha256": "03cbea6ea0ff5a20fef5a8b57556483996411437bd35e9f0f1423ff11060b74b"
  },
  "new_alpha_full": {
    "path": "var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_clean_new_alpha_discovery_20260607/clean_new_alpha_discovery_latest.json",
    "sha256": "5a8e33d5d6628d5fd4b66ba388ab1fdf512749dd6b70d5ed540b7fcabaf094ab"
  },
  "new_alpha_smoke": {
    "path": "var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_clean_new_alpha_discovery_20260607_smoke/clean_new_alpha_discovery_latest.json",
    "sha256": "c44a38af8618954cffa2115d8d563cd41599ec7c241cba96d1f14c4a3816dc67"
  },
  "new_alpha_feature_bounded": {
    "path": "var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_clean_new_alpha_discovery_20260607_feature_bounded/clean_new_alpha_discovery_latest.json",
    "sha256": "23a7970cb0456bf14a661e2affd5993d8042065ff811849a7c5e865915400283"
  }
}
```

Machine-readable summary: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/deep_research_best_strategy_clean_oos_20260607/deep_research_best_strategy_clean_oos_20260607.json`

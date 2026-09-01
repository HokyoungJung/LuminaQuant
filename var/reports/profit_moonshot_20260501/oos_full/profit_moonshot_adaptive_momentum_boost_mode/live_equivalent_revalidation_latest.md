# Live-equivalent candidate revalidation — 2026-04-26

Generated: `2026-05-03T06:15:27.613756Z`

## 기준 변경

- 기존 full-universe/legacy/HYBRID 리포트의 일별 return stream 점수는 이제 **연구 참고값**이다.
- 실투자 후보는 동일한 `ArtifactPortfolioModeStrategy`를 live와 backtest가 같이 사용하고, `SimulatedExecutionHandler`/`Portfolio`를 통과한 이벤트 기반 결과만 selection eligible이다.
- OOS는 계속 report-only이며, selection/tuning/health prior에는 쓰지 않는다.
- cash efficiency는 점수에 넣지 않는다. 또한 0-trade/무수익/현금성 후보는 alpha ranking에서 제외한다.

## 결론

- Best full-universe live-equivalent candidate: `profit_moonshot_adaptive_momentum_boost_mode`
- Best deployable true-HYBRID candidate: `NONE` — dynamic/true HYBRID는 아직 live-equivalent engine validation 미완료다.

## 왜 이전 val return을 그대로 쓰면 안 되는가

이전 랭킹은 저장된 artifact/일별 return stream을 재조합했다. 거래 엔진의 주문 크기, 수수료/슬리피지, 체결 이벤트, 컴포넌트별 EXIT 처리, live portfolio mode symbol universe를 같은 경로로 강제하지 않았다. 따라서 실투자 승격 기준으로는 모두 재검증 대상이다.

## Live portfolio mode preflight / revalidation

| rank | mode | status | alpha | score | val ret | val Sharpe | val MDD | trades train/val | blocker |
|---:|---|---|---|---:|---:|---:|---:|---:|---|
| 1 | `profit_moonshot_adaptive_momentum_boost_mode` | `live_equivalent_validated` | yes | 10.0575 | +0.51% | 0.0148 | +1.36% | 361/56 | BTC/USDT,ETH/USDT,BNB/USDT,SOL/USDT,TRX/USDT |

## Research artifact reset sample

| research rank | name | previous score | mapped live mode | reset status |
|---:|---|---:|---|---|
| 1 | `legacy_metric_no_highvol_baseline_raw_score` | 95.4113 | `legacy_no_highvol_hybrid_mode` | `requires_live_equivalent_engine_backtest` |
| 2 | `legacy_metric_highvol_active_raw_score` | 92.5170 | `-` | `research_only_not_live_selectable` |
| 3 | `final_scaled_v35_train_high_vol` | 90.3846 | `-` | `research_only_not_live_selectable` |
| 4 | `full_hybrid_val_only_no_cash_penalty` | 90.1061 | `-` | `research_only_not_live_selectable` |
| 5 | `legacy_full_hybrid_wave2_val_only` | 90.1061 | `-` | `research_only_not_live_selectable` |
| 6 | `legacy_metric_highvol_optional_raw_score` | 89.5358 | `-` | `research_only_not_live_selectable` |
| 7 | `source_three_way_regime` | 89.0714 | `three_way_regime` | `requires_live_equivalent_engine_backtest` |
| 8 | `final_scaled_v36_dynamic_high_vol` | 88.5026 | `-` | `research_only_not_live_selectable` |
| 9 | `static_challenger_static_anchor_val_return` | 88.3893 | `-` | `research_only_not_live_selectable` |
| 10 | `static_challenger_static_dominant_val_return` | 88.0278 | `-` | `research_only_not_live_selectable` |
| 11 | `final_scaled_baseline_no_hv` | 87.7476 | `-` | `research_only_not_live_selectable` |
| 12 | `source_soft_three_way_regime` | 87.3457 | `soft_three_way_regime` | `requires_live_equivalent_engine_backtest` |
| 13 | `legacy_robust_val_low_cash_no_oos_health` | 86.6257 | `-` | `research_only_not_live_selectable` |
| 14 | `source_static_blend_76_24` | 86.4183 | `static_blend_76_24` | `requires_live_equivalent_engine_backtest` |
| 15 | `retuned_full_universe_hybrid_online` | 85.8314 | `-` | `research_only_not_live_selectable` |
| 16 | `full_hybrid_val_low_cash` | 85.2615 | `-` | `research_only_not_live_selectable` |
| 17 | `legacy_full_hybrid_val_low_cash` | 85.2615 | `-` | `research_only_not_live_selectable` |
| 18 | `legacy_robust_val_aggressive_cash_no_oos_health` | 85.2440 | `-` | `research_only_not_live_selectable` |
| 19 | `composite_trend_30m_0.45_0.60_0.20_0.80` | 83.6935 | `-` | `research_only_not_live_selectable` |
| 20 | `composite_trend_30m_0.60_0.75_0.20_0.80` | 83.6556 | `-` | `research_only_not_live_selectable` |
| 21 | `composite_trend_30m_0.60_0.75_0.25_0.80` | 83.6556 | `-` | `research_only_not_live_selectable` |
| 22 | `composite_trend_30m_0.45_0.75_0.20_0.80` | 83.6332 | `-` | `research_only_not_live_selectable` |
| 23 | `composite_trend_30m_0.45_0.60_0.25_0.80` | 83.5911 | `-` | `research_only_not_live_selectable` |
| 24 | `composite_trend_30m_0.45_0.75_0.25_0.80` | 83.5757 | `-` | `research_only_not_live_selectable` |
| 25 | `composite_trend_30m_0.60_0.60_0.25_0.80` | 83.5624 | `-` | `research_only_not_live_selectable` |
| 26 | `composite_trend_30m_0.60_0.60_0.20_0.80` | 83.5528 | `-` | `research_only_not_live_selectable` |
| 27 | `composite_trend_30m_0.75_0.75_0.20_0.80` | 83.4760 | `-` | `research_only_not_live_selectable` |
| 28 | `composite_trend_30m_0.75_0.75_0.25_0.80` | 83.4760 | `-` | `research_only_not_live_selectable` |
| 29 | `composite_trend_30m_0.75_0.60_0.25_0.80` | 82.8943 | `-` | `research_only_not_live_selectable` |
| 30 | `composite_trend_30m_0.75_0.60_0.20_0.80` | 82.8469 | `-` | `research_only_not_live_selectable` |
| 31 | `legacy_static_anchor_true_hybrid_val` | 82.7210 | `-` | `research_only_not_live_selectable` |
| 32 | `legacy_static_anchor_true_hybrid_low_cash` | 82.6909 | `-` | `research_only_not_live_selectable` |
| 33 | `composite_trend_30m_0.60_0.60_0.25_0.95` | 81.5092 | `-` | `research_only_not_live_selectable` |
| 34 | `composite_trend_30m_0.60_0.60_0.20_0.95` | 81.4826 | `-` | `research_only_not_live_selectable` |
| 35 | `full_hybrid_val_very_low_cash` | 81.3806 | `-` | `research_only_not_live_selectable` |
| 36 | `legacy_full_hybrid_val_very_low_cash` | 81.3806 | `-` | `research_only_not_live_selectable` |
| 37 | `composite_trend_30m_0.45_0.60_0.20_0.95` | 80.8341 | `-` | `research_only_not_live_selectable` |
| 38 | `composite_trend_30m_0.45_0.60_0.25_0.95` | 80.8294 | `-` | `research_only_not_live_selectable` |
| 39 | `composite_trend_30m_0.75_0.75_0.20_0.95` | 80.7708 | `-` | `research_only_not_live_selectable` |
| 40 | `composite_trend_30m_0.75_0.75_0.25_0.95` | 80.7708 | `-` | `research_only_not_live_selectable` |

## 명시적 caveats

- `1`개 mode가 live-equivalent engine backtest와 profit alpha gate를 모두 통과했다. selection eligibility는 양(+)의 validation return/Sharpe/Sortino와 active MDD/trade/liquidation gate 통과 후보에만 부여한다.
- 이 리포트의 핵심 변경은 `좋아 보이는 연구 점수`를 promotion evidence로 쓰지 않고, live-equivalent engine path를 통과한 후보만 승격시키는 것이다.
- profit alpha gate는 val return/Sharpe/Sortino가 양수이고, train/val 거래 수와 active MDD/liquidation gate를 통과한 후보만 alpha selection eligible로 인정한다.
- OOS는 report-only다. OOS raw-first coverage가 부족한 경우에도 train/val selection score에는 반영하지 않는다.

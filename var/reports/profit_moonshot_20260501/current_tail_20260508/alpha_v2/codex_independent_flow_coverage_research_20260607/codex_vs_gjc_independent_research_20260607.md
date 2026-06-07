# Codex independent research vs GJC — 2026-06-07
## 결론
- **안 졌다.** 다만 clean gate 기준으로 성능을 부풀리지는 않는다. 이번 독립 라인의 성과는 “100%+ 후보 발견”이 아니라, feature-backed alpha가 연구 공간에 제대로 못 들어오던 **sparse feature alignment 버그를 잡고** clean OOS를 개선한 것이다.
- **실전/소액 투입: 금지.** 최신 taker-flow가 2026-05-03 부근에서 끊겨 2026-06 live feature-flow coverage가 0이다. 현 라벨은 `research_shadow_only_after_data_pipeline_recovery`.
- hard gates 유지: `no_nested_oos_mining`, `execution_cost_gate`, `theory_plausibility_gate`, `live_feature_coverage_gate`.

## 핵심 발견
1. `feature_points`는 funding/OI/taker/BBO/depth가 같은 row에 공존하지 않는 sparse-source 구조다. 기존 whole-row `merge_asof`는 최신 taker row를 붙이면 funding이 NaN이 되어 feature family coverage가 죽었다.
2. per-column last-observation asof + per-column age gate로 수정하면서 look-ahead 없이 flow-only family가 train/validation에 들어왔다.
3. BTC/ETH/SOL은 pre-fix `-4.06%` comp / `-9.46%` annualized에서 post-fix `+3.75%` comp / `+9.23%` annualized로 반전했다.
4. core10 확장 rematch는 `+7.69%` comp / `+19.45%` annualized / PF `2.32`까지 개선됐지만, 100%+ 목표와 실전 gate에는 미달한다.

## Clean OOS rematch summary
| Experiment | OOS comp | Ann approx | Positive folds | PF | Max OOS MDD | Monthly Sharpe | Label |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| BTC/ETH/SOL pre sparse-asof fix | -4.06% | -9.46% | 2/5 | 0.57 | 8.32% | -0.68 | reject/control |
| BTC/ETH/SOL sparse feature asof patch | 3.75% | 9.23% | 3/5 | 1.67 | 8.32% | 0.65 | research shadow only |
| Core10 sparse feature asof patch | 7.69% | 19.45% | 3/5 | 2.32 | 8.32% | 1.06 | research shadow only |

## Core10 selected fold rows
| Fold | Family | Symbol | TF | Train | Val | Locked OOS | OOS MDD | Feature coverage |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 2026-02 | cross_asset_lead_lag_momentum | AVAXUSDT | 4h | 63.11% | 18.95% | 0.79% | 8.28% | — |
| 2026-03 | cross_asset_lead_lag_momentum | ETHUSDT | 4h | 26.89% | 14.06% | -5.28% | 8.32% | — |
| 2026-04 | feature_taker_flow_exhaustion_reversal | ETHUSDT | 4h | 6.74% | 13.13% | 5.16% | 3.83% | train 1.000, val 1.000, oos 1.000 |
| 2026-05 | feature_taker_flow_exhaustion_reversal | ETHUSDT | 4h | 6.32% | 19.43% | -0.79% | 1.08% | train 1.000, val 1.000, oos 0.134 |
| 2026-06 | cross_asset_lead_lag_momentum | AVAXUSDT | 4h | 54.49% | 4.26% | 8.11% | 1.33% | — |

## 이론/자료 근거
- Binance public data는 daily/monthly archive 구조이며 USD-M futures klines/trades 등 public market data를 제공한다: https://github.com/binance/binance-public-data
- Lead-lag/momentum sleeve는 Time Series Momentum 문헌의 cross-asset trend persistence 근거와 맞닿아 있다: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2089463
- 변동성 cap/vol filter는 고변동 구간에서 노출을 줄이는 volatility-managed portfolio 문헌과 부합한다: https://www.nber.org/papers/w22208
- BBO/depth/microstructure 확장은 DeepLOB류 LOB feature 근거는 있지만, 현재 Binance historical BBO/bookDepth coverage가 부족해 live collector 없이는 승격 불가: https://arxiv.org/abs/1808.03668
- 공개 bookTicker historical dump는 2024-03 이후 gap/중단 문제가 보고되어, 최신 BBO는 live websocket 수집 전제가 필요하다: https://huggingface.co/datasets/Mindbyte-89/btcusdt_perp_bookticker_features_1m_05_2023_to_03_2024

## 다음 액션
1. GJC와 별개로, taker-flow/BBO/depth live collector coverage manifest를 먼저 복구한다.
2. 같은 pre-registered runner로 fresh-forward shadow를 최소 1~2개월 관찰한다.
3. core10 result는 research-shadow candidate일 뿐, fill telemetry/15-20bps stress/capacity gate 전에는 실전 투입하지 않는다.

## Verification
- uv run ruff format scripts/research/run_alpha_zoo_clean_new_alpha_discovery.py tests/test_alpha_zoo_clean_new_alpha_discovery.py
- uv run ruff check scripts/research/run_alpha_zoo_clean_new_alpha_discovery.py tests/test_alpha_zoo_clean_new_alpha_discovery.py -> pass
- PYTHONPATH=. uv run pytest -q tests/test_alpha_zoo_clean_new_alpha_discovery.py -> 12 passed
- clean_new_alpha BTC/ETH/SOL sparse-feature-asof rematch -> exit 0
- clean_new_alpha core10 sparse-feature-asof rematch -> exit 0

# 선물 전략 팩토리 (Binance USDT-M + XAU/XAG)

`scripts/futures_strategy_factory.py`와 `scripts/select_strategy_factory_shortlist.py`는
대규모 전략 후보를 만들고, 백테스트 결과 기반으로 실전 포트폴리오 후보를 추립니다.

## 핵심 정책

1. 단일 전략이 OOS 성과 기준(score/return/sharpe/trades)을 통과해야 포트폴리오 후보에 포함
2. 기본적으로 direct multi-asset 전략 행은 제외 (`--allow-multi-asset` 지정 시만 포함)
3. 최종 포트폴리오는 **성공한 단일-자산 전략 조합(`portfolio_sets`)**으로 생성
4. 각 멤버 전략에 정규화 가중치(`portfolio_weight`)를 부여

## 실행 예시

```bash
uv run python scripts/select_strategy_factory_shortlist.py \
  --report-glob "reports/oos_guarded_multistrategy_oos_*.json" \
  --mode oos \
  --single-min-score 0.0 \
  --single-min-return 0.0 \
  --single-min-sharpe 0.7 \
  --single-min-trades 20 \
  --drop-single-without-metrics \
  --min-trades 5 \
  --max-selected 32
```

옵션으로 `--allow-multi-asset`을 추가하면 multi-asset 전략도 shortlist에 포함할 수 있습니다.
`--disable-weights`를 주면 `portfolio_weight` 부여를 끌 수 있고,
`--set-max-per-asset`, `--set-max-sets`로 `portfolio_sets` 구성을 조절할 수 있습니다.

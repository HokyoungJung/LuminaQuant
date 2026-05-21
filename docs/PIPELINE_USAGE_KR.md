# 파이프라인 사용법

## 백테스트

```bash
lq-public backtest --data sample_data/sample_ohlcv.csv --fast-window 3 --slow-window 8
```

백테스트는 로컬 샘플 봉 데이터, long/flat 샘플 전략, 고정 초기 현금,
단순 수수료 모델만 사용합니다.

## Paper-live 리플레이

```bash
lq-public paper-live --data sample_data/sample_ohlcv.csv --fast-window 3 --slow-window 8
```

Paper-live 리플레이는 같은 샘플 봉이 시간 순서대로 들어오는 것처럼
처리합니다. 결과에는 실제 주문 라우팅이 꺼져 있음을 보여주는 safety
블록이 포함됩니다.

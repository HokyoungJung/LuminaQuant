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

## 샘플 파라미터 최적화

```bash
lq-public optimize --data sample_data/sample_ohlcv.csv --fast-grid 2,3,4 --slow-grid 6,8,10
```

optimizer는 샘플 이동평균 window만 대상으로 하는 범용 grid search입니다.
비공개 연구 목적함수나 프로덕션 파라미터는 포함하지 않습니다.

## Rust 커널 확인

```bash
lq-public backtest --engine rust --data sample_data/sample_ohlcv.csv
cargo test --manifest-path native/rust_backtest_kernel/Cargo.toml
```

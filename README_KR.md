# LuminaQuant 공개용 파이프라인

이 저장소는 공개 가능한 테스트용 샘플만 포함합니다.

포함 항목:

- 결정론적 샘플 OHLCV 데이터
- 교육용 이동평균 샘플 전략 1개
- 백테스트 실행 파이프라인
- 로컬 paper-live 리플레이 파이프라인
- CLI와 CI 테스트

제외 항목:

- 실제/독점 전략
- 프로덕션 데이터와 데이터 수집 코드
- 연구 기록, 최적화 산출물, 운영 설정
- 거래소 커넥터, 주문 실행 코드, 자격 증명

`paper-live`는 로컬 시뮬레이터이며 실제 주문을 낼 수 없습니다.


## Generic metrics / optimization

공개 optimizer는 샘플 이동평균 window만 탐색합니다. 점수는
`total_return - 2 * max_drawdown - 0.0001 * trade_count`인 범용 예시이며
민감 전략이나 연구 로직을 포함하지 않습니다.

```bash
lq-public optimize --data sample_data/sample_ohlcv.csv --fast-grid 2,3,4 --slow-grid 6,8,10
```

## 선택적 Rust 커널

`native/rust_backtest_kernel/`에 source checkout용 Rust 백테스트 커널을
포함했습니다. Python 샘플 백테스트 요약과 parity를 맞추며 CI에서 검사합니다.

```bash
lq-public backtest --engine rust --data sample_data/sample_ohlcv.csv
cargo test --manifest-path native/rust_backtest_kernel/Cargo.toml
```

## 문서

- [공개 범위](docs/PUBLIC_SCOPE_KR.md)
- [파이프라인 사용법](docs/PIPELINE_USAGE_KR.md)
- [Public scope](docs/PUBLIC_SCOPE.md)
- [Pipeline usage](docs/PIPELINE_USAGE.md)

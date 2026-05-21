# LuminaQuant 공개용 파이프라인

이 저장소는 공개 가능한 테스트용 샘플만 포함합니다.

포함 항목:

- 결정론적 샘플 OHLCV 데이터
- `src/lumina_quant/sample_strategy.py`의 교육용 이동평균 샘플 전략 1개
- 백테스트 실행 파이프라인
- 로컬 paper-live 리플레이 파이프라인
- 범용 metrics/scoring 및 샘플-only 튜닝
- 공개용 샘플 TOML 설정 파일
- CLI와 CI 테스트

제외 항목:

- 실제/독점 전략
- 프로덕션 데이터와 데이터 수집 코드
- 연구 기록, private 최적화 산출물, 운영 설정
- 거래소 커넥터, 주문 실행 코드, 자격 증명

`paper-live`는 로컬 시뮬레이터이며 실제 주문을 낼 수 없습니다.


## Generic metrics / optimization

공개 optimizer는 기본적으로 Optuna로 동작하며 deterministic grid도
지원합니다. import 가능한 strategy class와 TOML search space만 주면
샘플 목적함수 `total_return - 2 * max_drawdown - 0.0001 * trade_count`로
튜닝합니다. 민감 전략이나 연구 로직은 포함하지 않습니다.

```bash
lq-public optimize --data sample_data/sample_ohlcv.csv --n-trials 16
lq-public optimize --method grid --data sample_data/sample_ohlcv.csv --fast-grid 2,3,4 --slow-grid 6,8,10
lq-public optimize --config sample_configs/public_sample_pipeline.toml
```

## 공개용 샘플 config

`sample_configs/public_sample_pipeline.toml`에는 샘플 데이터 경로,
strategy class path, strategy params, Rust/Python backtest engine 선택,
optimization method, Optuna trial count/seed, 시작 현금, 수수료 설정만
들어 있습니다. CLI 인자는 config 값을 덮어쓰므로 private 파일 없이도
같은 공개 파이프라인을 smoke-test할 수 있습니다.

```bash
lq-public backtest --config sample_configs/public_sample_pipeline.toml
lq-public backtest --config sample_configs/public_sample_pipeline.toml --engine python
lq-public paper-live --config sample_configs/public_sample_pipeline.toml
lq-public optimize --config sample_configs/public_sample_pipeline.toml --fast-grid 2,3 --slow-grid 6,8
```

## 내 데이터와 전략 연결하기

외부 사용자는 파이프라인을 고치지 않고 로컬 CSV와 import 가능한 strategy
class만 넣으면 됩니다. strategy class는 `on_bar(bar)`를 구현하고
`lumina_quant.models.Signal`을 반환해야 합니다. 생성자 값은 CLI의
`--strategy-param key=value` 또는 TOML `strategy_params`로 넣고, Optuna
탐색 범위는 `[optimization.search_space.<param>]` 아래에 둡니다.

```bash
lq-public backtest --data your_ohlcv.csv --strategy your_pkg.module:YourStrategy --strategy-param lookback=20
lq-public optimize --config sample_configs/public_sample_pipeline.toml
```

## 선택적 Rust 커널

`native/rust_backtest_kernel/`에 source checkout용 Rust 백테스트 커널을
포함했습니다. bundled sample Python 백테스트 요약과 parity를 맞추고,
`--engine rust` 또는 `[backtest].engine = "rust"`로 선택되며 CI에서
검사합니다. Rust 경로는 generic native strategy ABI를 설계하기 전까지
bundled sample strategy 전용입니다. 외부 Python strategy class는 Python
파이프라인으로 실행됩니다.

```bash
lq-public backtest --engine rust --data sample_data/sample_ohlcv.csv
cargo test --manifest-path native/rust_backtest_kernel/Cargo.toml
```

## 문서

- [공개 범위](docs/PUBLIC_SCOPE_KR.md)
- [파이프라인 사용법](docs/PIPELINE_USAGE_KR.md)
- [Public scope](docs/PUBLIC_SCOPE.md)
- [Pipeline usage](docs/PIPELINE_USAGE.md)

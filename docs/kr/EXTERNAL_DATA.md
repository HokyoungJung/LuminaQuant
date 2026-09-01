# 외부 데이터 계약

LuminaQuant는 사용자 보유 데이터를 백테스트와 라이브 양쪽에서 사용할 수 있도록 canonical contract를 제공합니다.

## 백테스트 외부 데이터

필수 OHLCV 컬럼:
- `datetime`
- `open`
- `high`
- `low`
- `close`
- `volume`

지원 형식:
- CSV
- parquet
- 테스트/내부용 in-memory fixture

### CLI
```bash
uv run lq backtest --data-source external --external-data-root var/data/external/backtest
uv run lq optimize --data-source external --external-data-root var/data/external/backtest
```

### 설정
```yaml
backtest:
  data_source: external
  external:
    source_kind: csv
    root_path: var/data/external/backtest
    symbol_map:
      BTC/USDT: BTCUSDT.csv
```

명시적 `external` 모드는 **fail-fast** 입니다. 데이터가 없으면 자동 fallback 하지 않고 즉시 오류를 냅니다.

## 라이브 외부 데이터

```yaml
live:
  market_data_source: external
  external:
    source_kind: jsonl   # jsonl | parquet | pipe
    path: var/data/external/live_windows.jsonl
    schema: market_window_v1
    poll_seconds: 2
    allow_stale_seconds: 45
```

### `market_window_v1` JSONL
라인 단위 JSON:
- `time`
- `window_seconds`
- `bars_1s`
- `event_time_watermark_ms`
- `commit_id`
- `lag_ms`
- `is_stale`

`bars_1s`의 각 row 형식:
- `(timestamp_ms, open, high, low, close, volume)`

### `ohlcv_1s_v1` parquet
동일한 canonical 1초 OHLCV 컬럼을 가진 parquet 파일을 읽어 내부적으로 `MARKET_WINDOW`로 변환합니다.

## 전략 계약
전략은 vendor raw payload 대신 canonical contract를 소비해야 합니다.

추가된 선언형 필드:
- `required_inputs`
- `required_features`
- `preferred_contract`
- optional `calculate_signals_context(context)`

기존 `calculate_signals` / `calculate_signals_window` 도 계속 지원됩니다.

# Slippage Calibration / Monitoring Spec

작성일: 2026-03-19  
대상 저장소: `LuminaQuant`

## 목적

현재 LuminaQuant에는 **백테스트용 slippage model**이 존재한다.  
실거래 전환 전에는 이것을 **paper/testnet 기반의 realized slippage evidence**로 보정해야 한다.

이 문서는 그 측정/보정 규격을 정의한다.

---

## 1. 현재 상태

### 이미 있는 것

- `config.yaml`
  - `execution.slippage_rate`
  - `backtest.slippage_rate`
- `src/lumina_quant/backtesting/execution_sim.py`
  - spread
  - slippage
  - commission
  - latency
  - liquidity cap
- `tests/test_cost_calibration.py`
- `scripts/run_cost_aware_framework.py`

### 아직 없는 것

- live/paper realized slippage bps 축적 표준
- symbol / regime / volatility bucket별 실측 분해
- backtest 가정과 paper 실측을 연결하는 공식 gate

---

## 2. 측정 대상

각 체결/주문마다 최소 아래를 기록해야 한다.

- `run_id`
- `env` (`paper` / `testnet` / `real`)
- `symbol`
- `side`
- `order_type`
- `limit_reference_price`
- `limit_price`
- `limit_price_mode`
- `limit_price_offset_ticks`
- `time_in_force`
- `submit_ts_utc`
- `ack_ts_utc`
- `fill_ts_utc`
- `decision_price`
- `expected_mid_price`
- `expected_mark_price`
- `fill_price`
- `quantity`
- `notional`
- `timeout_flag`
- `partial_fill_flag`
- `cancel_flag`
- `reduce_only_flag`
- `position_side`

계산 필드:
- `fill_latency_ms`
- `realized_slippage_bps`
- `spread_bps_at_submit`
- `fee_bps`
- `all_in_cost_bps`

---

## 3. 권장 로그 형식

형식:
- JSONL

예시:

```json
{
  "run_id": "paper-20260319T120000Z",
  "env": "paper",
  "symbol": "BTC/USDT",
  "side": "BUY",
  "order_type": "LMT",
  "limit_reference_price": 73500.25,
  "limit_price": 73500.35,
  "limit_price_mode": "one_tick_worse",
  "limit_price_offset_ticks": 1,
  "time_in_force": "GTC",
  "submit_ts_utc": "2026-03-19T12:00:01.000Z",
  "ack_ts_utc": "2026-03-19T12:00:01.120Z",
  "fill_ts_utc": "2026-03-19T12:00:01.350Z",
  "decision_price": 73500.25,
  "expected_mid_price": 73500.10,
  "expected_mark_price": 73500.18,
  "fill_price": 73503.80,
  "quantity": 0.02,
  "notional": 1470.076,
  "timeout_flag": false,
  "partial_fill_flag": false,
  "cancel_flag": false,
  "reduce_only_flag": false,
  "position_side": "LONG",
  "fill_latency_ms": 350,
  "realized_slippage_bps": 0.50,
  "spread_bps_at_submit": 0.20,
  "fee_bps": 4.0,
  "all_in_cost_bps": 4.70
}
```

---

## 4. 집계 단위

아래 4개 bucket으로 나눠서 본다.

1. **symbol**
2. **timeframe / strategy family**
3. **volatility regime**
4. **liquidity / spread regime**

최소 요약값:
- count
- median slippage bps
- p90 slippage bps
- p95 slippage bps
- max slippage bps
- timeout rate
- partial fill rate

---

## 5. calibration 규칙

### 1차 보정

backtest 기본값과 paper 실측값을 비교해서,

- `median realized_slippage_bps`
- `p90 realized_slippage_bps`
- `timeout / cancel impact`

를 기준으로 symbol bucket별 조정값을 만든다.

### 2차 보정

다음 조건이 있으면 추가 penalty를 더한다.

- 고변동 regime
- spread 확장 구간
- partial fill 반복
- timeout 반복

---

## 6. 승인 기준

실거래 전환 전 최소한 아래를 만족해야 한다.

- [ ] realized slippage가 backtest 가정과 큰 괴리가 없을 것
- [ ] 특정 symbol에서 tail risk가 반복적으로 튀지 않을 것
- [ ] timeout / partial fill이 전략 edge를 훼손하지 않을 것
- [ ] all-in cost 반영 후에도 기대 edge가 유지될 것

---

## 7. alert 기준(초안)

예시:

- `realized_slippage_bps > configured_slippage_bps * 2`
  - warning
- 동일 symbol에서 연속 3회 timeout
  - high
- partial fill rate가 최근 1시간 기준 임계치 초과
  - high
- spread_bps 급증 + slippage 급증 동시 발생
  - block candidate

---

## 8. 지금 당장 필요한 것

우선 구현/운영해야 하는 순서:

1. paper/testnet 체결 로그 JSONL 스키마 고정
2. run_id 기준 집계 스크립트 추가
3. symbol / regime별 slippage summary 생성
4. backtest slippage config와 paper summary를 연결하는 gate 작성

현재 제공되는 helper:

```bash
cd /home/hoky/Quants-agent/LuminaQuant
uv run python scripts/ops/summarize_fill_slippage.py --input-jsonl /path/to/fills.jsonl
```

---

## 9. 현재 판단

현재 상태는:

- **백테스트용 slippage model은 존재**
- **실거래용 calibration evidence는 아직 부족**

즉,
- **실거래 전환 전 반드시 paper/testnet 기반 보정이 필요**

---

## 10. 권고

다음 단계에서 추천:

- paper 운영 시작
- fill / timeout / cancel / partial fill 로그 축적
- 2주 이상 누적 후 slippage calibration 요약 생성
- 그 뒤에야 real mode 여부를 논의

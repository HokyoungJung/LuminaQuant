# Paper Trading Readiness 점검서

작성일: 2026-03-19  
대상 저장소: `LuminaQuant`

## 목적

실거래 이전에 반드시 거쳐야 하는 **paper / testnet 운영 단계**를 정의한다.
핵심 목표는 다음 5개다.

1. 실제 주문 경로가 안전하게 동작하는지 확인
2. heartbeat / reconciliation / stale-feed 처리가 운영 가능한지 확인
3. realized slippage / timeout / partial fill 데이터를 확보
4. risk freeze / flatten 경로를 재현
5. real mode 전환 조건을 수치로 고정

---

## 1. 현재 코드 기준 준비도

### 이미 있는 것

- `config.yaml`
  - `live.mode: "paper"`
  - `testnet: true`
  - `require_real_enable_flag: true`
- `configs/profiles/paper.yaml`
- `configs/profiles/real.yaml`
- `src/lumina_quant/live/trader.py`
  - heartbeat
  - position reconciliation
  - order reconciliation
  - stale data block / recovery
  - flatten-all / trade freeze
- `src/lumina_quant/live/execution_live.py`
  - order timeout
  - open order reconciliation
  - client order id dedupe
- `src/lumina_quant/live/shadow_live_runner.py`
  - baseline vs candidate divergence 비교용 shadow helper
- `src/lumina_quant/live/execution_live.py` / `src/lumina_quant/exchanges/binance_futures_exchange.py`
  - 2026-05-25 기준 paper/testnet 전용 Binance USD-M conditional algo 보호 주문 경로
  - entry fill 이후 `STOP_MARKET` / `TAKE_PROFIT_MARKET` 제출
  - `/fapi/v1/algoOrder` request payload allowlist 적용; parent/protection telemetry는 local record에만 저장
- `AlphaZooOptunaHybridLiveStrategy` handoff artifact
  - `ready_for_real=false`, `real_money_execution=false`, `real_execution_allowed=false` 유지
  - selected frozen source assets `ETHUSDT`, `SOLUSDT`, `TRXUSDT` asset-generic regression coverage

### 아직 고정해야 하는 것

- paper 운영 기간
- paper pass/fail 지표
- operator check cadence
- 실제 Binance testnet conditional algo fill / sibling cancellation / reconciliation telemetry
- paper → real 승격 gate

---

## 2. 운영 단계

## Phase 0 — Static preflight

목표:
- 코드/설정/최신 포트폴리오 상태를 고정한다.

필수 체크:
- [ ] latest-tail refresh / validation 최신 상태 확인
- [ ] incumbent가 live candidate로 유지된 최종 decision artifact 확인
- [ ] 관련 테스트 통과
- [ ] `live.mode = paper`
- [ ] `testnet = true`
- [ ] `require_real_enable_flag = true`

권장 확인:
- [ ] 현재 profile이 `paper`로 로딩되는지 확인
- [ ] 거래소 API key가 paper/testnet용인지 확인
- [ ] dashboard 연결 상태 확인

---

## Phase 1 — Dry-run / no-risk launch

목표:
- 실제 의사결정 loop는 돌리되, 주문 제출/체결 영향을 최소화한 상태로 점검한다.

확인 항목:
- [ ] heartbeat가 일정 주기로 남는지
- [ ] stale data alert가 정상 기록되는지
- [ ] reconciliation loop가 예외 없이 도는지
- [ ] candidate/live decision path가 멈추지 않는지

관측 KPI:
- heartbeat 누락 여부
- stale data event count
- reconciliation error count
- crash / restart count

pass 조건:
- 중대한 예외 없이 지속 구동
- stale / reconciliation 오류가 폭증하지 않을 것

---

## Phase 2 — Paper / testnet order flow

목표:
- 실제 주문과 유사한 경로를 paper/testnet에서 반복 검증한다.

확인 항목:
- [ ] submit → ack → fill / cancel / timeout 상태 전이
- [ ] entry fill 이후 paper/testnet `STOP_MARKET` / `TAKE_PROFIT_MARKET` protective algo order 제출 확인
- [ ] protective leg trigger 후 sibling cancellation/reconciliation 정책 확인
- [ ] open order snapshot readiness
- [ ] timeout 후 cancel/reconcile
- [ ] reduce-only 주문 허용 여부
- [ ] trade freeze 중 신규 진입 차단 여부
- [ ] flatten-all 발동 경로

수집할 지표:
- order submit count
- order timeout count
- cancel count
- partial fill count
- reconciliation drift count
- average fill latency

pass 조건:
- timeout / cancel / reconciliation이 일관되게 동작
- drift 이벤트가 누적되지 않을 것

---

## Phase 3 — Paper slippage / cost calibration

목표:
- backtest 가정과 paper 결과의 괴리를 수치화한다.

반드시 수집할 것:
- symbol별 realized slippage bps
- volatility regime별 slippage
- partial fill / timeout 동반 비용
- fee / turnover drag

기본 비교 대상:
- `config.yaml`의 `execution.slippage_rate`
- `src/lumina_quant/backtesting/execution_sim.py`의 fill model

pass 조건:
- realized slippage가 backtest 가정을 크게 벗어나지 않을 것
- 특정 심볼/시간대에서 극단적 outlier가 반복되지 않을 것

---

## Phase 4 — Operator drill

목표:
- 운영자가 사고 상황에서 시스템을 통제할 수 있는지 확인한다.

리허설 항목:
- [ ] trade freeze ON
- [ ] flatten-all trigger
- [ ] stale data 발생 시 행동
- [ ] reconciliation drift 발생 시 행동
- [ ] restart 후 recovery

pass 조건:
- 수동 개입 절차가 1회성 improvisation이 아니라 문서대로 재현 가능

---

## 3. 필수 pass / fail 기준

## Pass

- [ ] paper profile / testnet 설정 고정
- [ ] 최소 운영 기간 충족
- [ ] heartbeat 안정
- [ ] reconciliation drift 통제 가능
- [ ] stale data block / recovery 확인
- [ ] timeout / cancel / partial fill 처리 확인
- [ ] realized slippage 측정 완료
- [ ] operator drill 완료

## Fail

다음 중 하나라도 만족하면 real 전환 금지:

- heartbeat 끊김 반복
- stale data alert 빈발
- order timeout / reconciliation drift 누적
- slippage outlier 지속
- freeze / flatten 경로 미검증
- operator 수동 중지 절차 미확립

---

## 4. 권장 관찰 기간

최소 권장:

- **연속 2주 paper/testnet**

강한 권장:

- **정상장 + 변동성 확장 구간 포함 3~4주**

이유:
- 단순 정상장에서는 slippage / timeout / stale feed 리스크가 과소평가될 수 있음

---

## 5. 실무 운영 체크리스트

매일:
- [ ] latest validation artifact timestamp 확인
- [ ] heartbeat 상태 확인
- [ ] stale/reconciliation/timeout 이벤트 확인
- [ ] 전일 realized slippage 요약 확인

매주:
- [ ] symbol별 slippage distribution 업데이트
- [ ] kill-switch 리허설
- [ ] paper PnL / drawdown / turnover 점검
- [ ] incumbent와 paper 실제 체결 품질 차이 점검

---

## 6. real 전환 직전 필수 문서

real 전환 전에 아래 문서가 최신이어야 한다.

- `01-live-trading-checklist.md`
- `03-aggressive-sleeve-local-retune-plan.md` (추가 성능 개선 필요 시)
- `04-paper-trading-runbook.md`
- `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_integer_leverage_optuna_hybrid_decision_20260524/paper_testnet_live_decision_latest.md`
- `var/reports/exact_window_backtests/followup_status/portfolio_live_readiness_decision_latest.md` (legacy incumbent 경로 참고용)

---

## 7. 현재 기준 판정

현재 저장소는:

- **paper/testnet 시작은 가능**
- **Alpha Zoo Optuna hybrid handoff도 paper/testnet-only로 저장됨**
- **real 전환은 아직 금지**

즉,
- **paper readiness: 진행 가능**
- **paper protective-order readiness: 코드/테스트는 있음, 실제 testnet fill telemetry 필요**
- **real readiness: 추가 운영 검증 필요, 현재 artifact veto 유지**

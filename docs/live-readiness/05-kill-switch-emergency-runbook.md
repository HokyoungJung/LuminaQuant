# Kill-Switch / Emergency Runbook

작성일: 2026-03-19  
대상 저장소: `LuminaQuant`

## 목적

실거래 또는 paper 운영 중 문제가 생겼을 때
**무엇을 멈추고, 무엇을 유지하고, 어떤 순서로 대응할지**를 정리한다.

이 문서는 소프트웨어 경로와 운영자 수동 개입 경로를 함께 다룬다.

---

## 1. 현재 시스템에 존재하는 안전장치

코드 레벨:
- `src/lumina_quant/risk_manager.py`
  - intraday drawdown breach
  - rolling loss breach
  - symbol exposure / total exposure cap
  - trade freeze / reduce-only allowance
- `src/lumina_quant/live/trader.py`
  - stale data block
  - flatten-all queue
  - trade freeze on/off
  - fail-fast / hard halt
  - reconciliation drift logging
- `src/lumina_quant/live/execution_live.py`
  - order timeout
  - cancel + reconcile
  - duplicate client_order_id 방지

운영 레벨:
- `uv run lq live --stop-file /tmp/lq.stop`
- `touch /tmp/lq.stop` 로 graceful shutdown
- emergency fallback은 host/systemd/process kill로 별도 확보 필요

---

## 2. 사고 유형별 대응

## A. stale data / data freshness breach

징후:
- stale data alert
- materialized staleness breach
- heartbeat는 살아 있으나 trading block 발생

예상 시스템 동작:
- 신규 의사결정 차단
- 경고 메시지
- fresh window 2회 확인 후 재개

운영자 대응:
- [ ] data source / network 상태 확인
- [ ] stale가 일시적인지 확인
- [ ] 복구되지 않으면 stop-file로 중지
- [ ] 반복 발생 시 실거래 금지

---

## B. reconciliation drift

징후:
- local position과 exchange position 불일치
- open order snapshot readiness 문제
- drift event 증가

예상 시스템 동작:
- drift risk event logging
- notifier 경고

운영자 대응:
- [ ] drift symbol / delta 확인
- [ ] open orders / positions 수동 조회
- [ ] paper면 세션 중지 후 원인 분석
- [ ] real이면 신규 진입 freeze 우선
- [ ] 필요 시 flatten-all 절차 실행

---

## C. order timeout / stuck order

징후:
- timeout 증가
- cancel 후 reconcile 필요

예상 시스템 동작:
- timeout 후 cancel 시도
- 최신 상태 재조회
- terminal / timeout_reconciled 상태 처리

운영자 대응:
- [ ] timeout count가 정상 범위인지 확인
- [ ] 특정 symbol / 특정 세션에 집중되는지 확인
- [ ] 반복되면 execution path paper 유지
- [ ] real 전환 금지

---

## D. intraday drawdown / rolling loss breach

징후:
- risk manager breach
- trading_frozen
- flatten-all trigger 가능

운영자 대응:
- [ ] 신규 진입 즉시 차단 확인
- [ ] reduce-only만 허용되는지 확인
- [ ] 필요 시 flatten-all
- [ ] 당일 재가동 금지 원칙 적용

---

## E. hard halt / fail-fast

징후:
- live data fail-fast
- critical exception
- ordered shutdown 진입

운영자 대응:
- [ ] 즉시 stop-file 또는 프로세스 중지
- [ ] 마지막 로그 / risk event / heartbeat 확보
- [ ] 재시작 전 원인 분석 완료

---

## 3. 운영자 실행 절차

## Level 1 — Graceful stop

실행:

```bash
touch /tmp/lq-paper.stop
```

또는 helper:

```bash
cd /home/hoky/Quants-agent/LuminaQuant
bash scripts/ops/stop_live_session.sh
```

조건:
- 시스템이 응답 중
- orderly shutdown 가능

---

## Level 2 — Trade freeze 유지 + 수동 점검

조건:
- stale / drift / timeout가 반복되지만 즉시 강제 종료까지는 아님

조치:
- [ ] 신규 진입 중단 상태 확인
- [ ] reduce-only만 허용되는지 확인
- [ ] positions / orders 수동 조회

---

## Level 3 — Flatten-all

조건:
- 포지션 무결성 의심
- 위험 이벤트 반복
- 운영자가 포지션 유지 불가로 판단

주의:
- 현재 코드에는 flatten-all queue 경로가 있으나
- **운영자 수동 trigger 절차를 별도로 고정하는 것이 필요**

즉, 현재는 “기능 존재” 수준이고,
실전 운영용으로는 다음이 추가로 필요하다.

- [ ] flatten-all operator command 표준화
- [ ] 실행 후 확인 절차
- [ ] 실패 시 재시도 절차

---

## Level 4 — Emergency kill

조건:
- graceful stop 불능
- 프로세스 hang
- 외부 거래소 상태 불안정

조치:
- host/systemd/process kill 사용
- kill 후 반드시:
  - [ ] open positions 확인
  - [ ] open orders 확인
  - [ ] 다음 재시작 금지 until reconciliation 완료

---

## 4. 운영자 체크리스트

### 시작 전
- [ ] stop-file 경로 준비
- [ ] dashboard / logs tail 확보
- [ ] latest validation artifact 확인

### 운영 중
- [ ] stale alert 여부
- [ ] reconciliation drift 여부
- [ ] timeout 증가 여부
- [ ] heartbeat 공백 여부

### 사고 후
- [ ] 마지막 heartbeat / risk event 저장
- [ ] open orders / positions 수동 조회
- [ ] 재시작 전 원인 기록

---

## 5. 현재 기준 평가

현재 상태:
- **software kill path: 부분 준비**
- **operator emergency runbook: 이번 문서로 초안 마련**
- **실거래 전 추가 구현 필요**

가장 중요한 부족점:
- flatten-all operator trigger 표준화
- stop / freeze / flatten 상태 전이 표준 운영절차 확정
- alarm escalation 기준 고정

---

## 6. 권고

실거래 전에 반드시 해야 할 것:

1. paper 환경에서 stale / drift / timeout / drawdown breach 리허설
2. flatten-all 수동 절차 고정
3. emergency kill 후 recovery 절차 고정
4. 운영자 1인이 아니라도 재현 가능한 runbook으로 확정

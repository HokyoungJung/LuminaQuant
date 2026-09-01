# Live Operations Runbook

작성일: 2026-03-27  
대상 저장소: `LuminaQuant`

## 목적

이 문서는 현재 LuminaQuant live runtime의 **실행 전 점검**, **기동 상태 해석**,
**운영 중 감시**, **중지/장애 대응**을 한 장의 runbook으로 정리한다.

핵심 목표:

1. 운영자가 live runtime의 시작 상태를 정확히 해석할 것
2. `poll/ws` transport와 `MARKET_DATA_SOURCE`의 우선순위를 혼동하지 않을 것
3. 문제가 생겼을 때 graceful stop / fail-fast / emergency 대응 순서를 재현 가능하게 만들 것

관련 문서:
- `docs/live-readiness/01-live-trading-checklist.md`
- `docs/live-readiness/04-paper-trading-runbook.md`
- `docs/live-readiness/05-kill-switch-emergency-runbook.md`

---

## 1. 현재 runtime contract 요약

현재 live runtime은 `src/lumina_quant/system_assembly.py`의 단일 contract를 통해 조립된다.

- `build_live_runtime_contract(transport=...)`
- `build_system(mode, transport=...)`

의미:

- CLI가 직접 data handler / execution handler / portfolio wiring을 따로 결정하지 않는다
- `live`와 `sandbox` 모두 동일한 assembly 경로를 사용한다
- live portfolio 경계는 `src/lumina_quant/live/portfolio.py`의 lazy boundary 뒤에 있다

운영자가 기억할 것:

- **runtime wiring의 진실원은 system assembly 하나**
- CLI에서 transport를 요청해도, 실제 effective transport는 market data source에 따라 달라질 수 있음

---

## 2. transport precedence

현재 우선순위:

1. 사용자가 `--transport {poll,ws}` 요청
2. `live.market_data_source` 확인
3. `market_data_source=committed` 이고 `--transport=ws` 이면
   - **effective transport는 강제로 `poll`**
   - CLI는 경고를 출력

즉:

| market_data_source | requested transport | effective transport |
|---|---|---|
| `committed` | `poll` | `poll` |
| `committed` | `ws` | `poll` |
| `external` / exchange-direct 계열 | `poll` | `poll` |
| `external` / exchange-direct 계열 | `ws` | `ws` |

운영 규칙:

- committed reader를 쓰는 세션에서는 websocket transport를 “요청”해도 실제로는 poll로 간주한다
- 경고가 보이면 **오류가 아니라 안전한 downgrade**

---

## 3. startup state 해석

`LiveTrader`는 startup state를 다음 중 하나로 기록/알림한다.

### 3.1 `ready`

의미:
- 초기화 완료
- startup reconciliation 성공
- 또는 user stream 미사용 경로에서 정상 진입

운영자 행동:
- heartbeat / drift / stale alert만 계속 감시
- paper/testnet evidence 수집 계속

### 3.2 `degraded`

의미:
- startup reconciliation timeout 등으로 완전한 준비는 아니지만
- fallback polling 경로로 운영 지속

운영자 행동:
- 신규 진입은 보수적으로 해석
- drift / stale / user stream health를 더 자주 확인
- real 전환 판단 근거로 사용하지 말 것

### 3.3 `failed_init`

의미:
- exchange init / handler init / recovery service init / pre-loop sync 등에서 실패
- notifier가 실패 알림을 보내고 audit run을 `FAILED`로 닫으려 시도

운영자 행동:
- 즉시 원인 로그 확보
- 재시작 전에 exchange / config / DSN / data source 상태 확인
- 반복되면 real 금지

---

## 4. 실행 전 점검

```bash
cd /home/hoky/Quants-agent/LuminaQuant

uv run python scripts/ops/live_readiness_preflight.py
uv run pytest -q tests/test_live_fail_fast_missing_committed_data.py tests/test_live_trader_startup_hardening.py tests/test_system_assembly.py
```

최소 확인 항목:

- [ ] `live.mode` 가 의도한 값인지
- [ ] `live.testnet` / `require_real_enable_flag` 상태가 맞는지
- [ ] PostgreSQL DSN 사용 가능
- [ ] `MARKET_DATA_SOURCE` 가 의도한 경로인지
- [ ] `ORDER_STATE_SOURCE` 가 의도한 경로인지
- [ ] stop-file 경로를 미리 정했는지

---

## 5. 권장 실행 명령

### 5.1 paper / testnet

```bash
cd /home/hoky/Quants-agent/LuminaQuant

uv run lq live \
  --transport poll \
  --stop-file /tmp/lq-paper.stop \
  --run-id paper-$(date -u +%Y%m%dT%H%M%SZ)
```

### 5.2 websocket 요청이 필요한 경우

```bash
cd /home/hoky/Quants-agent/LuminaQuant

uv run lq live \
  --transport ws \
  --stop-file /tmp/lq-live.stop \
  --run-id live-$(date -u +%Y%m%dT%H%M%SZ)
```

주의:

- `MARKET_DATA_SOURCE=committed` 이면 실제 runtime transport는 poll로 고정된다
- 따라서 websocket 실험은 exchange-direct / external market data source일 때만 의미가 있다

---

## 6. 런타임 중 확인 항목

반드시 볼 것:

- [ ] startup notifier 메시지 (`ready` / `degraded` / `failed_init`)
- [ ] heartbeat 정상 여부
- [ ] stale data alert 발생 여부
- [ ] reconciliation drift 누적 여부
- [ ] order timeout / cancel 비율
- [ ] audit run status / risk events

권장 확인:

- dashboard overview / risk-health / workflows
- latest validation artifacts
- operator stop-file 위치

---

## 7. 정상 중지 절차

### Level 1 — graceful stop

```bash
touch /tmp/lq-paper.stop
```

또는:

```bash
cd /home/hoky/Quants-agent/LuminaQuant
bash scripts/ops/stop_live_session.sh
```

기대 결과:

- trader loop 종료
- audit run이 `STOPPED`로 닫힘
- 다음 시작 전 stale/positions 확인 가능

---

## 8. 장애 대응

### 8.1 startup이 `failed_init`

확인:

- 마지막 notifier message
- audit run metadata
- stderr / logs

조치:

- [ ] exchange connectivity 확인
- [ ] config/profile 확인
- [ ] DSN / storage 접근성 확인
- [ ] data source / committed artifact 존재 확인

### 8.2 startup이 `degraded`

조치:

- [ ] 해당 세션을 “정상 full-ready”로 간주하지 말 것
- [ ] fallback polling 창에서 drift/stale 경고 추적
- [ ] evidence 수집 후 원인 분석

### 8.3 fail-fast / hard halt

조치:

- [ ] 즉시 stop-file 또는 프로세스 중지
- [ ] 마지막 risk event 저장
- [ ] open orders / open positions 수동 확인
- [ ] 원인 분석 전 재시작 금지

---

## 9. 운영 후 보관해야 할 증거

- run id
- startup state (`ready` / `degraded` / `failed_init`)
- audit run 종료 상태 (`STOPPED` / `FAILED`)
- stale / drift / timeout 관련 risk events
- dashboard screenshots 또는 summary
- 사용한 profile / transport / market_data_source / order_state_source

---

## 10. 지금 기준 운영 판단

현재 코드 기준으로는:

- live startup failure semantics는 충분히 강함
- runtime contract boundary는 중앙화됨
- live portfolio import boundary는 lazy 경계로 분리됨
- committed + ws 요청도 안전하게 poll로 downgrade됨

즉 운영자는 이제:

- **startup state를 해석**
- **effective transport를 혼동하지 않고**
- **graceful stop / degraded / fail-fast 대응 순서를 지키는 것**

에 집중하면 된다.


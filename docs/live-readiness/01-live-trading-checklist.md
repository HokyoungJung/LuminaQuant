# 실거래 전환 체크리스트

작성일: 2026-03-19  
대상 저장소: `LuminaQuant`  
현재 포트폴리오 상태: **legacy incumbent 문서 + 2026-05-25 Alpha Zoo paper/testnet override 참고**

## 목적

이 문서는 현재 선택된 포트폴리오를 실제 거래로 전환하기 전에 확인해야 할
실행 안정성 항목을 정리한다.

본 문서는 다음 최신 검증 결과를 전제로 한다.

- latest-tail refresh cutoff: `2026-03-19T11:39:51Z`
- latest-tail common feature tail: `2026-03-19T11:39:00Z`
- incumbent locked OOS end: `2026-03-19T11:39:00Z`
- OOS start: `2026-02-01T00:00:00Z`

관련 최종 의사결정 아티팩트:

- `var/reports/exact_window_backtests/followup_status/portfolio_live_readiness_decision_latest.json`
- `var/reports/exact_window_backtests/followup_status/portfolio_live_readiness_decision_latest.md`

## 2026-05-25 최신 Alpha Zoo paper/testnet override

아래 2026-03 legacy incumbent 체크리스트는 일반 live 전환 기준으로 유지한다. 현재 private Alpha Zoo 경로의 최신 결론은 다음 artifact와 문서를 우선한다.

- 최신 paper/testnet handoff: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_integer_leverage_optuna_hybrid_decision_20260524/paper_testnet_live_decision_latest.json`
- 요약 문서: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_integer_leverage_optuna_hybrid_decision_20260524/paper_testnet_live_decision_latest.md`
- 운영 런북: `docs/live-readiness/04-paper-trading-runbook.md`
- 연구 노트: `docs/research_note/research_note.md`

현재 판정:

- `AlphaZooOptunaHybridLiveStrategy`는 paper/testnet monitoring 대상으로 유지한다.
- real-money는 아직 금지한다: `ready_for_real=false`, `real_money_execution=false`, `real_execution_allowed=false`.
- live 주문은 기본 limit-first다. 진입/숏/exit/flatten은 `LMT`, `one_tick_worse` 가격 정책(BUY +1 tick, SELL -1 tick)을 기본으로 쓰며 시장가는 명시적 opt-in(`live.default_order_type=MKT`, `live.allow_market_orders=true`)이 필요하다.
- entry fill 이후 paper/testnet 전용 Binance USD-M conditional algo `STOP`/`TAKE_PROFIT` limit 보호 주문 경로가 있다.
- 거래소 request payload는 문서화된 `/fapi/v1/algoOrder` allowlist만 보내며, parent/protection telemetry는 local reconciliation record에만 저장한다.
- selected frozen source asset 적용성은 `ETHUSDT`, `SOLUSDT`, `TRXUSDT`에 대해 검증했다.
- 실제 testnet fill/slippage/reconciliation/protective-order telemetry와 2주 이상 관찰 전에는 real 전환 논의 금지.

---

## 1. 현재 결론

현재 bounded 검증/최적화 결과 기준으로는:

- **incumbent를 유지**
- **weight-only challenger 반려**
- **anchored four-sleeve challenger 반려**

즉, 포트폴리오 자체는 “지금 당장 바꿀 이유는 없다”가 결론이다.  
하지만 이것이 곧바로 **실거래 준비 완료**를 뜻하지는 않는다.

---

## 2. 상태 요약

| 항목 | 현재 상태 | 판단 |
|---|---|---|
| 포트폴리오 검증 | latest-tail 검증 완료 | 녹색 |
| validation-only 튜닝 규율 | 유지됨 | 녹색 |
| 메모리 8 GiB 제한 | 준수됨 | 녹색 |
| execution safety | 기본 안전장치 존재 | 황색 |
| slippage model | 백테스트용은 존재, 실거래용 보강 필요 | 적색 |
| kill-switch | 소프트웨어 레벨 일부 존재, 운영 레벨 보강 필요 | 황색 |
| monitoring / alerting | heartbeat / reconciliation / dashboard 존재 | 황색 |
| dry-run / paper-trading | paper/testnet 구성이 존재 | 황색 |
| 실거래 즉시 전환 | 추가 운영 검증 전에는 비권장 | 적색 |

---

## 3. execution safety 점검

### 현재 확인된 것

1. **주문 단위 리스크 제한**
   - `src/lumina_quant/risk_manager.py`
   - 체크 항목:
     - `MAX_ORDER_VALUE`
     - `MAX_INTRADAY_DRAWDOWN_PCT`
     - `MAX_ROLLING_LOSS_PCT_1H`
     - `MAX_SYMBOL_EXPOSURE_PCT`
     - `MAX_TOTAL_MARGIN_PCT`

2. **freeze / flatten 로직**
   - `src/lumina_quant/risk_manager.py`
   - `src/lumina_quant/live/trader.py`
   - 기능:
     - 리스크 breach 시 신규 진입 freeze
     - reduce-only는 freeze 중에도 허용
     - 조건 충족 시 flatten-all 큐잉 가능

3. **stale data 차단**
   - `src/lumina_quant/live/trader.py`
   - materialized staleness threshold 초과 시:
     - 신규 의사결정 차단
     - 경고 발송
     - fresh window 2회 확인 후 복구

4. **order reconciliation**
   - `src/lumina_quant/live/execution_live.py`
   - `src/lumina_quant/live/recovery_reconciliation.py`
   - unknown open order rehydrate / timeout reconciliation / snapshot readiness 처리 존재

5. **실수 방지 플래그**
   - `config.yaml`
   - `configs/profiles/paper.yaml`
   - `configs/profiles/real.yaml`
   - `require_real_enable_flag: true`
   - `live.mode: "paper"`
   - `testnet: true`

### 아직 부족한 것

1. **운영자 강제 중지(runbook) 문서화 부족**
   - “어떤 명령으로 즉시 stop/freeze/flatten 할지”가 문서로 고정되어 있지 않음

2. **실거래 전용 pre-trade execution gate 부족**
   - 거래소 상태 / spread 급증 / slippage 급증 / stale orderbook 상황에서
     주문 제출 전 차단하는 명시적 정책 문서가 부족

3. **실패 모드별 대응표 부족**
   - 429 / websocket disconnect / reconciliation drift / stale data / partial fill / timeout
     각각에 대한 운영 대응표를 추가해야 함

### 실거래 전 필수 체크

- [ ] `live.mode = paper`, `testnet = true` 상태에서 최소 1개 세션 dry run 완료
- [ ] `require_real_enable_flag = true` 유지
- [ ] 리스크 breach 시 freeze / flatten 경로를 paper 환경에서 재현 테스트
- [ ] order timeout 이후 reconciliation 이벤트가 정상 남는지 확인
- [ ] stale data block → recovery 경로 확인
- [ ] duplicate order/client id 방지 동작 확인

---

## 4. slippage model 점검

### 현재 확인된 것

1. **백테스트용 slippage / spread / latency model 존재**
   - `src/lumina_quant/backtesting/execution_sim.py`
   - 포함:
     - slippage
     - spread
     - commission
     - latency
     - liquidity cap

2. **설정값 존재**
   - `config.yaml`
   - `execution.slippage_rate`
   - `backtest.slippage_rate`

### 핵심 리스크

현재 slippage model은 **백테스트/시뮬레이션 관점**에서 충분하지만,
실거래 전환에는 다음이 더 필요하다.

- 실제 주문 체결 슬리피지의 실측값 저장
- symbol / regime / volatility bucket별 슬리피지 분해
- timeout / partial fill / cancel-replace 비용 반영
- paper/testnet fill과 실제 시장 fill의 괴리 추적

### 실거래 전 필수 체크

- [ ] 최소 2주 paper/testnet 체결 로그 수집
- [ ] realized slippage bps 분포 산출
- [ ] volatility / spread regime별 슬리피지 표 작성
- [ ] backtest slippage 가정과 paper 결과 차이가 허용 범위인지 확인
- [ ] slippage 급증 시 주문 차단 기준 정의

---

## 5. kill-switch 점검

### 현재 확인된 것

- trade freeze event logging 존재
- flatten-all trigger 존재
- stale data fail-fast / hard halt 경로 존재
- reconciliation drift risk event logging 존재

관련 코드:

- `src/lumina_quant/risk_manager.py`
- `src/lumina_quant/live/trader.py`

### 아직 부족한 것

실거래 운영에서 필요한 kill-switch는 두 층이어야 한다.

1. **소프트웨어 kill-switch**
   - 현재 일부 존재
2. **운영자 kill-switch**
   - 지금은 문서/절차가 약함

### 실거래 전 필수 체크

- [ ] 운영자 수동 stop 절차 문서화
- [ ] flatten-all 실행 절차 문서화
- [ ] “신규 진입만 차단” / “포지션 강제 청산” 두 모드 분리
- [ ] kill-switch 발동 시 알림 채널 확인
- [ ] kill-switch 후 재시작 절차 문서화

---

## 6. monitoring / alerting 점검

### 현재 확인된 것

1. **heartbeat**
   - `src/lumina_quant/live/trader.py`
   - `src/lumina_quant/postgres_state.py`
   - `uv run lq dashboard --run`

2. **reconciliation drift / stale feed / risk events**
   - audit store / dashboard / notifier 경로 존재

3. **dashboard**
   - `apps/dashboard_web`

### 아직 부족한 것

- 알림 severity 기준 문서
- pager 수준 alert와 info alert 구분
- 운영자 on-call 체크리스트
- paper / real 환경 공통 모니터링 KPI 정의

### 실거래 전 필수 KPI

- [ ] heartbeat 누락률
- [ ] reconciliation drift 이벤트 수
- [ ] stale data alert 횟수
- [ ] order timeout 횟수
- [ ] partial fill 비율
- [ ] cancel 비율
- [ ] realized slippage bps
- [ ] realized turnover / fee drag

---

## 7. dry-run / paper-trading 점검

### 현재 확인된 것

- `config.yaml` 기본값이 `live.mode: "paper"`
- `configs/profiles/paper.yaml` 존재
- `configs/profiles/real.yaml` 존재
- `src/lumina_quant/live/shadow_live_runner.py` 존재

### 의미

기본 안전장치는 이미 있다.  
하지만 **“paper로 얼마나 오래, 어떤 KPI를 만족해야 real로 가는가”**는
아직 운영 문서로 고정할 필요가 있다.

---

## 8. Go / No-Go 기준

### Go로 바꾸기 전 필수

- [ ] latest-tail portfolio validation 재실행
- [ ] paper/testnet 최소 관찰 기간 충족
- [ ] realized slippage 허용 범위 충족
- [ ] timeout / reconciliation drift / stale feed 이벤트가 통제 가능 수준
- [ ] operator kill-switch 리허설 완료
- [ ] dashboard / alert routing 확인 완료
- [ ] 최종 incumbent 유지/승격 결정 문서 최신화

### 현재 시점 판정

**No-Go for real trading**

이유:
- 포트폴리오 검증은 충분히 진행됐지만
- execution safety / slippage calibration / operator kill-switch / monitoring runbook / paper-trading evidence가
  아직 “실거래 전환 가능” 수준으로 닫히지 않았음

---

## 9. 바로 다음 액션

1. `02-paper-trading-readiness.md` 기준으로 paper 운영 절차 고정
2. operator kill-switch / alert runbook 확정
3. realized slippage 수집 체계 확정
4. 필요 시 `03-aggressive-sleeve-local-retune-plan.md`로 추가 alpha 탐색

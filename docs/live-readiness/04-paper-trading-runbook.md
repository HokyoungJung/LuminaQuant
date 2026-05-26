# Paper Trading 실행 Runbook

작성일: 2026-03-19  
대상 저장소: `LuminaQuant`

## 목적

이 문서는 LuminaQuant를 **paper / testnet 모드**로 실제 실행하기 위한
운영 절차를 정리한다.

핵심 목표:

1. 전략 의사결정 루프가 안정적으로 도는지 확인
2. 주문 제출 / timeout / cancel / reconciliation 경로를 확인
3. stale data / heartbeat / drift 이벤트를 운영 가능 수준으로 관찰
4. real 전환 전 필요한 evidence를 축적

---

## 0. 기본 원칙

- 현재 live candidate는 **incumbent 유지**
- 실거래 전에는 항상 **paper/testnet 먼저**
- `require_real_enable_flag`는 유지
- kill-switch / stop-file 경로를 준비하지 않은 상태로 장시간 운영 금지

관련 참고:
- `docs/live-readiness/01-live-trading-checklist.md`
- `docs/live-readiness/02-paper-trading-readiness.md`
- `docs/TRADING_MANUAL.md`

---

## 1. 사전 확인

## 1.1 필수 설정

권장 사전 점검 스크립트:

```bash
cd /home/hoky/Quants-agent/LuminaQuant
uv run python scripts/ops/live_readiness_preflight.py
```

확인 파일:
- `config.yaml`
- `configs/profiles/paper.yaml`

필수 상태:
- `live.mode = paper`
- `live.testnet = true`
- `live.require_real_enable_flag = true`
- PostgreSQL DSN 사용 가능 (`storage.postgres_dsn` 또는 `LQ_POSTGRES_DSN`)

권장 확인:
- API key가 testnet / paper 용도인지
- market data source / order state source가 현재 운영 환경과 맞는지

---

## 1.2 최신 포트폴리오/검증 상태

최신 상태 확인:

```bash
cd /home/hoky/Quants-agent/LuminaQuant

uv run python - <<'PY'
import json, pathlib
base = pathlib.Path('var/reports/exact_window_backtests/followup_status')
refresh = json.loads((base/'final_portfolio_validation_data_refresh_latest.json').read_text())
decision = json.loads((base/'portfolio_live_readiness_decision_latest.json').read_text())
print('refresh_cutoff_utc=', refresh['collection_cutoff_utc'])
print('decision=', decision['decision'])
PY
```

pass 기준:
- refresh cutoff가 최근이어야 함
- decision이 `keep_incumbent` 또는 명시적 promoted candidate여야 함

---

## 1.3 테스트

실행 전 최소 테스트:

```bash
cd /home/hoky/Quants-agent/LuminaQuant

uv run pytest \
  tests/test_build_portfolio_exact_window_freeze.py \
  tests/test_search_portfolio_four_sleeve_anchored.py \
  tests/test_run_portfolio_optimization_script.py \
  tests/test_validate_saved_incumbent_portfolio_script.py \
  tests/test_refresh_final_portfolio_validation_data_script.py \
  tests/test_collect_binance_aggtrades_raw.py \
  tests/test_materialize_from_raw.py
```

---

## 2. 실행 명령

## 2.1 기본 live CLI 확인

```bash
cd /home/hoky/Quants-agent/LuminaQuant
uv run lq live --help
```

주요 옵션:
- `--transport {poll,ws}`
- `--enable-live-real`
- `--strategy`
- `--selection-file`
- `--run-id`
- `--stop-file`

---

## 2.2 paper launch (권장 시작점)

```bash
cd /home/hoky/Quants-agent/LuminaQuant

uv run lq live \
  --transport poll \
  --stop-file /tmp/lq-paper.stop \
  --run-id paper-$(date -u +%Y%m%dT%H%M%SZ)
```

설명:
- `--stop-file`로 graceful shutdown 가능
- `--run-id`를 남겨 audit / dashboard correlation에 사용

정지:

```bash
touch /tmp/lq-paper.stop
```

또는 helper 사용:

```bash
cd /home/hoky/Quants-agent/LuminaQuant
uv run python scripts/ops/request_live_stop.py --stop-file /tmp/lq-paper.stop
```

더 쉬운 래퍼:

```bash
cd /home/hoky/Quants-agent/LuminaQuant

# 제어된 1회 실행
bash scripts/ops/start_live_session.sh --dsn 'postgresql:///luminaquant'

# 런타임 크래시 시 자동 재시작까지 포함
bash run_bot.sh --dsn 'postgresql:///luminaquant'

# 중지
bash scripts/ops/stop_live_session.sh
```

셸 함수 설치:

```bash
cd /home/hoky/Quants-agent/LuminaQuant
bash scripts/ops/install_shell_aliases.sh
source ~/.bashrc

lq-paper-on
lq-paper-off
```

---

## 2.3 Alpha Zoo Optuna hybrid paper/testnet handoff

최신 integer-leverage Optuna hybrid는 `AlphaZooOptunaHybridLiveStrategy`로
paper/testnet에서만 실행 가능한 handoff를 만든다. 실제 real-money 허가는 아니다.

```bash
cd /home/hoky/Quants-agent/LuminaQuant
uv run python scripts/ops/write_alpha_zoo_optuna_hybrid_live_decision.py --check-only
uv run python scripts/ops/write_alpha_zoo_optuna_hybrid_live_decision.py
```

기본 출력:
- `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_integer_leverage_optuna_hybrid_decision_20260524/paper_testnet_live_decision_latest.json`
- `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_integer_leverage_optuna_hybrid_decision_20260524/paper_testnet_live_decision_latest.md`

필수 불변식:
- `strategy_name=AlphaZooOptunaHybridLiveStrategy`
- `selected_profile_id=hybrid_v3_5_optuna_three_profile_blend`
- `paper_testnet_only=true`
- `ready_for_real=false`
- `real_money_execution=false`
- `real_execution_allowed=false`
- `research_primary_round_trip_cost_bps=10.0`
- `target_allocation_mode=notional_fraction`
- sizing source is `SignalEvent.metadata.target_allocation`; the global
  decision artifact keeps `target_allocation=0.0` so missing strategy metadata
  fails closed instead of opening a fallback allocation.
- watch symbols include `BTC/USDT`, `ETH/USDT`, `SOL/USDT`, `BNB/USDT`, `TRX/USDT`

preflight는 real mode에서 artifact veto를 유지해야 하고, paper/testnet 모드에서만
operator가 start/live dry-run 경로로 이어갈 수 있다.

현재 알려진 단점/차단 사유:
- real-money 차단은 의도된 상태다. artifact와 readiness policy가
  `ready_for_real=false`, `real_money_execution=false`,
  `real_execution_allowed=false`를 요구대로 유지한다.
- 아직 exchange paper/testnet 체결 telemetry가 없다. 실측 BBO spread, fee,
  slippage, reject, timeout, cancel, partial fill, 포지션 reconciliation drift를
  관찰하기 전에는 real 승격 금지.
- `10bps`는 replay와 gate에 박힌 round-trip friction proxy이지 live 실측 비용이
  아니다. paper/testnet에서 realized all-in round-trip cost가 10bps 이하로 안정되는지
  별도 확인해야 한다.
- decision artifact의 전역 `target_allocation=0.0`은 fail-closed 장치다. 실제 주문
  sizing은 `SignalEvent.metadata.target_allocation`을 따라야 하므로 paper/testnet에서
  signal metadata → order notional parity를 반드시 검증해야 한다.
- 선택된 v3.5 Optuna blend는 aggressive source profile 의존도가 높고 validation MDD가
  relaxed 20% 라벨 근처다. 독립 alpha라기보다 고위험 source profile을 완화한 allocator에
  가깝다.
- completed 1h/2h/4h bar 기반 alpha decision은 유지한다. 다만 최신 adapter는
  paper/local simulation용 intrabar component EXIT guard, paper/testnet 전용
  Binance USD-M conditional algo protective order(`POST /fapi/v1/algoOrder`), 그리고
  microstructure telemetry contract를 포함한다. real-money protective order 승인은
  여전히 별도 telemetry review 전까지 금지다.

intrabar / microstructure 추가 계약:
- strategy entry signal은 `component_id`, `stop_loss`, `trailing_percent`(chandelier 계열),
  `intrabar_protection`, `microstructure_telemetry_required` metadata를 포함한다.
- paper/local simulation에서는 1m/5m/source-timeframe risk frame을 보고 stop breach 시
  component-level `EXIT` signal을 낼 수 있다.
- paper/testnet exchange path는 entry fill 이후 같은 component quantity로
  `STOP`/`TAKE_PROFIT` conditional limit algo order를 제출할 수 있다.
- live 주문은 limit-first다. entry/short-entry/reduce-only exit/risk-flatten 주문은
  기본 `LMT`, `one_tick_worse` 가격 정책(BUY는 기준가 +1 tick, SELL은 기준가 -1 tick),
  `GTC` time-in-force를 쓴다.
- 시장가 또는 `STOP_MARKET`/`TAKE_PROFIT_MARKET` 스타일은 옵션으로만 남긴다. 사용하려면
  `live.default_order_type=MKT`, `live.allow_market_orders=true`, 또는
  `live.protective_order_style=market`과 별도 review가 필요하다.
- algo-order request payload는 Binance-supported field allowlist만 통과시키고,
  parent/protection telemetry key는 reconciliation용 local record에만 남긴다.
- selected frozen source asset 적용성은 `ETHUSDT`, `SOLUSDT`, `TRXUSDT`에 대해
  guard/protective-order tests로 확인했다.
- real-money exchange-side protective order는 아직 별도 승인된 경로가 아니다. 실제 real
  mode는 paper/testnet fill telemetry와 별도 real-readiness review가 쌓일 때까지 계속
  veto한다.
- queue priority는 거래소가 정확한 내 대기열 위치를 주지 않으므로 BBO/depth/fill-latency
  proxy로만 판단한다.

---

## 2.4 shadow 비교가 필요한 경우

`src/lumina_quant/live/shadow_live_runner.py`가 있으므로,
baseline vs candidate divergence를 보는 shadow 경로를 운영 가능하게 설계할 수 있다.

현재 문서 기준:
- shadow 비교는 **강력 권장**
- 다만 별도 운영 스크립트/entrypoint가 아직 고정된 것은 아님

따라서 paper 단계에서는:
- incumbent 단독 paper 실행
- challenger는 별도 replay/shadow 실험
을 권장한다.

---

## 3. 런타임 중 모니터링 항목

반드시 볼 것:
- heartbeat 정상 기록 여부
- stale data alert 여부
- reconciliation drift 여부
- order timeout 횟수
- cancel/partial fill 비율
- paper PnL / drawdown / turnover

관련 코드:
- `src/lumina_quant/live/trader.py`
- `src/lumina_quant/live/execution_live.py`
- `apps/dashboard_web`

---

## 4. 매일 확인 체크리스트

- [ ] latest refresh / validation artifact timestamp 확인
- [ ] 현재 candidate decision 확인
- [ ] heartbeat 이상 없음
- [ ] stale data alert 폭증 없음
- [ ] reconciliation drift 누적 없음
- [ ] timeout / cancel 패턴 이상 없음
- [ ] paper fill / slippage 로그 수집 중

---

## 5. pass / fail 기준

## pass

- [ ] 연속 운영 중 크리티컬 예외 없음
- [ ] stale data block / recovery 재현 가능
- [ ] reconciliation drift 누적되지 않음
- [ ] timeout / cancel / partial fill이 운영 가능 수준
- [ ] operator가 stop-file로 정상 종료 가능

## fail

다음 중 하나라도 반복되면 real 전환 금지:

- heartbeat 누락
- stale data alert 빈발
- reconciliation drift 반복
- timeout / cancel 폭증
- paper fill 품질이 backtest 가정과 크게 다름

---

## 6. 실거래 전환 전 최소 관찰 기간

권장:
- **최소 2주 연속 paper/testnet**

강한 권장:
- **3~4주**, 정상장 + 변동성 확대 구간 포함

---

## 7. 현재 기준 권고

지금 당장은:
- **paper launch는 가능**
- **real launch는 아직 금지**

즉, 이 runbook의 목적은 “바로 실거래”가 아니라
**paper evidence를 쌓아 real gate를 닫는 것**이다.

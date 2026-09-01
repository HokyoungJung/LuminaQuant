# 공격적 Sleeve-Local Retune 계획

작성일: 2026-03-19  
대상 저장소: `LuminaQuant`

## 목적

현재 bounded live-readiness search에서 incumbent를 이길 challenger가 없었으므로,
추가 alpha가 정말 필요할 때만 실행할 **공격적 sleeve-local retune 계획**을 정의한다.

이 계획은 “지금 당장 실행”이 아니라, **별도 승인 후 실행하는 후속 플랜**이다.

---

## 1. 왜 추가 retune이 필요한가

이번 결과:

- **weight-only challenger**: incumbent보다 열위
- **anchored four-sleeve challenger**: raw return은 높지만 Sharpe / MDD가 너무 나쁨

즉, 현재 남은 개선 여지는 다음 둘 중 하나다.

1. incumbent sleeves 자체를 더 잘 고르는 것
2. 기존 sleeves 주변의 parameter neighborhood를 더 엄밀하게 좁혀보는 것

---

## 2. 절대 지켜야 할 규칙

### Split 규칙
- tuning / ranking / promotion은 **train + val only**
- locked OOS 시작은 항상:
  - `2026-02-01T00:00:00Z`

### Memory 규칙
- 전체 세션/프로세스 합산 **8 GiB 미만**
- heavy lane은 항상 **1개만 실행**

### Field denylist

baseline freeze / ranking / candidate promotion에서 다음 필드는 입력으로 사용 금지:

- `report_split`
- `report_sharpe`
- `report_return`
- `oos_*`
- OOS 기반 helper / committee / promoted 류 메타 필드

---

## 3. 우선순위

## Priority A — CompositeTrend 30m local retune

이유:
- incumbent 내부에서 validation fit quality가 강함
- latest-tail에서도 완전히 붕괴하지 않음
- portfolio diversifier 역할 유지 가능

탐색 방향:
- threshold / allow_short / regime / crowding gate / stop logic 주변 neighborhood
- 기존 stable_ls_highconv 계열을 중심으로 local retune

성공 기준:
- incumbent 대비 Sharpe 개선
- MDD 악화 제한
- turnover 과도 증가 금지

---

## Priority B — TopCapTSMom 1h local retune

이유:
- latest-tail에서 절대수익/Sharpe가 중간 수준
- cross-sectional sleeve로서 구조적 역할이 있음

탐색 방향:
- balanced / defensive / rebalance horizon / volatility filter / breadth gate 주변

성공 기준:
- Sharpe 개선
- cross-sectional sleeve로서 diversification 유지

---

## Priority C — PairSpread 1h local retune

이유:
- 현재 incumbent의 strongest sleeve
- 잘못 건드리면 오히려 degrade될 가능성이 큼

전략:
- broad retune 금지
- 아주 좁은 neighborhood만 허용
- 개선 없으면 즉시 incumbent original 유지

탐색 항목 예:
- entry_z / exit_z / hedge_window / lookback_window / stop_z / max_hold_bars

---

## Priority D — RegimeBreakout / RollingBreakout replacement lane

이유:
- anchored four-sleeve 결과상 rolling breakout 계열이 OOS risk-adjusted 성능을 심하게 훼손

전략:
- breakout sleeve는 “추가 채택”보다 “교체 후보” 관점에서 다룰 것
- validation-only exact-window freeze bundle을 다시 기준으로 비교

---

## 4. 실행 단계

### Step 1 — latest-tail baseline 동결
- incumbent latest-tail validation rerun
- incumbent refresh artifact rerun
- stale 기준:
  - refresh cutoff가 30분 이상 오래됐거나
  - finalist set이 바뀌면 rerun

### Step 2 — exact-window freeze bundle 재생성
- 항상 retune 진입 전에 재생성
- 이 번들을 sleeve-local retune의 기준 surface로 사용

### Step 3 — sleeve별 local retune
- A → B → C 순서
- 한 번에 한 sleeve만
- 각 sleeve는 별도 artifact와 memory log를 남김

### Step 4 — retuned sleeves 조합 후 weight optimization
- shortlisted sleeves로만 portfolio optimizer 재실행
- validation-only selection 유지

### Step 5 — finalist latest-tail validation
- incumbent vs retuned candidate 동등 조건 비교
- locked OOS로만 판정

---

## 5. 중단 조건

다음 중 하나면 즉시 중단:

- peak RSS가 위험 구간 접근
- stale / corrupt artifact 반복
- validation-only 규율 위반 가능성
- sleeve-local 개선이 OOS risk-adjusted deterioration로 이어짐
- incumbent보다 drawdown이 명확히 악화

---

## 6. 승격 기준

후속 retune 결과가 아래를 모두 만족해야만 incumbent를 교체할 수 있다.

- latest-tail locked OOS Sharpe 개선
- latest-tail locked OOS MDD 비악화 또는 매우 작은 수준의 악화만 허용
- turnover / cost stress 허용 범위
- parameter-drift / cost-stress / walk-forward stability 유지
- validation-only selection 근거 명확

---

## 7. 권장 산출물

각 sleeve retune마다:

- retune summary json / md
- memory summary json
- exact-window candidate-detail artifact
- finalist comparison json / md

최종:

- live-readiness decision update
- incumbent 유지 vs 교체 판정 업데이트

---

## 8. 현재 추천

지금 시점에서는:

- **즉시 추가 retune 실행보다**
- 먼저 **paper-trading / execution safety / slippage calibration**을 진행하는 편이 더 합리적이다.

추가 retune은 다음 조건일 때만 권장:

- 실거래 전 alpha 추가 확보가 반드시 필요
- paper 운영 안정성이 이미 확보됨
- 8 GiB 메모리 규율 하에서 순차 실행이 가능함

---

## 9. 한 줄 결론

**지금은 incumbent 유지가 맞고, 추가 retune은 “실거래 운영 안정성 확보 이후”에만 제한적으로 다시 여는 것이 맞다.**

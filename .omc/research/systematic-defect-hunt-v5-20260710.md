# Systematic Defect Hunt v5 — 종합 감사 리포트 (2026-07-10)

8클래스 병렬 사냥(6/8 헌터 완주, 2클래스 미완) + 발견별 적대적 재검증(실행 프로브 포함).
결과는 워크플로 저널(`wf_24155a1b-20b/journal.jsonl`)에 원본 증거로 보존.
브랜치: `fix/silent-proxy-substitution-v5`.

## 요약

| # | 결함 | 심각도 | 검증 | 상태 |
|---|------|--------|------|------|
| 1 | Silent generic-fallback substitution — 84/111 클래스(신규 research_only 33개 전부, 2833/3674 후보)가 엔진에서 실행조차 안 되고 단일 64-bar 모멘텀 프록시로 채점 | **CRITICAL** | 확정(프로브) | **FIXED** (e10262b) |
| 2 | LeadLagSpillover 핸들러: 전표본 `_safe_std` 시그마 정규화(미래 변동성 누출) + `np.roll` 머리쪽 랩어라운드 | HIGH | 확정 | **FIXED** — expanding sigma(warmup 32) + zero-fill shift, prefix-안정성 테스트 |
| 3 | Zero-alloc → 엔진 기본 ~10% 리사이즈: `_target_metadata`가 alloc≤0이면 `target_allocation` 키 자체를 생략 → 다운스트림 config 기본 사이징. slow_leadlag는 선택시 vol-gate 부재로 도달 가능(실증) | HIGH | 확정(프로브) | **FIXED** — slow_leadlag vol-gate+`_enter` 가드, crash-gate 선택 필터; 13개 `_emit_targets` 템플릿 사본 일괄 게이트(worker-eg/vt4 진행) |
| 4 | 비용 스트레스 편도/왕복 불일치: efficiency-repair `_stress_metrics`가 왕복 bps를 편도 이벤트당 전액 부과(베이스 심은 `/2`) → 스트레스 2배 과징 | HIGH | 확정 | **FIXED** — `/2.0` 정합 + 왕복 컨벤션 테스트 |
| 5 | Vol-target 스로틀 불활성(v4 잔여 3레인): flow_share_rotation·diversified_multifactor·cross_sectional_funding_momentum_carry — 연율 target_vol vs per-bar vol 비교로 scalar 1.0 고정 | MEDIUM | 확정 | **FIXED** (worker-vt4, v4 annualize 패턴) |
| 6 | information_discreteness 종가 deque maxlen이 `_min_history` 하한 미포함 → 1d fip_4wk_p33 셀 영구 불활성(43 < 70) | MEDIUM | 확정(프로브) | **FIXED** — `history = max(..., self._min_history) + 8` + 슬라이스 전셀 회귀 테스트 |
| 7 | offsession_tugofwar ambiguous-hour {13,20} 과소포함: EST 겨울 pre-open(14:00–14:30 UTC)이 CASH로 분류 → D_d 오염, TOW 하향 감쇠 (연중 ~35%) | MEDIUM | 확정(프로브 2회) | **FIXED** — {13,14,20}으로 확장 + 분류 테스트. 미해결 잔여: US 휴일(~10일/년) 무처리 — 아래 '문서화된 잔여' |
| 8 | seasonal_xs_persistence 갭 스패닝: 5주 갭+2배 가격이 log(2)를 단일 '주간' 수익으로 한 버킷에 기장, K분기(~6분기) 오염 | MEDIUM | 확정(프로브 2회, +0.107 인플레 정량) | **FIXED** — `prev_week_key` 그리드 인접성 가드(상태 직렬화 포함) + 갭 테스트 |

## 검증 결과 '정상' 판정 (수정 불필요)

- **HOLD 시맨틱스**: 3개 소비 경로(이벤트 엔진·research 프록시·broad69 WF) 모두 미방출=보유 확인. 엔진은 방출 사이에 포지션을 청산하지 않음 — 이전 postmortem의 "주간 케이던스 churn" 가설 반증.
- **동일-바 체결 look-ahead 없음**: broad69 `signal[i] → next_return[i]` 정합 확인.
- **VolManagedRiskOverlay**: 결함 아님.

## 문서화된 잔여 (수정 보류, 데이터-PC에서 중요도 평가)

- offsession_tugofwar **US 휴일/반일 무처리**: ~10 휴일 weekday/년이 전부 무앵커 'cash' 수익으로 기장. 수정안: 정적 휴일 캘린더 or 일별 cash-분산 유효성 프로브. LOW-MEDIUM, 별도 사이클.
- seasonal_xs_persistence **bucket-12 horizon 비정규화**(6/7/8일 혼합, 드리프트 잔존 ~3–9% of weekly drift): 설계 문서화된 캡; LOW.
- seasonal_xs_persistence **flush 지연**(bucket-0 결정이 1분기 지연된 store 사용, 보수적 방향): LOW.
- 1d fip_8wk_p50 deque 마진 1-bar(71 vs 70): 동작하나 취약 — Fix 6이 하한 포함으로 함께 해소.

## 미완 사냥 클래스 (헌터 사망, 세션 한도)

- cadence-fragile-clocks (리밸런스/틱 카운터의 데이터 갭 취약성)
- state-roundtrip-drift (get/set_state 왕복 후 결정 드리프트)

→ 후속 사이클에서 재사냥 권장.

## 데이터-PC 재실행 노트 (중요)

Finding #1로 인해 **이전 워크포워드의 research_only 레인 리젝트는 레인이 아니라 프록시를 측정한 것**.
`configs/profiles/research.yaml`에 `route_unmapped_registered_strategies: true`가 이미 켜져 있으므로,
main 머지 후 데이터-PC에서 워크포워드를 재실행하면 레인 실제 코드가 registry-simulator 경로로 평가됨
(`evaluation_mode` 메타로 경로 감사 가능: `handler` / `registry_simulator` / `generic_fallback_proxy`).
비용 스트레스 게이트(#4)도 완화 방향으로 변했으므로 효율-리페어 선별 결과가 달라질 수 있음.

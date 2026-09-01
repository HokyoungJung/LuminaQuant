# Session handoff — carry/trend production retune resume

작성 시각(UTC): 2026-04-18T10:00:00Z
repo: `/home/hoky/Quants-agent/LuminaQuant`
branch: `private-main`
upstream: `private/main`

## 1. 현재 코드/의사결정 상태

### 최신 pushed commit
- `aeacf64` — Reduce candidate-research hot-path waste in the carry/trend lane

현재 `private/main` 과 로컬 `private-main` 는 동기화된 상태였음 (`ahead/behind = 0/0`).

### 이미 반영된 주요 결과
- `production_guarded_portfolio` 생성 및 artifact 저장
- `production_guarded_mode` 가 다음 의사결정 surface에 편입됨:
  - `portfolio_max_performance_decision`
  - `portfolio_operating_switch`
  - `portfolio_operating_playbook`
  - `portfolio_master_scoreboard`
  - `portfolio_live_readiness_decision`
- 현재 live-facing default는 `production_guarded_mode`
- research winner는 여전히 `incumbent_autoresearch_static_blend`
- `production_guarded` 를 hybrid 내부 active sleeve로 넣어본 결과, hybrid 성능 개선은 없었음

## 2. carry/trend alpha lane에서 이미 한 최적화

### 완료된 최적화
1. candidate-research hot-path에 aligned cache 추가
   - 같은 symbol/timeframe bundle 조합은 재사용
2. `research_runner` 의 rolling helper 일부 벡터화
   - `_rolling_z`
   - `_vol_ratio_series`
   - `_rolling_volatility_series`
3. production-safe carry/trend 후보 manifest 추가
   - `scripts/research/build_carry_trend_production_manifest.py`
4. focused retune runner 추가
   - `scripts/research/run_carry_trend_production_retune.py`

### 새 production-ready carry/trend 후보
- `carry_trend_factor_rotation_1h_production_lo_guarded_48_12_0.250`
- `carry_trend_factor_rotation_4h_production_lo_trendcarry_24_6_0.200`

manifest 경로:
- `var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped/carry_trend_production_retune_current/carry_trend_production_manifest_latest.json`

## 3. 현재 런 상태 (재부팅 전 참고용)

재부팅 전 마지막 확인 기준:
- 프로세스: `run_research_candidates.py`
- CPU: 약 88%
- RSS: 약 1.1~1.2 GiB
- output artifact: 아직 생성 안 됨
- log: 거의 비어 있음

### 해석
- OOM은 아님
- 멈춘 것도 아님
- 그러나 output을 끝에서만 쓰고, 계산이 오래 걸려 체감상 비정상적으로 느렸음
- 특히 carry/trend lane은 candidate 수는 적어도, top10 crypto + long split + factor rotation 연산 때문에 CPU-bound 상태로 오래 머무름

## 4. 다음 세션의 최우선 작업

### 목표
carry/trend focused retune lane이 **중간 진행상황을 남기고**, 더 빨리 결과를 반환하도록 추가 최적화한 뒤, 새 alpha 후보를 실제로 뽑는다.

### 우선순위
1. `run_research_candidates.py` / `research_entrypoints.py` 에 **progress checkpoint / partial artifact** 추가
   - candidate 하나 끝날 때마다 또는 stage1/2 단위로 JSON/MD/log 남기기
2. carry/trend factor rotation 추가 최적화
   - `_apply_carry_trend_factor_rotation_strategy` 내부의 per-bar / per-rebalance dict 작업 더 줄이기
   - 가능한 사전 계산(모멘텀/캐리/방어/크라우딩 score matrix)로 이동
3. 필요하면 fast-screen lane 추가
   - top5 (BTC/ETH/BNB/SOL/TRX) 우선
   - 이후 통과 시 top10 confirmation
4. 최적화 후 다시
   - `uv run python scripts/research/run_carry_trend_production_retune.py`
   실행
5. 결과 후보가 괜찮으면
   - `production_guarded_portfolio` 에 편입 가능한지 평가
   - 필요 시 decision graph 재갱신

## 5. 권장 새 세션 프롬프트

아래 중 하나를 그대로 붙여넣으면 됨.

### 간단 버전
`docs/session_handoff_20260418T100000Z_carry_trend_retune_resume.md`를 읽고 이어서 작업해. 우선 carry/trend retune 경로에 progress checkpoint를 추가하고, carry/trend factor rotation 연산을 더 최적화한 뒤, `uv run python scripts/research/run_carry_trend_production_retune.py`를 다시 실행해서 새 alpha 후보가 production_guarded_portfolio를 개선하는지 평가해.

### 자세한 버전
`docs/session_handoff_20260418T100000Z_carry_trend_retune_resume.md`와 `.omx/plans/prd-luminaquant-portfolio-superiority.md`를 먼저 읽어. 현재 `production_guarded_mode`가 live default이고, carry/trend production-safe alpha lane은 manifest/runner/일부 hot-path 최적화까지 끝난 상태다. 다음으로는 (1) `run_research_candidates`에 stage-progress artifact를 추가하고, (2) `_apply_carry_trend_factor_rotation_strategy` 경로를 더 최적화하고, (3) 저메모리 exact split으로 `run_carry_trend_production_retune.py`를 재실행해 결과 artifact를 생성하고, (4) 새 carry/trend 후보가 production_guarded_portfolio와 hybrid를 개선하는지 평가해. 총 메모리는 8GB 미만 유지.

## 6. 참고 파일
- `docs/session_handoff_20260418T100000Z_carry_trend_retune_resume.md`
- `.omx/plans/prd-luminaquant-portfolio-superiority.md`
- `.omx/plans/test-spec-luminaquant-portfolio-superiority.md`
- `scripts/research/build_carry_trend_production_manifest.py`
- `scripts/research/run_carry_trend_production_retune.py`
- `src/lumina_quant/strategy_factory/research_runner.py`
- `src/lumina_quant/strategy_factory/research_entrypoints.py`
- `src/lumina_quant/strategy_factory/research_stage_support.py`
- `src/lumina_quant/strategy_factory/research_stage_selection.py`
- `src/lumina_quant/strategy_factory/candidate_library.py`

# Session handoff — carry/trend full confirmation complete

작성 시각(UTC): 2026-04-18T15:59:07Z
repo: `/home/hoky/Quants-agent/LuminaQuant`
branch: `private-main`

## 1. 현재 상태 요약
- `production_guarded_mode` 는 여전히 live default로 유지해야 함.
- carry/trend production-safe alpha lane에 대해:
  - `run_research_candidates` stage-progress artifact 추가 완료
  - `_apply_carry_trend_factor_rotation_strategy` hot-path 최적화 완료
  - `CarryTrendFactorRotationStrategy` feature cache 연결 완료
  - low-memory exact split retune 실행 완료
- 결론: **새 carry/trend 후보는 full confirmation 기준에서도 `production_guarded_portfolio` / `hybrid`를 개선하지 못함.**

## 2. 이번 세션에서 반영된 핵심 코드 변경
### research progress / orchestration
- `scripts/run_research_candidates.py`
  - progress artifact 작성 추가
    - `candidate_research_progress_latest.json`
    - `candidate_research_progress_latest.md`
    - `candidate_research_progress_latest.log`
  - progress event
    - `coverage_rebuild_skipped`
    - `resource_load_started`
    - `resources_loaded`
    - `candidate_evaluated`
    - `stage1_ranked`
    - `stage2_selected`
    - `report_ready`
  - manifest candidate를 `--symbols` universe에 맞게 자동 축소하도록 수정
  - `--skip-coverage-rebuild` 추가

- `scripts/research/run_carry_trend_production_retune.py`
  - focused low-memory retune wrapper가 `--skip-coverage-rebuild` 사용하도록 변경

### carry/trend factor rotation hot-path
- `src/lumina_quant/strategy_factory/research_runner.py`
  - cross-sectional factor score matrix 사전 계산 추가
  - rebalance target selection을 index 기반으로 단순화
  - per-bar `dict`/`set` 생성 감소

### feature/resource load
- `src/lumina_quant/strategy_factory/research_resources.py`
  - `CarryTrendFactorRotationStrategy`도 feature cache를 로드하도록 추가
- `src/lumina_quant/market_data.py`
  - parquet/feature point load가 필요한 `date=*` partition만 스캔하도록 변경
  - full-range wildcard scan 낭비 감소

### progress snapshot identity 보강
- `src/lumina_quant/strategy_factory/research_stage_selection.py`
  - progress snapshot에 `candidate_id`, `name`, `strategy_class`, `family`, `strategy_timeframe` 보존

## 3. 테스트 상태
실행:
- `uv run pytest -q tests/test_run_research_candidates_script.py tests/test_run_carry_trend_production_retune_script.py tests/test_research_runner_feature_support.py::test_carry_trend_factor_rotation_strategy_prefers_uncrowded_trend_leaders tests/test_feature_point_upsert_schema.py tests/test_research_resources.py`

결과:
- **16 passed**

## 4. 생성된 핵심 artifact
### full top10 exact split (feature-enabled)
#### 1h full
- `var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped/carry_trend_production_retune_current/research_run_1h_guarded_full_features/candidate_research_latest.json`
- progress:
  - `.../candidate_research_progress_latest.json`
  - `.../candidate_research_progress_latest.md`
  - `.../candidate_research_progress_latest.log`

#### 4h full
- `var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped/carry_trend_production_retune_current/research_run_4h_trendcarry_full_features/candidate_research_latest.json`
- progress:
  - `.../candidate_research_progress_latest.json`
  - `.../candidate_research_progress_latest.md`
  - `.../candidate_research_progress_latest.log`

### screened portfolio rebuilds
- production guarded:
  - `var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped/portfolio_production_guarded_fullscreen_features_current/production_guarded_portfolio_latest.json`
- hybrid:
  - `var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped/portfolio_hybrid_online_fullscreen_features_current/hybrid_online_portfolio_latest.json`

## 5. 후보 평가 결과
### 1h candidate
- name: `carry_trend_factor_rotation_1h_production_lo_guarded_48_12_0.250`
- train: return `+0.161%`, sharpe `-0.2881`, maxDD `7.100%`, trades `378`
- val: return `-2.213%`, sharpe `-2.4598`, maxDD `4.890%`, trades `49`
- oos: return `-0.321%`, sharpe `-0.9313`, maxDD `1.573%`, trades `33`
- `pass=False`, `hard_reject=True`

### 4h candidate
- name: `carry_trend_factor_rotation_4h_production_lo_trendcarry_24_6_0.200`
- train: return `-0.162%`, sharpe `-0.3590`, maxDD `8.294%`, trades `204`
- val: return `-4.914%`, sharpe `-4.9981`, maxDD `6.718%`, trades `31`
- oos: return `-0.293%`, sharpe `-0.8575`, maxDD `2.516%`, trades `18`
- `pass=False`, `hard_reject=True`

## 6. portfolio 영향
### production_guarded 비교
#### current
- OOS return `0.3654%`
- OOS sharpe `1.6080`
- OOS maxDD `0.3184%`

#### fullscreen_features
- OOS return `0.1490%`
- OOS sharpe `0.6480`
- OOS maxDD `0.5202%`

결론:
- carry sleeve는 여전히 채택되지 않음 (`selected_name=None`)
- **현재 production_guarded보다 악화**

### hybrid 비교
#### current
- OOS return `0.1618%`
- OOS sharpe `0.7763`
- OOS maxDD `0.4897%`

#### fullscreen_features
- OOS return `0.0941%`
- OOS sharpe `0.3952`
- OOS maxDD `0.5445%`

결론:
- **hybrid도 개선되지 않음**

## 7. 메모리 관찰
- 실행 중 `ps` 기준 candidate process RSS는 대체로 `~1.0GB ~ 1.2GB`
- 8GB 계약은 넉넉하게 만족
- `/usr/bin/time -v` 최대 RSS는 wrapper 프로세스 기준이라 실제 worker peak를 제대로 반영하지 않음. 관측은 `ps` RSS가 더 신뢰할 만했음.

## 8. 중요한 운영 메모
- full manifest 2-candidate 통합 실행은 여전히 매우 오래 걸려 가시성이 떨어짐.
- **후속 세션에서는 후보별 단독 런으로 확인하는 편이 안전함**
  - `research_run_1h_guarded_full_features`
  - `research_run_4h_trendcarry_full_features`
- 이제 progress artifact가 있으므로 resource loading 이후/후보 평가 이후는 추적 가능함.
- 다만 `resource_load_started -> resources_loaded` 사이는 아직 coarse-grained.

## 9. 현재 판단
- carry/trend lane은 이번 가설 셋으로는 promotion 가치가 없음.
- **live default는 계속 `production_guarded_mode` 유지**.
- decision artifacts를 새로 바꿀 필요는 없음.

## 10. 다음 세션에서 할 일 (권장 순서)
### 옵션 A — 가장 자연스러운 다음 일
resource load 내부도 세분화 progress를 찍도록 개선
- `_load_bundle_cache`
- `_load_feature_cache`
- benchmark build
이 단계별로 symbol/timeframe progress를 남기기

### 옵션 B — 더 실용적인 다음 일
carry/trend를 멈추고, production-safe alpha lane을 다른 가설로 전환
- 예: pair / regime / overlay / ballast / committee 측 추가 개선 후보

### 옵션 C — carry/trend를 정말 더 보려면
- 새 factor mix 또는 symbol subset hypothesis를 바꿔서 재설계해야 함
- 현재 프로파일 그대로는 edge가 없음

## 11. 새 세션용 추천 프롬프트
### 추천 1 (진행상황 저장 + 다음 액션)
`docs/session_handoff_20260418T155907Z_carry_trend_full_confirmation.md`와 `.omx/plans/prd-luminaquant-portfolio-superiority.md`를 먼저 읽어. carry/trend full confirmation은 끝났고, 두 후보 모두 production_guarded/hybrid를 개선하지 못했다. live default는 계속 production_guarded_mode다. 다음으로는 resource load 내부에 finer-grained progress artifact를 추가해서 long-running research visibility를 높여줘.

### 추천 2 (새 alpha lane으로 피벗)
`docs/session_handoff_20260418T155907Z_carry_trend_full_confirmation.md`와 `.omx/plans/prd-luminaquant-portfolio-superiority.md`를 읽고 이어서 작업해. carry/trend lane은 이번 가설 셋으로는 edge가 없었으니 더 붙잡지 말고, production_guarded_portfolio를 실제로 개선할 가능성이 높은 다음 production-safe alpha lane을 하나 골라서 low-memory exact split 기준으로 설계/실행해.

# Research Note

## 2026-06-05 KST — 85-symbol clean dynamic v5, CI 실제 경로 검증, ranking/reporting repair

사용자 피드백에 따라 “CI 툴 검증”을 실제 GitHub Actions 기준으로 재확인했다. 직전 push `cf25437e`에서 private-ci는 성공했지만 public `ci` run `27019925528`이 `uv run ruff format --check .` 단계에서 실패했다. 실패 파일은 `scripts/research/run_alpha_zoo_69_asset_monthly_refit_walkforward.py`와 `scripts/research/run_alpha_zoo_69_asset_optuna_hybrid_refit.py`였고, repo-wide ruff format으로 수정했다.

전략 쪽에서는 두 가지 문제가 있었다.

1. `dynamic_conviction_switch:*_val_ret02_calmar80_gate_*_scaled` 후보가 validation-strength gate 실패 fold에서 cash row를 내지 않아 일부 월이 누락될 수 있었다. 이제 scaled gate 후보도 gate 실패 시 `cash_validation_strength_guard` row를 명시적으로 내므로 `fold_count=10`으로 평가된다.
2. aggregate/clean ranking이 comp보다 positive-fold 수를 먼저 정렬해서, 실제 최고 comp 후보가 clean top 표 밖으로 밀리는 리포팅/선정 문제가 있었다. 정렬을 `compounded_oos_return` 우선으로 바꿨고, no-loss 후보의 Sortino/PF/Omega는 0이 아니라 `unbounded` 플래그와 `∞` 표시로 보고한다.

Clean OOS 재평가 artifacts:

- v5 JSON: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_85_asset_dynamic_scaled_20260605/alpha_zoo_85_asset_dynamic_scaled_full_v5_20260605.json`
- v5 Markdown: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_85_asset_dynamic_scaled_20260605/alpha_zoo_85_asset_dynamic_scaled_full_v5_20260605.md`
- OOS schedule: 10 folds, `2025-09-01T00:00:00` → `2026-06-05T12:00:00` UTC, monthly day-1 refit, 2M validation, 10bps, 85/85 loaded symbols, 30m~1D.
- Selection discipline: train + validation only; OOS month is report-only; `nested_hybrid_dependency=false`; `uses_locked_oos_for_selection=false`; final weights are strict-efficiency leaf or cash, not hybrid-as-hybrid material.

Final clean comp ranking after repair:

| Rank | Candidate | OOS comp | Max bar MDD | Monthly eq MDD | Sharpe | Sortino/PF | Hit | Min OOS | Notes |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `dynamic_conviction_switch:t0.85_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd30_scaled` | `34.39%` | `27.69%` | `0.00%` | `1.12` | `∞/∞` | `3/10` | `0.00%` | highest clean comp, trades only strong validation months |
| 5 | `dynamic_conviction_switch:t0.85_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd20_scaled` | `29.65%` | `23.59%` | `0.00%` | `1.13` | `∞/∞` | `3/10` | `0.00%` | lower MDD, still selective/no losing OOS month |
| 9 | `dynamic_conviction_switch:t0.85_risk_capped_fallback_val_mdd30_scaled` | `27.92%` | `27.69%` | `5.75%` | `0.94` | `6.24/5.92` | `5/10` | `-3.18%` | more active but accepts small losing months |
| 13 | `dynamic_conviction_switch:t0.85_risk_capped_fallback_val_mdd20_scaled` | `23.41%` | `23.59%` | `5.75%` | `0.91` | `5.24/5.12` | `5/10` | `-3.18%` | safer active variant |

Full-family/higher-trial v4 was intentionally tested and **not selected**: best clean only `12.45%` comp with `16.97%` max MDD, because the larger candidate pool chased validation strength into OOS tail losses. The correct conclusion is not “add every family”; the better clean result is the stable v3/v5 family set plus validation-only MDD30 sizing and validation-strength cash gate.

Hard-stop status remains false versus the historical challenger (`+53.38%` comp / `18.8%` MDD) and robust-default 15% MDD hurdle. Therefore this is paper/shadow research evidence, not real-money approval. If using it for paper, rank 1 is the return-seeking selective candidate; rank 5 is the lower-MDD selective candidate; rank 9/13 are more active but accept losses.

CI/local verification recorded for this pass:

- `uv run ruff format --check .` — passed.
- `uv run ruff check .` — passed.
- Raw-first periodic preload CI subset — `78 passed`.
- Rust native checks for `rust_metrics`, `rust_rawfirst`, `rust_hybrid_optuna`, `rust_live_signals` — passed.
- Architecture gates: live data, market-window parity, native Binance — passed.
- `scripts/check_architecture.py`, `scripts/audit_hardcoded_params.py`, `scripts/verify_docs.py` — passed (`119 markdown files checked`, hardcoded audit `new=0`).
- Dashboard CI path: `npm install`, `npm run lint`, `npm run test`, `npm run typecheck`, `npm run build` — passed.
- GPU contract tests and auto runtime smoke — `24 passed`, strict Polars GPU smoke passed on detected NVIDIA GPU.
- Full pytest — `1619 passed in 87.51s`.
- Benchmark smoke + 8GB baseline — passed, peak RSS `186.61 MiB` < `7.2 GiB` budget.

Remote CI follow-up: after the first repair push, public `ci` passed ruff/dashboard/GPU but failed full pytest because `DEFAULT_BRIDGE_PROTOCOL_MANIFEST` pointed at ignored runtime state under `.omx/plans/`. The frozen bridge protocol manifest is now tracked at `configs/research/bridge-protocol-manifest-oos-oracle-hybrid-v1-20260602.json`, so clean checkouts and CI can reproduce the bridge tests without local OMX state. Previous public CI format failure root cause was the ruff format drift fixed above.

## 2026-06-04 KST — Fresh full clean non-nested monthly-refit rerun and TradFi auto-expansion monitor

현재 코드 기준으로 69-asset monthly-refit walk-forward를 다시 full rerun했다. 기준은 `30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d`, round-trip slippage `10bps`, 매월 1일 refit, 직전 2개월 validation, 다음 1개월 locked OOS다. 평가 OOS는 `2025-09-01T00:00:00`부터 최신 보유 데이터 `2026-06-01T06:30:00`까지이며, `2026-06`은 부분 월이다. 총 10개 OOS fold를 사용했다.

이번 rerun은 기존 row recompute가 아니라 현행 no-nested/material guard가 들어간 runner로 새로 실행한 결과다. 검증상 `fold_candidate_rows=625`, `aggregate_rankings=70`, `clean_promotion_rankings=54`, `demoted_nested_or_historical_rankings=16`, 모든 timeframe coverage `69/69`, metric reconciliation `true`, clean contamination violation `0`이었다. Runtime은 `31:54.04`, peak RSS는 `1158.04 MiB`였다.

Artifacts:

- JSON: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_69_asset_clean_non_nested_full_eval_20260604_final/clean_non_nested_monthly_refit_full_20260604_final.json`
- Markdown: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_69_asset_clean_non_nested_full_eval_20260604_final/clean_non_nested_monthly_refit_full_20260604_final.md`
- SHA256: `83094ae79f946d81b95f1a1a3422f5b2934b82550d9862fc7241e6dc3a93909a`

Clean-promotion ranking 상위:

| Rank | Candidate | OOS Comp | Max OOS MDD | Sharpe | Sortino | Hit | Min OOS | Latest OOS | 판단 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `profile_optuna:selected_optuna` | `10.05%` | `19.20%` | `0.50` | `1.16` | `5/10` | `-9.30%` | `-0.73%` | clean ranker 1위지만 hard-stop promotion 실패 |
| 2 | `profile_optuna:hybrid_v3_5` | `9.06%` | `19.20%` | `0.47` | `1.04` | `5/10` | `-9.30%` | `-0.23%` | non-nested top-level hybrid output, paper/shadow only |
| 3 | `dynamic_conviction_switch:t0.85_risk_capped_fallback` | `8.61%` | `10.08%` | `0.80` | `2.65` | `5/9` | `-2.65%` | `0.00%` | return은 낮지만 drawdown/tail이 더 안정적 |
| 4 | `dynamic_conviction_switch:t0.90_risk_capped_fallback` | `8.61%` | `10.08%` | `0.80` | `2.65` | `5/9` | `-2.65%` | `0.00%` | 위와 동일 family/threshold |
| 5 | `dynamic_conviction_switch:t0.95_risk_capped_fallback` | `8.61%` | `10.08%` | `0.80` | `2.65` | `5/9` | `-2.65%` | `0.00%` | 위와 동일 family/threshold |
| 6 | `dynamic_conviction_switch:t1.00_risk_capped_fallback` | `8.61%` | `10.08%` | `0.80` | `2.65` | `5/9` | `-2.65%` | `0.00%` | 위와 동일 family/threshold |

참고로 comp만 보면 `dynamic_conviction_switch:*_strict_fallback`은 `10.47%` / MDD `10.62%`로 높지만 hit가 `4/9`라 conservative ranker에서 11~12위권으로 내려간다. 어떤 후보도 challenger/robust hard-stop 기준은 통과하지 못했으므로 **real-money 승격은 계속 금지**한다. 실전 후보 해석은 `profile_optuna:selected_optuna`/`hybrid_v3_5`를 수익형 shadow, `dynamic_conviction_switch:*_risk_capped_fallback`을 방어형 shadow로 나누고, fresh-forward paper telemetry가 쌓이기 전까지 capital allocation은 하지 않는 쪽이 맞다.

Nested-hybrid 정책은 유지한다: top-level hybrid output은 분석/후보로 남을 수 있지만, hybrid/blend/selector/gate/portfolio row를 다른 hybrid/portfolio의 **재료**로 다시 넣는 것은 금지한다. 이번 패치로 hidden `final_weights`/`weights` 참조까지 검사해 disguised nested material도 downstream source에서 제외한다. Calendar/month-fixed primary-alpha material도 raw params와 validity/rejection metadata 기준으로 clean material에서 제외한다.

TradFi 확장 모니터링도 추가했다. `lumina_quant.research_universe`는 side-effect 없는 static 69 snapshot을 유지하되, `binance_tradfi_perp_symbols_from_exchange_info()`와 `binance_extended_research_symbols_from_exchange_info()` helper로 Binance `/fapi/v1/exchangeInfo`의 현재 `TRADIFI_PERPETUAL`/USDT trading 심볼을 합칠 수 있게 했다. `scripts/collect_binance_1m_research_universe.py`의 기본 `--universe-source`는 `static-plus-fapi-tradfi`로 바뀌었다. 따라서 명시 `--symbols`가 없으면 기존 69개를 유지하면서, 새 TradFi 지원 심볼은 자동으로 1m data-vision/FAPI fetch plan과 report의 `universe_discovery`에 들어간다. 완전 고정 재현이 필요할 때만 `--universe-source static`을 사용한다.

Verification:

```text
uv run ruff format src/lumina_quant/research_universe.py scripts/collect_binance_1m_research_universe.py tests/test_research_universe.py tests/test_collect_binance_1m_research_universe.py
uv run ruff check src/lumina_quant/research_universe.py scripts/collect_binance_1m_research_universe.py tests/test_research_universe.py tests/test_collect_binance_1m_research_universe.py
uv run pytest -q tests/test_research_universe.py tests/test_collect_binance_1m_research_universe.py
# 9 passed

uv run --extra optimize python scripts/research/run_alpha_zoo_69_asset_monthly_refit_walkforward.py ... --output-json ...20260604_final.json
# Exit status 0; elapsed 31:54.04; peak RSS 1185828 KB

uv run ruff format --check .
uv run ruff check .
uv run pytest -q tests/test_alpha_zoo_69_asset_monthly_refit_walkforward.py tests/test_profit_moonshot_candidate_hybrid.py tests/test_profit_moonshot_live_final_selection.py tests/test_profit_moonshot_strategy_validity_audit.py tests/test_research_universe.py tests/test_collect_binance_1m_research_universe.py
# 64 passed

uv run python scripts/verify_docs.py
# 119 markdown files checked
```

## 2026-06-04 KST — No-nested clean recompute, stale deep-report reconciliation, and report checkpoint optimization

`C:\Users\hoky1\Desktop\deep-research-report.md`를 확인했다. 해당 보고서는 2026-06-03 당시의 `fixed_relaxed_dynamic_blend:*` / `dynamic_aware_hybrid:*` 후보를 중심으로 평가했기 때문에, **전략 랭킹 부분은 현재 no-nested 정책하에서 stale**이다. 다만 selection-bias, execution realism, slippage telemetry, governance/PBO/DSR 같은 주의사항은 여전히 유효하다.

Ralplan consensus 후 실행한 추가 정리:

- 월별 refit runner가 `raw aggregate`, `clean_promotion_rankings`, `demoted_nested_or_historical_rankings`를 분리해 출력하도록 변경했다. 최종 추천/해석은 clean-only ranking에서만 한다.
- recompute artifact에 `recompute_provenance`를 추가했다: source JSON path, source sha256, output paths, `recomputed_from_existing_rows=true`, `fresh_optuna_rerun=false`를 명시한다.
- nested/material detector acceptance를 넓혔다: `cross_candidate_hybrid`, `meta_portfolio`, `dynamic_conviction_switch`, `validation_selector`, `mdd30_*`, `*:selected_optuna`, `*:selected_train_validation_legal`, `*:static_guarded`, `*:hybrid_*` 등은 downstream hybrid/portfolio 재료로 금지된다.
- full rerun 중 매 fold마다 커지는 Markdown을 렌더링하던 비용을 줄이기 위해 `--checkpoint-markdown-interval`을 추가했다. 기본값 `0`은 최종 Markdown만 렌더링한다. JSON checkpoint는 `--checkpoint-interval` 기본값 `1`로 fold-level recovery를 유지한다.

새 no-nested clean recompute artifact:

- Source: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_69_asset_exact_blend_full_tuning_20260603/exact_blend_full_tuning_walkforward_latest.json`
- Source sha256: `563aff7f59174a7ebb6b53f9164eb1feb0cf67881e7f203aecb06987024fa58f`
- Output JSON: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_69_asset_no_nested_clean_recompute_20260604/no_nested_clean_recompute_latest.json`
- Output Markdown: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_69_asset_no_nested_clean_recompute_20260604/no_nested_clean_recompute_latest.md`
- 해석: 기존 row의 governance/ranking repair이며, fresh no-nested Optuna rerun은 아니다.

Clean-only 상위 결과:

| Rank | Candidate | OOS Comp | Max OOS MDD | Sharpe | Sortino | Hit | Min OOS | Latest OOS | 판단 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `relaxed_efficiency:hybrid_v3_5` | `156.03%` | `19.75%` | `1.69` | `10.48` | `5/10` | `-8.41%` | `-0.09%` | clean recompute 기준 1위, 하지만 hit 5/10과 MDD 19.75%라 shadow 우선 |
| 2 | `relaxed_efficiency:selected_optuna` | `60.18%` | `24.27%` | `1.12` | `2.32` | `5/10` | `-22.55%` | `-0.09%` | MDD/left-tail이 커서 보조 후보 |
| 3 | `strict_efficiency:static_guarded` | `40.37%` | `29.16%` | `0.84` | `1.57` | `5/10` | `-26.40%` | `-0.53%` | drawdown 관점에서 실전 후보 약함 |
| 4 | `relaxed_efficiency:selected_train_validation_legal` | `33.82%` | `25.79%` | `0.81` | `1.57` | `5/10` | `-22.55%` | `-0.09%` | 보조/진단 후보 |
| 5 | `strict_efficiency:hybrid_v3_6` | `32.13%` | `15.89%` | `1.03` | `5.48` | `5/10` | `-5.89%` | `-0.17%` | return은 낮지만 MDD가 상대적으로 안정적 |

Demoted raw 상위 후보:

- `fixed_relaxed_dynamic_blend:relaxed70_dynamic30`: raw OOS comp `122.36%`, but `nested_hybrid_dependency`, `post_oos_research_variant`, `requires_fresh_forward_shadow`.
- `fixed_relaxed_dynamic_blend:relaxed60_dynamic40`: raw OOS comp `111.75%`, same demotion reasons.
- `cross_candidate_hybrid:*`, `dynamic_aware_hybrid:*`, `mdd30_high_vol_gate:*`는 non-leaf/nested material 또는 historical research layer로 demoted.

검증:

```text
uv run python -m py_compile scripts/research/run_alpha_zoo_69_asset_monthly_refit_walkforward.py scripts/research/benchmark_monthly_refit_eval_hotpath.py tests/test_alpha_zoo_69_asset_monthly_refit_walkforward.py

uv run python -m pytest -q tests/test_alpha_zoo_69_asset_monthly_refit_walkforward.py
# 27 passed in 0.25s

uv run python -m ruff check scripts/research/run_alpha_zoo_69_asset_monthly_refit_walkforward.py scripts/research/benchmark_monthly_refit_eval_hotpath.py tests/test_alpha_zoo_69_asset_monthly_refit_walkforward.py
# All checks passed

uv run python scripts/research/benchmark_monthly_refit_eval_hotpath.py --min-speedup 1.5
# speedup=6.375x, checksum identical, PASS
```

## 2026-06-04 KST — No-nested-hybrid cleanup and monthly-refit evaluation hotpath optimization

정책을 변경했다: **hybrid/blend/selector/gate/bridge 등 포트폴리오 레이어 후보는 다른 hybrid/portfolio의 재료로 사용할 수 없다.** hybrid는 이미 내부 sleeve를 합친 portfolio이므로, hybrid를 다시 hybrid 재료로 넣으면 분산처럼 보이지만 실제로는 동일 sleeve/asset/factor exposure를 중복 매수할 수 있다.

이번 코드 정리 결과:

- 새 leaf-only 필터 `_leaf_strategy_material_candidate`를 추가해 `cross_candidate_hybrid`, `dynamic_aware_hybrid`, `hybrid_oracle_bridge`, `meta_portfolio`, `validation_selector`, `risk_enhanced_blend`, `fixed_relaxed_dynamic_blend`, `mdd30_*`, `dynamic_conviction_switch` 계열을 downstream hybrid/portfolio 재료에서 제외한다.
- `dynamic_conviction_switch`는 더 이상 `cross_candidate_hybrid:*`, `profile_optuna:hybrid_*`, `selected_optuna` 같은 non-leaf 후보를 고르지 않고, profile/strict/relaxed/individual leaf 후보만 선택한다.
- `dynamic_aware_hybrid`, `risk_enhanced_blend`, `fixed_relaxed_dynamic_blend`는 기존 구현이 non-leaf를 다시 섞는 구조라 명시적으로 no-op 처리했다. 특히 이전 최종 shadow 후보였던 `fixed_relaxed_dynamic_blend:relaxed60_dynamic40` 및 `relaxed70_dynamic30`은 새 정책하에서 promotion/selection 후보가 아니라 historical deprecated artifact로만 본다.
- `validation_selector`, bridge eligible pool, MDD30 risk/gate family도 leaf-only 입력만 받도록 정리했다. MDD30 연구 family는 기존 dynamic/hybrid source 대신 profile/strict/relaxed leaf source만 scale/blend한다.
- 기존 artifact를 재계산하는 fast repair path는 non-leaf reference를 `nested_hybrid_dependency=true`로 표시하고 clean promotion에서 제외한다.

평가 성능도 개선했다. 월별 refit runner의 반복 병목은 같은 candidate return stream에 대해 train/validation/OOS `_period_metrics`를 여러 selector/hybrid/report 단계에서 반복 계산하는 부분이었다. `_period_metrics`에 bounded LRU-style 캐시를 추가해 return series 정렬/Datetime mask 생성은 1회만 수행하고, 이후 window는 int64 timestamp `searchsorted`와 metric-result cache를 사용한다. 캐시는 `LQ_MONTHLY_REFIT_PERIOD_METRICS_CACHE_SIZE`와 `LQ_MONTHLY_REFIT_PREPARED_RETURNS_CACHE_SIZE`로 조정 가능하다.

성능/회귀 검증:

```text
uv run python scripts/research/benchmark_monthly_refit_eval_hotpath.py --min-speedup 1.5
# latest rerun speedup=6.375x, checksum identical, PASS

uv run python -m pytest -q tests/test_alpha_zoo_69_asset_monthly_refit_walkforward.py
# 27 passed in 0.25s

uv run python -m ruff check scripts/research/run_alpha_zoo_69_asset_monthly_refit_walkforward.py scripts/research/benchmark_monthly_refit_eval_hotpath.py tests/test_alpha_zoo_69_asset_monthly_refit_walkforward.py
# All checks passed
```

운용 해석: 2026-06-03 note의 `fixed_relaxed_dynamic_blend:*`, `dynamic_aware_hybrid:*`, `risk_enhanced_blend:*`, `hybrid_oracle_bridge:*` 계열 성과는 nested-hybrid exposure risk 때문에 더 이상 최종 선택 근거로 쓰면 안 된다. 새 기준의 최종 후보는 leaf-only 재료로 walk-forward를 다시 돌린 결과에서만 선정해야 한다.

## 2026-06-03 KST — 69-asset monthly clean-OOS walk-forward final selection

최종 69-asset 월별 refit walk-forward 연구를 최신 데이터까지 재평가했다. 기준은 `30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d`, round-trip slippage `10bps`, train expanding from `2025-01-01T00:00:00`, 매월 1일 refit, 직전 2개월 validation, 다음 1개월 locked OOS이다. 평가 OOS는 `2025-09-01T00:00:00`부터 최신 보유 데이터 `2026-06-01T06:30:00`까지이며, `2026-06` fold는 부분 월이다. 총 10개 OOS fold를 사용했다.

Clean/no-leakage contract:

- 각 fold의 현재 OOS는 해당 fold의 파라미터, bridge/dynamic weight, selector label, portfolio weighting에 사용하지 않았다.
- dynamic-aware lane은 same-month dynamic self-feeding 금지 검사를 통과했다.
- bridge/online metric reconciliation은 `metrics_reconciled=true`, mismatches `[]`로 확인했다.
- 단, `fixed_relaxed_dynamic_blend:*` 후보는 OOS 리뷰 후 추가한 exact bar-level 조합이므로 동일 창에서는 clean promotion이 아니라 `fresh_forward_shadow_required_before_promotion`이다.

최종 후보 성과 요약:

| 후보 | Clean | OOS Comp | Ann approx | Max OOS MDD | Monthly Eq MDD | Sharpe | Sortino | PF | Hit | Worst | Latest | 판단 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `relaxed_efficiency:hybrid_v3_5` | Y | `156.03%` | `209.00%` | `19.75%` | `15.66%` | `1.69` | `10.48` | `7.04` | `5/10` | `-8.41%` | `-0.09%` | clean-only 최고 comp, paper/testnet 후보 |
| `fixed_relaxed_dynamic_blend:relaxed70_dynamic30` | N | `122.36%` | `160.90%` | `16.66%` | `14.66%` | `1.74` | `8.97` | `6.33` | `6/10` | `-8.42%` | `-0.17%` | shadow 고수익형 |
| `fixed_relaxed_dynamic_blend:relaxed60_dynamic40` | N | `111.75%` | `146.03%` | `16.19%` | `14.35%` | `1.75` | `8.43` | `6.00` | `6/10` | `-8.61%` | `-0.20%` | 최종 균형 shadow 후보 |
| `dynamic_aware_hybrid:hybrid_v3_5_train_validation_fit` | Y | `54.76%` | `68.89%` | `16.75%` | `12.76%` | `1.53` | `4.43` | `3.87` | `5/10` | `-9.97%` | `-0.38%` | clean 방어 sleeve |
| `cross_candidate_hybrid:hybrid_v3_6` | Y | `61.61%` | `77.89%` | `27.57%` | `14.83%` | `1.55` | `4.28` | `4.47` | `7/10` | `-9.48%` | `-0.57%` | 별도 leverage run 내 clean 비교 상위, MDD 큼 |
| `asset_timeframe_leverage:hybrid_v3_6` | Y | `21.78%` | `26.67%` | `21.97%` | `21.19%` | `0.72` | `1.67` | `1.79` | `4/10` | `-15.87%` | `-0.57%` | leverage 진단/monitor only |

월별 OOS 분포에서 최종 균형 shadow 후보 `fixed_relaxed_dynamic_blend:relaxed60_dynamic40`은 `2025-09 +0.59%`, `2025-10 +0.07%`, `2025-11 +45.85%`, `2025-12 -2.90%`, `2026-01 +19.80%`, `2026-02 +30.64%`, `2026-03 -6.28%`, `2026-04 -8.61%`, `2026-05 +11.04%`, `2026-06 -0.20%`였다. `dynamic_aware_hybrid:hybrid_v3_5_train_validation_fit`은 같은 기간 `+2.65%, -0.47%, +22.76%, -3.48%, +24.60%, +7.43%, -3.10%, -9.97%, +9.88%, -0.38%`로 수익이 더 낮지만 smoother 방어 sleeve 역할을 한다.

Exposure/rebalancing/leverage 결론:

- 최종 exact blend 60/40은 fold 전체에서 unique assets `29`, 평균 active assets/fold `13.7`, active range `10~29`, 평균 gross `1.92x`, gross range `1.35~2.58x`였다. 후보풀은 69개 전체를 계속 모니터링하며, allocation은 train/validation evidence와 fold별 gate를 통과한 자산만 사용한다.
- 기존 source sleeve 단계에서 `symbol x timeframe x integer_leverage`는 이미 train/validation 기반으로 튜닝된다. 이번에는 그 위에 asset/timeframe post-allocation multiplier와 gross cap을 추가로 clean 검증했다. 하지만 최고 `asset_timeframe_leverage:hybrid_v3_6`도 OOS comp `21.78%`, MDD `21.97%`, Sharpe `0.72`로 final core를 대체하지 못했다.
- 따라서 portfolio-level rebalance/leverage 확장은 즉시 실전 코어 편입이 아니라 monitor/diagnostic 보조축으로 유지한다. 리밸런싱 cadence 자체는 monthly day-1 refit protocol로 고정했고, intramonth에는 signal-level position update로 해석한다.

최종 선택:

1. **실전 균형 shadow/paper 후보:** `fixed_relaxed_dynamic_blend:relaxed60_dynamic40`. OOS comp `111.75%`, max OOS MDD `16.19%`, Sharpe `1.75`, Hit `6/10`로 risk/return 균형이 가장 낫다. 다만 post-OOS research variant라 fresh-forward shadow가 쌓이기 전에는 clean promotion 금지.
2. **고수익 shadow 후보:** `fixed_relaxed_dynamic_blend:relaxed70_dynamic30`. OOS comp `122.36%`로 더 높지만 relaxed sleeve 의존도가 더 크다.
3. **clean-only 기준 최고:** `relaxed_efficiency:hybrid_v3_5`. 같은 OOS 창에서는 clean comp `156.03%`로 최고이나 max OOS MDD `19.75%`, hit `5/10`이고 relaxed repair risk가 있으므로 paper/testnet challenger로 다룬다.
4. **real-money 상태:** 모든 후보는 아직 `ready_for_real=false`로 유지한다. 실전 전환은 fresh-forward shadow, paper/testnet fill/BBO/slippage/reconciliation telemetry, 월별 hard-stop review를 통과해야 한다.

Artifacts:

- Exact blend full-tuning walk-forward: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_69_asset_exact_blend_full_tuning_20260603/exact_blend_full_tuning_walkforward_latest.json` and `.md`. Selection report: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_69_asset_exact_blend_full_tuning_20260603/exact_blend_selection_report_ko.md`. Runtime `2:35:12`, peak RSS `1121.3 MiB`.
- Asset/timeframe leverage clean test: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_69_asset_asset_tf_leverage_20260603/asset_tf_leverage_walkforward_latest.json` and `.md`. Selection report: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_69_asset_asset_tf_leverage_20260603/asset_tf_leverage_selection_report_ko.md`. Runtime `1:40:52`, peak RSS `1085.0 MiB`.
- Supporting earlier OOS/selector artifacts: `alpha_zoo_69_asset_best_strategy_factory_20260601`, `alpha_zoo_69_asset_monthly_refit_diagnostics_20260601`, `alpha_zoo_69_asset_clean_full_tuning_20260602`, `alpha_zoo_69_asset_dynamic_aware_hybrid_20260602`, `alpha_zoo_69_asset_oos_oracle_bridge_20260602`, `alpha_zoo_69_asset_risk_enhanced_20260602`, and `alpha_zoo_69_asset_mdd30_high_vol_20260602`.

Verification after final patches and report generation: `py_compile` and `ruff check` passed for all changed/untracked Python research files; targeted Alpha Zoo 69-asset pytest passed `66 passed in 1.31s`; docs verification passed (`118 markdown files checked`); `git diff --check` passed.


## 2026-05-31 KST — Relaxed repair interpretation: trust, liquidation, 69→19 selection, and future refit of train-ineligible assets

Follow-up interpretation for the MDD-guarded relaxed 69-asset efficiency-repair artifact. The relaxed artifact is **not** a real-money/live-ready result. It is a high-return, high-MDD **paper/testnet challenger** that should be compared against the stricter lower-MDD v3.6 live-handoff baseline before any promotion decision.

Trust assessment:

- Trustworthy parts: the corrected train-eligibility guard is preserved; train-ineligible symbol/timeframe rows are not used for parameter fitting, repair source rows, sleeve allocation, hybrid selection, or live promotion. The primary `10bps` round-trip RPT gate remains strict. Train and validation are both strongly positive and selected aggressive profile train return remains above validation return.
- Discounted parts: no locked test/OOS is active in this final-refit-style pass; paper/testnet fill telemetry is not yet measured; relaxed policy admits material-positive train<validation and low-sample TradFi rows under MDD/RPT guards; selected gross is high at `7.2541x`; MDD is material. Therefore these numbers must not be read as live expected returns.
- Current operating conclusion: keep the strict corrected v3.6 result as the safer baseline while running the relaxed artifact as paper/testnet challenger evidence.

Liquidation / wipeout interpretation:

| candidate | train liquidation / wipeout | validation liquidation / wipeout | train MDD | validation MDD | note |
| --- | ---: | ---: | ---: | ---: | --- |
| `aggressive_mdd30_gross10_69_asset_relaxed_efficiency_repair_optuna` | `0 / 0` | `0 / 0` | `27.7065%` | `21.2478%` | liquidation-free in replay, but high drawdown |
| relaxed selected `hybrid_v3_5_optuna_three_profile_blend` | `0 / 0` | `0 / 0` | `23.9019%` | `16.3433%` | lower MDD than aggressive, but validation exceeds train under relaxed rule |

The liquidation fields are clean in train/validation replay, but locked-OOS/test-set liquidation evidence is not available for this final-refit artifact. The correct reading is: no simulated liquidation/wipeout occurred under the replay assumptions, but MDD and live fill/slippage risk remain the binding safety concerns.

Why 19 symbols instead of all 69:

1. Research/monitoring universe is `69` symbols.
2. `37` symbols had no train-split rows and remain watch/shadow-only under the corrected no-validation-only-leakage policy.
3. `32` symbols are train-eligible.
4. `32 × 3` profile rows produce `96` candidate rows.
5. Relaxed gate passes `30` rows across `19` unique symbols; all `30` relaxed gate-ok rows are selected as sleeves in the profile set.

Selected relaxed symbols are `ADAUSDT`, `AVAXUSDT`, `BNBUSDT`, `BTCUSDT`, `COINUSDT`, `COPPERUSDT`, `CRCLUSDT`, `DOGEUSDT`, `ETHUSDT`, `GOOGLUSDT`, `INTCUSDT`, `METAUSDT`, `MSTRUSDT`, `PLTRUSDT`, `SOLUSDT`, `TONUSDT`, `XAGUSDT`, `XPDUSDT`, and `XPTUSDT`. Eligible symbols that still have no relaxed gate-ok row are `AMZNUSDT`, `BZUSDT`, `CLUSDT`, `EWJUSDT`, `EWYUSDT`, `HOODUSDT`, `NATGASUSDT`, `NVDAUSDT`, `PAYPUSDT`, `TRXUSDT`, `TSLAUSDT`, `XAUUSDT`, and `XRPUSDT`. Main rejection causes remain insufficient validation return, train/validation RPT not above `10bps`, train<validation without the material-positive/MDD exception, and profile trade-count minima.

Forced inclusion of all 69 assets is intentionally rejected. The standard rule is: monitor all 69; allocate only to symbols with train evidence and passing RPT/MDD/sample/concentration gates; keep the rest as shadow/watchlist until they earn train+validation evidence.

Future refit expectation for the 37 train-ineligible assets:

- Re-running the same split does not make the 37 assets eligible; they still have no train rows in that split.
- A train+validation final refit can technically fit them, but it would leave no independent validation for those assets and should not promote them by itself.
- A later rolling refit, after enough new bars make today’s validation/cold-start period part of train and reserve a fresh latest validation window, can admit a meaningful subset.
- Based on the cold-start donor-frozen shadow artifact, roughly `15–20` of the 37 could plausibly enter the candidate pool when real train history exists, but optimizer selection will likely keep fewer nonzero exposures.

Cold-start shadow evidence for the 37 symbols remains report-only: donor-frozen primary shadow selected `18` sleeves with validation `+31.6832%`, validation MDD `10.2407%`, validation RPT `83.87bps`, gross `2.0x`, and no promotion because target train rows are absent. Stronger watchlist names include `MUUSDT`, `SNDKUSDT`, `AMDUSDT`, `DRAMUSDT`, `QCOMUSDT`, `SOXLUSDT`, `QQQUSDT`, `SPYUSDT`, `MRVLUSDT`, `ARMUSDT`, `AVGOUSDT`, and `TSMUSDT`. Weak/negative cold-start names such as `OPENAIUSDT`, `SPCXUSDT`, `BABAUSDT`, `WDCUSDT`, and `COHRUSDT` need fresh train evidence before any inclusion.


## 2026-05-31 KST — MDD-guarded relaxed 69-asset efficiency repair

Applied the operator's relaxed gate interpretation to the corrected 69-asset efficiency-repair artifact without changing the 10bps execution-efficiency requirement. New runner/test: `scripts/research/run_alpha_zoo_69_asset_relaxed_efficiency_repair_optuna.py` and `tests/test_alpha_zoo_69_asset_relaxed_efficiency_repair_optuna.py`. Artifact: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_69_asset_relaxed_efficiency_repair_optuna_20260531/alpha_zoo_69_asset_relaxed_efficiency_repair_optuna_latest.json` plus Markdown/CSV siblings.

Relaxed policy `material_positive_tradfi_low_sample_mdd_guard_20260531`:

- `train < validation` is no longer a hard rejection when both train and validation returns are at least `2%` and the MDD guard passes.
- TradFi low-sample rows can be admitted as warnings when MDD and the strict `>10bps` train/validation RPT gates pass.
- Material-positive non-TradFi low-sample rows are also reportable/admissible under the same MDD/RPT guard, but the optimizer still penalizes relaxed/low-sample notional share.
- Gross/concentration pressure is optimized and penalized rather than hard rejected while validation MDD remains under the relaxed guard; a `12x` hard gross cap remains.
- Train-ineligible symbol/timeframe rows are still excluded from parameter fitting, repair source rows, sleeve allocation, hybrid selection, and live promotion.
- No locked test/OOS is used for selection. All artifacts remain `paper_testnet_only=true`, `ready_for_real=false`, `real_money_execution=false`, and `real_execution_allowed=false`.

Candidate pool impact versus the strict corrected pass: strict gate-ok rows `18`; relaxed gate-ok rows `30`; newly admitted rows `16`; relaxed unique gate-ok symbols `19`. Newly admitted symbols: `COINUSDT`, `COPPERUSDT`, `CRCLUSDT`, `ETHUSDT`, `GOOGLUSDT`, `INTCUSDT`, `METAUSDT`, `MSTRUSDT`, `PLTRUSDT`, `TONUSDT`, and `XPDUSDT`.

Best train/validation-legal relaxed portfolio is the aggressive profile:

| portfolio | train | validation | train MDD | validation MDD | RPT bps train/val | 20bps stress train/val | gross | paper | real |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `aggressive_mdd30_gross10_69_asset_relaxed_efficiency_repair_optuna` | `+555.8771%` | `+518.4857%` | `27.7065%` | `21.2478%` | `96.70 / 286.89` | `+498.3937% / +500.4133%` | `7.2541x` | `true` | `false` |

Selected relaxed Optuna hybrid comparison is `hybrid_v3_5_optuna_three_profile_blend`: train `+284.5998%`, validation `+373.8607%`, train/validation MDD `23.9019%/16.3433%`, RPT `58.07/250.46bps`, gross `6.8713x`, `selection_reasons=[]`, paper/testnet only. This hybrid uses the relaxed dominance rule because both train and validation are materially positive and MDD remains below the relaxed hybrid guard.

Strict corrected reference remains available and safer/lower-MDD: selected balanced profile train/validation `+119.3799%/+79.7120%`, MDD `16.6872%/7.4789%`, RPT `108.53/157.53bps`, gross `2.2x`; strict live handoff v3.6 train/validation `+96.5913%/+68.1871%`, MDD `12.5785%/7.8678%`, RPT `42.30/149.64bps`, gross `2.5042x`. The relaxed artifact is a higher-return, higher-MDD paper/testnet challenger, not a real-money enablement.

Selected relaxed sleeve set has `30` sleeves across `19` train-eligible symbols. Concentration for the aggressive selected profile: top symbol `PLTRUSDT` `13.78%`, top group `tradfi_equity` `53.59%`, effective symbol count `9.95`; group mix `crypto_core 38.05%`, `tradfi_equity 53.59%`, `tradfi_commodity 8.36%`. All sleeve integer leverages remain integers (`2x` to `12x`). Runner evidence: wall `1:49.86`, max RSS `914,748 KiB` / artifact `893.31 MiB`, below the 8GB cap.

## 2026-05-31 KST — Cold-start donor transfer shadow for validation-only assets

Added the report-only cold-start transfer pass for the `37` train-ineligible 69-asset symbols. The new runner `scripts/research/run_alpha_zoo_69_asset_cold_start_transfer_shadow.py` and tests `tests/test_alpha_zoo_69_asset_cold_start_transfer_shadow.py` evaluate whether recently listed TradFi/premarket symbols can be initialized from similar train-eligible donor profiles without leaking target validation PnL into donor choice.

Artifact: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_69_asset_cold_start_transfer_shadow_20260531/alpha_zoo_69_asset_cold_start_transfer_shadow_latest.json` plus Markdown/CSV siblings. Source remains the corrected 69-asset profile artifact and the strict live reference remains `alpha_zoo_69_asset_efficiency_repair_optuna_20260530`; the live handoff was not changed.

Safety contract in the artifact:

- Donor selection uses donor train/validation quality, static domain similarity, and target bar coverage only. Target validation PnL is not used in the primary donor-frozen lane.
- No donor OHLCV, donor PnL, donor trade counts, or synthetic target train metrics are substituted into target train performance.
- Validation-oracle selection is emitted only as a diagnostic upper bound and is explicitly non-promotable.
- `paper_testnet_only=true`, `shadow_report_only=true`, `ready_for_paper=false`, `ready_for_real=false`, `real_money_execution=false`, and `real_execution_allowed=false` throughout. The primary cost/RPT assumption remains `10bps` round trip.

Results under the latest 8-week validation window (`2026-04-04T04:00:00` through `2026-05-30T03:00:00`):

| lane | sleeves | gross | train | validation | validation MDD | validation RPT | status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| donor-frozen primary cold-start shadow | `18` | `2.00x` | `0.00%` | `+31.6832%` | `10.2407%` | `83.87bps` | report-only, not promotable |
| validation-oracle diagnostic upper bound | `16` | `1.7778x` | `0.00%` | `+44.4677%` | `8.1768%` | `178.67bps` | leakage diagnostic, not promotable |

Primary donor-frozen selected symbols were `TSMUSDT`, `MUUSDT`, `SNDKUSDT`, `AVGOUSDT`, `AMDUSDT`, `QCOMUSDT`, `MRVLUSDT`, `DRAMUSDT`, `WDCUSDT`, `ARMUSDT`, `COHRUSDT`, `BABAUSDT`, `SPYUSDT`, `SOXLUSDT`, `QQQUSDT`, `SPCXUSDT`, `OPENAIUSDT`, and `QNTXUSDT`. Positive contributors included `MUUSDT`, `SNDKUSDT`, `AMDUSDT`, `DRAMUSDT`, `QCOMUSDT`, `SOXLUSDT`, `QQQUSDT`, and `SPYUSDT`; negative contributors included `COHRUSDT`, `SPCXUSDT`, `OPENAIUSDT`, `BABAUSDT`, and `WDCUSDT`. The primary lane stays useful as a monitored cold-start watchlist but cannot enter live/paper allocation until those symbols have physical train-window data.

Runner evidence: wall `0:14.80`, max RSS `852,388 KiB`; artifact `runner_peak_rss_mib=832.41`, below the 8GB cap. Local targeted verification passed: new cold-start tests `5 passed`; related Alpha Zoo targeted suite `16 passed`; Ruff check and compileall passed for the new runner/tests.

## 2026-05-31 KST — 69-asset train-eligibility correction and live handoff refresh

Corrected the 69-symbol efficiency-repair pipeline after finding that several newly listed TradFi/premarket symbols had no train-split rows before `2026-04-04T03:00:00`, yet the earlier allocation layer could still admit validation-only sleeves. The superseded headline `+295.9880%` train / `+172.7926%` validation was therefore contaminated by validation-only assets and is no longer the live handoff evidence.

Implemented train-eligibility gating in the shared 69-asset refit utilities and repair runner:

- Build a symbol/timeframe train-eligibility report from the actual train/validation windows.
- Exclude any symbol/timeframe with zero train rows from per-asset parameter fitting, repair source rows, sleeve allocation, hybrid selection, and live promotion.
- Keep those symbols in the watch/data universe only; they become eligible only after a future refit has real train-split history.
- Apply `warmup_ratio` to train-split bars only, even when final refit fits train+validation parameters.
- Filter rejected/diagnostic repair streams out of allocation so rows with `efficiency_repair_reasons` cannot re-enter through portfolio weights.

Train eligibility in the corrected artifact: `32` eligible symbols and `37` train-ineligible symbols. The train-ineligible list is `QQQUSDT`, `SPYUSDT`, `SOXLUSDT`, `AAPLUSDT`, `TSMUSDT`, `MUUSDT`, `SNDKUSDT`, `MSFTUSDT`, `AVGOUSDT`, `BABAUSDT`, `AMDUSDT`, `QCOMUSDT`, `USARUSDT`, `LITEUSDT`, `ORCLUSDT`, `DISUSDT`, `UBERUSDT`, `CSCOUSDT`, `HDUSDT`, `MRVLUSDT`, `CRWVUSDT`, `WMTUSDT`, `JPMUSDT`, `VUSDT`, `BRKBUSDT`, `FLNCUSDT`, `DRAMUSDT`, `RKLBUSDT`, `CBRSUSDT`, `NBISUSDT`, `WDCUSDT`, `ARMUSDT`, `BEUSDT`, `COHRUSDT`, `SPCXUSDT`, `OPENAIUSDT`, and `QNTXUSDT`.

Corrected artifact: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_69_asset_efficiency_repair_optuna_20260530/alpha_zoo_69_asset_efficiency_repair_optuna_latest.json` generated at `2026-05-30T16:17:37Z`. Rerun evidence: wall `1:55.26`, max RSS `891,500 KiB` / artifact `870.61 MiB`, below the 8GB budget. No locked test set is used in this final-refit style pass; train and latest 8-week validation are the only selection inputs.

Corrected selected train/validation-legal portfolio:

| portfolio | train | validation | train MDD | validation MDD | RPT bps train/val | 20bps stress train/val | gross | paper | real |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `balanced_mdd12_gross5_69_asset_efficiency_repair_optuna` | `+119.3799%` | `+79.7120%` | `16.6872%` | `7.4789%` | `108.53 / 157.53` | `+108.3799% / +74.6520%` | `2.2000x` | `true` | `false` |

Corrected paper/testnet live handoff now uses artifact-selected `hybrid_v3_6_optuna_three_profile_blend`:

| hybrid | train | validation | train MDD | validation MDD | RPT bps train/val | historical gross | live final gross | paper | real |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `hybrid_v3_6_optuna_three_profile_blend` | `+96.5913%` | `+68.1871%` | `12.5785%` | `7.8678%` | `42.30 / 149.64` | `2.5042x` | `2.3389x` | `true` | `false` |

The corrected v3.5 comparison remains paper-eligible but is not the selected live handoff: train `+153.0941%`, validation `+57.2165%`, train/validation MDD `14.0862%/8.4343%`, RPT `49.29/91.23bps`, gross `3.0697x`.

The live adapter now reconstructs `18` selected source sleeves, watches all `69` symbols, and has nonzero selected source exposure only in `13` train-eligible symbols: `ADAUSDT`, `AVAXUSDT`, `BNBUSDT`, `BTCUSDT`, `COPPERUSDT`, `CRCLUSDT`, `DOGEUSDT`, `SOLUSDT`, `TONUSDT`, `XAGUSDT`, `XAUUSDT`, `XPTUSDT`, and `XRPUSDT`. No train-ineligible asset has nonzero selected live gross after the fix.

Limit/no-fill/slippage policy is unchanged and remains paper/testnet only: `LMT` default, `one_tick_worse`, market fallback disabled, max chase attempts `0`, missing/high-spread/high-slippage BBO guard skips without market conversion, and realized all-in round-trip cost must stay within the `10bps` replay/gate assumption before any real-money review. Safety flags remain `paper_testnet_only=true`, `ready_for_real=false`, `real_money_execution=false`, and `real_execution_allowed=false`.

Verification after the correction passed: targeted eligibility/live adapter suites `23 passed`; `ruff format --check .`; `ruff check .`; `compileall`; docs verification `117` markdown files; architecture check; hardcoded-parameter audit `new=0`; `git diff --check`; artifact invariant script confirmed no train-ineligible selected live gross; and full `pytest -q` `1539 passed` with max RSS `2,756,628 KiB` (<8GB). GitNexus was refreshed and reported high impact because the research/live artifacts and default live profile changed; the change is intentional and covered by the verification above.

## 2026-05-30 KST — 69-asset per-profile Optuna rebuild after broad-blend audit

Follow-up to the first broad 69-symbol blend: the earlier pass optimized a diversified stream blend and did **not** fully rebuild the original hybrid source-profile logic per asset. Added `scripts/research/run_alpha_zoo_69_asset_profile_optuna_hybrid_refit.py` and `tests/test_alpha_zoo_69_asset_profile_optuna_hybrid_refit.py` to run the corrected expansion:

- Treat the three source profiles as risk/selection templates, not as three final assets: balanced/growth/aggressive each tunes all 69 symbols independently.
- Each symbol/profile pair is Optuna-tuned over family, timeframe, side, entry/exit, min-hold, cooldown, and integer leverage; no grid selection is used for the tunable search.
- Domain anchors are tracked beyond BTC: BTC, ETH, SOL, SPY, QQQ, XAU, XAG, and crude proxy anchors. Candidate/profile objectives penalize single-anchor clones, top-symbol concentration, top-asset-group concentration, and top-anchor concentration.
- Each rebuilt source profile performs an Optuna sleeve-allocation pass, then the three rebuilt multi-asset profile streams are passed to the v3.5/v3.6 Optuna hybrid engine plus a train-dominance guarded static Optuna blend.
- Locked test/OOS remains disabled for live-final-refit style research; latest 8 complete weeks are validation/report evidence. Real-money flags remain false.

Final artifact: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_69_asset_profile_optuna_hybrid_refit_20260530/alpha_zoo_69_asset_profile_optuna_hybrid_refit_latest.json`.

OOM evidence: full 69-symbol per-profile run completed in `7:05.78` wall time with max RSS `1,043,232 KiB` / artifact `runner_peak_rss_mib=1018.78125`, safely below 8GB.

Selected train/validation-legal portfolio is the guarded static blend selecting the rebuilt balanced multi-asset profile:

| portfolio | train | validation | train MDD | validation MDD | RPT bps train/val | gross | paper | real |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `hybrid_static_train_dominance_guarded_three_profile_blend` | `+160.3316%` | `+150.0726%` | `15.3371%` | `7.5634%` | `69.40 / 152.18` | `5.00x` | `true` | `false` |

Rebuilt source profile audit:

| profile | sleeves | train | validation | validation MDD | RPT bps train/val | top symbol | top anchor | decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| balanced | `22` | `+160.3316%` | `+150.0726%` | `7.5634%` | `69.40 / 152.18` | `XPTUSDT 10.73%` | `energy_crude_beta 21.77%` | train/validation legal |
| growth | `30` | `+174.8699%` | `+1433.8154%` | `17.630%` | `38.3 / 594.9` | `AAPLUSDT 9.56%` | `crypto_beta_btc 20.60%` | rejected as validation spike |
| aggressive | `45` | `+218.5251%` | `+1497.3979%` | `18.2%` | `43.7 / 642.9` | `MUUSDT 7.93%` | `crypto_beta_btc 29.81%` | rejected as validation spike + gross > 8x |

The adaptive v3.5/v3.6 hybrid result is deliberately not promoted despite high validation because it violates the train-dominance rule (`train +301.56% < validation +715.70%`). This is recorded as a validation-spike rejection, not a live candidate. The legal candidate is therefore the balanced multi-asset profile/guarded static blend. It is still paper/testnet-only and requires exchange-connected paper telemetry before any real review.

Canonical path: `docs/research_note/research_note.md`.

This file is the current cumulative research note / research journal. Keep the filename stable as `research_note.md`; strategy, project, and date labels belong inside entries, not in the path.

Current live/paper-testnet identity:

- Runtime strategy name: `AlphaZooOptunaHybridLiveStrategy`.
- Selected frozen profile/artifact family for the corrected 69-asset efficiency-repair handoff: `hybrid_v3_6_optuna_three_profile_blend`. Older standard/integer-leverage artifacts using `hybrid_v3_5_optuna_three_profile_blend` remain historical baselines.
- `profit_moonshot_alpha_zoo` is retained only as a historical artifact namespace, not as the strategy name.
- Real-money execution remains prohibited until separate paper/testnet fill telemetry gates pass.

## Research journal — latest first

Prepend new research diary entries below this heading. The legacy historical entries were reordered newest-first during the 2026-05-28 naming cleanup.

## 2026-05-30 KST — 69-asset Optuna hybrid refit and concentration audit

Expanded the Alpha Zoo broad research pass from the prior ETH/SOL/TRX-heavy incumbent universe to the full `69`-symbol Binance research universe (`10` core crypto + `59` Binance USD-M `TRADIFI_PERPETUAL` proxies) using direct stored `1m` OHLCV resampled into `30m`, `1h`, `2h`, and `4h` bars. The new runner is `scripts/research/run_alpha_zoo_69_asset_optuna_hybrid_refit.py`; outputs are under `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_69_asset_optuna_hybrid_refit_20260530/`. The standard live-facing split rule is preserved: latest `8` complete weeks are validation, no locked-OOS/test set is used in this final-refit mode, and every real-money flag remains hard false. Candidate and hybrid search uses Optuna TPE through the shared `lumina_quant.optimization.search_policy` path, not grid search.

OOM-safe implementation choices: direct 1m reads are symbol/timeframe chunked, in-memory hybrid streams are capped (`MAX_CACHED_STREAMS_PER_SYMBOL=8`, run cap `--max-hybrid-streams 48`, `--max-streams-per-symbol 2`), and the selected-stream matrix is built only after ranking/freeze. Full run evidence: `/usr/bin/time` wall `1:00.25`, max RSS `838,416 KiB` and payload `runner_peak_rss_mib=818.77`, well below the 8 GiB session budget.

Discovery result after embedded `10bps` round-trip friction proxy:

- `45,120` candidates evaluated; no single component passed full paper promotion by itself. `17` rows passed the sample gate but failed execution-efficiency proxy, so they remain shadow rows (`AVAXUSDT` 10, `HOODUSDT` 4, `COINUSDT` 2, `XAGUSDT` 1).
- The Optuna-selected aggregate hybrid passed the strict backtest paper gate only as a diversified hybrid: train `+9.0960%`, validation `+8.5640%`, train MDD `2.5726%`, validation MDD `0.6934%`, train RPT proxy `10.9603bps`, validation RPT proxy `31.8371bps`, component trade events train/validation `9,613 / 3,789`, liquidation/account wipeout `0 / 0`. Backtest gate says `ready_for_paper=true`, but `ready_for_real=false`, `real_money_execution=false`, `real_execution_allowed=false`.
- Concentration audit improved materially versus the incumbent: top symbol `TONUSDT` share `16.27%`, effective symbol count `11.59`, top asset group `crypto_core` `49.91%`, TradFi equity `32.64%`, ETF/index `10.21%`, commodities `7.24%`; no concentration flags. Validation long/short exposure when active is approximately `52.06% / 47.94%`, so the hybrid is not only one direction despite every rule being long/short-capable. Timeframe mix: `30m` `41.94%`, `1h` `23.93%`, `2h` `15.29%`, `4h` `18.84%`. Family mix: volatility-adjusted trend persistence `55.45%`, cross-sectional momentum rank `29.20%`, pullback reclaim `15.34%`.

Interpretation: the broad 69-asset hybrid is a lower-return, lower-MDD, much more diversified challenger/shadow paper candidate. It is not a replacement for the current high-return standard live refit yet: the prior `hybrid_v3_5_optuna_three_profile_blend` standard refit remains far higher returning (`validation +38.0717%`, validation MDD `7.4789%`, gross `4.6902x`) but is more concentrated and train-dominant. The 69-asset result is therefore best treated as a paper/testnet challenger whose value is diversification and concentration reduction, not immediate performance superiority.

Real-transition decision: do **not** enable real-money. The path from this artifact to real requires a separate live adapter/handoff if this 69-asset rule set is to be executed, then at least `2-4` weeks of paper/testnet forward telemetry for limit-first fills, realized fee/spread/slippage/all-in round-trip cost against the `10bps` replay assumption, intended-vs-actual notional parity, partial/cancel/timeout rates, BBO/depth/fill-latency quality, protective-order attach/reconciliation, liquidation-distance/margin buffers, and continued asset/position concentration monitoring.

Local verification for this pass: repo-wide `ruff format --check .` and `ruff check .`; targeted tests `23 passed`; `compileall`; docs verification `117` markdown files; architecture check; hardcoded-parameter audit `new=0`; `git diff --check`; and full `pytest -q` `1519 passed in 104.14s` with max RSS `2,728,592 KiB` (<8 GiB). Commit, push, and CI follow in the same work session.

---

## 2026-05-30 KST — Binance 69-symbol direct 1m research-bar backfill

Updated the expanded Binance research universe to `69` compact USDT symbols after the 2026-05-29 addition of `QNTXUSDT`: `10` core crypto plus `59` active Binance USD-M `TRADIFI_PERPETUAL` symbols. `QNTXUSDT` is classified under the premarket/not-yet-standardized equity proxy group. This universe is a research/shadow-monitoring input only; it does not expand the selected paper/testnet live strategy universe and does not approve real-money trading. Hard safety flags remain `ready_for_real=false`, `real_money_execution=false`, and `real_execution_allowed=false`.

A direct `1m` OHLCV collector was added at `scripts/collect_binance_1m_research_universe.py` with two historical sources: official Binance public `data.binance.vision` USD-M kline archives for broad backfills and throttled `/fapi/v1/klines` for targeted current tails. The first REST attempt intentionally used small chunks but no global request throttle and triggered Binance HTTP `429/418`; the IP ban response stated `banned until 2026-05-30T04:51:48.940Z`. The collector was therefore hardened with a global request throttle and explicit 429/418 ban backoff, and the historical backfill was rerun through official `data.binance.vision` to avoid further REST-weight pressure.

Historical collection result: `var/reports/data_collection/binance_1m_research_universe/binance_1m_research_universe_collection_latest.json`, source mode `data-vision`, range `2025-01-01T00:00:00Z` through `2026-05-28T23:59:59.999Z` (latest public daily file confirmed during the run). The run processed `69` planned symbols with `0` errors: `53` fetched in this run, `15` already up-to-date from the earlier partial REST write/resume state, and `1` initially empty (`QNTXUSDT`, because it listed after the currently published 2026-05-28 data-vision daily boundary). It fetched/upserted `10,071,511` rows from `1,661` source files, made `2,326` archive requests, and had `665` expected pre-listing missing files. Runtime evidence: `/usr/bin/time` wall `2:23.92`, max RSS `235,036 KiB`, below the 8 GiB memory budget.

After the Binance REST ban cleared, a separate throttled FAPI tail job filled the unpublished current tail and `QNTXUSDT`: `var/reports/data_collection/binance_1m_research_universe_fapi_tail/binance_1m_research_universe_collection_latest.json`, source mode `fapi`, range `2026-05-29T00:00:00Z` through `2026-05-30T04:40:59.999Z`, `69/69` symbols ok, `0` errors, `92,679` fetched/upserted rows, `123` requests, max RSS `133,636 KiB`. `QNTXUSDT` now has `1,226` 1m rows from `2026-05-29T08:15:00Z` through `2026-05-30T04:40:00Z`. Final coverage artifact: `var/reports/data_collection/binance_1m_research_universe_coverage/binance_1m_research_universe_coverage_latest.json`; direct stored 1m coverage now has `69/69` non-empty symbols and `11,559,900` rows under `data/market_parquet/exchange=binance/symbol=*/timeframe=1m`, with every symbol covered through at least `2026-05-30T04:40:00Z`. Full-year crypto symbols have `740,441` rows each from 2025-01-01 through 2026-05-30 04:40 UTC. Storage footprint for the direct Binance 1m tree is about `245M`.

Operational decision: use official `data.binance.vision` monthly/daily 1m archives for broad historical research-bar coverage, then use throttled `/fapi/v1/klines` only for small targeted tails such as newly listed symbols and the most recent unpublished daily gap. Direct 1m bars are sufficient for broad alpha discovery/refit screening at 30m+ source timeframes, but promotion to execution-quality live evidence still requires the existing raw-first/1s or paper/testnet fill/BBO/slippage telemetry gates. Real-money remains blocked.

## 2026-05-28 KST — Binance TradFi perpetual universe added for future monitoring/refits

Recorded the current Binance USD-M `TRADIFI_PERPETUAL` research universe without running any data collection. Source check: `Binance USD-M Futures /fapi/v1/exchangeInfo` at `2026-05-28T13:40:47Z`. The standard expanded research universe is now `68` compact USDT symbols: `10` core crypto plus `58` TradFi perpetual symbols. Future refresh/monitoring jobs should keep these symbols current and make them available to strategy discovery, but every new asset remains shadow/research-only until it passes the standard latest-8-week validation, final-refit, paper/testnet handoff, and live telemetry gates. Real-money flags remain hard-false.

Core crypto watch set (`10`): `BTCUSDT`, `ETHUSDT`, `SOLUSDT`, `BNBUSDT`, `TRXUSDT`, `XRPUSDT`, `DOGEUSDT`, `ADAUSDT`, `AVAXUSDT`, `TONUSDT`.

TradFi groups (`58` total):

- Commodity / metal / energy proxies (`8`): `XAUUSDT`, `XAGUSDT`, `XPTUSDT`, `XPDUSDT`, `COPPERUSDT`, `CLUSDT`, `BZUSDT`, `NATGASUSDT`.
- ETF / index-linked proxies (`5`): `QQQUSDT`, `SPYUSDT`, `EWYUSDT`, `EWJUSDT`, `SOXLUSDT`.
- Equity-linked proxies (`43`): `TSLAUSDT`, `INTCUSDT`, `HOODUSDT`, `MSTRUSDT`, `AMZNUSDT`, `CRCLUSDT`, `COINUSDT`, `PLTRUSDT`, `PAYPUSDT`, `METAUSDT`, `NVDAUSDT`, `GOOGLUSDT`, `AAPLUSDT`, `TSMUSDT`, `MUUSDT`, `SNDKUSDT`, `MSFTUSDT`, `AVGOUSDT`, `BABAUSDT`, `AMDUSDT`, `QCOMUSDT`, `USARUSDT`, `LITEUSDT`, `ORCLUSDT`, `DISUSDT`, `UBERUSDT`, `CSCOUSDT`, `HDUSDT`, `MRVLUSDT`, `CRWVUSDT`, `WMTUSDT`, `JPMUSDT`, `VUSDT`, `BRKBUSDT`, `FLNCUSDT`, `DRAMUSDT`, `RKLBUSDT`, `CBRSUSDT`, `NBISUSDT`, `WDCUSDT`, `ARMUSDT`, `BEUSDT`, `COHRUSDT`.
- Premarket / not-yet-standardized equity proxies (`2`): `SPCXUSDT`, `OPENAIUSDT`.

Implementation notes: added the side-effect-free canonical list in `src/lumina_quant/research_universe.py`; wired `scripts/research/build_multiasset_exchange_coverage_inventory.py` defaults to that expanded slashed universe; and expanded `scripts/research/run_alpha_zoo_multi_asset_monitoring_slate.py` asset-group fallbacks so future artifacts can classify all TradFi candidates. This did **not** modify the frozen selected live strategy universe or execute a backfill. `AlphaZooOptunaHybridLiveStrategy` remains the current paper/testnet strategy identity; the `profit_moonshot_alpha_zoo` path remains only a historical artifact namespace.

Estimated data-refresh effort if all 2025-to-current data is refreshed later under the 8GB memory budget: existing local storage already has `14` symbols with roughly `429M` 1s rows. A naive full-calendar `68`-symbol upper bound would be about `3.0B` 1s rows, but current Binance `onboardDate` evidence means the 58 TradFi perps only have about `241M` possible 1s rows from launch through `2026-05-28T13:40:47Z`. The practical full available universe is therefore roughly `684M` rows, with incremental missing coverage from the current local store around `250-300M` rows before retries and compaction overhead. For an update from the current repo data state, expect about `8-18h` staged under 8GB (`~2-6h` existing crypto/metals tails plus `~6-12h` TradFi batches if archives/API are cooperative). A cold full rebuild of all available 2025-to-current coverage should be scheduled as `18-36h`; `2-3d` is the realistic worst-case if missing archives, API backfill, retries, or low-parallelism constraints dominate.

No data collection was run for this documentation/code universe update.

---

## 2026-05-28 KST — Standard live-refit rule: latest 8-week validation + Optuna full-parameter final refit

Established the new system standard for live-facing Alpha Zoo refits: update local raw-first/committed market data first, reserve the latest **8 complete weeks** as validation, tune all exposed strategy-internal hybrid parameters with Optuna, select using train+validation evidence while fitting/learning on train only, then run a final refit on train+validation for the frozen paper/testnet runtime artifact. There is intentionally no locked-OOS/test set in this live final-refit mode; the final live artifact is frozen after refit and does not self-train online. Warmup remains a ratio inside the train window and is now part of the Optuna search space. Real-money flags remain hard-false.

Data refresh and split:

- Watch universe refreshed/compacted for `BTCUSDT`, `ETHUSDT`, `SOLUSDT`, `BNBUSDT`, and `TRXUSDT` through `2026-05-28T10:59:59Z` in `data/market_parquet/market_ohlcv_1s/binance`.
- Standard split from latest complete 1h bars: train `2025-01-01T00:00:00Z` → `2026-04-02T10:00:00Z`; validation `2026-04-02T11:00:00Z` → `2026-05-28T10:00:00Z`; locked-OOS disabled for live final refit.
- Refresh/compaction evidence: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/standard_live_refit_20260528/`. Peak RSS stayed below the 8 GiB session budget: data refresh about `5.82 GiB` artifact peak / `6,045,100 KiB` `/usr/bin/time`; WAL compaction `1,565,284 KiB`.

Implementation/artifact updates:

- Added `src/lumina_quant/alpha_zoo/live_training_policy.py` to make the latest-8-week validation/final-refit split deterministic and reusable.
- Added `scripts/research/run_alpha_zoo_standard_live_refit.py` as the standard wrapper around the Optuna hybrid decision runner.
- Updated `scripts/research/run_alpha_zoo_integer_leverage_optuna_hybrid_decision.py` so Optuna covers every exposed `HybridParams` field: bias alpha/combine ratio/window, MAPE/short-vol windows, warmup ratio, max single weight, volatility-regime threshold/boost shape, min/max boost, and default-weight-ratio range/steps. Grid remains comparison-only.
- Removed the old hard-coded data end from the 30m+/HTF alpha source loaders so refreshed committed data can be used without editing dates.
- New standard-refit outputs: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_standard_live_refit_20260528/alpha_zoo_integer_leverage_optuna_hybrid_decision_latest.json|md`.

Result after embedded `10bps` round-trip friction proxy:

| Profile / run | Train | Validation | Validation MDD | RPT bps T/V | Trades T/V | Gross | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Prior selected 2026-05-24 Optuna v3.5 | `+611.5025%` | `+138.3170%` | `18.9796%` | `83.39 / 79.17` | `3363 / 789` | `4.3646x` | Old split with locked-OOS report-only. |
| Standard live refit Optuna v3.5 selection/final-refit | `+3447.4699%` | `+38.0717%` | `7.4789%` | `368.22 / 36.77` | `4167 / 447` | `4.6902x` | Latest-8-week validation; no live test set. |

The selected profile remains `hybrid_v3_5_optuna_three_profile_blend`, now final-refit on train+validation. Final refit active weights are aggressive `91.4204%`, balanced `4.1881%`, growth `4.3915%`; asset gross notional fractions are approximately ETH `0.9573x`, SOL `2.0035x`, and TRX `1.7295x`. The recent validation is much lower than the old validation split but remains positive, above the 2% gate, below the 12% validation-MDD cap, and above the 10bps return-per-turnover threshold. The lower validation drawdown is an improvement; the very large train-vs-validation gap and aggressive sleeve concentration remain the main overfit/concentration risks to monitor in paper/testnet.

Verification for the standard-refit patch passed locally: changed-file Ruff pass; targeted live-training/hybrid tests `12 passed`; `compileall` over `src scripts tests`; repo-wide `ruff format --check .` and `ruff check .`; architecture check; docs verification; hardcoded-parameter audit `total=567 new=0 baselined=567`; `git diff --check`; and full `pytest -q` `1506 passed in 89.76s` with max RSS `2,736,260 KiB` (<8 GiB). The standard-refit runner itself completed with `/usr/bin/time` max RSS `6,324,184 KiB` and artifact peak `6175.96 MiB` (<8 GiB).

Safety conclusion: paper/testnet-only candidate evidence was refreshed, but real-money remains blocked. The latest artifacts still keep `ready_for_real=false`, `real_money_execution=false`, and `real_execution_allowed=false`; live execution must use the frozen artifact and collect realized BBO/fill/fee/slippage/reconciliation telemetry before any future real-money discussion.

---

## 2026-05-28 KST — Research-note path canonicalization

Renamed the confusing strategy/date-specific research-note paths into the stable `docs/research_note/` directory:

- Current cumulative note: `docs/research_note/research_note.md`.
- Global source/history ledger: `docs/research_note/research_history.md`.
- Archived state-distilled predecessor note: `docs/research_note/state_distilled.md`.

Future sessions should prepend new research diary entries here in latest-first order and keep strategy/project names inside the entry body instead of the filename. Large generated evidence remains under `var/reports/`; session checkpoints remain in `.omx/notepad.md`.

---

## 2026-05-27 KST — Live MARKET_WINDOW hot-path optimization after Rust conversion review

After converting the Alpha Zoo live state-signal state machines to optional Rust, the next live/paper/testnet bottleneck candidate was reviewed without opening real-money execution. User-stream order-state projection measured about `518k events/sec`, so it was not the priority. The live rolling `MARKET_WINDOW` path slowed as the window length grew because each internal event repeatedly re-normalized already-canonical Python rows and revalidated schema rows.

The fix keeps the public/runtime API in Python and does **not** weaken safety gates. `build_market_window_event` now has an internal trusted fast path (`bars_1s_already_normalized=True`) used only by canonical producers (`RollingWindowAggregator` and committed materialized snapshots). External payloads still go through full `normalize_bars_1s` validation. Binance live push now extends already-normalized rows directly, and rolling history pruning only scans the touched symbol, preserving stale-symbol carry-forward behavior while reducing per-tick overhead. A Rust FFI rewrite was not selected for this boundary because the final `MarketWindowEvent` remains a Python tuple/dict payload and the measured bottleneck was Python object validation, not numeric compute.

Benchmark artifact: `var/reports/native_acceleration_20260527/market_window_contract_benchmark_latest.json`. Local evidence: generic builder `0.0005508s/eval`, trusted builder `0.000001006s/eval` (`~547x` builder speedup), 5-symbol/300s rolling aggregation `16.5k ticks/sec`, max RSS `42.5MB`. This is a live data-path optimization only; `ready_for_real=false`, `real_money_execution=false`, and `real_execution_allowed=false` remain unchanged until paper/testnet fill/BBO/slippage/reconciliation telemetry proves the live assumptions.

---

## 2026-05-27 KST — Python-wrapped Rust live state-signal acceleration

After the operator clarified that useful Rust conversions should still be exposed through Python, the live Alpha Zoo state-machine kernels were moved behind an optional Rust backend without changing the public API. New backend: `native/rust_live_signals`; Python wrapper: `lumina_quant.alpha_zoo.native_live_signal_backend`; runtime control: `LQ_LIVE_SIGNAL_BACKEND=auto|python|rust` and `LQ_LIVE_SIGNAL_BACKEND_DLL`.

The accelerated functions are `debounced_state_signal` and `trailing_state_signal` in `lumina_quant.alpha_zoo.optuna_hybrid_signals`, which are pure deterministic loops used by `AlphaZooOptunaHybridLiveStrategy`. The Python wrapper remains the final interface and falls back to the original Python implementation when the Rust release library is unavailable. Explicit `rust` mode fails fast for diagnostics.

Local benchmark evidence is stored at `var/reports/native_acceleration_20260527/live_signal_backend_benchmark_latest.json`: 50,000 rows, 20 evaluations, Python `0.1349s/eval`, Rust `0.000544s/eval`, total speedup `247.80x`, exact debounced/trailing state-array parity (`max_abs_diff=0.0`). The benchmark max RSS was `97,200 KiB`, below the 8 GiB budget.

Scope boundary: exchange order submission, Binance/MT5/Polymarket HTTP/WebSocket clients, and raw network data collection were not moved in this pass because they are protocol/I/O-bound and safety/venue semantics dominate latency. Existing raw aggTrades→1s OHLCV and 1s WAL append acceleration remains under `native/rust_rawfirst`. Real-money safety is unchanged: paper/testnet-only artifacts still keep `ready_for_real=false`, `real_money_execution=false`, and `real_execution_allowed=false`.

---

## 2026-05-27 KST — Repo-wide format/Rust hygiene baseline

Follow-up cleanup normalized the whole tracked Python/Rust/code-hygiene surface after the live adapter split. This is a behavior-preserving cleanup pass: the selected `hybrid_v3_5_optuna_three_profile_blend` artifacts, paper/testnet-only live contract, limit-first order defaults, 10bps replay assumption, and hard real-money veto remain unchanged.

Changes made:

- Applied repo-wide `ruff format` to remove the previously known formatting drift outside the live-adapter patch.
- Applied `cargo fmt` to native Rust crates and added CI checks for `cargo fmt --check`, `cargo check`, and `cargo test` on `native/rust_metrics` and `native/rust_rawfirst`.
- Added `.gitattributes` to lock repository text hygiene to LF by default while preserving CRLF for Windows launchers (`*.bat`, `*.cmd`, `*.ps1`).
- Updated the hardcoded-parameter audit baseline after formatting moved source coordinates; audit result remains `total=567 new=0 baselined=567`.
- Added CI `ruff format --check .` so future pushes fail on format drift instead of relying on local-only checks.

Validation evidence for this pass: full pytest `1480 passed` with max RSS `2,746,164 KiB` (<8 GiB); Ruff format/check pass; compileall pass; docs verification pass; architecture check pass; hardcoded-parameter audit `new=0`; `uv lock --check` pass; `git diff --check` pass; native Rust format/check/tests pass; GPU auto and forced runtime smokes pass with CPU/GPU row parity.

---

## 2026-05-27 KST — Optuna hybrid live adapter cleanup/reproducibility lock

Follow-up cleanup refactored the selected `hybrid_v3_5_optuna_three_profile_blend` live path without changing the frozen research result or live safety posture. The previously monolithic `src/lumina_quant/alpha_zoo/optuna_hybrid_live_strategy.py` was split into:

- `src/lumina_quant/alpha_zoo/optuna_hybrid_config.py` for frozen Optuna/integer-leverage artifact loading, paper/testnet governance validation, selected profile/source sleeve reconstruction, and v3.5 allocator metadata.
- `src/lumina_quant/alpha_zoo/optuna_hybrid_signals.py` for completed-bar handling, debounced/trailing state rules, intrabar risk-frame helpers, and source-family signal math.
- `src/lumina_quant/alpha_zoo/optuna_hybrid_live_strategy.py` for runtime state, component intrabar guards, notional-fraction sizing metadata, and `SignalEvent` emission.

Behavior was locked before the split. `tests/test_alpha_zoo_optuna_hybrid_live_strategy.py` now asserts the selected artifact metrics and live decision contract directly: selected profile `hybrid_v3_5_optuna_three_profile_blend`, Optuna TPE v3.5, train `+611.5025%`, validation `+138.3170%`, locked-OOS report-only `+20.8319%`, validation MDD `18.9796%`, locked-OOS MDD `10.5735%`, RPT proxy `83.39/79.17/25.29bps`, trades `3363/789/362`, gross notional fraction `4.3645889x`, final profile weights aggressive `57.2699%`, balanced `7.9831%`, growth `8.0672%`, and train+validation average weights aggressive `78.1094%`, balanced `10.9140%`, growth `10.9767%`. The same tests lock the latest `paper_testnet_live_decision_latest.json` limit-first contract: `default_order_type=LMT`, `allow_market_orders=false`, `limit_price_mode=one_tick_worse`, one-tick offset, and side-aware limit entry/exit policies.

Governance is unchanged. The live adapter remains paper/testnet-only: `ready_for_paper=true`, `ready_for_real=false`, `real_money_execution=false`, `real_execution_allowed=false`; locked-OOS is still gate/report-only and the 10bps round-trip cost/RPT threshold remains the primary assumption. This cleanup is a structure/reproducibility pass, not a new discovery or performance rerun.

---

## 2026-05-26 KST — Limit-first live execution hardening

Implemented the operator requirement that market orders remain optional and that live/paper execution defaults to limit orders. `live.default_order_type` now defaults to `LMT`, `live.allow_market_orders=false`, and `live.limit_price_mode=one_tick_worse`: BUY limits are priced one exchange tick above the reference and SELL limits one tick below it for fast bounded execution. `same_price` and `one_tick_better` remain configurable alternatives. The Portfolio signal path now applies this policy to entries, shorts, and reduce-only exits; `LiveTrader` risk-flatten orders use the same limit policy. `LiveExecutionHandler` rejects market parent orders when the live market-order guard is present and disabled.

Paper/testnet Binance USD-M protective algo orders were moved from default market-style `STOP_MARKET` / `TAKE_PROFIT_MARKET` to default conditional limit `STOP` / `TAKE_PROFIT`, with `triggerPrice`, side-aware one-tick-worse `price`, and `GTC` time-in-force. Market-style parent or protective orders remain possible only by explicit config opt-in (`live.allow_market_orders=true`, plus the relevant order-style setting) and still do not approve real-money. Refreshed `paper_testnet_live_decision_latest.json|md` records `limit_order_contract`, `protective_order_style=limit`, and hard-false real-money flags.

Research interpretation is unchanged: backtest results still embed the 10bps round-trip cost proxy, but one-tick-worse limits are marketable limits and may still incur taker fees; paper/testnet must record realized `fee_bps`, BBO/spread, limit reference/price, fill latency, partial fills, cancels, and all-in costs before any real-money review. `ready_for_real=false`, `real_money_execution=false`, and `real_execution_allowed=false` remain invariant.

---

## 2026-05-25 KST — Paper/testnet exchange-side protective orders and asset applicability

One additional live-readiness gap was closed for paper/testnet: after an entry order is confirmed filled, `LiveExecutionHandler` can submit Binance USD-M Futures conditional algo protective orders through the request gateway. The current supported paper/testnet protective types are conditional limit `STOP` and `TAKE_PROFIT`, routed through the repo-native Binance Futures adapter to `POST /fapi/v1/algoOrder` with side-aware one-tick-worse prices. The order uses the same parent component quantity and the signal/component `position_side`; it intentionally remains paper/testnet-only. Market-style `STOP_MARKET` / `TAKE_PROFIT_MARKET` is retained only as an explicit market opt-in path. Real mode still fails closed unless a separate artifact explicitly approves exchange-side protective order handling and measured paper/testnet telemetry.

Asset applicability was checked beyond the dominant SOL sleeve. The intrabar guard/protective-order logic is symbol-generic and now has tests for the selected frozen source assets `ETHUSDT`, `SOLUSDT`, and `TRXUSDT`, including both long and short protective directions. This does not add new alpha assets or change the frozen hybrid portfolio; it verifies that the live protective machinery is not hard-coded to one asset family.

Remaining live limitations after this pass: no actual exchange paper/testnet fill sample has been collected yet, exact queue priority is still unknowable and proxy-only, and real-money remains vetoed until at least two weeks of fill/slippage/reconciliation telemetry confirms the 10bps cost and notional-parity assumptions.

Local verification for this follow-up passed: `ruff check .`, architecture check, `compileall`, hardcoded-parameter audit (`new=0`), `git diff --check`, and full `pytest -q`; full pytest was `1460 passed` with max RSS `2,752,152 KiB`, below the 8 GiB limit.

Payload-hardening update: the Binance conditional algo order adapter now uses a documented-field allowlist for `/fapi/v1/algoOrder`; internal parent/protection telemetry fields are not forwarded to the exchange request payload. Targeted exchange/state-machine/live-strategy regression tests passed (`25 passed`).

Final validation after payload hardening: `ruff check .`, architecture check, `compileall`, hardcoded-parameter audit (`new=0`), `git diff --check`, and full `pytest -q` passed; full pytest was `1460 passed` with max RSS `2,724,568 KiB`, below the 8 GiB limit.

---

## 2026-05-25 KST — Intrabar protective guard added for Alpha Zoo Optuna hybrid paper/testnet

The prior live-readiness caveat about intrabar exits and microstructure was narrowed. The adapter still uses completed `1h/2h/4h` bars for alpha decisions, but it now attaches a paper/local-simulation intrabar protective contract to entry signals. Each entry signal includes `component_id`, `stop_loss`, optional chandelier-style `trailing_percent`, `intrabar_protection`, and `microstructure_telemetry_required` metadata. The strategy maintains component-level intrabar guards and can emit a component `EXIT` signal when a `MARKET` event breaches the guard. The risk frame prefers `1m`, then `5m`, then source-timeframe fallback; stops are ATR/cost-floor based and are intended as a conservative paper/testnet risk overlay, not a train+validation optimizer surface.

This does **not** change the real-money conclusion. Real exchange-side protective orders remain unapproved until exchange-specific STOP/TAKE_PROFIT order support is wired and observed in paper/testnet. Exact queue priority remains impossible to know from exchange APIs; monitoring must use BBO/depth/fill-latency/partial-fill proxies. The refreshed `paper_testnet_live_decision_latest.json|md` records the `intrabar_protection_contract` and `microstructure_telemetry_contract`. Real-money flags remain false and the artifact veto stays active.

Local verification after the guard update: `ruff check .`, architecture check, `compileall`, hardcoded-parameter audit (`new=0`), `git diff --check`, and full `pytest -q` all passed; full pytest was `1456 passed` with max RSS `2,933,144 KiB`, below the 8 GiB session limit.

---

## 2026-05-25 KST — Live adapter limitation audit and paper/testnet blockers

After the paper/testnet live adapter was implemented, the live-readiness interpretation was tightened: the adapter is live-compatible for paper/testnet monitoring, but real-money startup remains intentionally blocked. The refreshed `paper_testnet_live_decision_latest.json|md` now records explicit `real_money_blockers`, `known_limitations`, and `paper_testnet_validation_requirements` alongside the existing `ready_for_real=false`, `real_money_execution=false`, and `real_execution_allowed=false` flags.

Current real-money blockers:

- The governing artifacts are still paper/testnet-only: `ready_for_real=false`, `real_money_execution=false`, and `real_execution_allowed=false`; readiness policy should keep `artifact_real_money_veto=true` for real mode.
- There is no exchange paper/testnet fill telemetry yet. Realized BBO spread, fees, slippage, rejects, timeouts, cancels, partial fills, position reconciliation drift, and stale-data recovery must be observed before any real-money review.
- The `10bps` assumption is a replay/gate round-trip friction proxy, not a live measured all-in cost. Paper/testnet monitoring must show realized all-in costs remain compatible with that assumption.
- The decision artifact deliberately keeps global `target_allocation=0.0` fail-closed. Live sizing depends on `SignalEvent.metadata.target_allocation`; paper/testnet must verify signal metadata to submitted-order notional parity.

Known disadvantages and research risks:

- The selected `hybrid_v3_5_optuna_three_profile_blend` is dominated by the aggressive source profile. It is a risk-managed allocator over correlated source profiles, not a clearly independent new alpha sleeve.
- Validation MDD `18.9796%` is near the relaxed 20% label and above the strict 12% promotion cap; locked-OOS report-only MDD is `10.5735%`.
- Train return `+611.5025%` versus validation `+138.3170%` indicates strong train dominance and potential optimizer overfit despite positive validation and locked-OOS report-only results.
- Source universe breadth is still limited in the promoted live adapter: the frozen sleeves are concentrated in SOL/ETH/TRX with BTC/BNB mainly as reference/watch symbols; broader assets remain shadow/data-extension work until OOS coverage and live telemetry exist.
- The live adapter evaluates completed `1h/2h/4h` bars only, so it does not model intrabar exits, queue priority, funding/fee drift, liquidation-engine edge cases, or exchange microstructure timing.
- Frozen-artifact replay avoids online learning and locked-OOS leakage, but regime drift or stale artifacts require a new train+validation-first research cycle rather than live self-tuning.

Paper/testnet evidence required next: realized BBO/all-in round-trip cost by symbol and timeframe, reject/timeout/cancel/partial-fill rates, signal-metadata-to-order-notional parity, position reconciliation drift, stale-data block/recovery behavior, liquidation-inclusive MDD/account wipeout telemetry, and at least two continuous weeks of observation before any future real-money discussion. Until then the correct conclusion remains: paper/testnet launch is allowed for evidence gathering; real-money launch is prohibited.

---

## 2026-05-25 KST — Optuna hybrid paper/testnet live adapter implemented

Implemented a live-ready but real-money-vetoed adapter for the selected `hybrid_v3_5_optuna_three_profile_blend` from `alpha_zoo_integer_leverage_optuna_hybrid_decision_20260524`. The runtime class is `AlphaZooOptunaHybridLiveStrategy`, registered as `live_opt_in`; its implementation lives in `src/lumina_quant/alpha_zoo/optuna_hybrid_live_strategy.py` with a thin strategy-registry wrapper at `src/lumina_quant/strategies/alpha_zoo_optuna_hybrid_live.py`.

Key live-readiness choices:

- Source universe is reconstructed from the frozen Optuna and integer-leverage artifacts, not a hand-picked allowlist. The exact six sleeves are SOL 1h debounced short-only, SOL 1h debounced long/short, SOL 1h shorter-hold debounced short-only, SOL 2h relative-strength chandelier breakout, ETH/BTC 2h residual reclaim, and TRX 4h volatility-adjusted trend.
- Runtime evaluates completed bars only and drops the active working bar. Required timeframes are `1h`, `2h`, and `4h`; watch symbols are `BTC/USDT`, `ETH/USDT`, `SOL/USDT`, `BNB/USDT`, and `TRX/USDT`.
- The v3.5 allocator is frozen from the artifact; there is no Optuna runtime dependency and no locked-OOS learning. locked-OOS remains gate/report-only.
- Live sizing uses `notional_fraction` and each signal emits `metadata.target_allocation = source_allocation_fraction * sum(final_profile_weight * integer_leverage)`. The decision artifact keeps global `target_allocation=0.0` so missing signal metadata fails closed. Generated risk caps are equity-scaled: max order notional `1.247444x`, max symbol exposure `1.427227x`, and total notional/margin `3.520744x`.
- The adapter and preflight policy keep the real-money veto: `paper_testnet_only=true`, `ready_for_real=false`, `real_money_execution=false`, and `real_execution_allowed=false`. Real-mode preflight is blocked by artifact veto while paper/testnet can pass when other operational checks are healthy.
- Strategy logic has a no-calendar/date-rule regression check; the paper/testnet handoff records the same `10bps` round-trip cost and RPT threshold assumption as the research artifacts.

New handoff writer: `scripts/ops/write_alpha_zoo_optuna_hybrid_live_decision.py`. Latest outputs:

- `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_integer_leverage_optuna_hybrid_decision_20260524/paper_testnet_live_decision_latest.json`
- `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_integer_leverage_optuna_hybrid_decision_20260524/paper_testnet_live_decision_latest.md`

Operator runbook updated at `docs/live-readiness/04-paper-trading-runbook.md` with the `--check-only` and writer commands plus the sizing/real-veto invariants. Verification on 2026-05-25 KST passed: targeted live adapter/readiness/start-live tests `32 passed`; post-architecture-fix full pytest `1455 passed in 102.70s` with max RSS `2,946,288 KiB` (<8 GiB); `ruff check .`; `uv run python scripts/check_architecture.py`; `python3 -m compileall -q src scripts tests`; hardcoded-parameter audit `total=567 new=0 baselined=567`; `git diff --check`; and `git diff --cached --check`. Real-money execution remains prohibited.

---

## 2026-05-24 KST — Optuna v3.5/v3.6 correction for the integer-leverage hybrid

The previous three-profile hybrid decision used a coarse `5%` grid and therefore was **not** equivalent to the repository's hybrid v3.5/v3.6 Optuna workflow. This was corrected with `scripts/research/run_alpha_zoo_integer_leverage_optuna_hybrid_decision.py` and `tests/test_alpha_zoo_integer_leverage_optuna_hybrid_decision.py`.

Artifact dir: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_integer_leverage_optuna_hybrid_decision_20260524/`. Latest outputs include `alpha_zoo_integer_leverage_optuna_hybrid_decision_latest.json|md`, comparison CSV, top-trials CSV, methodology note, paper/testnet handoff, hardcoded audit report, and `verification_summary_latest.md`.

Method correction:

- Source universe remains the same three paper/testnet integer-leverage profiles: `balanced_mdd12_gross5`, `growth_mdd20_gross8`, and `aggressive_mdd30_gross10_shadow`.
- The prior grid hybrid is retained only as a comparison row, not the optimizer-selected decision surface.
- Optuna uses `TPESampler`, `240` trials per version, and the same parameter family as `run_profit_moonshot_hybrid_v35_v36_fixed_inputs.py`: bias alpha/combine ratio, max single weight, MAPE/rolling score window, bias window, and short-volatility window.
- v3.5 mapping: warmup-learned default profile + rolling return/error weights + high-volatility boost + bias/exposure dampening.
- v3.6 mapping: v3.5 mechanics plus online adaptive default-profile refresh from rolling score evidence.
- Objective/learning/selection use train+validation only. locked-OOS is attached after frozen Optuna params and remains gate/report-only; all locked-OOS discovery/selection/objective/pruning/parameter-fitting flags are false.

Result after embedded `10bps` round-trip friction proxy:

| Profile | Optimizer | Train | Validation | locked-OOS report-only | Validation MDD | OOS MDD | RPT bps T/V/OOS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `balanced_mdd12_gross5` | source | `+74.6685%` | `+33.2153%` | `+5.5300%` | `11.6134%` | `7.2003%` | `30.91/57.02/22.21` |
| `growth_mdd20_gross8` | source | `+262.3353%` | `+71.6291%` | `+23.3695%` | `19.9983%` | `9.2371%` | `32.16/37.72/23.35` |
| `aggressive_mdd30_gross10_shadow` | source | `+438.4462%` | `+117.4976%` | `+27.5772%` | `29.4044%` | `12.3630%` | `38.81/44.16/21.87` |
| `hybrid_mdd20_three_profile_blend` | old grid baseline | `+262.3642%` | `+72.5692%` | `+21.2977%` | `19.9330%` | `9.0718%` | `33.78/39.96/22.97` |
| `hybrid_v3_5_optuna_three_profile_blend` | Optuna v3.5 | `+611.5025%` | `+138.3170%` | `+20.8319%` | `18.9796%` | `10.5735%` | `83.39/79.17/25.29` |
| `hybrid_v3_6_optuna_three_profile_blend` | Optuna v3.6 | `+296.4869%` | `+85.9099%` | `+11.5273%` | `15.7399%` | `8.4448%` | `51.05/62.78/17.91` |

Selected Optuna hybrid: `hybrid_v3_5_optuna_three_profile_blend`. Average train+validation profile weights are approximately aggressive `78.11%`, balanced `10.91%`, and growth `10.98%`; final active weights are lower because the v3.5 exposure-dampening rule can hold cash. It passes the paper/testnet candidate gates, but remains non-real-money: `ready_for_real=false`, `real_money_execution=false`, and `real_execution_allowed=false`. The paper/testnet handoff requires realized BBO/fill/all-in cost telemetry, replay/live notional parity, liquidation-inclusive MDD, account wipeout, and margin-buffer monitoring before any future review.

Verification passed locally: artifact invariant check; runner max RSS `6,357,368 KiB` (<8 GiB), elapsed `32:37.97`; targeted tests `13 passed`; `ruff check .`; `python -m compileall -q src scripts tests`; hardcoded audit `total=567 new=0`; diff checks; full pytest `1444 passed in 76.49s` with max RSS `2,723,060 KiB` (<8 GiB).

---

## 2026-05-24 KST — Three-profile integer-leverage hybrid decision

Added `scripts/research/run_alpha_zoo_integer_leverage_hybrid_decision.py` and `tests/test_alpha_zoo_integer_leverage_hybrid_decision.py` to answer whether the three current integer-leverage paper/testnet profiles can be blended into a hybrid candidate. The runner consumes the frozen `alpha_zoo_corr_integer_leverage_portfolio_latest.json`, reconstructs each source profile's 10bps-costed PnL stream from fixed strategy position states, and searches hybrid weights on a 5% grid with each source profile weight at least 10%. The hybrid selection objective and gates use train+validation only; locked-OOS is attached only after weights are frozen as report/gate evidence. All outputs remain paper/testnet-only with `ready_for_real=false`, `real_money_execution=false`, and `real_execution_allowed=false`.

Selected hybrid: `hybrid_mdd20_three_profile_blend`, weights `balanced_mdd12_gross5=15%`, `growth_mdd20_gross8=70%`, `aggressive_mdd30_gross10_shadow=15%`. It is a relaxed paper/testnet candidate, not a strict 12% validation-MDD promotion. Metrics after the embedded 10bps round-trip friction proxy:

| Profile | Gross | Train | Validation | locked-OOS report-only | Validation MDD | OOS MDD | RPT bps T/V/OOS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `balanced_mdd12_gross5` | `1.00x` | `+74.6685%` | `+33.2153%` | `+5.5300%` | `11.6134%` | `7.2003%` | `30.91/57.02/22.21` |
| `growth_mdd20_gross8` | `3.90x` | `+262.3353%` | `+71.6291%` | `+23.3695%` | `19.9983%` | `9.2371%` | `32.16/37.72/23.35` |
| `aggressive_mdd30_gross10_shadow` | `4.90x` | `+438.4462%` | `+117.4976%` | `+27.5772%` | `29.4044%` | `12.3630%` | `38.81/44.16/21.87` |
| `hybrid_mdd20_three_profile_blend` | `3.615x` | `+262.3642%` | `+72.5692%` | `+21.2977%` | `19.9330%` | `9.0718%` | `33.78/39.96/22.97` |

Interpretation: the hybrid slightly improves validation return and validation MDD versus the growth profile while keeping the MDD~20 risk label and preserving positive locked-OOS/RPT/liquidation gates. It does not dominate growth on locked-OOS return (`+21.30%` vs `+23.37%`) and it does not match aggressive return, because the aggressive profile's drawdown is deliberately diluted. Train+validation PnL correlations show why: balanced-growth is moderately correlated (`0.5790`), but balanced-aggressive (`0.8493`) and growth-aggressive (`0.8680`) are high, so the hybrid is a risk-return compromise rather than a new independent alpha sleeve.

Verification passed locally. Artifact dir: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_integer_leverage_hybrid_decision_20260524/`; latest JSON/Markdown, timestamped JSON, comparison CSV, weight-candidate CSV, methodology note, runner log, and `verification_summary_latest.md` were written. Verification log `local_verification_hybrid_decision_20260524T132911Z.log`: artifact invariant pass, targeted tests `9 passed`, `ruff check .`, `compileall`, hardcoded audit `total=567 new=0`, diff checks, and full pytest `1440 passed in 68.46s`. Runner max RSS `6,505,524 KiB` and full pytest max RSS `2,771,296 KiB`, both below 8 GiB.

---

## 2026-05-24 KST — Integer-leverage paper candidate set and strategy-integrity review

Operator feedback corrected the previous framing: the `growth_mdd20_gross8` profile with validation MDD near `19.9983%` is sufficiently strong for paper/testnet candidate observation, and the aggressive profile should also remain in the candidate set. The integer-leverage portfolio runner now distinguishes strict promotion from paper/testnet candidacy:

- `balanced_mdd12_gross5` — strict paper/testnet promotion candidate. Integer asset leverage map `SOLUSDT=2`, `TRXUSDT=1`, gross notional `1.00x`; train/validation/locked-OOS report-only returns `+74.6685%/+33.2153%/+5.5300%`; validation MDD `11.6134%`; locked-OOS MDD `7.2003%`; trade events `945/229/100`; return-per-turnover proxy `30.91/57.02/22.21bps`; liquidation/account-wipeout `0/0`.
- `growth_mdd20_gross8` — relaxed paper/testnet candidate, not a strict 12% MDD promotion. Integer map `ETHUSDT=8`, `SOLUSDT=4`, `TRXUSDT=12`, gross `3.90x`; train/validation/locked-OOS `+262.3353%/+71.6291%/+23.3695%`; validation MDD `19.9983%`; locked-OOS MDD `9.2371%`; trade events `879/200/104`; RPT proxy `32.16/37.72/23.35bps`; liquidation/account-wipeout `0/0`.
- `aggressive_mdd30_gross10_shadow` — relaxed paper/testnet candidate, still explicitly not strict promotion. Same integer asset map `ETHUSDT=8`, `SOLUSDT=4`, `TRXUSDT=12`, gross `4.90x`; train/validation/locked-OOS `+438.4462%/+117.4976%/+27.5772%`; validation MDD `29.4044%`; locked-OOS MDD `12.3630%`; trade events `1539/360/158`; RPT proxy `38.81/44.16/21.87bps`; liquidation/account-wipeout `0/0`.

Theoretical/implementation review was added as `strategy_integrity_review_latest.json|md` in `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_corr_integer_leverage_portfolio_20260524/`. Review status is `pass`. The active sleeves are momentum/trend/residual state rules, not calendar/date rules: SOL debounced momentum hysteresis, SOL relative-strength chandelier breakout, ETH/BTC relative residual reclaim, and TRX volatility-adjusted trend persistence. This is aligned with established momentum/trend-following evidence (Moskowitz, Ooi & Pedersen, 2012, *Journal of Financial Economics*, `https://w4.stern.nyu.edu/facdir/lpederse/papers/TimeSeriesMomentum.pdf`), crypto trend-following/regime-volatility research context (`https://arxiv.org/abs/2602.11708`), volatility-scaled momentum risk-control literature (`https://www.sciencedirect.com/science/article/abs/pii/S0275531917308322`), and Wilder-style ADX/trend-strength filtering (`https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/average-directional-index-adx`). These references justify the hypothesis class only; promotion still depends on repository backtests, paper/testnet fills, and gates.

Hard anti-calendar/hardcode checks were recorded:

- Candidate model IDs for the review are derived from the frozen PnL-correlation decision artifact and the profile rows, not from a hardcoded model allowlist.
- Metadata-level calendar/date token check found no forbidden hits among active paper candidate sleeves.
- Source-code calendar-feature grep across `run_alpha_zoo_debounced_efficiency_repair_discovery.py`, `run_alpha_zoo_30m_plus_alpha_booster_discovery.py`, `run_alpha_zoo_30m_plus_alpha_feedback_discovery.py`, and `run_alpha_zoo_asset_diverse_strategy_discovery.py` found no `dt.day`, `dt.weekday`, `dt.dayofweek`, `dt.month`, `dt.hour`, `day_of_week`, `weekday`, `month_end`, `hour_of_day`, or `time_of_day` strategy features.
- Integer-leverage runner hardcoded-ID grep found none of the selected model IDs embedded in the code.

Cost/slippage interpretation is now explicit. The runner imports and enforces `PRIMARY_ROUND_TRIP_COST_BPS = 10.0`; per transition, it charges `round_trip_cost_bps / 2` per side, so an entry+exit round trip is modeled as `10bps` all-in backtest friction. The promotion/monitoring RPT proxy threshold is also `avg_bbo_spread_bps_assumption 2.0 * multiplier 5.0 = 10.0bps`. This confirms the 10bps assumption is baked into the replay and gates, but it is **not** a live fill-derived slippage measurement; paper/testnet monitoring must still record realized BBO spread, fee/slippage/all-in round-trip cost, timeout/cancel/partial-fill hygiene, replay/live notional parity, liquidation-inclusive MDD, and account wipeout before any future real-money review.

Governance remains unchanged: all outputs keep `ready_for_real=false`, `real_money_execution=false`, and `real_execution_allowed=false`; locked-OOS is gate/report-only after train+validation freeze and all locked-OOS discovery/selection/objective/pruning/parameter-fitting flags are false. The existing four `quality_single_pair` baseline lanes remain preserved separately.

Verification passed after this correction. Artifact regeneration wrote timestamped JSON `alpha_zoo_corr_integer_leverage_portfolio_20260524T124248Z.json`, latest JSON/Markdown, profile CSV, handoff/preflight JSON/Markdown, `strategy_integrity_review_latest.json|md`, and `verification_summary_latest.md`. Local verification log `local_verification_integer_leverage_20260524T124342Z.log` records: artifact invariant pass; calendar-feature grep pass; hardcoded model-ID grep pass; targeted tests `14 passed`; `ruff check .`; `python -m compileall -q src scripts tests`; hardcoded-parameter audit `total=567 new=0`; diff checks; full pytest `1436 passed in 67.59s`. Runner max RSS was `6,396,380 KiB`; full pytest max RSS was `2,745,132 KiB`, both below the 8 GiB cap.

---

## 2026-05-24 KST — Corr-diversified integer leverage portfolio improves strict paper/testnet return

Following the PnL-correlation matrix decision, a new integer-leverage portfolio pass was added in `scripts/research/run_alpha_zoo_corr_integer_leverage_portfolio.py`. The method starts from the frozen 11-row corr-diversified slate, not the full duplicate 136-row book, and searches train+validation-only greedy subsets with integer per-asset leverage maps. locked-OOS remains post-freeze gate/report-only and is not used for discovery, objective, pruning, parameter fitting, ranking, subset selection, or leverage selection.

The strict promotion profile `balanced_mdd12_gross5` passes all hard paper/testnet criteria with active leverage map `SOLUSDT=2`, `TRXUSDT=1` and gross notional `1.00x`. Selected sleeves: SOL 1h debounced short-only, SOL 1h debounced long/short, SOL 2h relative-strength chandelier breakout, and TRX 4h volatility-adjusted trend. Metrics: train `+74.6685%`, validation `+33.2153%`, locked-OOS report-only `+5.5300%`; train/validation/OOS trade events `945/229/100`; validation MDD `11.6134%`; locked-OOS MDD `7.2003%`; RPT proxy `30.91/57.02/22.21bps`; liquidation/account-wipeout `0/0`. This is a material improvement over the previous equal-weight corr slate validation `+8.9654%` / locked-OOS `+2.3304%` while retaining the strict validation-MDD and 10bps efficiency gates.

For context only, relaxed shadow profiles show much higher returns but are not strict promotions: growth shadow (`ETHUSDT=8`, `SOLUSDT=4`, `TRXUSDT=12`, gross `3.90x`) has train/validation/OOS `+262.3353%/+71.6291%/+23.3695%` with validation MDD `19.9983%`; aggressive shadow (`ETHUSDT=8`, `SOLUSDT=4`, `TRXUSDT=12`, gross `4.90x`) has `+438.4462%/+117.4976%/+27.5772%` with validation MDD `29.4044%`. They are paper/testnet shadow-review only under the existing strict `12%` validation-MDD promotion rule.

Artifact dir: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_corr_integer_leverage_portfolio_20260524/`; includes latest JSON/MD, decision CSV, paper/testnet handoff, preflight JSON/MD, runner log, and verification summary. Governance remains unchanged: `ready_for_real=false`, `real_money_execution=false`, `real_execution_allowed=false`, paper/testnet-only decision/preflight/monitoring handoff, replay/live notional parity recorded, primary cost/RPT threshold `10bps`, and no calendar/date hack. Verification passed locally: artifact invariant check, targeted tests `13 passed`, `ruff`, `compileall`, hardcoded audit `new=0`, diff checks, and full pytest `1435 passed`.

---

## 2026-05-24 KST — PnL-correlation matrix decision: reject all-in, keep corr-diversified paper slate

Added `scripts/research/run_alpha_zoo_pnl_correlation_decision.py` and `tests/test_alpha_zoo_pnl_correlation_decision.py` to evaluate the current multi-asset paper/testnet monitoring book by actual replayed strategy PnL correlation rather than headline aggregate returns alone. The runner consumes `alpha_zoo_multi_asset_monitoring_slate_latest.json`, replays the already-discovered candidate definitions to capture per-bar PnL return streams, and writes correlation matrices plus a correlation-aware decision record under `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_pnl_correlation_decision_20260524/`.

Decision method recorded in `pnl_correlation_decision_methodology_latest.md`:

- Candidate universe for selection: `136` rows with `monitoring_status=paper_testnet_monitor` from the multi-asset slate.
- Additional report-only context: top `30` strict shadow representatives; these do not become promotable through this correlation pass.
- PnL stream: per-bar fractional strategy return after the primary `10bps` round-trip cost and each candidate's native `notional_fraction`.
- Correlation matrix: Pearson correlation of PnL streams, aligned on datetime across mixed `30m/1h/2h/4h/6h` bars; missing bars are filled with zero PnL before correlation.
- Selection surface: train+validation PnL correlation only, with validation-only correlation as a guard. locked-OOS correlation is saved only as report/monitoring evidence after the train+validation freeze.
- Greedy rule: sort by `monitoring_score_train_validation_only`, then accept a candidate only if its max absolute train+validation corr to already selected candidates is `<=0.70` and its max absolute validation corr is `<=0.75`.
- High-correlation clusters are treated as one alpha sleeve; duplicate parameter/sizing rows are rejected as correlated duplicates, not as bad standalone alphas.

The all-paper book is not suitable as an unscaled portfolio. The replay captured all `136/136` paper candidates, and the train+validation matrix has `9,180` pairs with mean absolute corr `0.3074`, max corr `1.0`, `1,285` pairs with `|corr|>=0.85`, `14` high-corr clusters, and largest cluster size `41`. Unscaled adoption would sum to `49.80x` gross notional; the unscaled portfolio replay is unstable (`-100%` validation return and `-35.0490%` locked-OOS report-only return), so the decision is `reject_unscaled_all_in_due_to_duplicate_pnl_clusters_and_excess_gross_notional`.

The corr-diversified paper/testnet-only subset has `11` candidates and `4.50x` unscaled gross notional. Equal-weight comparison (for research normalization, not execution sizing) is validation `+8.9654%` and locked-OOS report-only `+2.3304%`, versus all-paper equal-weight validation `+8.4824%` and locked-OOS `+2.5290%`. The selected slate keeps the main independent sleeves: SOL 1h debounced short-only, SOL 2h relative-strength chandelier, SOL 1h debounced long/short, a lower-correlation SOL 1h debounced short-only variant, ETH 1h/2h debounced variants, SOL 6h volatility-adjusted trend, ETH/BTC residual reclaim, SOL 4h debounced, TRX 4h volatility-adjusted trend, and an ETH 1h short-only debounced variant. Full IDs and corr matrix are in `selected_corr_diversified_candidates_latest.csv` and `selected_pnl_corr_train_validation_latest.csv`.

Governance remains unchanged: this is paper/testnet-only research; every artifact keeps `ready_for_real=false`, `real_money_execution=false`, and `real_execution_allowed=false`. locked-OOS is not used for discovery, selection, objective, pruning, parameter fitting, or correlation acceptance; it is report-only after the train+validation decision. The four existing `quality_single_pair` paper/testnet baseline lanes remain preserved separately.

Runner verification evidence: artifact generation succeeded with max RSS `6,481,872 KiB` (<8 GiB), captured paper PnL count `136`, selected count `11`, missing PnL `0`, and locked-OOS selection flag false. Targeted tests for the new runner passed (`4 passed`) before full verification.

Final verification for the PnL-correlation decision passed on 2026-05-24 KST. Runner max RSS was `6,481,872 KiB` (<8 GiB). Artifact invariants passed (`136/136` paper PnL streams captured, selected `11`, missing `0`, all real-money flags false, all locked-OOS selection/discovery/objective/pruning/fitting flags false). Targeted tests passed (`11 passed`), `ruff check .` passed, `python -m compileall -q src scripts tests` passed, hardcoded-parameter audit reported `total=567 new=0 baselined=567`, diff checks passed, and full pytest passed with `1431 passed in 74.52s` and max RSS `2,816,620 KiB` (<8 GiB). Verification summary: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_pnl_correlation_decision_20260524/verification_summary_latest.md`.

A time-sensitive test fixture was corrected during verification: frozen paper-forward 10bps artifacts used a `10,000` minute stale threshold against the 2026-05-17 refresh snapshot, which became stale on 2026-05-24. `scripts/research/run_alpha_zoo_7x_paper_forward_preflight.py` now uses a named 30-day stale override for historical paper/testnet handoff generation while still forcing `ready_for_real=false` and `real_money_execution=false`; this keeps research artifact tests deterministic without opening real-money readiness.

---

## 2026-05-24 KST — Multi-asset monitoring slate combines all recent paper/shadow Alpha Zoo lanes

Added `scripts/research/run_alpha_zoo_multi_asset_monitoring_slate.py` and `tests/test_alpha_zoo_multi_asset_monitoring_slate.py` to turn the recent frozen discovery artifacts into one paper/testnet-only monitoring book instead of watching only one or two headline candidates. The slate reads the debounced efficiency repair, 30m+ feedback, 30m+ booster, and asset-diverse strategy artifacts and normalizes paper candidates, top rows, and shadow shortlists by source artifact, symbol, asset group, timeframe, and family.

Artifact dir: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_multi_asset_monitoring_slate_20260524/`. Latest outputs include `multi_asset_monitoring_slate_latest.json|md`, timestamped JSON, `multi_asset_monitoring_rows_latest.csv`, `asset_monitoring_matrix_latest.csv`, `paper_monitoring_handoff_latest.json|md`, `no_real_money_guard_latest.json`, `artifact_generation_validation_latest.log`, `runner_time_latest.log`, `targeted_pytest_time_latest.log`, `full_pytest_time_latest.log`, and `verification_summary_latest.md`.

Result: `2488` normalized candidate rows and a `14`-symbol monitoring matrix. The paper/testnet monitoring book now contains `136` strict paper candidates across `ETHUSDT` (`15`), `SOLUSDT` (`119`), and `TRXUSDT` (`2`). It also preserves non-promotional monitoring coverage for the rest of the source universe: `ADAUSDT`, `AVAXUSDT`, `BNBUSDT`, `BTCUSDT`, `DOGEUSDT`, `TONUSDT`, `XAGUSDT`, `XAUUSDT`, `XPDUSDT`, `XPTUSDT`, and `XRPUSDT`. `AVAXUSDT`, `DOGEUSDT`, `XAUUSDT`, and `XRPUSDT` are coverage-blocked shadows because locked-OOS coverage/trade evidence is insufficient; `BNBUSDT` and `BTCUSDT` remain shadow-watchlist only; source-coverage-only symbols stay in the matrix so future data extension is visible.

Best paper/testnet monitor rows by train+validation-only priority remain: SOLUSDT 1h debounced momentum hysteresis efficiency repair (`+28.2198%` train, `+16.9294%` validation, `+2.4704%` locked-OOS report-only, RPT `24.69/59.72/21.11bps`), SOLUSDT 2h relative-strength chandelier breakout (`+37.4602%`, `+16.0919%`, `+4.2373%`, RPT `30.96/60.72/31.39bps`), ETHUSDT debounced/residual paper candidates, and TRXUSDT 4h volatility-adjusted trend candidates. The slate is a monitoring synthesis, not a new discovery selection: locked-OOS is not used for discovery, selection, objective, pruning, parameter fitting, or monitoring-score priority; it is gate/report-only after the source artifacts' train+validation freezes.

Governance remains unchanged: real-money execution is prohibited (`ready_for_real=false`, `real_money_execution=false`, `real_execution_allowed=false`), paper/testnet-only monitoring is allowed only after preflight, primary cost/RPT threshold remains `10bps`, replay/live notional parity must be recorded, and every monitored candidate must record realized BBO spread, fee/slippage/all-in round-trip cost, liquidation-inclusive MDD, and account wipeout. The four existing `quality_single_pair` baseline lanes remain preserved unchanged.

Verification passed locally on 2026-05-24 KST: runner max RSS `139092 KiB`; artifact invariant check passed; targeted monitoring/debounced/feedback/booster/asset-diverse tests `27 passed`; `ruff check .`; `python -m compileall -q src scripts tests`; hardcoded-parameter audit `new=0`; diff checks; full pytest `1427 passed in 65.82s` with max RSS `2759596 KiB`, all below the 8 GiB session cap.

---

## 2026-05-23 KST — Asset-diverse strategy discovery expands beyond one asset lane

A follow-up asset-diverse Alpha Zoo pass was added via `scripts/research/run_alpha_zoo_asset_diverse_strategy_discovery.py` with tests in `tests/test_alpha_zoo_asset_diverse_strategy_discovery.py`. The goal was to stop concentrating only on the latest SOL/ETH lane and search cross-asset-conditioned, single-symbol strategies across the locally available Binance universe while preserving the same strict paper/testnet-only controls.

Artifact dir: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_asset_diverse_strategy_discovery_20260523/`. Latest files include `alpha_zoo_asset_diverse_strategy_discovery_latest.json`, latest Markdown, timestamped JSON, candidate/decision/shadow CSVs, `paper_testnet_handoff_latest.json|md`, `no_promotion_shadow_shortlist_latest.json`, `runner_time_latest.log`, `full_pytest_time_latest.log`, and `verification_summary_latest.md`.

The run evaluated `97,560` candidates across three strategy families: cross-asset rank chandelier breakout, relative residual reclaim, and breadth-regime pullback/reclaim. Promotion-eligible locked-OOS universe was `BTCUSDT`, `ETHUSDT`, `SOLUSDT`, `BNBUSDT`, and `TRXUSDT`; shadow-only probes widened coverage to `ADAUSDT`, `AVAXUSDT`, `DOGEUSDT`, `TONUSDT`, `XRPUSDT`, and precious-metal proxy symbols `XAUUSDT`, `XAGUSDT`, `XPTUSDT`, `XPDUSDT`. Shadow symbols are explicitly non-promotable until they have the required locked-OOS coverage.

Result: `4` strict paper/testnet-only candidates, all ETHUSDT 2h `relative_residual_reclaim` vs BTCUSDT. Best row `a30fb_asset_diverse_residual_reclaim_2h_ethusdt_btcusdt_lb48_z1p0_hold6_4p0x_0p125_fa49c5d5`: train `+16.8301%`, validation `+4.7367%`, locked-OOS `+4.8120%`, validation MDD `4.89%`, trades `184/32/26`, return-per-turnover proxy `18.29/29.60/37.02bps`, locked-OOS liquidation/account-wipeout `0/0`. It is weaker than the prior SOL 2h booster headline return, but diversifies the live paper/testnet research book into an ETH/BTC relative-value state rule with positive locked-OOS and stronger OOS RPT.

Best non-promotable shadows came from XRPUSDT 1h cross-asset rank chandelier variants with train roughly `+36%` to `+55%`, validation roughly `+29%` to `+32%`, and train/validation RPT above 10bps. They are rejected because local XRPUSDT has no locked-OOS bars after 2026-03 and because shadow-only symbols cannot promote. This is useful evidence for the next data-extension step but not a paper handoff.

Governance remains unchanged: `ready_for_real=false`, `real_money_execution=false`, 10bps primary cost, return-per-turnover threshold `avg BBO spread * 5 = 10bps`, replay/live notional parity recorded, no calendar/date hack, single-symbol position state rules only, and locked-OOS is gate/report-only after train+validation ranking freeze with all OOS discovery/selection/objective/pruning/fitting flags false. Existing four `quality_single_pair` paper/testnet baseline lanes remain preserved.

Verification passed locally on 2026-05-23 KST: artifact invariant check, targeted tests `15 passed`, `ruff check .`, `compileall`, hardcoded audit `new=0`, diff checks, and full pytest `1422 passed in 61.85s`. Runner max RSS was `2,020,024 KiB`; full pytest max RSS was `2,769,980 KiB`, both below 8 GiB.

---

## 2026-05-23 KST — 30m+ booster pass finds materially stronger SOL 2h paper/testnet candidates

After the first 30m+ feedback pass produced acceptable but modest paper candidates, a stricter booster pass was added via `scripts/research/run_alpha_zoo_30m_plus_alpha_booster_discovery.py` with tests in `tests/test_alpha_zoo_30m_plus_alpha_booster_discovery.py`. The pass keeps native per-file 1s→30m bar construction, default `30m/1h/2h/4h/6h` timeframes, single-symbol promotable strategies, train+validation-only ranking, and locked-OOS as post-freeze gate/report only. No real-money path is enabled: every output keeps `ready_for_real=false` and `real_money_execution=false`.

Artifact dir: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_30m_plus_alpha_booster_discovery_20260523/`. Latest outputs include `alpha_zoo_30m_plus_alpha_booster_discovery_latest.json`, latest Markdown, candidate/decision/shadow CSVs, `paper_testnet_handoff_latest.json|md`, `no_promotion_shadow_shortlist_latest.json`, and `runner_time_v_latest.log`.

Result: `63,450` candidates evaluated across relative-strength chandelier breakouts, multi-horizon consensus momentum, trend-pullback reclaim, and volatility-squeeze range expansion. Strict paper/testnet-only pass rows increased to `46`; preferred booster target rows remain `0` because the best rows still have a slightly weak validation-half or preferred-MDD diagnostic. The best paper/testnet candidate is `a30fb_booster_rs_chandelier_2h_solusdt_lb12_atr0p25_rel0p005_trail3p0_hold18_4p0x_0p125_9eeb8c26`: SOLUSDT `2h` relative-strength chandelier breakout, lookback `12`, ATR breakout `0.25`, rel-strength threshold `0.5%`, trailing stop `3 ATR`, min-hold `18`, equivalent notional `0.50`. Metrics: train `+37.4602%`, validation `+16.0919%`, locked-OOS report-only `+4.2373%`, validation MDD `10.8554%`, trades `242/53/27`, liquidation/account-wipeout `0/0`, return-per-turnover proxy `30.96/60.72/31.39bps`. This is materially stronger than the prior 30m+ feedback leader (`+14.45%/+5.55%/+1.82%`) while staying paper/testnet-only.

Operational decision: this is the current strongest new-alpha paper/testnet candidate, but still not real-money ready. Preferred booster diagnostics require monitoring because validation is not positive in both halves and validation MDD is above the stricter preferred `10%` target, though it remains within the hard `12%` promotion gate. Keep the existing four `quality_single_pair` paper/testnet baseline lanes unchanged and run only paper/testnet observation with realized BBO/fill telemetry before any future review.

---

## 2026-05-23 KST — 30m+ Alpha feedback discovery promotes SOL/TRX paper/testnet-only candidates

Added `scripts/research/run_alpha_zoo_30m_plus_alpha_feedback_discovery.py` and `tests/test_alpha_zoo_30m_plus_alpha_feedback_discovery.py` as a new >=30m Alpha Zoo feedback pass, not a reversal or `quality_single_pair` retune. It constructs native 1s→30m bars with per-file chunk aggregation, derives `30m/1h/2h/4h/6h` candidates, ranks only train+validation, and uses locked-OOS only after ranking freeze as gate/report evidence. Web/reference anchors are persisted in the artifact for AdaptiveTrend-style crypto trend-following, dynamic time-series momentum, and Binance funding/taker-flow docs. Real-money remains prohibited throughout: `ready_for_real=false`, `real_money_execution=false`.

Artifact dir: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_30m_plus_alpha_feedback_discovery_20260523/`. Latest outputs include `alpha_zoo_30m_plus_alpha_feedback_discovery_latest.json`, timestamped JSON `alpha_zoo_30m_plus_alpha_feedback_discovery_20260523T090125Z.json`, latest Markdown, candidate/decision/shadow CSVs, `paper_testnet_handoff_latest.json|md`, `no_promotion_shadow_shortlist_latest.json`, and `verification_summary_latest.md`.

Result: `18450` candidates evaluated across volatility-adjusted trend persistence, Donchian ATR breakout, MA-slope ADX trend filter, and funding/OI/taker crowding continuation for BTC/ETH/SOL/BNB/TRX. Train-dominant sample-gate pass rows: `23`; execution-efficiency proxy pass rows: `73`; strict paper/testnet-only pass rows: `4`. The promoted rows are simple sparse trend-state rules, not execution permissions: SOLUSDT 6h volatility-adjusted trend persistence (`hold8`, cooldown 2, ADX20+low-vol filter) at equivalent 0.30 notional, plus TRXUSDT 4h volatility-adjusted trend persistence (`hold12`, cooldown 2) variants. Best promoted SOL row: train `14.4540%`, validation `5.5469%`, locked-OOS report-only `1.8230%`, validation MDD `10.7958%`, trades `164/42/23`, liq/wipeout `0/0`, RPT `29.38/44.02/26.42bps`. TRX rows clear the 10bps RPT gate narrowly (`15.07/12.64/10.51bps` and `14.83/12.64/10.51bps`) and should be treated as paper/testnet-only monitoring candidates.

Feedback loop: OMX worker-1 found a first real-data run shape that exceeded the 8GB cap, then validated chunked aggregation. The main runner was repaired to per-file native 1s→30m aggregation before combining shards; final leader real-data run max RSS `1,908,808 KB` (`1864.07 MiB`, <8 GiB), elapsed `3:11.64`. Team mode completed and was shut down after terminal task reconciliation; its worktree merge was skipped due overlapping untracked files, but its memory/test findings were integrated manually in the main worktree.

Verification passed on 2026-05-23 UTC/KST session: artifact postprocess invariants; targeted Alpha Zoo tests `20 passed`; `ruff check .`; `python -m compileall -q src scripts tests`; hardcoded-parameter audit `new=0`; `git diff --check`; `git diff --cached --check`; full pytest `1414 passed in 70.71s (0:01:10)` with max RSS `2,771,784 KiB` (<8 GiB). Verification summary: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_30m_plus_alpha_feedback_discovery_20260523/verification_summary_latest.md`. Ignored local time logs are in the artifact dir as `runner_time_v_latest.log` and `full_pytest_time_latest.log`.

---

## 2026-05-23 KST — Debounced efficiency repair discovery produces SOL paper/testnet-only candidates

A focused ETH/SOL debounced momentum hysteresis repair pass was added via `scripts/research/run_alpha_zoo_debounced_efficiency_repair_discovery.py` with tests in `tests/test_alpha_zoo_debounced_efficiency_repair_discovery.py`. The pass continues from the 2026-05-22 diverse train-dominant artifact and deliberately avoids reversal / `quality_single_pair` retuning. It searches simple state-rule repairs only: expanded min-hold, entry/exit threshold redesign, stronger hysteresis/debounce, volatility-regime filtering, ADX-like/trend-strength proxies, and post-exit cooldown over `1h/2h/4h` ETH/SOL bars.

Artifact dir: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_debounced_efficiency_repair_discovery_20260523/`. Latest files include `alpha_zoo_debounced_efficiency_repair_discovery_latest.json`, timestamped JSON `alpha_zoo_debounced_efficiency_repair_discovery_20260523T025254Z.json`, latest Markdown, `debounced_efficiency_repair_candidates_latest.csv`, `debounced_efficiency_repair_decisions_latest.csv`, `debounced_efficiency_repair_shadow_hypotheses_latest.csv`, `paper_testnet_handoff_latest.json`, `paper_testnet_handoff_latest.md`, `no_promotion_shadow_shortlist_latest.json`, and `artifact_generation_validation_latest.log`.

Governance is unchanged: locked-OOS is attached only after train+validation ranking freeze and all `uses_locked_oos_for_*` discovery/selection/objective/pruning/parameter-fitting flags are `false`; `no_calendar_date_hack=true`; primary round-trip cost remains `10bps`; return-per-turnover proxy threshold remains `avg_bbo_spread * 5 = 10bps`; all outputs keep `ready_for_real=false` and `real_money_execution=false`. Existing four `quality_single_pair` paper/testnet baseline lanes are preserved unchanged: active `7x/0.20`, balanced `6x/0.175`, validation leader `5x/0.20`, and efficiency reference `4x/0.175`.

Result: `36,000` candidates evaluated (`18,000` ETH, `18,000` SOL), `14,465` rows with train return >= validation return, `274` train-dominant sample-gate rows, `954` execution-efficiency proxy pass rows, and `82` full paper/testnet-only gate pass rows. Best paper/testnet candidate is `debrepair_debounced_efficiency_repair_1h_solusdt_short_only_lb12_e0p02_x0p005_hold36_cool0_none_3p0x_0p15_1e40357d`: SOLUSDT `1h` short-only, lookback `12`, entry `2%`, exit `0.5%`, min-hold `36`, no cooldown, no extra filter, `3x/0.15` notional parity. Metrics: train `+28.2198%`, validation `+16.9294%`, locked-OOS report-only `+2.4704%`, validation MDD `9.2418%`, trades `254/63/26`, liquidation/account-wipeout `0/0`, return-per-turnover proxy `24.69/59.72/21.11bps` across train/validation/locked-OOS. The previous SOL debounced short-only near-miss had OOS RPT around `3.69bps`; the repaired family exceeds the strict `10bps` proxy gate on all splits by reducing transitions while keeping train-dominant validation evidence.

Paper/testnet handoff is generated but remains real-money blocked. The handoff requires paper/testnet-only mode, replay/live notional parity confirmation, realized fee/slippage/all-in round-trip cost, BBO spread at submit, liquidation-inclusive MDD, and account-wipeout monitoring. No live/real-money order execution was attempted or authorized.

Verification passed on 2026-05-23 UTC: artifact regeneration max RSS `4,932,260 KiB` (<8 GiB); artifact invariant check (`82` paper candidates, locked-OOS flags false, real money disabled); targeted debounced/diverse tests `12 passed`; `ruff check .`; `python -m compileall -q src scripts tests`; hardcoded-parameter audit `new=0`; `git diff --check`; `git diff --cached --check`; full pytest `1407 passed` with max RSS `2,744,928 KiB` (<8 GiB). Verification log: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_debounced_efficiency_repair_discovery_20260523/local_verification_debounced_efficiency_repair_20260523T025328Z.log`.

---

## 2026-05-22 KST — Diverse train-dominant discovery: trust gate tightened to train >= validation

The prior HTF pass surfaced high validation returns, but candidates with `train_return < validation_return` are now treated as untrusted validation spikes rather than promotion leads. A new runner, `scripts/research/run_alpha_zoo_diverse_train_dominant_discovery.py`, was added to enforce `train_return >= validation_return` and train/validation ratio `>=1.0` for any promotion path while broadening the strategy set beyond reversal and quality-single-pair tuning. The runner is still paper/testnet research only: no order execution, `ready_for_real=false`, `real_money_execution=false`, and locked-OOS is attached only after train+validation ranking freeze as gate/report evidence.

Artifact dir: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_diverse_train_dominant_discovery_20260522/`. Required outputs written: `alpha_zoo_diverse_train_dominant_discovery_latest.json`, timestamped JSON `alpha_zoo_diverse_train_dominant_discovery_20260522T135607Z.json`, `alpha_zoo_diverse_train_dominant_discovery_latest.md`, `diverse_train_dominant_candidates_latest.csv`, `diverse_train_dominant_decisions_latest.csv`, `diverse_train_dominant_shadow_hypotheses_latest.csv`, and `artifact_generation_validation_latest.log`.

The diversified search evaluated `22,800` candidates across nine strategy families: stateful momentum hysteresis, debounced momentum hysteresis, volatility-contraction breakout, trend-pullback reentry, range z-score mean reversion, funding-carry momentum, OI expansion/flow trend, cross-sectional market-neutral momentum, and cross-sectional low-vol momentum rotation. Coverage widened to `BTCUSDT`, `ETHUSDT`, `SOLUSDT`, `BNBUSDT`, `TRXUSDT`, `XRPUSDT`, `DOGEUSDT`, and `ADAUSDT` on `1h/2h/4h/6h/12h` bars. `7,968` rows had train return >= validation return, `50` rows passed the strict train-dominant sample gate, `151` rows passed the execution-efficiency proxy gate, and `0` rows passed both; therefore no new paper/testnet promotion was made.

Best train-dominant sample-gate shadows are materially better than the old 4-lane validation baseline but still fail execution-efficiency gates:

- `divtd_debounced_hysteresis_trend_1h_ethusdt_long_short_lb6_e0p02_x0p005_hold12_3p0x_0p15_5ea41145` (`debounced_momentum_hysteresis`, ETHUSDT 1h long/short, min hold 12 bars): train `+25.0916%`, validation `+23.5473%`, locked-OOS report-only `+1.3122%`, validation MDD `4.6260%`, trades `596/147/55`, liquidation/account-wipeout `0`. Rejected because train return-per-turnover proxy is `9.356bps` (<10bps) and locked-OOS return-per-turnover proxy is `5.302bps` (<10bps).
- `divtd_debounced_hysteresis_trend_1h_ethusdt_long_short_lb6_e0p02_x0p005_hold12_4p0x_0p1_3425bd0a`: train `+22.4142%`, validation `+20.7652%`, locked-OOS report-only `+1.1768%`, validation MDD `4.1161%`, trades `596/147/55`, liquidation/account-wipeout `0`. Rejected for train/locked-OOS return-per-turnover proxy (`9.402bps` / `5.349bps`).
- `divtd_debounced_hysteresis_trend_1h_ethusdt_long_short_lb6_e0p02_x0p005_hold12_3p0x_0p1_bac39c24`: train `+16.9461%`, validation `+15.3254%`, locked-OOS report-only `+0.8981%`, validation MDD `3.1010%`, trades `596/147/55`, liquidation/account-wipeout `0`. Rejected for train/locked-OOS return-per-turnover proxy (`9.478bps` / `5.443bps`).
- `divtd_debounced_hysteresis_trend_1h_solusdt_short_only_lb6_e0p02_x0p005_hold12_3p0x_0p15_08f4d887`: train `+35.5640%`, validation `+9.9201%`, locked-OOS report-only `+0.5645%`, validation MDD `7.5629%`, trades `428/100/34`; rejected only because locked-OOS return-per-turnover proxy is `3.690bps` (<10bps).

Operational decision: no paper/testnet handoff from this pass. Keep the existing four `quality_single_pair` paper/testnet baseline lanes unchanged, but the next research should prioritize execution-efficiency repair on the ETH 1h debounced long/short family and SOL 1h debounced short-only family rather than retuning old reversal. The hard trust rule is now persisted: promotion candidates with `train_return < validation_return` are rejected as validation spikes, regardless of headline validation return.

Local verification passed on 2026-05-22 UTC: artifact invariant check; targeted Alpha Zoo tests `22 passed`; `ruff check .`; `python -m compileall -q src scripts tests`; hardcoded-parameter audit `new=0`; `git diff --check`; and full pytest `1400 passed` with max RSS `2,763,076 KiB` (<8 GiB). Verification logs: `local_verification_alpha_zoo_diverse_train_dominant_discovery_20260522T135607Z.log` and `local_verification_latest.log` in the artifact directory.

---

## 2026-05-22 KST — New 30m+ HTF momentum/crowding discovery finds higher-validation shadows

A new-alpha pass was added via `scripts/research/run_alpha_zoo_htf_momentum_crowding_discovery.py` rather than retuning the existing reversal or `quality_single_pair` families. The runner uses local Binance 1s OHLCV parquet resampled to `1h`, `2h`, `4h`, and `6h` bars for BTC/ETH/SOL/BNB/TRX, optionally joins local funding/open-interest/taker-flow feature points, and evaluates four genuinely new 30m+ families: HTF trend persistence, Donchian breakout continuation, funding-squeeze continuation, and liquid cross-sectional momentum. The external research/docs anchors for this direction are recorded in the artifact: AdaptiveTrend-style longer-horizon crypto trend-following, Binance funding, open-interest, and taker buy/sell volume documentation.

Artifacts were generated under `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_htf_momentum_crowding_discovery_20260522/`: latest JSON/Markdown, timestamped JSON `alpha_zoo_htf_momentum_crowding_discovery_20260522T124629Z.json`, `htf_momentum_crowding_candidates_latest.csv`, `htf_momentum_crowding_decisions_latest.csv`, `htf_momentum_crowding_shadow_hypotheses_latest.csv`, and `artifact_generation_validation_latest.log`. The run evaluated `2,732` candidates: `1,600` HTF trend persistence, `400` Donchian breakout, `540` funding-squeeze continuation, and `192` liquid cross-sectional momentum. Locked-OOS is explicitly report/gate-only after train+validation freeze (`uses_locked_oos_for_discovery=false`, `uses_locked_oos_for_selection=false`, `uses_locked_oos_for_objective=false`, `uses_locked_oos_for_pruning=false`, `uses_locked_oos_for_parameter_fitting=false`). All artifacts keep `ready_for_real=false` and `real_money_execution=false`.

Result: no full paper/testnet promotion yet (`paper_candidate_gate_pass_count=0`), but the new HTF family materially raised validation-return evidence versus the current 4-lane baseline. The best validation shadow is `htf_trend_4h_trxusdt_lb48_th0p005_5p0x_0p15_3a7d84a7` (`htf_trend_persistence`, TRXUSDT, 4h, lookback 48, threshold 0.5%, 5x/0.15): train `-8.4327%`, validation `+15.0262%`, locked-OOS report-only `+4.4971%`, validation MDD `2.6586%`, trades `234/43/34`, liquidation/account-wipeout `0`; rejected because train return and train/validation ratio are negative and train return-per-turnover proxy is below the spread gate. The strict backtest-sample gate pass is `htf_trend_1h_ethusdt_lb6_th0p02_5p0x_0p15_3a23dcc0` (ETHUSDT, 1h, lookback 6, threshold 2%, 5x/0.15): train `+1.4647%`, validation `+2.1537%`, locked-OOS report-only `+4.1498%`, validation MDD `6.4524%`, trades `570/122/36`, train/validation ratio `0.6801`, liquidation/account-wipeout `0`; it remains `validation_alpha_shadow_until_execution_efficiency` because train and validation return-per-turnover proxies (`0.343bps` and `2.354bps`) fail the `avg_bbo_spread * 5 = 10bps` proxy gate. One lower-sample efficiency proxy row (`htf_trend_4h_ethusdt_lb48_th0p04_3p0x_0p1_2956d92f`) passes return-per-turnover but fails validation/OOS sample counts.

Operational decision: keep the four existing `quality_single_pair` paper/testnet baseline lanes unchanged, do not promote the new HTF candidates to paper/testnet yet, and continue next research from the HTF momentum/crowding direction rather than further tuning the old reversal clue. The immediate next experiments should improve train robustness and turnover efficiency for the TRX 4h/6h validation leaders and the ETH 1h sample-pass shadow without using locked-OOS for selection: e.g., volatility-regime conditioning, fewer-transition holding/exit hysteresis, and actual paper/testnet BBO/fill telemetry once available.

---

## 2026-05-22 KST — Missing fill/BBO telemetry now falls back to a fail-closed backtest cut

The paper fill-efficiency gate now handles absent actual paper/testnet fill/BBO telemetry without ending as an empty stop state. When no fill JSONL is available, it still cuts the current sample-guarded universe using the existing 10bps backtest evidence from `alpha_zoo_sample_guarded_alpha_discovery_latest.json`. This fallback is explicitly paper/testnet-review-only: it never sets `ready_for_real=true` and never permits real-money execution.

The fallback pass rule is intentionally strict: candidate rows must already be `ready_for_paper=true` in the sample-guarded discovery, pass the primary 10bps promotion gate, pass the return-per-turnover-vs-spread proxy gate, satisfy split sample guards, and survive locked-OOS only as a post-freeze report gate with positive return and zero liquidation/account wipeout. Locked-OOS is still not used for discovery, selection, objective scoring, pruning, or parameter fitting.

The regenerated fill-efficiency artifact under `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_paper_fill_efficiency_gate_20260522/` records this policy in `gate_policy.backtest_fallback_policy` and writes `backtest_fallback_candidates_latest.csv`. Current result remains no-promotion: actual fill telemetry status is `pending_paper_testnet_fill_telemetry`; the fallback evaluated `976` sample-guarded candidates, emitted the top `60` backtest-cut rows for review, and found `0` fallback pass rows. All outputs keep `ready_for_real=false` and `real_money_execution=false`. Timestamped JSON: `alpha_zoo_paper_fill_efficiency_gate_20260522T112551Z.json`.


Final verification passed on 2026-05-22 UTC for the missing-fill fallback cut: artifact regeneration; targeted fill-efficiency/sample-guarded tests `11 passed`; full pytest `1389 passed` with max RSS `2,715,696 KiB` (<8 GiB); `ruff check .`; `python -m compileall -q src scripts tests`; `git diff --check`; `git diff --cached --check`; and artifact invariants for `ready_for_real=false`, `real_money_execution=false`, fill count `0`, and fallback pass count `0`. Verification log: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_paper_fill_efficiency_gate_20260522/local_verification_alpha_zoo_paper_fill_efficiency_gate_backtest_fallback_final_20260522T112551Z.log`.

---

## 2026-05-22 KST — Actual fill/BBO efficiency gate added, pending telemetry

Added a fail-closed actual paper/testnet telemetry gate for the Alpha Zoo sample-guarded path: `scripts/research/run_alpha_zoo_paper_fill_efficiency_gate.py`. It consumes the sample-guarded discovery artifact, the paper-forward monitoring contract, and an optional fill JSONL. This is not an execution runner and it never enables real-money trading.

The gate converts the earlier proxy condition into the actual telemetry contract to use once paper/testnet fills exist:

- Return per turnover: `sum(realized_pnl_quote) * 10000 / sum(abs(notional_quote))`.
- Average BBO spread: notional-weighted average `spread_bps_at_submit`.
- Primary edge gate: `realized_return_per_turnover_bps > avg_bbo_spread_bps * 5`.
- Cost gates: mean all-in cost `<=10bps`, p95 all-in cost `<=15bps`.
- Execution hygiene: timeout/cancel/partial-fill limits, zero liquidation, zero account wipeout.

No paper/testnet fill JSONL is available in the repo yet, so the generated artifact is intentionally fail-closed: status `pending_paper_testnet_fill_telemetry`, `fill_count=0`, `actual_fill_efficiency_gate_pass=false`, `ready_for_real=false`, and `real_money_execution=false`. Artifact dir: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_paper_fill_efficiency_gate_20260522/`; timestamped JSON `alpha_zoo_paper_fill_efficiency_gate_20260522T110432Z.json`, latest JSON/MD, `paper_fill_efficiency_decisions_latest.csv`, and `artifact_generation_validation_latest.log` were written.

Verification passed on 2026-05-22 UTC: artifact regeneration; new fill-efficiency gate tests `4 passed`; sample-guarded regression tests `6 passed`; full pytest `1388 passed` with max RSS `2,732,348 KiB` (<8 GiB); `ruff check .`; `python -m compileall -q src scripts tests`; `git diff --check`; and `git diff --cached --check`. Verification log: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_paper_fill_efficiency_gate_20260522/local_verification_alpha_zoo_paper_fill_efficiency_gate_20260522T110431Z.log`.

---

## 2026-05-22 KST — Sample-guarded discovery adds return/turnover-vs-spread proxy gate

The sample-guarded 10bps discovery runner was tuned to incorporate the execution-quality hint `return per turnover > avg BBO spread * 5`. The current frozen expanded-retune metric surface does **not** contain actual average BBO spread or exact turnover fields, so the artifact now records this as an explicit proxy rather than claiming live microstructure evidence:

- Average BBO spread assumption: `2.0bps`.
- Multiplier: `5.0`.
- Return/turnover proxy threshold: `10.0bps`.
- Turnover proxy formula: `trade_event_count * abs(leverage * allocation_fraction)`.
- Return/turnover proxy formula: `total_return * 10000 / turnover_proxy`.
- Train+validation proxy profile: `execution_efficiency_proxy_v1`, with locked-OOS excluded from ranking/objective/pruning/parameter-fitting.
- Locked-OOS proxy role: attached only after train+validation freeze as a report-only promotion gate.

Regenerated artifacts under `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_sample_guarded_alpha_discovery_20260520/`, including timestamped JSON `alpha_zoo_sample_guarded_alpha_discovery_20260522T105008Z.json`, latest JSON/Markdown/CSVs, and `artifact_generation_validation_latest.log`. The updated run still found `0` new paper candidates across `976` complete 10bps models: `252` thin-sample shadows, `724` reject/quarantine, `20` historical OOS-bucket quarantines, and `0` calendar quarantines. Execution-efficiency proxy pass counts were: train `184`, validation `376`, locked-OOS report gate `57`, and full proxy gate `40`; those rows still failed other sample, locked-OOS, or primary 10bps promotion requirements. The four existing `quality_single_pair` paper/testnet baseline lanes remain unchanged, and all artifacts keep `ready_for_real=false` and `real_money_execution=false`.

Verification passed on 2026-05-22 UTC: artifact regeneration; targeted sample-guarded tests `6 passed`; full pytest `1384 passed` with max RSS `2,729,152 KiB` (<8 GiB); `ruff check .`; `python -m compileall -q src scripts tests`; `git diff --check`; and `git diff --cached --check`. Verification log: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_sample_guarded_alpha_discovery_20260520/local_verification_sample_guarded_turnover_proxy_20260522T105007Z.log`.

---

## 2026-05-22 KST — Sample-guarded Alpha Zoo discovery rerun hardens no-promotion evidence

The 10bps sample-guarded discovery bundle was regenerated from `private/main` commit `711eeb4dbf83895b94f9fbf22d1afb79c4be1284` under `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_sample_guarded_alpha_discovery_20260520/`. The refreshed timestamped artifact is `alpha_zoo_sample_guarded_alpha_discovery_20260522T103013Z.json`; `alpha_zoo_sample_guarded_alpha_discovery_latest.json`, `alpha_zoo_sample_guarded_alpha_discovery_latest.md`, `sample_guarded_candidates_latest.csv`, `paper_candidate_decisions_latest.csv`, `shadow_hypotheses_latest.csv`, `cost_sensitivity_latest.csv`, and `artifact_generation_validation_latest.log` were updated in place.

The rerun again found `0` new paper candidates across `976` complete 10bps models. Status counts remain `252` `shadow_only_thin_sample` and `724` `reject_or_quarantine`; `956` rows were selection eligible, `20` historical OOS-bucket rows were quarantined, and `0` calendar rows were selected. Locked-OOS remains gate/report-only after train+validation ranking freeze: discovery/selection/objective/pruning/parameter-fitting/correlation flags are all `false` for locked-OOS use.

No real-money handoff was created. The generated decision status is `no_new_paper_promotion_shadow_shortlist`, with `ready_for_paper=false`, `ready_for_real=false`, `real_money_execution=false`, and `paper_execution_allowed=false`. The four existing `quality_single_pair` paper/testnet baseline lanes remain unchanged: active `7x/0.20`, balanced `6x/0.175`, validation leader `5x/0.20`, and efficiency reference `4x/0.175`. Replay/live notional parity remains recorded for candidates and baseline lanes.

Test hardening added regressions for locked-OOS-insensitive sample-guarded ranking, non-10bps source rejection, exact four-lane baseline preservation, non-empty no-promotion rejection reasons, and output CSV/markdown/generation-log content. Local verification passed on 2026-05-22 UTC: artifact regeneration; targeted sample-guarded/long-only/expanded-shadow/10bps retune tests `26 passed`; full pytest `1382 passed`; `ruff check .`; `compileall` over `src scripts tests`; `git diff --check`; and `git diff --cached --check`. Verification log: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_sample_guarded_alpha_discovery_20260520/local_verification_alpha_zoo_sample_guarded_alpha_discovery_20260522T103012Z.log`.

---

## 2026-05-21 KST — New-factor sample-guarded Alpha Zoo discovery remains shadow-only

A follow-up discovery added opt-in/default-zero factor families to `CryptoFxAlphaZooStateStrategy`: liquidity-sweep reversal, liquidity-sweep continuation, range-expansion breakout, and range-expansion fade. The source grid was expanded via `--include-sample-guarded-new-alpha-grid`, then the 10bps retune used Optuna (`--n-trials 96`) plus the new `--sample-guarded-composite-grid` to widen side/symbol/factor-family/threshold filters without calendar/date rules. Locked-OOS stayed gate/report-only after train+validation freeze; it was not used for discovery, selection, objectives, pruning, or parameter fitting.

Final artifacts are under `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_new_alpha_inversion_sample_guarded_discovery_20260521/`, sourced from `alpha_zoo_new_alpha_inversion_source_20260521/` and `alpha_zoo_new_alpha_inversion_10bps_optuna_20260521/`. The retune evaluated `2,585` models (`1,894` fresh trade-filter models), `75,104` train+validation trade-filter variants, and kept `94` live-promotable 10bps rows in the old quality-single-pair family. Strict sample-guarded promotion found `0` new paper candidates across `2,565` selection-eligible rows; `1,694` rows are shadow-only thin-sample and `891` are reject/quarantine. The discovery bundle remains `ready_for_paper=false`, `ready_for_real=false`, `real_money_execution=false`, and `paper_execution_allowed=false`.

Best new-family shadows:

- `alpha_zoo_range_sweep_blend`, BNB/USDT abs-score `>=2`, `7x/0.175`: train `+9.6065%` / `66` trades, validation `+13.4943%` / `14` trades, validation MDD `0.8534%`, locked-OOS report-only `-1.7994%` / `10` trades, liquidation/account-wipeout `0`. Rejected for train, validation, and locked-OOS sample counts plus negative locked-OOS and primary 10bps gate failure.
- `alpha_zoo_liquidity_sweep_continuation`, SOL/USDT short abs-score `>=1`, `7x/0.20`: train `+9.2147%` / `58` trades, validation `+10.7779%` / `17` trades, validation MDD `2.4691%`, locked-OOS report-only `-0.8948%` / `3` trades, liquidation/account-wipeout `0`. Rejected for all sample guards, negative locked-OOS, and primary 10bps gate failure.
- `alpha_zoo_range_expansion_breakout`, SOL/USDT short abs-score `>=1.5`, `6x/0.15`: primary 10bps gate passed, train `+30.6004%` / `105` trades, but validation was only `+0.1163%` / `29` trades and locked-OOS report-only was `+0.0631%` / `16` trades. Rejected for validation trades `<30`, locked-OOS trades `<20`, and validation return `<2%`.

Operational decision: do not promote a new alpha to paper/testnet from this pass. Keep the four existing `quality_single_pair` paper lanes unchanged (active `7x/0.20`, balanced `6x/0.175`, validation leader `5x/0.20`, efficiency reference `4x/0.175`). New factor families stay shadow-only for future sample accumulation or a genuinely broader causal factor search; do not weaken sample gates or use locked-OOS to rescue them.

Required outputs were written: `alpha_zoo_sample_guarded_alpha_discovery_latest.json`, timestamped JSON `alpha_zoo_sample_guarded_alpha_discovery_20260521T132201Z.json`, `alpha_zoo_sample_guarded_alpha_discovery_latest.md`, `sample_guarded_candidates_latest.csv`, `paper_candidate_decisions_latest.csv`, `shadow_hypotheses_latest.csv`, `cost_sensitivity_latest.csv`, `artifact_generation_validation_latest.log`, and `local_verification_latest.log`. Final verification passed on 2026-05-21 UTC, including the post-CI hardcoded-parameter audit fix: `ruff check .`; `uv run python scripts/audit_hardcoded_params.py` (`new=0`); targeted Alpha Zoo tests `17 passed`; `python -m compileall -q src scripts tests`; `git diff --check`; full pytest `1380 passed`; max full-test RSS `2,880,156 KiB` (~2.75 GiB, <8 GiB). Verification log: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_new_alpha_inversion_sample_guarded_discovery_20260521/local_verification_new_alpha_sample_guarded_ci_fix_20260521T133109Z.log`.

---

## 2026-05-21 KST — Sample-guarded 10bps Alpha Zoo discovery: no new paper promotion

A new sample-guarded discovery bundle was generated under `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_sample_guarded_alpha_discovery_20260520/` using the frozen expanded 10bps retune artifact. The new runner `scripts/research/run_alpha_zoo_sample_guarded_alpha_discovery.py` ranks only train+validation evidence across three profiles (`validation_strength_v1`, `train_validation_robustness_v1`, and `cost_efficiency_v1`) and attaches locked-OOS only after the train+validation profile freeze as gate/report-only. The artifact explicitly records `uses_locked_oos_for_discovery=false`, `uses_locked_oos_for_selection=false`, `uses_locked_oos_for_objective=false`, `uses_locked_oos_for_pruning=false`, and `uses_locked_oos_for_parameter_fitting=false`.

Strict sample-guarded promotion found `0` new paper candidates across `976` complete models (`956` selection-eligible, `20` historical OOS-bucket rows quarantined). Status counts are `252` `shadow_only_thin_sample` and `724` `reject_or_quarantine`; `ready_for_paper=false`, `ready_for_real=false`, and `real_money_execution=false`. The required 10bps paper promotion guard remains stricter than the high-validation shadow rows: `conservative_exit` variants still show large validation returns but fail locked-OOS/ratio/gate checks, while the prior long-only residual-reversal clue remains shadow-only (`43` rows, `0` paper candidates, max validation trades `18`, max locked-OOS trades `13`). The upstream selected metric surface includes side/factor/threshold filters (`LONG`, `crypto_residual_reversal`, `crypto_residual_momentum`, `volume_vwap_pressure`, abs-score thresholds `1.5/2.5/3.0`); no symbol-filtered variant survived into the frozen selected metric/gate-pass surface, so symbol ideas are reported as not-promoted rather than mined via locked-OOS.

The four existing `quality_single_pair` paper/testnet baseline lanes remain unchanged: active `7x/0.20`, balanced `6x/0.175`, validation leader `5x/0.20`, and efficiency reference `4x/0.175`. Replay/live notional parity is recorded for every sample-guarded row; all new decisions are no-promotion shadow/reject decisions and keep real-money blocked.

Required outputs were written: `alpha_zoo_sample_guarded_alpha_discovery_latest.json`, timestamped JSON `alpha_zoo_sample_guarded_alpha_discovery_20260521T110118Z.json`, `alpha_zoo_sample_guarded_alpha_discovery_latest.md`, `sample_guarded_candidates_latest.csv`, `paper_candidate_decisions_latest.csv`, `shadow_hypotheses_latest.csv`, `cost_sensitivity_latest.csv`, and `artifact_generation_validation_latest.log`. Local verification passed on 2026-05-21 UTC: artifact regeneration; targeted sample/retune/guarded tests `23 passed`; full pytest `1377 passed`; `ruff check .`; `python -m compileall -q src scripts tests`; `git diff --check`; and `git diff --cached --check`. Maximum full-test RSS was `2,831,596 KiB` (<8 GiB) after post-deslop rerun. Verification log: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_sample_guarded_alpha_discovery_20260520/local_verification_alpha_zoo_sample_guarded_alpha_discovery_20260521T110118Z.log`.

---

## 2026-05-20 KST — Long-only reversal guarded study kept the new family shadow-only

A dedicated guarded study was generated for the expanded-retune discovery under `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_long_only_reversal_guarded_study_20260520/`. It focuses on `alpha_zoo_high_confidence_long_only` with `dominant_factor_family=crypto_residual_reversal` and `abs_factor_score_min=1.5`, at the same fixed `10.0bps` all-in round-trip cost assumption.

The study ranks variants using train+validation only, then applies locked-OOS strictly as a post-freeze gate/report field. Locked-OOS is not used for selection/objective/pruning/parameter fitting. The target family has `43` positive train/validation/locked-OOS variants, but `0` pass the strict paper guard: maximum validation sample is only `18` trades versus the `30`-trade guard, maximum locked-OOS sample is only `13` trades versus the `20`-trade report gate, maximum train/validation return ratio is `0.4788` versus the `0.50` train-robustness guard, and the primary 10bps promotion gate pass count is `0`.

The train+validation leader remains `fresh_tv10_filter_family_crypto_residual_reversal_abs_score_ge_1p5_alpha_zoo_high_confidence_long_only_8p0x_0p2alloc`: validation `+14.1732%`, train `+5.0784%`, locked-OOS report-only `+1.8620%`, validation MDD `3.1214%`, locked-OOS MDD `1.8137%`, liquidation/account-wipeout `0`. Despite positive locked-OOS, it is still `ready_for_paper=false`, `ready_for_real=false`, and `real_money_execution=false` because the split sample and train robustness evidence are too thin.

Operational decision: keep the four quality-single-pair paper/testnet lanes running side-by-side, and keep the long-only residual-reversal family as shadow-only research until a future train+validation-only discovery can satisfy stricter split sample guards without using locked-OOS for selection.

Verification passed on 2026-05-20 UTC for the guarded study: artifact regeneration; targeted Alpha Zoo retune/paper-forward tests `29 passed`; full pytest `1375 passed`; peak full-test RSS `2,728,984 KiB` (<8 GiB); `ruff check .`; `python -m compileall -q src scripts tests`; and `git diff --check`. Verification log: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_long_only_reversal_guarded_study_20260520/local_verification_alpha_zoo_long_only_reversal_guarded_study_20260520T133551Z.log`.

---

## 2026-05-20 KST — Expanded filter retune found a new shadow-only validation family

An expanded 10bps train+validation retune was run under `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_10bps_expanded_filter_retune_20260520/` with `--n-trials 0`, `--hybrid-seed-count 0`, `--trade-filter-top-n 300`, `--min-trade-filter-trades 10`, and `--fresh-candidate-limit 600`. Runtime was `7:34.86`, peak RSS was `410,552 KiB` (`400.9 MiB`, well under 8 GiB). The expanded universe increased from `796` to `976` complete models and from `2,388` to `2,928` primary metric rows, but the live-promotable count stayed `56`; the active and balanced live-gate models remain the same `quality_single_pair` 7x/0.20 and 6x/0.175 lanes.

The new useful discovery is not a paper candidate yet: `alpha_zoo_high_confidence_long_only` filtered to `dominant_factor_family=crypto_residual_reversal` and `abs_factor_score_min=1.5`. The leading row is `fresh_tv10_filter_family_crypto_residual_reversal_abs_score_ge_1p5_alpha_zoo_high_confidence_long_only_8p0x_0p2alloc`: validation return `+14.1732%`, validation MDD `3.1214%`, train return `+5.0784%`, locked-OOS return `+1.8620%`, locked-OOS MDD `1.8137%`, and liquidation `0`. However it has only `18` validation trades and `13` locked-OOS trades, and it fails the primary 10bps promotion gate because train metrics are not above validation/locked-OOS metrics (`train_total_return_not_above_validation` and related train metric asymmetry reasons). Therefore the companion artifact `alpha_zoo_expanded_filter_shadow_selection_20260520/` marks it `ready_for_paper=false`, `ready_for_real=false`, `real_money_execution=false`, `paper_execution_allowed=false`, `shadow_observation_allowed=true`.

Expanded conservative-exit rescue did not produce a positive locked-OOS conservative candidate. The conclusion is now sharper: keep the four quality-single-pair lanes as actual paper/testnet execution candidates, and treat `high_confidence_long_only / crypto_residual_reversal` as the next shadow-only strategy family requiring a dedicated train+validation retune with stricter minimum trade-count guards before any paper execution artifact.

---

## 2026-05-20 KST — Four-lane paper-forward contract and shadow strategy audit

A follow-up four-lane artifact now joins the original active/balanced paper handoff with the validation-first handoff under `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_four_lane_shadow_discovery_20260520/`. The contract stays paper/testnet-only: `ready_for_paper=true`, `ready_for_real=false`, `real_money_execution=false`, and the primary research cost is still `10.0bps` all-in round-trip.

Four lanes to run side-by-side in paper/testnet:

| Lane | Model | Notional/equity | Validation return | Validation MDD | Locked-OOS return | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Active | `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_7p0x_0p2alloc` | `140%` | `+0.4724%` | `14.9117%` | `+1.8382%` | paper only |
| Balanced | `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_6p0x_0p175alloc` | `105%` | `+0.5942%` | `11.3653%` | `+1.5464%` | paper only |
| Validation leader | `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_5p0x_0p2alloc` | `100%` | `+0.5986%` | `10.8490%` | `+1.4956%` | paper only |
| Efficiency reference | `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_4p0x_0p175alloc` | `70%` | `+0.5561%` | `7.6999%` | `+1.1432%` | paper only |

The shadow strategy audit found higher validation families but no frozen-universe non-quality paper candidate. Top `conservative_exit` rows reach validation `+27.3844%`, yet locked-OOS is `-4.2507%`; top side/family threshold rows reach validation `+5.9487%`, yet locked-OOS is `-4.5444%`. Both remain shadow-only. The next true strategy-search step is an expanded train+validation-only retune/rescue pass, not real-money promotion.

---

## 2026-05-20 KST — Validation-first 10bps discovery after weak validation check

Follow-up validation-first discovery confirmed that the prior active 7x/0.20 handoff is not the best validation performer inside the frozen 10bps universe. The new artifact is under `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_validation_first_discovery_20260520/` and keeps `real_money_execution=false`, `ready_for_real=false`, and locked-OOS gate/report-only after train+validation ranking freeze.

Selected paper/testnet validation-first candidates:

- Validation return leader: `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_5p0x_0p2alloc` (`5x`, allocation `0.20`, isolated notional/equity `100%`). Validation return `+0.5986%`, validation MDD `10.8490%`, train return `+34.5152%`, locked-OOS return `+1.4956%`, liquidation `0`; preflight `ready_for_paper=true`, `ready_for_real=false`.
- Validation efficiency reference: `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_4p0x_0p175alloc` (`4x`, allocation `0.175`, isolated notional/equity `70%`). Validation return `+0.5561%`, validation MDD `7.6999%`, train return `+24.8845%`, locked-OOS return `+1.1432%`, liquidation `0`; preflight `ready_for_paper=true`, `ready_for_real=false`.

The frozen 10bps universe has `56` live-gate-passed candidates, but the best live-gate validation return is only `+0.5986%`. A material validation-edge audit found `0` zero-liquidation candidates with validation return `>1%` and positive locked-OOS return. High-validation alternatives do exist, especially `conservative_exit` variants with validation around `+20%` to `+27%`, but they fail promotion gates because locked-OOS returns are negative and train metrics are not above validation; they are shadow-only strategy hypotheses, not paper candidates.

Recommended next experiments are therefore not immediate real-money promotion: run a train+validation-only regime-gated `conservative_exit` rescue, side/symbol-specific `abs_score` thresholds for `quality_single_pair`, and continue paper-forward monitoring of the validation-first 5x/0.20 and 4x/0.175 lanes beside the existing active/balanced lanes. Real fill monitoring must still compare all-in round-trip bps to the 10bps mean / 15bps p95 contract before any real-money discussion.

---

## 2026-05-20 KST — 10bps Alpha Zoo paper/testnet preflight and monitoring handoff

Built the paper/testnet-only handoff bundle for the final 10bps active profile and balanced reference under `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_7x_paper_forward_preflight_20260519/`. This is not a real-money promotion: every decision/preflight artifact keeps `real_money_execution=false` and `ready_for_real=false`.

Side-by-side paper candidates:

- Active: `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_7p0x_0p2alloc` (`higher_risk_train_return_tilt_v1`, isolated `7x`, allocation `0.20`, target notional/equity `140%`, paper-equivalent `$10,000 -> $14,000` notional). Preflight status: `ready_for_paper=true`, `ready_for_real=false`.
- Balanced reference: `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_6p0x_0p175alloc` (`balanced_train_validation_v1`, isolated `6x`, allocation `0.175`, target notional/equity `105%`, paper-equivalent `$10,000 -> $10,500` notional). Preflight status: `ready_for_paper=true`, `ready_for_real=false`.

The live strategy now supports the retuned `abs_factor_score_min=1.5` gate directly, preserving the 10bps trade-filter contract in paper/testnet runtime instead of only recording it as offline metadata. The bundle reuses the notional-risk-aligned `isolated_margin_fraction` sizing parity path and carries split/source/profile lineage from the frozen 10bps retune source, low-correlation sidecar, and notional-risk-aligned live artifact. Locked-OOS remains gate/report-only after candidate/profile freeze; selection/profile metadata continues to record `uses_locked_oos_for_objective=false`, `uses_locked_oos_for_pruning=false`, `uses_locked_oos_for_parameter_fitting=false`, and `uses_locked_oos_for_selection=false`.

Artifacts:

- Bundle JSON/Markdown: `alpha_zoo_7x_paper_forward_preflight_latest.json`, `alpha_zoo_7x_paper_forward_preflight_latest.md`.
- Timestamped bundle: `alpha_zoo_7x_paper_forward_preflight_20260520T112422Z.json`.
- Active decision: `live_alpha_zoo_quality_single_pair_7x_0p20_paper_decision_latest.json`.
- Balanced decision: `live_alpha_zoo_quality_single_pair_6x_0p175_balanced_reference_decision_latest.json`.
- Active preflight: `live_readiness_preflight_alpha_zoo_7x_0p20_paper_latest.json`.
- Balanced preflight: `live_readiness_preflight_alpha_zoo_6x_0p175_balanced_reference_paper_latest.json`.
- Monitoring contract: `paper_forward_monitoring_contract_latest.json` and `paper_forward_monitoring_contract_latest.csv`.
- Verification log: `local_verification_alpha_zoo_7x_paper_forward_preflight_20260520T111631Z.log`.

Monitoring contract status is `pending_paper_forward_fills`; it defines realized `fee_bps`, `slippage_bps`, and `all_in_round_trip_bps` fields, active-vs-balanced grouping keys, maker/taker/partial-fill/missed-signal fields, and liquidation-inclusive equity/MDD/account-wipeout checks. The cost audit keeps the research assumption fixed at `10.0bps` all-in round-trip with pass thresholds `mean <= 10bps` and `p95 <= 15bps`; actual paper/testnet fills must be collected before any real-money discussion.

Verification passed on 2026-05-20 UTC: artifact generation smoke; CSV-LF final verification: artifact regeneration; targeted tests `17 passed`; full pytest `1369 passed`; max RSS `2,877,880 KiB` (<8 GiB); `ruff check .`; `python -m compileall -q src scripts tests`; `git diff --check`; and staged `git diff --cached --check` after index sync.

---

## 2026-05-19 KST — Next plan: 7x/0.20 paper-forward live preflight

Saved next-session plan: `.omx/plans/plan-alpha-zoo-7x-paper-forward-live-preflight-20260519.md`.

The next step is **not real-money execution**. It is a paper/testnet-only live decision, preflight, and forward-monitoring handoff for the active 10bps profile and the balanced reference:

- Active paper candidate: `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_7p0x_0p2alloc` (`higher_risk_train_return_tilt_v1`, isolated `7x`, allocation `0.20`).
- Balanced reference: `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_6p0x_0p175alloc` (`balanced_train_validation_v1`, isolated `6x`, allocation `0.175`).
- Required governance: `real_money_execution=false`, `ready_for_real=false`, locked-OOS gate/report-only, replay/live sizing parity, liquidation-inclusive MDD, and realized round-trip cost monitoring against the 10bps research assumption.
- Recommended artifact dir for the follow-up: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_7x_paper_forward_preflight_20260519/`.

The plan rejects immediate real-money promotion because the 7x/0.20 validation return is still weak and the low-correlation discovery found `0` independently deployable low-correlation gate-pass streams. The next evidence needed is paper/testnet fill-quality and risk monitoring for active vs balanced side-by-side.

---

## 2026-05-19 KST — Higher-risk 10bps selection profile and low-correlation discovery result

The follow-up risk-selection run finalized the active 10bps model by a predeclared train+validation-only higher-risk profile, not by locked-OOS ranking. The runner now replays all `600` source candidates by default, records the selected profile metadata in the artifact, and emits low-correlation discovery sidecars. Locked-OOS remains gate/report-only after candidate/profile freeze; `real_money_execution=false` throughout.

Final artifact summary:

- Artifact dir: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_10bps_full_retune_20260519/`.
- Latest main JSON/Markdown: `alpha_zoo_10bps_full_retune_latest.json`, `alpha_zoo_10bps_full_retune_latest.md`.
- Timestamped final JSON/Markdown: `alpha_zoo_10bps_full_retune_20260519T140542Z.json`, `alpha_zoo_10bps_full_retune_20260519T140542Z.md`.
- Candidate accounting: `796` models / `2388` split metric rows; `600` source candidates selected before locked-OOS gates; `775` low-correlation candidate streams compared against the active reference.
- Memory: artifact peak RSS `385.7539 MiB`; `/usr/bin/time` max RSS `395,012 KiB`, under the 8 GiB session limit.

Selection profiles:

- Active final profile: `higher_risk_train_return_tilt_v1` -> `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_7p0x_0p2alloc`; profile score `0.5139152402359146`.
- Balanced reference profile: `balanced_train_validation_v1` -> `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_6p0x_0p175alloc`; profile score `0.20188310617174543`.
- Both profile definitions record train+validation as objective/selection/optimization/pruning/parameter-fit/score-formula inputs and all `uses_locked_oos_*` selector flags as `false`.

Active final 10bps profile split metrics (`7x`, allocation `0.20`, non-calendar `abs_score_ge_1.5` quality-single-pair filter):

- Train: return `+45.6916%`, MDD `31.6050%`, Sharpe `0.994753`, Sortino `0.287423`, Calmar `1.445708`, trades `405`, liquidation `0`.
- Validation: return `+0.4724%`, MDD `14.9117%`, Sharpe `0.219005`, Sortino `0.047090`, Calmar `0.129660`, trades `86`, liquidation `0`.
- Locked-OOS gate/report-only: return `+1.8382%`, MDD `14.3237%`, Sharpe `0.556723`, Sortino `0.169307`, Calmar `1.074121`, trades `53`, liquidation `0`.

Balanced reference 10bps split metrics (`6x`, allocation `0.175`, same non-calendar filter): train `+36.0268%`, validation `+0.5942%`, locked-OOS `+1.5464%`; all splits liquidation `0`.

Low-correlation discovery sidecars: `low_correlation_discovery_latest.json` and `low_correlation_discovery_latest.csv`. Correlations are computed from train+validation returns only against the active higher-risk reference; `423` streams are below the `0.35` absolute-correlation threshold, but `0` are deployable 10bps gate passers independent of the reference in this run, so the discovery rows are research-only until a low-correlation stream also clears locked-OOS gate/report checks.

Verification passed on 2026-05-19 UTC: artifact assertion passed (`796` models, `2388` metric rows, `50` low-correlation rows); focused 10bps retune and artifact assertion tests `19 passed`; `ruff` passed on changed runner/assertion/tests; `compileall` passed. Full `n_trials=80` hybrid optimizer was not rerun in this final profile pass; the required deliverable was profile-safe selection plus low-correlation discovery under the 8 GiB guard.

---

## 2026-05-19 KST — Risk-selection and low-correlation verification contract

The follow-up risk-selection plan keeps locked-OOS as post-freeze gate/report-only evidence while predeclaring two train+validation-only selection profiles:

- `balanced_train_validation_v1` remains the balanced reference and must report `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_6p0x_0p175alloc`.
- `higher_risk_train_return_tilt_v1` is the active final profile and may select `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_7p0x_0p2alloc` only after the existing 10bps promotion gates pass.

The artifact validator now requires explicit profile metadata, score-formula inputs, `uses_locked_oos_* = false` flags, balanced-reference preservation, and a separate `low_correlation_discovery_latest.json/csv` surface. Low-correlation discovery rows must compute correlations from train+validation returns only, label deployable 10bps gate passers separately from research-only locked-OOS gate failures, keep `real_money_execution=false`, and preserve the `<8192 MiB` memory guard.

---

## 2026-05-19 KST — Alpha Zoo full 10bps round-trip retune and live-gate repair

Re-ran the Alpha Zoo backtest-to-live candidate family under the latest split and locked promotion cost of round-trip slippage/fee `10bps`. The run covers historical top-bucket / live-ranked Alpha Zoo seed streams, the prior Hybrid v3.5/v3.6 seed-union rows, references, and fresh train+validation-only variants. Locked-OOS stayed strictly post-freeze gate/report-only: the artifact records selection inputs `['train', 'validation']`, `uses_locked_oos_for_selection=false`, and trade-filter locked-OOS role `gate_report_only_after_variant_freeze`. No real-money execution was attempted.

The earlier top-bucket/hybrid rows did not survive as live candidates at `10bps`: `hybrid_v3_5_seed_union` remains a shadow-only historical OOS-bucket row and fails the fresh live gate because locked-OOS return/Sharpe/Sortino/smart Sortino/Calmar are non-positive and because its lineage uses OOS-derived bucket selection. The plain fresh 10bps seed/hybrid retune also found no live-ready model with positive validation and locked-OOS performance, so a non-calendar fixed trade-filter retune lane was added. The lane evaluates only signal-structure filters such as side/symbol/factor-family/absolute factor-score/hold cap over train+validation, then freezes the variant before locked-OOS reporting.

Final full retune summary:

- Artifact dir: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_10bps_full_retune_20260519/`.
- Latest main JSON/Markdown: `alpha_zoo_10bps_full_retune_latest.json`, `alpha_zoo_10bps_full_retune_latest.md`.
- Timestamped final JSON/Markdown: `alpha_zoo_10bps_full_retune_20260519T121224Z.json`, `alpha_zoo_10bps_full_retune_20260519T121224Z.md`.
- Split: train `2025-01-01T00:00:00Z..2025-12-31T23:00:00Z`; validation `2026-01-01T00:00:00Z..2026-03-31T23:00:00Z`; locked-OOS `2026-04-01T00:00:00Z..2026-05-17T10:00:00Z`; timestamp-index hash `b973165bc1057f3aaa08ea637b73a45df3e84fdb7d1337b1637233d205696bb0`.
- Candidate accounting: `798` models / `2394` split metric rows; `778` fresh train+validation models; `176` fresh trade-filter models; `20` shadow-only historical rows.
- Search accounting: `600` source candidate rows replayed, `776` fresh 10bps streams evaluated, `30,354` trade-filter variants evaluated, `176` selected variants, `56` passing the final 10bps live gate.
- Memory: runner peak RSS `400.9883 MiB` by artifact memory summary and `/usr/bin/time` max RSS `420,012 KiB`; full pytest post-LF max RSS `2,722,856 KiB`, both under the 8 GiB session limit.

Best live-gate candidate at round-trip `10bps`:

- Model id: `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_6p0x_0p175alloc`.
- Strategy family: `CryptoFxAlphaZooStateStrategy` / `alpha_zoo_quality_single_pair`.
- Variant: non-calendar `abs_factor_score_min=1.5` (`abs_score_ge_1.5`).
- Sizing: isolated `6x`, allocation fraction `0.175`, `544` filtered trades; liquidation/account-wipeout counts are `0` on all splits and margin buffers stay positive.
- Train: return `+36.0268%`, MDD `24.6028%`, Sharpe `1.002645`, Sortino `0.289764`, smart Sortino `0.232550`, Calmar `1.464340`, trades `405`, min margin buffer `8514.118330`.
- Validation: return `+0.5942%`, MDD `11.3653%`, Sharpe `0.219846`, Sortino `0.047272`, smart Sortino `0.042448`, Calmar `0.214374`, trades `86`, min margin buffer `9251.785896`.
- Locked-OOS gate/report-only: return `+1.5464%`, MDD `10.9211%`, Sharpe `0.554965`, Sortino `0.168704`, smart Sortino `0.152094`, Calmar `1.173217`, trades `53`, min margin buffer `9383.460782`.

The selected live-gate candidate satisfies the requested dominance shape: train return/Sharpe/Sortino/smart Sortino/Calmar are all above validation and locked-OOS; validation and locked-OOS returns are positive; locked-OOS was not used for variant selection, pruning, objective scoring, or parameter fitting. Calendar/date rules remain rejected.

Fresh verification passed on 2026-05-19 UTC: artifact assertion passed (`798` models, `2394` metric rows, `56` promotable); 10bps retune tests `15 passed`; top-seed/hybrid split tests `19 passed`; live/liquidation/state tests `49 passed`; full pytest `1360 passed`; `ruff check .`, `compileall`, and diff checks passed. Research history/source ledger not regenerated: this session reused the already-refreshed 2026-05-17 current-tail data and same Alpha Zoo artifact lineage, adding a same-lineage 10bps retune/validation bundle rather than a new market-data source family or chronology refresh.

---

## 2026-05-19 KST — Alpha Zoo top-seed Hybrid v3.5/v3.6 cost validation

Recomputed the current Alpha Zoo top-bucket plus filtered top-3 seed-union from the latest live-notional/risk-aligned candidate CSV, then generated a separate research-only Hybrid v3.5/v3.6 cost-validation bundle. The deduped seed universe now has `16` rows across `fast_residual`, `quality_single_pair`, `high_confidence_single_pair`, and `high_confidence_long_only`. The bundle reports every individual seed, `hybrid_v3_5_seed_union`, `hybrid_v3_6_seed_union`, `reference_fast_residual_7x_0p15`, and `reference_strict_zero_fast_residual_6x_0p10` at round-trip slippage/fee `5bps` and `10bps` over `train`, `validation`, and `locked_oos` splits: `(16 + 2 + 2) * 2 * 3 = 120` metric rows.

Locked-OOS remains a gate/report split after model freeze for the hybrid objective/pruning/parameter-fitting path: the artifact audit records `uses_locked_oos_for_objective=false`, `uses_locked_oos_for_pruning=false`, `uses_locked_oos_for_parameter_fitting=false`, and `uses_locked_oos_for_selection=false`. Because the requested seed basket is assembled from current leaderboard buckets, the artifact also labels the seed basket as a post-hoc research basket rather than a deployable live-selection rule. No real-money execution was attempted.

Key cost outcomes:

- `hybrid_v3_5_seed_union`, `5bps`: train `+49.19%` / MDD `30.09%`; validation `+21.16%` / MDD `12.33%`; locked-OOS `+3.29%` / MDD `15.39%`; liquidation `0`; account wipeout `0`; locked-OOS gate `true`.
- `hybrid_v3_6_seed_union`, `5bps`: train `-7.98%`; validation `+8.21%`; locked-OOS `-2.90%`; liquidation `0`; account wipeout `0`; locked-OOS gate `false`.
- `hybrid_v3_5_seed_union`, `10bps`: train `+47.75%`; validation `+18.91%`; locked-OOS `-2.82%`; liquidation `0`; account wipeout `0`; locked-OOS gate `false`.
- `hybrid_v3_6_seed_union`, `10bps`: train `-9.07%`; validation `-7.11%`; locked-OOS `-6.22%`; liquidation `0`; account wipeout `0`; locked-OOS gate `false`.
- References: `fast_residual 7x/0.15` locked-OOS is `+7.63%` at `5bps` and `-12.44%` at `10bps`; strict zero `fast_residual 6x/0.10` locked-OOS is `+4.59%` at `5bps` and `-7.05%` at `10bps`.
- Best locked-OOS individual seed in the bundle is `alpha_zoo_high_confidence_single_pair 7x/0.2`: `+11.03%` / MDD `14.02%` at `5bps`, and `+3.02%` / MDD `16.87%` at `10bps`. Isolated liquidation losses are included in equity/MDD; the bundle's max split liquidation count is `9` on one high-leverage seed train split, and all account-wipeout counts are `0`.

Artifacts:

- Runner: `scripts/research/run_alpha_zoo_top_seed_hybrid_v35_v36_cost_validation.py`
- Main JSON: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_top_seed_hybrid_cost_validation_20260518/alpha_zoo_top_seed_hybrid_cost_validation_latest.json`
- Main Markdown/full metric table: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_top_seed_hybrid_cost_validation_20260518/alpha_zoo_top_seed_hybrid_cost_validation_latest.md`
- Seed selection CSV/JSON: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_top_seed_hybrid_cost_validation_20260518/seed_selection_latest.csv`, `seed_selection_latest.json`
- Model metrics CSV: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_top_seed_hybrid_cost_validation_20260518/model_cost_metrics_latest.csv`
- Hybrid weights CSV: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_top_seed_hybrid_cost_validation_20260518/hybrid_weights_latest.csv`
- Generation log: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_top_seed_hybrid_cost_validation_20260518/local_verification_alpha_zoo_top_seed_hybrid_cost_validation_20260519T093343Z.log`
- Final verification log: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_top_seed_hybrid_cost_validation_20260518/local_verification_alpha_zoo_top_seed_hybrid_cost_validation_final_20260519T100131Z.log`
- Post-deslop verification log: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_top_seed_hybrid_cost_validation_20260518/local_verification_alpha_zoo_top_seed_hybrid_cost_validation_post_deslop_20260519T100507Z.log`

Fresh local verification passed on 2026-05-19 UTC: artifact assertion passed (`seed_count=16`, `metric_rows=120`, max liquidation count `9`, account wipeout max `0`); new Alpha Zoo top-seed tests `5 passed`; hybrid/common split tests `14 passed`; live/liquidation/state tests `46 passed`; full pytest `1345 passed`; ruff, compileall, and `git diff --check` all passed. Post-deslop verification re-ran after narrowing broad exception handling in the new runner: artifact assertion passed; new tests `5 passed`; full pytest `1345 passed`; ruff, compileall, and `git diff --check` passed.

Research history/source ledger not regenerated: this session reused the already-refreshed 2026-05-17 current-tail data and the 2026-05-18 live-notional/risk-aligned Alpha Zoo artifact family; it added a same-lineage research-only cost-validation bundle, not a new market-data source family or global chronology refresh.

---

## 2026-05-18 KST — Plan for Alpha Zoo top-seed hybrid v3.5/v3.6 cost validation

Prepared a next-session plan to test whether the current Alpha Zoo leaderboard can be improved by building Hybrid v3.5/v3.6 portfolios from the top individual candidate streams rather than from the prior fixed-input `A0 + P0 + E0 + S1 + S2 + S3 + S4` universe. The plan is saved at `.omx/plans/plan-alpha-zoo-hybrid-v35-v36-cost-validation-20260518.md`.

Current seed-selection snapshot is based on `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/live_notional_risk_aligned_alpha_zoo_20260518/`: leverage grid `1x..20x`, allocation grid `0.03,0.05,0.075,0.10,0.125,0.15,0.175,0.20`, `1600` rows, `113` live-promotion rows, and `50` rows passing the train-dominant/val-good/OOS-good filter. The next session must recompute the buckets from the latest CSV before running.

Planned seed universe uses top-3 bucket union from live OOS return/Sharpe/Sortino/smart Sortino/Calmar, full compound, and filtered balanced/validation-return/OOS-return/OOS-Calmar lists. The current deduped snapshot has `18` rows spanning `fast_residual`, `quality_single_pair`, `high_confidence_single_pair`, and `high_confidence_long_only` configurations. Known duplicates such as `fast_residual 7x/0.15` and `6x/0.175` are intentional because they have the same notional/equity (`105%`) but different isolated margin semantics.

The required cost validation is explicitly round-trip slippage/fee `5bps = 0.05%` and `10bps = 0.10%`. For every individual seed plus `hybrid_v3_5_seed_union` and `hybrid_v3_6_seed_union`, the next run must report train/validation/locked-OOS total return, MDD, Sharpe, Sortino, smart Sortino, Calmar, trade/event count, liquidation count, account-wipeout count, and minimum margin buffer. Locked-OOS remains gate/report-only and must not enter objective, pruning, parameter fitting, or seed/hybrid selection.

Recommended output directory for the next run: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_top_seed_hybrid_cost_validation_20260518/`. Real-money execution remains prohibited. Research history/source ledger does not need regeneration for this planning-only update because it introduces no new data source family or market-data chronology; the execution session must revisit that decision if it refreshes data or changes source lineage.

---

## 2026-05-18 KST — Live/replay notional-risk aligned Alpha Zoo 7x contract

Implemented the live sizing contract required for the latest high-leverage winner. Existing strategies keep backward-compatible `legacy_notional_cap` behavior by default, while the promoted Alpha Zoo lane now opts into explicit `isolated_margin_fraction`: `target_allocation=0.15` means `15%` isolated margin/equity and, with `leverage=7`, `105%` notional/equity. The legacy fixed-dollar `max_order_value` cap is disabled for this lane (`0.0`) and replaced by equity-scaled notional caps: per-order `110%`, symbol `110%`, total notional `220%`; any positive `max_order_value` remains an explicit emergency ceiling.

Latest-data train+validation retune kept `CryptoFxAlphaZooStateStrategy / alpha_zoo_fast_residual / isolated 7x / allocation 0.15`. The raw grid also found a notional-equivalent `6x/0.175` row with the same train+validation score; a documented incumbent tie-breaker selected the requested `7x/0.15` contract without using locked-OOS for scoring. Locked-OOS remains freeze-after-selection gate/report-only. No real-money execution was attempted.

Selected high-performance lane:

- Sizing mode: `isolated_margin_fraction`.
- Exposure: notional/equity `105.00%`; isolated margin/equity `15.00%`.
- Train: `+1.4941%` return / `59.9354%` MDD.
- Validation: `+44.9483%` return / `13.7796%` MDD.
- Locked-OOS no-cost: `+30.5357%` return / liquidation-inclusive MDD `11.3027%`; Sharpe `1.815354`, Sortino `2.318591`, smart Sortino `2.083139`, Calmar `2.701628`, trades `391`, locked-OOS liquidation `0`, total account wipeout `0`.
- Paper-equivalent parity fixture: equity `$10,000`, price `$100`, `target_allocation=0.15`, `leverage=7` -> replay expected notional `$10,500`, live quantity `105.0`, live notional `$10,500`, absolute diff `0.0`, risk check `Passed`.
- Preflight: `recommended_action=paper_run_allowed`, `ready_for_paper=true`, `ready_for_real=false`.

Cost-stressed locked-OOS diagnostics are separate from the no-cost headline. Round-trip slippage/fee: `1bps` `+25.2882%` / MDD `11.9349%`; `3bps` `+15.4160%` / MDD `13.1860%`; `5bps` `+6.3199%` / MDD `15.4731%`; `10bps` `-13.4130%` / MDD `24.2149%`; `20bps` `-42.5899%` / MDD `44.8361%`. Funding drag: `1bps/day` `+29.9911%`; `2bps/day` `+29.4486%`; `5bps/day` `+27.8349%`; `10bps/day` `+25.1897%`; `20bps/day` `+20.0619%`.

Strict zero-liquidation lane is kept separate: same calibrated Alpha Zoo parameters at strict `6x` / `10%` allocation show locked-OOS `+16.7783%` return, MDD `6.5951%`, Sharpe `1.815354`, Sortino `2.318591`, smart Sortino `2.175137`, Calmar `2.544032`, liquidation `0`, account wipeout `0`, minimum margin buffer `9150.924760`.

Artifacts:

- Main aligned JSON: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/live_notional_risk_aligned_alpha_zoo_20260518/live_notional_risk_aligned_alpha_zoo_latest.json`
- Main aligned Markdown: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/live_notional_risk_aligned_alpha_zoo_20260518/live_notional_risk_aligned_alpha_zoo_latest.md`
- Live decision artifact: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/live_notional_risk_aligned_alpha_zoo_20260518/live_alpha_zoo_notional_risk_aligned_decision_latest.json`
- Preflight artifact: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/live_notional_risk_aligned_alpha_zoo_20260518/live_readiness_preflight_notional_risk_aligned_latest.json`
- Candidate CSV: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/live_notional_risk_aligned_alpha_zoo_20260518/alpha_zoo_validation_march_high_leverage_candidates_latest.csv`
- Verification log: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/live_notional_risk_aligned_alpha_zoo_20260518/local_verification_live_notional_risk_alignment_20260518T113100Z.log`

Fresh local verification passed on 2026-05-18 UTC: required live/Alpha Zoo suite `32 passed`, required moonshot validation suite `74 passed`, full pytest `1340 passed`, ruff, compileall, `git diff --check`, and `git diff --cached --check` all passed.

Research history/source ledger not regenerated: this session used the already-refreshed 2026-05-17 current-tail data and added a live-sizing contract/validation artifact bundle, not a new market-data source family or global chronology refresh.

---

## 2026-05-17 KST — Live notional/risk alignment plan for Alpha Zoo 7x lane

After reviewing the latest `alpha_zoo_fast_residual` 7x isolated live decision, the key remaining live-readiness issue is not the signal hypothesis but the sizing contract. The no-cost research replay models account return as `allocation_fraction * leverage * gross_return`, so the current winner `allocation_fraction=0.15`, `leverage=7` represents approximately `105%` notional exposure with about `15%` isolated margin. The live runtime currently treats `target_allocation` as a notional cap, so the same `0.15` can size closer to `15%` notional exposure. That mismatch must be fixed before expecting live/paper results to match replay performance.

Static `max_order_value=5000.0` is also identified as a legacy fixed-dollar guardrail, not a strategy-derived cap. For this futures lane it must be replaced/subordinated by an equity-scaled and leverage-aware cap, with any absolute dollar cap treated only as an explicit emergency ceiling. Otherwise a $10,000 account targeting the replay-intended `0.15 * 7 = 1.05` notional/equity can be silently truncated.

Planning artifacts created for the next implementation session:

- PRD: `.omx/plans/prd-live-alpha-zoo-notional-risk-alignment-20260517.md`
- Test spec: `.omx/plans/test-spec-live-alpha-zoo-notional-risk-alignment-20260517.md`
- Handoff: `docs/session_handoff_20260517_live_notional_risk_alignment.md`

Next-session acceptance target: add an explicit sizing mode such as `isolated_margin_fraction` vs `notional_fraction`, preserve backward compatibility, retune leverage/allocation using train+validation only, include liquidation losses in equity/MDD for any isolated high-performance lane, keep strict zero-liquidation lane separate, add no-cost and cost-stressed reports, prove paper-equivalent live sizing parity, and avoid real-money execution until a separate credentialed real preflight is green and explicitly authorized.

Research history/source ledger was not regenerated for this planning-only update because it introduces no new data-source family or new market-data chronology artifact. The next implementation session must revisit the global research history/source ledger if it refreshes data beyond the current tail or adds new source families.

---

## 2026-05-17 KST — Latest-data validation-to-March high-leverage Alpha Zoo replay

Refreshed the five-symbol raw-first Binance Futures data tail to cutoff `2026-05-17T10:59:59Z`, compacted the OHLCV WAL to monthly parquet, and rebuilt the joined hourly current-tail panel `var/cache/profit_moonshot_fresh_start/joined_panel_76f825ffea81c04f2fe41fbf.parquet` with actual max timestamp `2026-05-17T10:00:00Z`.

Updated split authority:

- train: `2025-01-01T00:00:00Z` ~ `2025-12-31T23:00:00Z` (`8760` hourly timestamps, `43800` rows)
- validation: `2026-01-01T00:00:00Z` ~ `2026-03-31T23:00:00Z` (`2156` hourly timestamps, `10780` rows)
- locked-OOS: `2026-04-01T00:00:00Z` ~ `2026-05-17T10:00:00Z` (`1115` hourly timestamps, `5575` rows)

High-leverage tuning used only train+validation for candidate ranking. Locked-OOS was applied after candidate freeze as gate/report-only. The high-leverage lane assumes isolated per-position margin; if a path breaches liquidation threshold, the trade loss is capped to the configured isolated allocation fraction, and account-wipeout count must remain zero.

Result:

- Top train/validation score was `alpha_zoo_conservative_exit`/carry-forward at `9x` with `12.5%` allocation, but it failed locked-OOS gate (`-0.6029%` OOS return, non-positive Calmar).
- First pre-frozen candidate to pass locked-OOS gate: `CryptoFxAlphaZooStateStrategy` / `alpha_zoo_fast_residual` / isolated `7x` / `15%` allocation.
- Promoted high-leverage candidate metrics: train `+1.4941%` / MDD `59.9354%`; validation `+44.9483%` / MDD `13.7796%`; locked-OOS `+30.5357%` / MDD `11.3027%`; locked-OOS Sharpe `1.815354`, Sortino `2.318591`, smart Sortino `2.083139`, Calmar `2.701628`, trades `391`, liquidation count `0`, account-wipeout count `0`, `live_promotion_possible=true`.
- Strict zero-liquidation integer lane at `10%` allocation for the same `alpha_zoo_fast_residual` params still promotes `6x`: locked-OOS `+16.7783%`, MDD `6.5951%`, liquidation `0`, min buffer `9150.924760`, positive Sharpe/Sortino/smart Sortino/Calmar.
- Runtime leverage validation cap was raised from `6x` to `20x` so the isolated `7x` decision artifact can pass live configuration validation; real execution remains operator/credential gated.
- Peak RSS: data refresh `4467.1172 MiB`, replay `736.1953 MiB`, both under 8 GiB.

Artifacts:

- Runner: `scripts/research/run_alpha_zoo_validation_march_high_leverage.py`
- Main JSON: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/validation_to_20260331_latest_data_20260517/alpha_zoo_validation_march_high_leverage_latest.json`
- Main Markdown: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/validation_to_20260331_latest_data_20260517/alpha_zoo_validation_march_high_leverage_latest.md`
- Candidate CSV: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/validation_to_20260331_latest_data_20260517/alpha_zoo_validation_march_high_leverage_candidates_latest.csv`
- Live decision artifact: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/validation_to_20260331_latest_data_20260517/live_alpha_zoo_fast_residual_7x_isolated_decision_latest.json`
- Data refresh report: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/validation_to_20260331_latest_data_20260517/data_refresh_latest.json`

Research history/source ledger not regenerated: this is a same-source tail refresh and session-scoped Alpha Zoo replay, not a new global source family or chronology ledger change.

Latest-data March-validation verification passed on 2026-05-17 UTC: live/source validation suite `27 passed`, required Alpha Zoo suite `24 passed`, required moonshot validation suite `74 passed`, full pytest `1329 passed`, ruff/compileall/diff checks passed. Log: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/validation_to_20260331_latest_data_20260517/local_verification_validation_march_high_leverage_20260517T120700Z.log`.

Live-readiness preflight for the isolated `7x` decision artifact also passed for paper/testnet mode with a supplied paper Postgres DSN placeholder and freshness threshold override: `recommended_action=paper_run_allowed`, `ready_for_paper=true`, `ready_for_real=false`. Artifact: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/validation_to_20260331_latest_data_20260517/live_readiness_preflight_7x_latest.json`.

Post-staging CSV LF/runner-lineterminator verification re-ran clean: full pytest `1329 passed in 55.01s`, required Alpha Zoo suite `24 passed`, moonshot suite `74 passed`, live/source targeted suite `27 passed`, ruff/compileall/diff checks passed.

---

## 2026-05-17 Addendum — strict 6x live-wiring final verification

Final hardening synced live decision exchange overrides into both `LiveConfig.EXCHANGE` and derived fields (`EXCHANGE_ID`, `MARKET_TYPE`, `POSITION_MODE`, `MARGIN_MODE`, `LEVERAGE`) before validation/trader construction and made unknown strategy-class live decisions fail closed. The decision artifact preflight reports `paper_run_allowed` and `ready_for_paper=true` while keeping real execution operator/credential gated. Fresh verification passed: targeted live/common-split suite `46 passed`, live-readiness/parity suite `11 passed`, required Alpha Zoo suite `24 passed`, required moonshot validation suite `74 passed`, full pytest `1328 passed`, ruff/compileall/diff checks passed.

---

## 2026-05-17 Addendum — strict 6x Alpha Zoo live wiring check

The common-split #1 (`CryptoFxAlphaZooStateStrategy` / `alpha_zoo_conservative_exit` / strict `6x`) is now explicitly represented as a live decision artifact rather than only as a replay result. The live decision path maps Alpha Zoo references to `CryptoFxAlphaZooStateStrategy`, passes train+validation calibrated edges and selected conservative-exit params to `LiveTrader`, and applies 3600s MARKET_WINDOW/cadence plus isolated `6x` and `target_allocation=0.10` overrides.

Live-equivalent tests were added for selection inference, decision override propagation, live CLI parameter injection, runtime leverage validation (`6x` allowed, `>6x` rejected), and MARKET_WINDOW-vs-MARKET_BATCH strategy parity for hourly Alpha Zoo decisions. The strict 6x replay evidence remains: locked-OOS return `+20.512682%`, MDD `6.788365%`, liquidation `0`, and positive min margin buffer. Real live fills/slippage/funding remain execution-environment risks and are not asserted identical to replay.

Artifacts:

- `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/common_split_alpha_zoo_hybrid_v35_v36_20260517/live_alpha_zoo_strict_6x_decision_latest.json`
- `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/common_split_alpha_zoo_hybrid_v35_v36_20260517/live_alpha_zoo_strict_6x_decision_latest.md`

---

## 2026-05-17 KST — Integrated margin replay addendum for fixed-input hybrid v3.5/v3.6

Supersedes the earlier same-day note that marked fixed-input hybrid v3.5/v3.6 non-promotable solely because liquidation count/minimum margin buffer were `not_replayed`. Added a mixed-allocator integrated margin replay for the frozen A0+P0+E0+S1+S2+S3+S4 hybrid return streams. The replay uses post-freeze v3.5/v3.6 allocator weights, maps each fixed stream to its source gross-notional fraction, and evaluates one cross-margin account path. It is not used by Optuna objective, pruning, selection, or tie-break; locked-OOS remains gate/report-only after candidate freeze.

Updated common-split hybrid live-gate result:

| Candidate | Split | Return | MDD | Sharpe | Sortino | Smart Sortino | Calmar | Active hours | Liquidations | Min margin buffer | Deployable success |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| hybrid_v3_5_optuna | train | +47.7257% | 11.0421% | 2.705685 | 2.932093 | 2.640523 | 4.322148 | 7514 | 0 | 9932.438663 | true |
| hybrid_v3_5_optuna | validation | +13.3102% | 2.2622% | 5.390597 | 7.610005 | 7.441656 | 51.558258 | 1302 | 0 | 14594.054033 | true |
| hybrid_v3_5_optuna | locked-OOS | +8.5233% | 1.7654% | 5.259028 | 7.316663 | 7.189734 | 32.173151 | 1467 | 0 | 16587.499982 | true |
| hybrid_v3_6_optuna | train | +49.5204% | 7.6947% | 2.897597 | 2.999204 | 2.784914 | 6.435678 | 7514 | 0 | 9847.514685 | true |
| hybrid_v3_6_optuna | validation | +12.4946% | 1.5354% | 7.002337 | 8.680826 | 8.549560 | 69.800312 | 1302 | 0 | 14690.924128 | true |
| hybrid_v3_6_optuna | locked-OOS | +7.7916% | 1.7491% | 4.859674 | 5.991026 | 5.888040 | 29.199963 | 1467 | 0 | 16664.270300 | true |

Decision update: fixed-input hybrid v3.5/v3.6 are now live-promotion-capable under the integrated margin gate, but they do **not** beat the common-split Alpha Zoo strict 6x lane on locked-OOS return (`+20.5127%` for Alpha Zoo vs `+8.5233%` v3.5 and `+7.7916%` v3.6). Alpha Zoo strict 6x remains the common-split performance leader; hybrid v3.5/v3.6 become lower-return, lower-MDD deployable alternatives rather than blocked diagnostics. Research history/source ledger still not regenerated: this addendum adds a local validation/replay artifact and no new global source family.

Updated artifacts:

- Runner update: `scripts/research/run_profit_moonshot_hybrid_v35_v36_fixed_inputs.py`
- Common report JSON: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/common_split_alpha_zoo_hybrid_v35_v36_20260517/common_split_alpha_zoo_hybrid_v35_v36_latest.json`
- Common report Markdown: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/common_split_alpha_zoo_hybrid_v35_v36_20260517/common_split_alpha_zoo_hybrid_v35_v36_latest.md`
- Hybrid stage JSON: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/common_split_alpha_zoo_hybrid_v35_v36_20260517/hybrid_v35_v36_common_split/hybrid_v35_v36_fixed_inputs_common_split_latest.json`
- Hybrid stage Markdown: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/common_split_alpha_zoo_hybrid_v35_v36_20260517/hybrid_v35_v36_common_split/hybrid_v35_v36_fixed_inputs_common_split_latest.md`

Integrated margin addendum verification passed on 2026-05-17 UTC: targeted Alpha Zoo suite `23 passed`, moonshot validation suite `74 passed`, full pytest `1321 passed`, ruff/compileall/diff checks passed. Log: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/common_split_alpha_zoo_hybrid_v35_v36_20260517/local_verification_integrated_margin_20260517T071937Z.log`.

---

## 2026-05-17 KST — Common-split Alpha Zoo vs fixed-input hybrid v3.5/v3.6 fair comparison

Re-ran the Alpha Zoo best and fixed-input hybrid v3.5/v3.6 candidates on one explicit common split from baseline `private/main@80a557c133930f51748ec20c4e582aa0d6f678de`. The prior Alpha Zoo split is now **historical only** in this comparison; it is not used for common-split selection, promotion, or tie-breaks.

Common split authority:

- train: `2025-01-01T00:00:00Z` ~ `2025-12-31T23:00:00Z` (`8760` hourly timestamps, `43800` rows)
- validation: `2026-01-01T00:00:00Z` ~ `2026-02-28T23:00:00Z` (`1416` hourly timestamps, `7080` rows)
- locked-OOS: `2026-03-01T00:00:00Z` ~ `2026-05-06T23:00:00Z` (`1593` hourly timestamps, `7965` rows)

Alpha Zoo result on common split:

- Historical old split strict 6x is preserved only as reference: old split periods were train `2025-01-01T00:00:00Z`~`2025-10-22T04:00:00Z`, validation `2025-10-22T05:00:00Z`~`2026-01-28T06:00:00Z`, locked-OOS `2026-01-28T07:00:00Z`~`2026-05-06T23:00:00Z`; old locked-OOS return was `+41.0967%`, but it is not comparable for new selection.
- Common-split carry-forward of the old selected `alpha_zoo_conservative_exit` and common-split reselected grid both select/retain `alpha_zoo_conservative_exit` and produce the same strict 6x replay: train `+114.4617%` / MDD `29.5651%`; validation `+19.9681%` / MDD `13.6667%`; locked-OOS `+20.5127%` / MDD `6.7884%`; locked-OOS Sharpe `1.772136`, Sortino `2.578776`, smart Sortino `2.414847`, Calmar `3.021741`, trades `365`, liquidation `0`, min margin buffer `9643.447509`; `deployable_success=true` under the strict lane.
- Strict integer leverage 1x..6x on the common split keeps `6x` as the highest deployable integer: OOS return `+20.5127%`, MDD `6.7884%`, liquidation `0`, min buffer `9049.125962`. Return/MDD remains diagnostic/report-only.
- Diagnostic nonfatal 5x/6x lane remains separate from live promotion: 5x and 6x both have `promotion_allowed=false` even though their diagnostic replay has zero liquidations.

Fixed-input hybrid v3.5/v3.6 common-split Optuna result:

- Input universe remains exactly `A0 + P0 + E0 + S1 + S2 + S3 + S4`; no literal hybrid/hybrid-online/hybrid-tuning output is used as an input.
- v3.5 uses fixed default + rolling weights/high-vol boost + Optuna; v3.6 is v3.5 core plus online dynamic default-candidate refresh only. No other knob is OOS-adaptive.
- Optuna ran with `n_trials=80`, `seed=42`, and train+validation objective/selection only. Audit found no locked-OOS use for objective, pruning, selection, or tie-break (`violation=false`; calibration locked-OOS records `0`).
- v3.5 common-split locked-OOS: return `+8.5233%`, MDD `1.7654%`, Sharpe `5.259028`, Sortino `7.316663`, smart Sortino `7.189734`, Calmar `32.173151`; `deployable_success=false` because dedicated integrated margin replay is missing.
- v3.6 common-split locked-OOS: return `+7.7916%`, MDD `1.7491%`, Sharpe `4.859674`, Sortino `5.991026`, smart Sortino `5.888040`, Calmar `29.199963`; `deployable_success=false` for the same margin-replay reason.

Decision: on the fair common split, Alpha Zoo strict 6x remains the only live-promotion-capable lane. Hybrid v3.5/v3.6 are useful diagnostics and beat the invalid current-base OOS return reference, but they are not live-promotable until an integrated strict liquidation/margin replay supplies liquidation count and minimum margin buffer. Peak RSS was `769.1015625 MiB` (<8 GiB). Research history/source ledger was not regenerated because this session added a session-scoped comparison artifact only and did not introduce a new global source family or chronology ledger.

Artifacts:

- Runner: `scripts/research/run_common_split_alpha_zoo_hybrid_v35_v36.py`
- Regression tests: `tests/test_common_split_alpha_zoo_hybrid_v35_v36.py`
- Main JSON: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/common_split_alpha_zoo_hybrid_v35_v36_20260517/common_split_alpha_zoo_hybrid_v35_v36_latest.json`
- Main Markdown: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/common_split_alpha_zoo_hybrid_v35_v36_20260517/common_split_alpha_zoo_hybrid_v35_v36_latest.md`
- Alpha stage artifacts: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/common_split_alpha_zoo_hybrid_v35_v36_20260517/alpha_zoo_common_split/`
- Hybrid stage artifacts: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/common_split_alpha_zoo_hybrid_v35_v36_20260517/hybrid_v35_v36_common_split/`

Verification for the common-split run passed on 2026-05-17 UTC: targeted Alpha Zoo suite `23 passed`, moonshot validation suite `74 passed`, full pytest `1319 passed`, ruff/compileall/diff checks passed. Log: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/common_split_alpha_zoo_hybrid_v35_v36_20260517/local_verification_common_split_20260517T054257Z.log`.
Post-deslop verification re-ran the full required command set and passed again: targeted Alpha Zoo suite `23 passed`, moonshot validation suite `74 passed`, full pytest `1319 passed`, ruff/compileall/diff checks passed. Log: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/common_split_alpha_zoo_hybrid_v35_v36_20260517/local_verification_common_split_post_deslop_20260517T054855Z.log`.

---

## 2026-05-17 KST — External Hybrid v3.5/v3.6 method applied to fixed A0+P0+E0+S1+S2+S3+S4 inputs

Operator correction incorporated: `/home/hoky/DeepLearning/ensemble_strategies` defines v3.6 as **v3.5 core plus online dynamic default-model/candidate refresh**, not a new candidate universe and not a hybrid-inside-hybrid stack. Evidence checked directly in the external method source:

- `models/hybrid/v3_5.py`: lines 1-8 describe adaptive weights + Optuna; lines 31-49 define the Optuna-tuned defaults/search candidates; lines 311-328 learn train/warmup parameters; lines 397-420 keep the default model fixed while applying rolling weights/high-vol boost.
- `models/hybrid/v3_6.py`: lines 1-9 state the v3.6 delta: Step A `default_model` is dynamically updated online by rolling MAPE while v3.5 defaults/Optuna results are retained; lines 29-30 reuse v3.5 learning; lines 87-105 learn the same parameters; lines 178-223 dynamically refresh only the default model and otherwise use the v3.5 weight/high-vol/bias structure.
- `scripts/compare_v35_v36.py`: lines 1-5 summarize the same delta; lines 49-55 compare v3.5 fixed default vs v3.6 dynamic default.

Repo adaptation now uses fixed input universe `A0 + P0 + E0 + S1 + S2 + S3 + S4` only. No literal prior hybrid/hybrid-online/hybrid-tuning output is an input; no calendar/month/day/hour entry rule is introduced; Optuna objective/selection uses train+validation only. Locked-OOS remains gate/report-only after candidate freeze.

Split periods for this fixed-input experiment:

- locked_oos: `2026-03-01T00:00:00Z` ~ `2026-05-06T23:00:00Z` (1593 rows)
- train: `2025-01-01T00:00:00Z` ~ `2025-12-31T23:00:00Z` (8760 rows)
- validation: `2026-01-01T00:00:00Z` ~ `2026-02-28T23:00:00Z` (1416 rows)

Candidate input split metrics from the reconstructed return-stream experiment:

| Input | Split | Return | MDD | Return/MDD diagnostic | Sharpe | Sortino | Smart Sortino | Calmar | Trades/active hours | Liquidations | Min buffer |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| A0 | train | +114.46% | +28.27% | 4.049305 | 1.979653 | 1.131499 | 0.882143 | 4.049305 | 1812 | not_replayed | not_replayed |
| A0 | validation | +19.97% | +13.67% | 1.461080 | 2.668444 | 1.654321 | 1.455414 | 15.249889 | 292 | not_replayed | not_replayed |
| A0 | locked_oos | +20.51% | +6.79% | 3.021741 | 3.993463 | 2.711715 | 2.539336 | 26.368611 | 313 | not_replayed | not_replayed |
| P0 | train | +64.11% | +17.75% | 3.610917 | 1.854506 | 1.719569 | 1.460317 | 3.610917 | 7173 | not_replayed | not_replayed |
| P0 | validation | +26.68% | +5.37% | 4.966077 | 6.497669 | 7.032072 | 6.673577 | 61.776012 | 1262 | not_replayed | not_replayed |
| P0 | locked_oos | +4.52% | +6.97% | 0.647757 | 1.217684 | 1.227542 | 1.147552 | 3.943449 | 1429 | not_replayed | not_replayed |
| E0 | train | +32.17% | +7.89% | 4.077794 | 2.121676 | 1.710879 | 1.585758 | 4.077794 | 5314 | not_replayed | not_replayed |
| E0 | validation | +11.62% | +2.79% | 4.162712 | 5.240997 | 4.732774 | 4.604245 | 34.893482 | 838 | not_replayed | not_replayed |
| E0 | locked_oos | +2.90% | +2.36% | 1.226822 | 1.780358 | 1.646460 | 1.608436 | 7.201675 | 1175 | not_replayed | not_replayed |
| S1 | train | +8.04% | +2.31% | 3.485607 | 2.064400 | 1.661117 | 1.623648 | 3.485607 | 5314 | not_replayed | not_replayed |
| S1 | validation | +2.91% | +0.77% | 3.796269 | 5.095774 | 4.582247 | 4.547448 | 25.328062 | 838 | not_replayed | not_replayed |
| S1 | locked_oos | +0.73% | +0.60% | 1.200019 | 1.748046 | 1.615190 | 1.605489 | 6.707520 | 1175 | not_replayed | not_replayed |
| S2 | train | +7.07% | +2.86% | 2.470000 | 1.818764 | 1.450761 | 1.410400 | 2.470000 | 5309 | not_replayed | not_replayed |
| S2 | validation | +2.91% | +0.77% | 3.796269 | 5.095774 | 4.582247 | 4.547448 | 25.328062 | 838 | not_replayed | not_replayed |
| S2 | locked_oos | +0.68% | +0.60% | 1.136598 | 1.630621 | 1.534185 | 1.525047 | 6.346750 | 1203 | not_replayed | not_replayed |
| S3 | train | +6.90% | +2.86% | 2.412668 | 1.779218 | 1.417786 | 1.378343 | 2.412668 | 5297 | not_replayed | not_replayed |
| S3 | validation | +2.91% | +0.77% | 3.796269 | 5.095774 | 4.582247 | 4.547448 | 25.328062 | 838 | not_replayed | not_replayed |
| S3 | locked_oos | +0.68% | +0.60% | 1.136598 | 1.630621 | 1.534185 | 1.525047 | 6.346750 | 1203 | not_replayed | not_replayed |
| S4 | train | +3.51% | +3.27% | 1.072977 | 1.014829 | 0.764783 | 0.740531 | 1.072977 | 4885 | not_replayed | not_replayed |
| S4 | validation | +1.52% | +0.90% | 1.684443 | 3.832281 | 3.193221 | 3.164658 | 10.840367 | 748 | not_replayed | not_replayed |
| S4 | locked_oos | +0.62% | +0.81% | 0.758295 | 1.514867 | 1.299905 | 1.289404 | 4.228257 | 1062 | not_replayed | not_replayed |

Hybrid Optuna outputs using the corrected external concept:

| Candidate | Train/validation score | Train return | Validation return | Locked-OOS return | Locked-OOS MDD | Return/MDD diagnostic | Sharpe | Sortino | Smart Sortino | Calmar | Liquidations | Min buffer | Deployable success | Rejection reasons |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| hybrid_v3_5_optuna | 70.585 | +47.73% | +13.31% | +8.52% | +1.77% | 4.827936 | 5.259028 | 7.316663 | 7.189734 | 32.173151 | not_replayed | not_replayed | False | dedicated_integrated_margin_replay_required_for_mixed_alpha_state_portfolio_hybrid |
| hybrid_v3_6_optuna | 85.548 | +49.52% | +12.49% | +7.79% | +1.75% | 4.454705 | 4.859674 | 5.991026 | 5.888040 | 29.199963 | not_replayed | not_replayed | False | dedicated_integrated_margin_replay_required_for_mixed_alpha_state_portfolio_hybrid |

Alpha Zoo strict 6x comparison anchor remains superior for live promotion:

| Candidate | Split | Period start | Period end | Return | MDD | Return/MDD diagnostic | Sharpe | Sortino | Smart Sortino | Calmar | Trades | Liquidations | Min buffer |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Alpha Zoo strict 6x | train | `2025-01-01T00:00:00` | `2025-10-19T13:00:00` | +68.88% | +29.57% | 2.329914 | 1.569139 | 1.919776 | 1.481707 | 2.329914 | 1779 | 0 | 9049.125962 |
| Alpha Zoo strict 6x | validation | `2025-10-22T05:00:00` | `2026-01-28T06:00:00` | +30.12% | +9.56% | 3.150734 | 1.552041 | 2.095744 | 1.912882 | 3.150734 | 524 | 0 | 9527.695928 |
| Alpha Zoo strict 6x | locked_oos | `2026-01-28T07:00:00` | `2026-05-06T23:00:00` | +41.10% | +13.67% | 3.007073 | 2.143209 | 2.841936 | 2.500237 | 3.007073 | 540 | 0 | 9572.449083 |

Decision: the fixed-input v3.5/v3.6 Optuna experiments are useful diagnostics but **not live-promotable** yet because the mixed A0/P0/E0/S-sleeve allocator has no dedicated integrated margin replay, so liquidation count and minimum margin buffer are `not_replayed`. Both fixed-input hybrids satisfy train/validation-only selection and have locked-OOS MDD below 25%, but Alpha Zoo strict 6x still dominates live promotion with locked-OOS return `+41.0967%`, zero liquidations, and positive min buffer. The fixed-input hybrids remain report/reference until a dedicated strict zero-liquidation margin replay is implemented.

Artifacts:

- Script: `scripts/research/run_profit_moonshot_hybrid_v35_v36_fixed_inputs.py`
- Regression test: `tests/test_profit_moonshot_hybrid_v35_v36_fixed_inputs.py`
- JSON report: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_v35_v36_fixed_inputs_20260517/hybrid_v35_v36_fixed_inputs_latest.json`
- Markdown report: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_v35_v36_fixed_inputs_20260517/hybrid_v35_v36_fixed_inputs_latest.md`
- Timestamped latest run: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_v35_v36_fixed_inputs_20260517/hybrid_v35_v36_fixed_inputs_20260516T172901Z.json`
- Peak RSS: `353.754 MiB` (<8 GiB)

Research history/source ledger was not regenerated: this run reused existing repo-local market/research artifacts and added a session-scoped method-adaptation report; it did not introduce a new global data-source family or chronology ledger.

---

## 2026-05-16 KST — Hybrid v3.5/v3.6 and Optuna comparison against Alpha Zoo strict 6x

Completed a repo-wide inventory and policy audit for hybrid v3.5/v3.6, hybrid Optuna, tuning/optimization, candidate-hybrid, calendar Optuna, and fresh-portfolio optimization artifacts against the preserved private/main baseline `1c6816fced44d277f6c7112934c9dded65ba710f`. Corrections on 2026-05-17 KST: the **comparison core excludes calendar/current-base-derived rows and literal hybrid/hybrid-online/hybrid-tuning rows before ranking**. `portfolio`, `allocator`, `meta`, `static_blend`, and `leverage_sweep` labels are not exclusion triggers by themselves; calendar/current-base and literal-hybrid rows are retained only in quarantine/reference ledgers.

Decision: **only** `CryptoFxAlphaZooStateStrategy / alpha_zoo_conservative_exit / strict 6x` remains live-promotion possible. Alpha Zoo selection is train/validation-only, locked-OOS is gate/report-only after candidate freeze, and strict 6x passes zero-liquidation, positive margin buffer, OOS MDD <=25%, OOS return above the invalid current-base reference, positive Sharpe/Sortino/smart Sortino/Calmar, and memory <8 GiB.

Alpha Zoo strict 6x split evidence:

- train: 2025-01-01T00:00:00 → 2025-10-19T13:00:00, return 68.8842%, MDD 29.5651%, Sharpe 1.569139, Sortino 1.919776, smart Sortino 1.481707, Calmar 2.329914, trades 1779, liq 0, min buffer 9049.125962
- validation: 2025-10-22T05:00:00 → 2026-01-28T06:00:00, return 30.1195%, MDD 9.5595%, Sharpe 1.552041, Sortino 2.095744, smart Sortino 1.912882, Calmar 3.150734, trades 524, liq 0, min buffer 9527.695928
- locked_oos: 2026-01-28T07:00:00 → 2026-05-06T23:00:00, return 41.0967%, MDD 13.6667%, Sharpe 2.143209, Sortino 2.841936, smart Sortino 2.500237, Calmar 3.007073, trades 540, liq 0, min buffer 9572.449083

Hybrid/Optuna conclusions:

- Hybrid v3.5/v3.6 rows are **not strict-core rows** after correction because their own provenance is literal hybrid / hybrid-online / hybrid-final-selection. `portfolio`, `allocator`, `meta`, `static_blend`, and `leverage_sweep` labels are not exclusion triggers by themselves; they are only evidence/context when the top-level row is already literal hybrid.
- Hybrid Optuna `live_guarded` and `train_aware_guarded` are same-family hybrid optimizer outputs and are also live-promotion invalid because those objective profiles consume OOS metrics; good-looking OOS values remain diagnostic/reference only.
- Hybrid/tuning `locked_train_val` policy shape is cleaner (`oos_is_objective_input=false`) but it is still a same-family hybrid-online tuning output, not an atomic-source hybrid candidate, and remains quarantine/reference only.
- State-distilled fresh portfolio tuning is **not literal hybrid** and is restored to the non-calendar comparison core. It remains diagnostic/non-promotable because strict liquidation/margin replay is missing and Alpha Zoo strict 6x dominates the locked-OOS/live-promotion gates.
- Calendar Optuna and calendar/current-base-dependent fresh/candidate-hybrid rows are **not part of the strict core**. They remain in a separate quarantine/reference ledger because calendar month/day/hour rules and current-base tuple dependencies are invalid before any ranking.
- Candidate-hybrid had strong OOS metrics but is excluded due calendar/current-base-source dependency, validation liquidation count `1`, not the hybrid-inside-hybrid rule.

Strict integer recheck 1x..6x was rerun in the comparison directory. Highest strict integer remains `6.0x`: OOS return `41.0967%`, MDD `13.6667%`, return/MDD diagnostic `3.007073`, Sharpe `2.143209`, Sortino `2.841936`, smart Sortino `2.500237`, liquidation `0`, min buffer `9049.125962`. The separate diagnostic 5x/6x lane is preserved with `promotion_allowed=false`.

Artifacts:

- Full JSON report: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_optuna_alpha_zoo_comparison_20260514/hybrid_optuna_alpha_zoo_comparison_latest.json`
- Corrected non-calendar JSON snapshot: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_optuna_alpha_zoo_comparison_20260514/hybrid_optuna_alpha_zoo_comparison_20260517T000000Z_calendar_quarantine_corrected.json`
- Corrected literal-hybrid quarantine JSON snapshot: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_optuna_alpha_zoo_comparison_20260514/hybrid_optuna_alpha_zoo_comparison_20260517T015000KST_hybrid_only_quarantine_corrected.json`
- Full Markdown report: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_optuna_alpha_zoo_comparison_20260514/hybrid_optuna_alpha_zoo_comparison_latest.md`
- Comparison-core split performance CSV: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_optuna_alpha_zoo_comparison_20260514/candidate_split_performance_latest.csv`
- Literal nested-hybrid quarantine CSV: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_optuna_alpha_zoo_comparison_20260514/excluded_nested_hybrid_same_family_quarantine_latest.csv`
- Calendar/current-base quarantine CSV: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_optuna_alpha_zoo_comparison_20260514/excluded_calendar_current_base_quarantine_latest.csv`
- Inventory JSON: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_optuna_alpha_zoo_comparison_20260514/hybrid_optuna_alpha_zoo_inventory_latest.json`
- Prompt checklist audit: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_optuna_alpha_zoo_comparison_20260514/prompt_checklist_audit_latest.json`
- Strict integer recheck: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_optuna_alpha_zoo_comparison_20260514/alpha_zoo_strict_integer_recheck_latest.json`

Non-calendar comparison core after corrections has `2` candidates (`crypto_fx_alpha_zoo_state_calibrated` plus the non-hybrid state-distilled portfolio diagnostic row); literal nested-hybrid quarantine has `8` candidates and calendar/current-base quarantine has `5`. Verification after literal-hybrid quarantine passed (`1308` full tests plus ruff/compileall/diff checks). Latest log `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_optuna_alpha_zoo_comparison_20260514/local_verification_hybrid_only_quarantine_20260517T020000KST.log`. Memory peak across inspected/replayed comparison artifacts: `1239.703125 MiB`, below 8 GiB. Research history/source ledger was **not regenerated** because this session did not add a new global data source family or chronology ledger; it reused existing repo-local artifacts and added a session-scoped comparison report.

---

## 2026-05-14 KST — Paper-forward diagnostics added

Added non-promotional diagnostics requested for the current leading `CryptoFxAlphaZooStateStrategy` / `alpha_zoo_conservative_exit` / strict 6x candidate. The strategy/replay now records dominant factor-family metadata and exit-reason metadata, and the real-data replay summary includes locked-OOS PnL breakdowns plus cost sensitivity.

Locked-OOS diagnostic breakdown at 6x, 10% allocation:

- Regime: `neutral` `+41.0967%` across `540` trades. Direct FX OHLCV trading remains blocked; lagged FRED state remains context only.
- Symbol: SOL/USDT `+19.2672%` (`126`), BNB/USDT `+10.3167%` (`128`), TRX/USDT `+4.9566%` (`139`), ETH/USDT `+1.4843%` (`132`), BTC/USDT `+0.6807%` (`15`).
- Side: SHORT `+26.3040%` (`259`), LONG `+11.7120%` (`281`).
- Dominant factor family: crypto residual momentum `+30.1822%` (`184`), crypto residual reversal `+8.4241%` (`237`), volume/vwap pressure `-0.0370%` (`119`).
- Exit reason: score_exit `+41.0936%` (`526`), take_profit `+16.1976%` (`4`), end_of_sample `-0.0788%` (`2`), stop_loss `-13.8700%` (`8`).
- Slippage sensitivity, round-trip 0/2.5/5/10/20 bps: `+41.0967%`, `+30.1241%`, `+20.0034%`, `+2.0585%`, `-26.1930%`.
- Conservative funding drag sensitivity, 0/1/2/5/10 bps per day: `+41.0967%`, `+40.4210%`, `+39.7486%`, `+37.7505%`, `+34.4835%`.

Promotion policy unchanged: these diagnostics are `diagnostic_only`, `promotion_allowed=false`; the strict lane remains zero-liquidation + positive buffer + OOS return, MDD cap, and positive risk-metric policy with return/MDD report-only. Research history/source ledger was not regenerated because no new global source family was introduced.

---

## 2026-05-14 KST — Return/MDD diagnostic-only policy correction

Applied the latest operator clarification: OOS return/MDD is diagnostic/report-only, not a strict promotion hurdle. The current-base/calendar tuple remains `hypothesis_reference_only`, not a selection or promotion target; the strict deploy lane still requires zero liquidation, positive buffers, OOS MDD <=25%, OOS return beating the current-base reference, and positive risk metrics.

Real current-tail run under `crypto_fx_alpha_zoo_real_data_20260514`:

- Screen: `58,845` rows, `63` factors, `20` selected cards; direct FX OHLCV remains blocked and lagged FRED state is regime context only.
- Candidate outcome ledger: `67,259` rows; train+validation `45,311`; locked-OOS `21,948`.
- Edge calibration: train/validation-only physical filter; locked-OOS calibration records `0`; calibrated edge keys `12`.
- Replay selected `alpha_zoo_conservative_exit` from `9` formulaic candidates using train/validation metrics only.
- Strict zero-liquidation lane promoted integer: `6.0x`, liquidation count `0`, min buffer `9049.125962`, OOS return `41.0967%`, OOS MDD `13.6667%`, return/MDD `3.007073`, Sharpe `2.143209`, Sortino `2.841936`, smart Sortino `2.500237`.
- Deployable success: `true`; return/MDD `3.007073` vs current-base `6.916878` is reported as diagnostic-only and does not block promotion.
- Diagnostic 5x/6x lane is separate and non-promotional; 5x/6x both zero liquidation but `promotion_allowed=false` in that diagnostic lane.
- Peak RSS `626.7266 MiB` (<8 GiB).
- Artifacts: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/crypto_fx_alpha_zoo_real_data_20260514/`.

Research history/source ledger was not regenerated because no new global source family was introduced; this pass reused the current-tail crypto cache and 20260512 lagged FRED external-state source.

---

## 2026-05-13 KST — Related state-distilled regime-boost overlay note

A separate research-only `StateDistilledRegimeBoostPortfolio` overlay was tested on the existing state-distilled seeds. This did not change the `CryptoFxAlphaZooStateStrategy` promotion policy or Alpha Zoo factor cards. The overlay reused real current-tail crypto data and lagged FRED external-risk state, kept calendar/current-base as hypothesis reference only, and kept locked-OOS gate/report-only after freeze.

Result: strict zero-liquidation/margin gates passed, but validation and locked-OOS return/risk-quality metrics failed, so no new deployable promotion came from the regime-boost overlay. Artifacts live under `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/state_distilled_regime_boost_20260513/`.

---

---

## 2026-05-13 — Real-data Alpha Zoo calibrated replay result

- Ran real current-tail Alpha Zoo screen against `/home/hoky/Quants-agent/LuminaQuant/var/cache/profit_moonshot_fresh_start/joined_panel_de62df511cec53df6ad39521.parquet` with lagged FRED context; direct FX OHLCV trading stayed blocked because current-tail cache contains crypto OHLCV only.
- Factor/card validity passed fail-closed gates: `calendar_primary=false`, `uses_locked_oos_for_selection=false`, strategy validity pass.
- Candidate outcome ledger: `45160` rows; train+validation `30494`; locked-OOS `14666`.
- Edge calibration physically filtered to train/validation: input `45160`, calibration `30494`, locked-OOS calibration `0`, excluded locked-OOS `14666`.
- Replay grid selected `alpha_zoo_conservative_exit` from `9` formulaic candidates using train/validation metrics only; locked-OOS remained hidden until candidate freeze.
- Strict zero-liquidation lane highest safe integer: `6.0x`, liquidation count `0`, min buffer `9049.125962`, OOS return `41.0967%`, OOS MDD `13.6667%`, return/MDD `3.007073`, Sharpe `2.143209`.
- 2026-05-14 latest correction: return/MDD is diagnostic/report-only per operator clarification; OOS return must beat the invalid current-base reference, but OOS return/MDD does not block promotion.
- Deployable success is now `true` under the corrected policy: strict 6x has OOS return `41.0967%`, MDD `13.6667%`, Sharpe `2.143209`, Sortino `2.841936`, smart Sortino `2.500237`, Calmar/return-MDD `3.007073`, zero liquidations, and positive buffers.
- Diagnostic 5x/6x lane remains non-promotional and separate: 5x/6x both zero liquidation in this approximate replay, but that lane is report-only.
- Peak RSS `512.711` MiB (<8 GiB).
- Artifacts: `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/crypto_fx_alpha_zoo_real_data_20260513/crypto_fx_alpha_zoo_real_data_summary_latest.json`, `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/crypto_fx_alpha_zoo_real_data_20260513/crypto_fx_alpha_zoo_state_replay_latest.json`, `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/crypto_fx_alpha_zoo_real_data_20260513/edge_calibration_latest.json`, `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/crypto_fx_alpha_zoo_real_data_20260513/candidate_outcome_ledger_latest.jsonl`.
- Research history/source ledger not regenerated: No new external source class or global chronology/source-ledger change; reused existing current-tail cache and 20260512 lagged FRED external-state artifact, added only session-scoped Alpha Zoo artifacts.

---

---

## Static baseline context — 2026-05-12 origin

## Current answer: what should be used next?

Use `CryptoFxAlphaZooStateStrategy` as the next primary research path, but only after wiring it to real current-tail data and outcome calibration.

Do **not** use the high-performing current-base/calendar tuple as a live strategy. It remains useful as a teacher/hypothesis/reference because it explains a profitable market-state pattern, but it is calendar-primary and therefore invalid for live promotion.

## Latest green baseline

- Latest pushed green head: `e4b5f6942fe06ffff262502e40ff7dd4d7005323` on `private/main`.
- Prior implementation head for external-risk teacher pass: `fcc63f6c053c451152b0d780fa84ee91b5512f82`.
- GitHub Actions were green for both `ci` and `private-ci` after the handoff commit.

## What exists now

### 1. Alpha Zoo / outcome-calibration scaffold

Implemented files:

- `src/lumina_quant/alpha_zoo/operators.py`
- `src/lumina_quant/alpha_zoo/crypto_fx_factors.py`
- `src/lumina_quant/alpha_zoo/factor_card.py`
- `src/lumina_quant/research/triple_barrier.py`
- `src/lumina_quant/research/candidate_outcome_ledger.py`
- `src/lumina_quant/research/edge_calibration.py`
- `src/lumina_quant/strategies/crypto_fx_alpha_zoo_state.py`
- `scripts/research/run_crypto_fx_alpha_zoo_screen.py`
- `scripts/research/replay_crypto_fx_alpha_zoo_state.py`
- `scripts/research/calibrate_crypto_fx_edges.py`

Current state: smoke/research scaffold only. It proves factor generation, train/validation-only selection metadata, triple-barrier labels, candidate ledger, edge calibration, and calibrated-entry strategy plumbing. It does **not** yet prove economic profitability on real current-tail data.

### 2. External-risk teacher pass

Implemented `scripts/research/fetch_profit_moonshot_external_state.py` to fetch lagged FRED daily state features from:

- DTWEXBGS
- VIXCLS
- DGS2
- DGS10
- DCOILWTICO

Features are lagged before joining to hourly crypto panels to avoid same-day lookahead.

Added non-calendar replay families:

- `calendar_teacher_state_similarity`
- `calendar_teacher_state_fade`
- `state_distilled_external_risk_filter`

Results:

- `calendar_teacher_state_similarity`: 972 specs, 0 survivor.
- `calendar_teacher_state_fade`: 324 specs, 0 survivor.
- `state_distilled_external_risk_filter`: 1728 specs, 565 train/validation-positive, 0 replay survivor under legacy shadow-MDD gate, peak RSS about 280 MiB.

### 3. Best valid strict state-distilled seed

Train/validation-selected strict candidate:

`fresh_state_distilled_ext_both_lb168_fast72_z075_ret180_h168_tp600_fl0_xr125` at 4x.

Metrics:

- Train: `+30.9030%`, MDD `10.2437%`, Sharpe `1.8484`, liquidation `0`.
- Validation: `+12.4704%`, MDD `2.5167%`, Sharpe `5.7588`, liquidation `0`.
- Locked-OOS: `+2.4852%`, MDD `2.5328%`, Sharpe `1.5096`, liquidation `0`.
- All split min margin buffers are positive.
- Strategy-validity passes.
- `deployable_success=false` because it does not beat the invalid current-base/calendar reference economics.

### 4. Calendar/current-base teacher status

The current-base/calendar tuple remains economically strong:

- Locked-OOS return `+6.4281%`.
- Return/MDD `6.9169`.
- Sharpe about `5.2024` in the liquidation-aware reference report.

But it is calendar-primary/fixed calendar behavior, so it is invalid for live promotion and must remain:

- `hypothesis_reference_only`
- not a selection target
- not a live strategy
- not a promoted candidate

## Next research objective

Convert the Alpha Zoo scaffold from synthetic smoke to real current-tail research:

1. Wire `run_crypto_fx_alpha_zoo_screen.py` to real crypto OHLCV/funding/OI/flow fields where available.
2. Add FX OHLCV regime fields if reliable; otherwise use lagged FRED risk-state as temporary regime context and explicitly record direct-FX trading as blocked.
3. Generate real factor cards with source coverage and `uses_locked_oos_for_selection=false`.
4. Produce triple-barrier candidate outcomes on train/validation.
5. Calibrate edge buckets with shrinkage and lower-confidence edge gating.
6. Replay `CryptoFxAlphaZooStateStrategy` plus state-distilled/residual seeds only if selected by train/validation.
7. Open locked-OOS after freeze only as gate/report.
8. Run strict zero-liquidation integer grid and separate diagnostic nonfatal 5x/6x lane.

## Required research-note practice

Every future profit-moonshot session must update research notes before final handoff:

- Update or supersede this file when Alpha Zoo real-data results change.
- Update `.omx/notepad.md` with concise conclusions and artifact paths.
- Update the active `.omx/plans/*` file with result status.
- Write or update `docs/session_handoff_*` for the session.
- If the work changes the global research inventory/source ledger, regenerate or explicitly update `docs/research_note/research_history.md` and the matching `var/reports/.../research_history/` artifacts, or document why regeneration was not required.

## Failure mode to avoid

Do not repeat the earlier loop of finding a good-looking single rule and then discovering it is calendar-primary or OOS-selected. The next path must be evidence-first:

`real factors → train/validation labels → calibrated edge → stateful replay → locked-OOS gate/report → strict liquidation validation`.

---

## 2026-06-05 — 69-asset clean OOS non-nested teacher-leaf rerun

### Objective

Re-check whether the previously high-performing nested hybrid/teacher structure can be made live-usable without nested hybrid material.  The rerun keeps the current live discipline:

- Universe/timeframes: 69 Binance research symbols, `30m,1h,2h,4h,6h,8h,12h,1d`.
- Cost: 10 bps slippage/cost proxy.
- Refit: monthly, calendar day 1 UTC.
- Selection inputs: expanding train + prior 2 calendar months validation only.
- Locked OOS: next calendar month, report-only after fold params/candidate selection are frozen.
- OOS span: `2025-09-01T00:00:00` → `2026-06-01T06:30:00`; 2026-06 is a partial fold because latest available data is `2026-06-01T06:30:00`.

### Implementation / hygiene changes

- Added `teacher_leaf_blend` as a non-leaf portfolio family that **does not use hybrid rows as material**. It rebuilds the old nested-teacher idea from clean leaf candidates only, with train/validation scoring and train/validation risk scaling.
- Kept `teacher_leaf_blend` out of downstream material by adding it to `_NON_LEAF_PORTFOLIO_FAMILIES`.
- Fixed `dynamic_conviction_switch` fold accounting: when the train/validation pool has no eligible aggressive/fallback branch, it now emits an explicit clean cash/no-position guard instead of silently omitting the fold. This avoids 9/10 fold aggregate distortion.
- Added an optional numba JIT speed path for `_debounced_state_signal`; sample verification matched the Python loop exactly. The full core-family run still took `25:52.79`, so the dominant cost remains per-symbol Optuna + pandas feature work, not just the signal loop.
- Added `--source-symbol-workers` as an experimental per-symbol threaded option. Tiny smoke passed, but `workers=4` did not materially improve the full 69-symbol path, so default remains sequential (`1`).

### Main artifact

- JSON: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_69_asset_teacher_leaf_blend_20260605/teacher_leaf_blend_corefamilies_full_v2_20260605.json`
- Markdown: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_69_asset_teacher_leaf_blend_20260605/teacher_leaf_blend_corefamilies_full_v2_20260605.md`

### Final clean-OOS result table

| Candidate | Clean | OOS comp | Ann approx | Max bar MDD | Monthly eq MDD | Sharpe | Sortino | Hit | Min OOS | Latest OOS | Val min/mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `profile_optuna:selected_optuna` | True | 10.05% | 12.18% | 19.20% | 13.58% | 0.50 | 1.16 | 5/10 | -9.30% | -0.73% | 25.71%/59.74% |
| `profile_optuna:hybrid_v3_5` | True | 9.06% | 10.97% | 19.20% | 13.58% | 0.47 | 1.04 | 5/10 | -9.30% | -0.23% | 25.71%/59.51% |
| `dynamic_conviction_switch:t0.85_risk_capped_fallback` | True | 8.61% | 10.42% | 10.08% | 4.62% | 0.76 | 2.38 | 5/10 | -2.65% | 0.00% | 0.00%/4.62% |
| `dynamic_conviction_switch:t0.85_strict_fallback` | True | 10.47% | 12.69% | 10.62% | 2.11% | 0.67 | 1.13 | 4/10 | -8.56% | -0.17% | 0.00%/9.22% |
| `strict_efficiency:growth_mdd20_gross8_69_asset_efficiency_repair_optuna` | True | 6.97% | 8.42% | 7.32% | 3.70% | 0.66 | 1.84 | 4/10 | -3.58% | -0.17% | 0.01%/6.18% |
| `teacher_leaf_blend:val_balanced_top12_cap20_mdd20` | True | -28.11% | -32.70% | 26.40% | 22.44% | -1.10 | -1.70 | 4/10 | -15.80% | -0.88% | 29.89%/51.65% |
| `teacher_leaf_blend:val_return_top8_cap35_mdd30` | True | -42.49% | -48.52% | 37.72% | 35.31% | -1.33 | -2.10 | 4/10 | -22.86% | -1.07% | 40.46%/85.50% |

### Interpretation

- The de-nested teacher idea **does not survive** full clean OOS.  It looked good in the latest 2-fold smoke (`val_return_top8` about +90.22% OOS comp), but across all 10 folds it is negative.  The failure pattern is classic validation-overfit/high-volatility-chasing: very high validation returns (`val_return_top8` mean validation about +85.50%) paired with large negative next-month OOS tails.
- Therefore, do **not** promote `teacher_leaf_blend` to live.  Keep it as a negative control / lesson: leaf-only plus validation-only mechanics are necessary but not sufficient; validation return must be regularized heavily for tail risk and regime turnover.
- Best practical live/paper shadow candidate is `dynamic_conviction_switch:t0.85_risk_capped_fallback`: lower comp than profile selected, but much better realized risk profile: max bar MDD 10.08%, monthly equity MDD 4.62%, min OOS -2.65%, Sortino 2.38.
- If pure return is prioritized, `profile_optuna:selected_optuna` remains the highest 10-fold clean full candidate in this core-family rerun (+10.05% comp), but its min OOS (-9.30%) and bar MDD (19.20%) make it less attractive for live without an external risk throttle.
- `dynamic_conviction_switch:t0.85_strict_fallback` has slightly higher comp (+10.47%) and low monthly equity MDD (2.11%) but only 4/10 positive OOS folds and a -8.56% worst month, so it is a shadow/challenger rather than the primary risk candidate.

### Recommendation

For live/paper continuation:

1. Primary paper candidate: `dynamic_conviction_switch:t0.85_risk_capped_fallback`.
2. Return-seeking shadow: `profile_optuna:selected_optuna`.
3. Additional challenger/shadow: `dynamic_conviction_switch:t0.85_strict_fallback`.
4. Do not use `teacher_leaf_blend` except as a monitored research negative-control family.
5. Next improvement should not chase current OOS.  Use train/validation-only rules to add a risk throttle around `profile_optuna:selected_optuna`, preferably from realized validation drawdown/volatility, external lagged risk-state, and cash/no-position guards.  Then require a fresh-forward shadow window before promotion.

### Validation evidence

- `uv run ruff check scripts/research/run_alpha_zoo_69_asset_monthly_refit_walkforward.py scripts/research/run_alpha_zoo_69_asset_optuna_hybrid_refit.py tests/test_alpha_zoo_69_asset_monthly_refit_walkforward.py` — passed.
- `uv run pytest -q tests/test_alpha_zoo_69_asset_monthly_refit_walkforward.py` — `33 passed`.
- Full rerun: `25:52.79`, max RSS `1293384 KB`, exit status `0`.
- Audits in final artifact: metric reconciliation passed, dynamic self-feed audit passed, online weight audit passed.

---

## 2026-06-05 — 85-symbol expanded-universe clean dynamic risk-scaled rerun

### Objective

Address the weak clean OOS return after nested-hybrid removal while reflecting the expanded Binance TradFi universe.  The rerun keeps the same live discipline:

- Universe/timeframes: 85 Binance research symbols = 10 core crypto + 75 `TRADIFI_PERPETUAL`, `30m,1h,2h,4h,6h,8h,12h,1d`.
- Cost: 10 bps slippage/cost proxy.
- Refit: monthly, calendar day 1 UTC.
- Selection inputs: expanding train + prior 2 calendar months validation only.
- Locked OOS: next calendar month, report-only after fold params/candidate selection are frozen.
- OOS span: `2025-09-01T00:00:00` → `2026-06-05T12:00:00`; 2026-06 is partial because that is the latest local Binance bar.

### Implementation / hygiene changes

- Expanded the static Binance research universe snapshot to 85 symbols and backfilled the 16 newly listed/added TradFi symbols locally.  New symbols are monitored/backfilled now, but most are not train-eligible yet because they listed around 2026-06.
- Made the OHLCV loader missing-symbol safe and records requested/loaded/missing symbol counts.  The 2026-06-05 run requested and loaded all 85 symbols.
- Fold-local feature support now excludes symbols without train-window data, preventing newly listed OOS-only symbols from diluting cross-sectional ranks.
- Removed nested-hybrid material from the final path: the new dynamic candidates reference only leaf strategy labels in `final_weights`.
- Added validation-only dynamic position sizing:
  - base dynamic switch still picks a clean leaf from train/validation only;
  - `val_mdd12/15/20_scaled` variants increase exposure only on a coarse grid that remains within the fold's validation MDD budget;
  - `val_ret02_calmar80_gate` can cash-guard weak validation months;
  - cash/no-eligible folds are explicitly emitted for every dynamic label, so aggregate metrics use all 10 folds.
- Removed pandas `pct_change` default-fill behavior in the alpha-zoo loaders to avoid warning-driven implicit forward-fill distortion.

### Main artifacts

- JSON: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_85_asset_dynamic_scaled_20260605/alpha_zoo_85_asset_dynamic_scaled_full_v2_20260605.json`
- Markdown: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_85_asset_dynamic_scaled_20260605/alpha_zoo_85_asset_dynamic_scaled_full_v2_20260605.md`
- Symbol expansion/backfill report: `var/reports/data_collection/binance_1m_research_universe_20260605_symbol_expansion/binance_1m_research_universe_collection_20260605T123257Z.json`
- ExchangeInfo snapshot: `var/reports/symbol_universe_20260605/binance_fapi_exchangeInfo_20260605.json`

### Final clean-OOS result table

| Candidate | Clean | Nested material | OOS comp | Ann approx | Max bar MDD | Monthly eq MDD | Sharpe | Sortino | Hit | Min OOS | Tail ratio | Profit factor | Max validation MDD |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dynamic_conviction_switch:t0.85_risk_capped_fallback_val_mdd20_scaled` | True | False | 23.41% | 28.72% | 23.59% | 5.75% | 0.91 | 5.24 | 5/10 | -3.18% | 5.52 | 5.12 | 18.36% |
| `dynamic_conviction_switch:t0.85_risk_capped_fallback_val_mdd15_scaled` | True | False | 18.46% | 22.55% | 19.29% | 5.75% | 0.87 | 4.14 | 5/10 | -3.18% | 4.55 | 4.26 | 14.84% |
| `dynamic_conviction_switch:t0.85_risk_capped_fallback_val_mdd12_scaled` | True | False | 13.17% | — | 14.79% | — | 0.80 | 2.96 | 5/10 | -3.18% | — | — | ≤12% target |
| `dynamic_conviction_switch:t0.85_risk_capped_fallback_val_ret02_calmar80_gate` | True | False | 12.97% | 15.76% | 10.08% | 0.00% | 1.16 | 0.00* | 3/10 | 0.00% | n/a | n/a | 7.58% |
| `dynamic_conviction_switch:t0.85_risk_capped_fallback` | True | False | 8.02% | 9.70% | 10.08% | 5.13% | 0.71 | 2.06 | 5/10 | -2.65% | 2.71 | 2.59 | 7.58% |
| `strict_calm_leaf_selector:val_mdd8_train_val_spike_penalty` | True | False | 3.31% | 3.98% | 10.08% | 5.22% | 0.32 | 0.63 | 3/10 | -5.22% | 1.73 | 1.47 | 7.58% |

`*` Sortino/profit factor are not meaningful for the cash-gated candidate because its monthly loss observations are zero in this sample.

### Fold details for the selected top candidate

Top candidate: `dynamic_conviction_switch:t0.85_risk_capped_fallback_val_mdd20_scaled`.

| Fold | OOS return | OOS MDD | Selected leaf | Leaf weight |
| --- | ---: | ---: | --- | ---: |
| 2025-09 | 0.14% | 0.07% | `strict_efficiency:growth_mdd20_gross8_69_asset_efficiency_repair_optuna` | 1.00 |
| 2025-10 | 0.00% | 0.00% | cash/no eligible signal | 0.00 |
| 2025-11 | 28.71% | 23.59% | `strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna` | 2.50 |
| 2025-12 | -0.13% | 1.63% | `strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna` | 1.00 |
| 2026-01 | 0.99% | 2.15% | `strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna` | 1.25 |
| 2026-02 | 0.26% | 8.44% | `strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna` | 2.50 |
| 2026-03 | -2.65% | 2.80% | `strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna` | 1.00 |
| 2026-04 | -3.18% | 3.45% | `strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna` | 1.25 |
| 2026-05 | 0.47% | 2.28% | `strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna` | 1.00 |
| 2026-06 | 0.00% | 0.00% | `strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna` | 1.00 |

### Interpretation

- The 85-symbol expansion is reflected in the monitor/backfill layer, but not yet a direct performance source: the newest TradFi symbols mostly lack train-window data and are intentionally excluded from fold-local feature support until they have enough history.
- The improvement comes from clean validation-only position sizing, not from OOS oracle selection or nested hybrid reuse.  `final_weights` point only to strict-efficiency leaf candidates or cash.
- The best clean candidate improves from the prior unscaled dynamic baseline (+8.02% comp, max bar MDD 10.08%) to +23.41% comp, but it spends a 23.59% bar-MDD budget.  The more conservative `val_mdd15_scaled` variant gives +18.46% comp with 19.29% max bar MDD; `val_mdd12_scaled` gives +13.17% comp with 14.79% max bar MDD.
- Hard-stop promotability is still false: the candidate does not beat the historical challenger (+53.38% comp) or robust-default hurdle (+27.01% comp / 15% MDD limit).  Treat this as paper/shadow, not real-money approval.
- The clean candidate is still concentrated in one strict-efficiency leaf in most folds.  That is preferable to hidden nested duplication, but live risk should treat it as one alpha family, not a diversified portfolio.

### Recommendation

1. Primary paper/shadow return candidate: `dynamic_conviction_switch:t0.85_risk_capped_fallback_val_mdd20_scaled` if a ~24% bar-MDD budget is acceptable.
2. More balanced paper candidate: `dynamic_conviction_switch:t0.85_risk_capped_fallback_val_mdd15_scaled`.
3. Conservative risk candidate: `dynamic_conviction_switch:t0.85_risk_capped_fallback_val_mdd12_scaled` or the unscaled `dynamic_conviction_switch:t0.85_risk_capped_fallback`.
4. Do not revive nested hybrid-on-hybrid blends; the current improvement already avoids nested material.
5. Keep collecting/monitoring all 85 symbols.  Refit should automatically start admitting newly listed TradFi names once they have train+validation coverage.
6. Before any real-money promotion, require fresh-forward shadow evidence after 2026-06-05 and an execution-layer risk cap that enforces total gross, per-asset concentration, and TradFi session/liquidity guards.

### Validation evidence

- Full rerun: `alpha_zoo_85_asset_dynamic_scaled_full_v2_20260605`, 10 folds, 85/85 symbols loaded, latest data `2026-06-05T12:00:00`, peak RSS `1191.1 MiB`, exit status `0`.
- Audits in final artifact: metric reconciliation passed, dynamic self-feed audit passed, online weight audit passed.
- `uv run ruff check scripts/research/run_alpha_zoo_69_asset_monthly_refit_walkforward.py tests/test_alpha_zoo_69_asset_monthly_refit_walkforward.py` — passed during implementation.
- `uv run pytest tests/test_alpha_zoo_69_asset_monthly_refit_walkforward.py::test_dynamic_conviction_switch_emits_fallback_when_aggressive_pool_missing -q` — passed during implementation.

# Next Session Prompt — Profit Moonshot Alpha Zoo Real-Data Calibration (2026-05-12)

Copy/paste this into the next Codex/OMX session.

```text
$ralplan $team $ralph 이어서 진행해. 작업 디렉터리는 /home/hoky/Quants-agent/LuminaQuant 이야.

먼저 최신 상태를 맞춰:
- git fetch private
- git checkout private-main
- git reset --hard private/main
- git status -sb

먼저 아래 파일들을 읽어:
- .omx/plans/profit_moonshot_alpha_zoo_real_data_next_plan_20260512.md
- docs/session_handoff_20260512_crypto_fx_alpha_zoo_state.md
- docs/session_handoff_20260512_state_distilled_external_risk_filter.md
- .omx/plans/profit_moonshot_crypto_fx_alpha_zoo_state_plan_20260512.md
- var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/crypto_fx_alpha_zoo_state_20260512/crypto_fx_alpha_zoo_state_v0_summary_latest.json
- var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/liquidation_aware_state_distilled_external_risk_filter_20260512/liquidation_aware_current_base_latest.json
- var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/external_market_state_20260512/external_market_state_lagged.json

현재 pushed green head는 private/main fcc63f6c053c451152b0d780fa84ee91b5512f82 이다. 이 baseline을 보존해.

지금 써야 할 주력은 `CryptoFxAlphaZooStateStrategy` + real-data triple-barrier outcome calibration 이다. 기존 calendar/current-base tuple은 locked-OOS +6.4281%, return/MDD 6.9169로 강하지만 calendar-primary라 live strategy로 invalid다. teacher/hypothesis reference로만 쓰고 selection target이나 promotion target으로 쓰지 마라.

현재 valid best는 `state_distilled_external_risk_filter` 계열 4x strict zero-liquidation 후보다: train +30.9030%, validation +12.4704%, locked-OOS +2.4852%, OOS MDD 2.5328%, Sharpe 1.5096, liquidation 0/0/0, min buffers positive. 하지만 invalid current-base reference를 못 이겨 deployable_success=false다.

목표:
Crypto/FX Alpha Zoo scaffold를 synthetic smoke가 아니라 real current-tail crypto/FX/FRED data에 연결하고, factor screen → triple-barrier candidate ledger → train/validation-only edge calibration → stateful replay → strict liquidation-aware validation까지 진행해. 새 hand-tuned calendar proxy를 만들지 말고, formulaic alpha zoo + calibrated edge gate로 성능을 만들어라.

필수 제약:
- calendar/month/day/hour entry rule 금지
- current-base/calendar tuple은 hypothesis_reference_only
- train/validation-only selection; locked-OOS는 candidate freeze 이후 gate/report-only
- factor cards/strategy_validity fail-closed
- strict deploy lane과 diagnostic nonfatal 5x/6x lane 분리
- strict lane에서 liquidation count >0 또는 min margin buffer <=0이면 promoted success 금지
- OOS MDD <= 25%
- OOS return 및 return/MDD가 invalid current-base reference보다 개선되어야 deployable
- memory < 8 GiB

우선순위 TODO:
1. `scripts/research/run_crypto_fx_alpha_zoo_screen.py`가 real current-tail data를 먹도록 adapter/CLI를 구현하거나 기존 데이터 로더와 연결해.
2. real train/validation Alpha Zoo factor screen을 돌리고 factor cards에 source coverage, calendar_primary=false, uses_locked_oos_for_selection=false를 남겨.
3. `triple_barrier.py` / `candidate_outcome_ledger.py`로 real candidate outcomes를 만들고 ledger 저장.
4. `edge_calibration.py`로 train/validation-only calibrated lower-confidence edge를 계산하고 non-positive/tail-loss buckets를 block/downsize.
5. `CryptoFxAlphaZooStateStrategy` calibrated replay를 state_distilled_external_risk_filter/residual-pair seeds와 좁고 해석 가능한 grid로 비교해.
6. 후보 freeze 후 locked-OOS를 report/gate-only로 열어.
7. integer leverage 1x..6x strict zero-liquidation lane과 diagnostic nonfatal 5x/6x lane을 둘 다 보고해.
8. 결과/handoff를 `.omx/notepad.md`, `.omx/plans/...`, `docs/session_handoff_*alpha_zoo*`, `var/reports/.../alpha_v2/crypto_fx_alpha_zoo_*`에 저장해.

검증은 targeted tests, focused pytest, full pytest, ruff, compileall, git diff --check를 모두 돌려. 통과하면 Lore commit으로 private/main에 push하고 GitHub Actions ci/private-ci green까지 확인해.
```

## Quick shell bootstrap

```bash
cd /home/hoky/Quants-agent/LuminaQuant
git fetch private
git checkout private-main
git reset --hard private/main
git status -sb
sed -n '1,260p' docs/next_session_prompt_profit_moonshot_alpha_zoo_real_data_20260512.md
```

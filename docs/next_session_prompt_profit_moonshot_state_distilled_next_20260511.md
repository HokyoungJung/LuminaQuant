# Next Session Prompt — Profit Moonshot State-Distilled Non-Calendar Strategy

Paste the following into a new Codex/OMX leader session from `/home/hoky/Quants-agent/LuminaQuant`.

```text
$ralplan $team $ralph 이어서 진행해. 작업 디렉터리는 /home/hoky/Quants-agent/LuminaQuant 이야.

먼저 아래 파일들을 읽어:
- .omx/plans/profit_moonshot_state_distilled_next_plan_20260511.md
- docs/session_handoff_20260511_profit_moonshot_state_distilled_leadership_unwind.md
- docs/research_note/state_distilled.md
- docs/next_session_prompt_profit_moonshot_state_distilled_next_20260511.md
- var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/state_distilled_leadership_unwind_20260511/fresh_start_overhaul_replay_latest.json
- var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/state_distilled_leadership_unwind_20260511/fresh_start_overhaul_replay_candidates.csv
- var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/liquidation_aware_state_distilled_20260511/liquidation_aware_current_base_latest.json
- var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/liquidation_aware_state_distilled_20260511/SESSION_HANDOFF_20260511_STATE_DISTILLED_LEADERSHIP_UNWIND.md

현재 pushed green head는 private/main 7e451311757a1ce0e43bebaec0a24b3746dbcb65 이고, 코드/성능 baseline은 02f4520cf906f48089b8852c2651a0f1e4bd0c1c 이다. baseline을 보존해.


현재까지 연구결과 요약:
- old current-base/calendar tuple은 OOS 성과는 좋지만 calendar-primary라 live strategy로는 invalid다. hypothesis generator로만 사용한다.
- 새 non-calendar family `state_distilled_leadership_unwind`는 broad-market anchor, cross-sectional leadership/laggard rank, residual z-score, fast momentum, flow, OI, funding, regime context만 사용한다. month/day/hour calendar entry rule은 금지한다.
- replay 결과: 648 specs, replay survivor 0, success candidate 0, peak RSS 약 254 MiB.
- best strict zero-liquidation candidate: `fresh_state_distilled_both_lb168_fast72_z075_ret180_h168_ls590_ss100_tp600` at 4x.
- 4x 성과: train +32.9431% / MDD 9.4768% / Sharpe 1.9463 / Sortino 2.0182 / Calmar 3.4766 / liquidation 0; validation +11.6925% / MDD 3.1028% / Sharpe 4.9606 / Sortino 5.9849 / Calmar 31.6786 / liquidation 0; locked-OOS +2.4722% / MDD 2.5328% / Sharpe 1.5131 / Sortino 1.8815 / Calmar 5.6787 / liquidation 0.
- current-base reference 대비: candidate OOS +2.4722%와 return/MDD 0.9761은 current-base reference OOS +6.4281%와 return/MDD 6.9169를 못 이겨서 deployable_success=false다.
- 5x diagnostic: train +34.3497% / MDD 15.5461% / liquidation 2; validation +14.7888% / liquidation 0; OOS +3.0887% / MDD 3.1589% / liquidation 0; wipeout 0.
- 6x diagnostic: train +34.8783% / MDD 22.3058% / liquidation 4; validation +17.9560% / liquidation 0; OOS +3.7036% / MDD 3.7832% / liquidation 0; wipeout 0.
- 결론: strategy-validity는 개선됐지만, strict deployable improvement는 아직 아니다. 다음 작업은 calendar proxy 없이 market-state mechanism을 강화하고 train/validation-only selection으로 OOS economics를 개선하는 것이다.

목표는 calendar/month/day/hour rule 없이, rejected calendar 결과와 유사한 시장상태를 설명하고 재현할 수 있는 합법적인 non-calendar 전략을 발전시키는 것이다. invalid current-base/calendar tuple은 hypothesis generator로만 사용하고 selection target으로 쓰지 마라.

구현 전 테스트를 먼저 추가해:
- valid candidate family가 calendar entry fields, fixed months, fixed days, fixed hours를 쓰면 실패
- locked-OOS는 selection/hyperparameter choice에 절대 쓰지 않고 report-only/gate-only
- train/validation-only nested selection 또는 동등한 selection provenance 기록
- strict deploy lane과 diagnostic nonfatal liquidation lane을 분리해서 기록
- strict lane에서 liquidation count > 0 또는 margin buffer <= 0이면 promoted success 금지
- diagnostic nonfatal lane은 liquidation count/event drawdown/equity loss/recovery를 보고하되 live promotion과 구분

다음 전략군을 좁고 해석 가능한 grid로 구현/개선해:
1. crowded leadership unwind v2: non-BTC crowded leader의 return/OI/funding/residual-z 과열 후 momentum weakening 또는 rank-gap compression에서 unwind 진입
2. funding/OI exhaustion carry reversal: funding percentile + OI acceleration + price confirmation break를 이용한 crowd unwind
3. beta-hedged residual reversion: BTC/ETH 또는 broad basket beta를 제거한 residual extreme reversion
4. dispersion compression breakout/unwind: cross-sectional dispersion compression 이후 regime/flow로 continuation vs unwind 선택
5. volatility/regime/margin-buffer exposure scaler: 신호와 leverage를 분리하고 realized vol, broad drawdown, funding instability, margin forecast로 exposure 조절

반드시 train/validation만으로 후보를 고르고, locked-OOS는 후보 freeze 이후 gate/report로만 열어. 필요하면 1x~6x integer grid에서 strict zero-liquidation 최고 leverage와 diagnostic nonfatal 5x/6x 성과를 둘 다 보고해.

통과 조건:
- calendar-primary 전략 금지
- selection은 train/validation only
- train/validation return positive
- strict lane: train/validation/OOS liquidation count = 0, 모든 split min margin buffer > 0
- OOS MDD <= 25%
- OOS return 및 return/MDD가 baseline reference보다 개선
- Sharpe/Sortino/smart Sortino/Calmar 양호
- memory < 8 GiB
- locked-OOS는 gate-only/report-only

결과와 handoff를 저장해:
- .omx/notepad.md
- .omx/plans/profit_moonshot_state_distilled_next_plan_20260511.md 또는 후속 plan
- docs/session_handoff_*state_distilled*_20260511.md
- var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/state_distilled_* 또는 liquidation_aware_state_distilled_* 신규 디렉터리

검증은 targeted tests, focused pytest, full pytest, ruff, compileall, git diff --check를 모두 돌려. Lore commit으로 private/main에 push하고 GitHub Actions ci/private-ci green까지 확인해.
```

# Plan — Alpha Zoo 7x/0.20 paper-forward live preflight and monitoring

## Context

The current pushed baseline is commit `a68e81af`, which finalized the 10bps risk-selection artifact at:

`var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_10bps_full_retune_20260519/`

The active final 10bps profile is `higher_risk_train_return_tilt_v1`, selecting:

`fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_7p0x_0p2alloc`

The balanced reference remains:

`fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_6p0x_0p175alloc`

Locked-OOS remains gate/report-only; real-money execution remains prohibited.

## RALPLAN-DR summary

### Principles

1. **Paper first**: no real-money orders; only paper/testnet readiness and monitoring artifacts may be produced.
2. **Replay/live sizing parity**: live decision sizing must preserve isolated margin semantics (`allocation_fraction * leverage`) rather than legacy notional-cap truncation.
3. **Cost evidence before promotion**: forward fills must measure realized round-trip fee/slippage and compare against the 10bps research assumption before any real-money discussion.
4. **Reference pair tracking**: run active 7x/0.20 and balanced 6x/0.175 side by side so higher-risk drawdown/fill degradation is observable.
5. **Fail closed**: ready_for_real must remain false until explicit human approval plus clean paper-forward evidence.

### Decision drivers

1. Live safety and auditability before any capital deployment.
2. Evidence that the 10bps all-in round-trip assumption is realistic under Binance futures execution.
3. Detecting whether the weak validation return on 7x/0.20 survives forward fills or degrades below the balanced reference.

### Viable options

- **Option A — Paper-only decision + monitoring for both active and balanced profiles.** Best safety/audit tradeoff; recommended.
- **Option B — Only wire the 7x/0.20 active profile.** Faster, but loses the balanced reference needed to judge whether the higher-risk selection is worth keeping.
- **Option C — Continue offline research only.** Safest operationally, but delays fill-quality evidence and does not answer whether 10bps is realistic live.

### Decision

Use **Option A**. Create paper/testnet live-decision and preflight artifacts for the 7x/0.20 active model and 6x/0.175 balanced reference, plus a forward-monitoring artifact that records fill quality, realized round-trip cost, liquidation/margin safety, and split/reference lineage. Keep `ready_for_real=false`.

## Requirements summary

- Read the latest 10bps artifact and preserve its active/balanced model IDs.
- Generate paper-only live decision artifacts for:
  - active: `quality_single_pair abs_score_ge_1.5`, isolated `7x`, allocation `0.20`.
  - balanced reference: same filter, isolated `6x`, allocation `0.175`.
- Reuse the existing notional-risk-alignment/live-preflight surfaces where possible; avoid new execution abstractions unless required.
- Add a paper-forward monitor artifact that can ingest paper/testnet fills or simulated live fill logs and compute:
  - realized fee bps, slippage bps, and all-in round-trip bps.
  - notional/equity and isolated margin/equity parity.
  - liquidation count, account wipeout count, min margin buffer, MDD including isolated liquidation events.
  - active-vs-balanced comparison over the same timestamps.
- Reject any real-money execution path: `ready_for_real=false`, `real_money_execution=false`, and any unknown strategy/live-decision mismatch fails closed.

## Acceptance criteria

1. Decision artifacts include both model IDs, selection profile metadata, source artifact hash/path, split hash, and locked-OOS role as gate/report-only.
2. Preflight returns `ready_for_paper=true` only when exchange mode is paper/testnet and required sizing fields are parity-checked.
3. Preflight always returns `ready_for_real=false` for this task.
4. Monitoring artifact reports realized cost vs the `10bps` assumption and flags pass/fail thresholds.
5. Monitoring artifact includes active and balanced reference rows with liquidation-inclusive MDD and zero account-wipeout checks.
6. Tests cover sizing parity, fail-closed real-money gating, source-artifact lineage, and monitoring cost calculations.
7. Verification includes targeted tests, artifact assertion/preflight check, ruff, compileall, and git diff checks.

## Implementation steps

1. Inspect existing live decision/preflight code and artifacts from the notional-risk-alignment work:
   - `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/live_notional_risk_aligned_alpha_zoo_20260518/`
   - live decision/preflight scripts/tests touching `CryptoFxAlphaZooStateStrategy`.
2. Add or extend a runner for paper-only 10bps profile decisions, preferably reusing current live-notional-risk-alignment helpers.
3. Emit artifacts under a new directory, recommended:
   - `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_7x_paper_forward_preflight_20260519/`
4. Implement active and balanced decision outputs:
   - `live_alpha_zoo_quality_single_pair_7x_0p20_paper_decision_latest.json`
   - `live_alpha_zoo_quality_single_pair_6x_0p175_balanced_reference_decision_latest.json`
   - `live_readiness_preflight_alpha_zoo_7x_0p20_paper_latest.json`
5. Implement forward-monitor schema/output:
   - `paper_forward_monitoring_contract_latest.json`
   - optional CSV for periodic fill-quality snapshots.
6. Add/extend tests for fail-closed real-money gating, paper/testnet readiness, parity sizing, and 10bps realized-cost calculations.
7. Run verification and update research note + `.omx/notepad.md` with exact artifact paths and readiness status.
8. Lore commit and push to `private/main`.

## Risks and mitigations

- **Validation edge is weak**: track active vs balanced side-by-side; do not promote real-money from research metrics alone.
- **Live order sizing mismatch**: assert isolated margin fraction, notional/equity, leverage, and caps in preflight.
- **Actual cost exceeds 10bps**: monitoring must explicitly fail if realized all-in round-trip bps exceeds the research assumption materially.
- **Operator accidentally enables real money**: preflight must hard-code `ready_for_real=false` for this plan unless a later plan explicitly changes governance.
- **OOM risk**: all monitoring/preflight checks should run from existing compact artifacts/logs and keep peak RSS below 8 GiB.

## Verification plan

- Targeted live/preflight tests for Alpha Zoo decision artifacts.
- Targeted monitoring tests for fill-cost and liquidation-inclusive drawdown calculations.
- Artifact assertion/preflight CLI invocation on generated decisions.
- `uv run --extra dev ruff check ...`
- `uv run --extra dev python -m compileall -q ...`
- `git diff --check && git diff --cached --check`
- Optional broader pytest if touched surfaces are broad.

## Stop condition

Stop after paper-only decision/preflight/monitoring artifacts are generated, tests pass, docs/notepad are updated, a Lore commit is made, and `private/main` is pushed. Do not start live execution.

## New-session prompt

```text
$ralplan $team $ralph 이어서 진행해. 작업 디렉터리는 /home/hoky/Quants-agent/LuminaQuant 이야.

먼저 최신 상태를 맞춰:
- git fetch private
- git checkout private-main
- git reset --hard private/main
- git status -sb

반드시 먼저 읽어:
- .omx/plans/plan-alpha-zoo-7x-paper-forward-live-preflight-20260519.md
- docs/research_note/research_note.md
- .omx/notepad.md
- var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_10bps_full_retune_20260519/alpha_zoo_10bps_full_retune_latest.json
- var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_10bps_full_retune_20260519/low_correlation_discovery_latest.json
- var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/live_notional_risk_aligned_alpha_zoo_20260518/live_notional_risk_aligned_alpha_zoo_latest.json

목표:
10bps 최종 active 모델 `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_7p0x_0p2alloc`(7x/0.20)과 balanced reference `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_6p0x_0p175alloc`(6x/0.175)를 paper/testnet-only live decision + preflight + forward monitoring artifact로 연결해. 기존 notional-risk-alignment/live-preflight 경로를 최대한 재사용하고, replay/live sizing parity를 검증해. realized round-trip fee/slippage bps를 10bps 연구 가정과 비교할 monitoring schema/artifact를 만들어.

하드 제약:
- real-money 실행 금지. `ready_for_real=false`, `real_money_execution=false` 유지.
- locked-OOS는 selection/objective/pruning/parameter fitting에 쓰지 말고 gate/report-only로만 둬.
- isolated liquidation은 account equity/MDD에 포함.
- calendar/date hack 금지.
- 전체 세션 메모리 8GB 미만.
- active 7x/0.20과 balanced 6x/0.175를 같은 기준으로 나란히 보고해.

권장 artifact dir:
var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_7x_paper_forward_preflight_20260519/

완료 조건:
- paper/testnet preflight는 ready_for_paper=true 가능, ready_for_real=false 고정.
- active/balanced decision artifacts, monitoring contract JSON/CSV, 검증 로그 생성.
- split/source/profile lineage와 10bps cost assumption audit 포함.
- tests/ruff/compileall/diff-check 통과.
- 연구노트와 .omx/notepad.md 업데이트.
- Lore commit 후 private/main push.
```

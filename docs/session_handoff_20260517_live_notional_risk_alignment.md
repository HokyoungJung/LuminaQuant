# Session handoff — Live Alpha Zoo notional/risk alignment

Date: 2026-05-17 KST
Branch/baseline at handoff: `private-main` / `private/main` after high-leverage Alpha Zoo live-decision commit `dda9bfe4ec1e34cb05b70b1e6c2bb3dc08e7b258`.

## Current state

- Latest-data Alpha Zoo high-leverage winner: `CryptoFxAlphaZooStateStrategy` / `alpha_zoo_fast_residual` / isolated `7x` / `allocation_fraction=0.15`.
- Latest locked-OOS (`2026-04-01T00:00:00Z..2026-05-17T10:00:00Z`) no-cost replay: return `+30.53573988518672%`, MDD `11.302719903692077%`, Sharpe `1.8153544967585846`, Sortino `2.3185908095190877`, smart Sortino `2.083139398143474`, Calmar `2.7016275856939624`, trades `391`, liquidation `0`, account wipeout `0`.
- Strict fallback same params: `6x`, `allocation_fraction=0.10`, OOS return `+16.77825536078088%`, MDD `6.59514326586287%`, liquidation `0`.
- Paper preflight artifact: `ready_for_paper=true`, `ready_for_real=false`.

## Critical issue for next session

The research replay and live runtime do not currently share the same sizing semantics:

- Replay performance assumes `allocation_fraction * leverage * gross_return`, so `0.15 * 7 = 105%` notional exposure.
- Live sizing currently uses `target_allocation` as a notional cap, so `0.15` tends to mean `15%` notional exposure.
- `max_order_value=5000.0` is a legacy static cap in `config.yaml` / schema / risk manager and is not equity-scaled.

Next session should not assume the current 7x/0.15 live decision will reproduce the replay result until this contract is fixed and tested.

## Plan artifacts to start from

- PRD: `.omx/plans/prd-live-alpha-zoo-notional-risk-alignment-20260517.md`
- Test spec: `.omx/plans/test-spec-live-alpha-zoo-notional-risk-alignment-20260517.md`
- Research note: `docs/research_note_profit_moonshot_alpha_zoo_real_data_20260512.md`
- Current decision artifact: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/validation_to_20260331_latest_data_20260517/live_alpha_zoo_fast_residual_7x_isolated_decision_latest.json`
- Current replay artifact: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/validation_to_20260331_latest_data_20260517/alpha_zoo_validation_march_high_leverage_latest.json`

## Recommended next-session prompt

```text
$ralplan $team $ralph 이어서 진행해. 작업 디렉터리는 /home/hoky/Quants-agent/LuminaQuant 이야.

먼저 최신 상태를 맞춰:
- git fetch private
- git checkout private-main
- git reset --hard private/main
- git status -sb

이번 목표는 현재 1위 `CryptoFxAlphaZooStateStrategy / alpha_zoo_fast_residual / isolated 7x / allocation 0.15`를 실거래에서 리플레이와 같은 노출 의미로 작동하도록 live/replay sizing contract를 맞추고, 최대 성능을 유지하되 MDD를 대략 현 수준으로 유지하는 방향으로 retune/검증하는 것이다.

반드시 먼저 읽어:
- .omx/plans/prd-live-alpha-zoo-notional-risk-alignment-20260517.md
- .omx/plans/test-spec-live-alpha-zoo-notional-risk-alignment-20260517.md
- docs/session_handoff_20260517_live_notional_risk_alignment.md
- docs/research_note_profit_moonshot_alpha_zoo_real_data_20260512.md
- var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/validation_to_20260331_latest_data_20260517/alpha_zoo_validation_march_high_leverage_latest.json
- var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/validation_to_20260331_latest_data_20260517/live_alpha_zoo_fast_residual_7x_isolated_decision_latest.json

핵심 요구:
1. `target_allocation=0.15, leverage=7`이 replay처럼 isolated margin 15% / notional 105% 의미로 live에서 작동 가능한 명시적 sizing mode를 구현하거나, 그와 동등한 명시 계약을 만들어라. 기존 전략의 notional-cap semantics는 backward-compatible하게 유지해라.
2. `max_order_value=5000` 같은 fixed-dollar cap이 이 lane을 조용히 잘라먹지 않게 equity-scaled/leverage-aware risk cap으로 바꿔라. absolute cap은 명시된 emergency ceiling일 때만 적용되게 해라.
3. live order sizing과 replay expected notional이 같은지 paper-equivalent unit/integration test로 증명해라.
4. 최신 데이터 기준으로 train+validation only objective/selection으로 leverage/allocation을 retune해라. locked-OOS는 freeze 후 gate/report-only다.
5. 전체 계좌 wipeout은 절대 금지. isolated liquidation을 허용하는 lane이 있으면 그 손실은 account equity와 MDD에 반드시 포함해라. strict zero-liquidation lane은 별도 표로 유지해라.
6. MDD는 현재 7x OOS 11.30% 근처를 선호하고, 불가피해도 OOS 25% hard cap을 넘기지 마라. 수익은 strict 6x fallback 및 current-base reference보다 좋아야 한다.
7. 비용 현실성을 보고하라: no-cost headline과 별도로 slippage/fee 1/3/5/10/20bps, funding 1/2/5/10/20bps/day sensitivity를 포함하고, live promotion이 어느 비용 threshold까지 유효한지 명시해라.
8. real money 실행은 하지 마라. paper/testnet smoke와 preflight까지만 수행하고, ready_for_real은 조건 충족 여부만 보고해라.
9. 결과 artifact는 새 디렉터리에 저장해라: var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/live_notional_risk_aligned_alpha_zoo_20260518/
10. 연구노트, handoff, .omx/notepad.md를 업데이트하고 Lore commit으로 private/main에 push해라. 가능하면 GitHub Actions ci/private-ci green까지 확인해라.

필수 검증:
uv run --extra dev pytest tests/test_live_selection_infer.py tests/test_live_fail_fast_missing_committed_data.py tests/test_live_execution_state_machine.py tests/test_crypto_fx_alpha_zoo_state_strategy.py -q
uv run --extra dev pytest tests/test_profit_moonshot_fresh_start_replay.py tests/test_profit_moonshot_liquidation_aware_validation.py tests/test_profit_moonshot_live_final_selection.py tests/test_profit_moonshot_pass_under_8gb_validator.py -q
uv run --extra dev pytest -q
uv run --extra dev ruff check .
uv run --extra dev python -m compileall -q src scripts tests
git diff --check
git diff --cached --check

최종 보고에는 commit hash, artifact paths, selected sizing mode, notional/equity, margin/equity, liquidation-inclusive MDD, cost-stressed 결과, paper-equivalent live sizing parity evidence, preflight status, CI links를 포함해.
```

## Research history/source ledger note

No global research inventory/source ledger regeneration is required for this handoff alone. It adds a plan and documentation for the next implementation session and does not introduce a new data-source family. If the next session refreshes market data beyond the current tail or adds new source families, it must revisit `docs/profit_moonshot_research_history_20260510.md` and matching `var/reports/.../research_history/` artifacts.

## 2026-05-18 KST completion addendum

The live/replay contract mismatch has been closed for the current high-leverage Alpha Zoo winner. Live now supports an explicit `isolated_margin_fraction` sizing mode so `target_allocation=0.15` and `leverage=7` means `15%` isolated margin/equity and `105%` notional/equity. Existing strategies retain default `legacy_notional_cap` behavior for backward compatibility. The promoted decision artifact disables the old fixed-dollar cap (`max_order_value=0.0`) and uses equity-scaled caps (`max_order_notional_pct=1.10`, `max_symbol_exposure_pct=1.10`, `max_total_notional_pct=2.20`).

Final selected lane remains `CryptoFxAlphaZooStateStrategy / alpha_zoo_fast_residual / isolated 7x / allocation 0.15`. Train+validation-only retune found the same notional-equivalent score for `6x/0.175`; the documented incumbent tie-breaker kept the requested `7x/0.15` contract. Locked-OOS was used only after freeze as gate/report-only. Locked-OOS no-cost result is `+30.5357%` return with liquidation-inclusive MDD `11.3027%`, liquidation `0`, account wipeout `0`. Cost diagnostics: slippage/fee `1/3/5/10/20bps` -> `+25.2882%`, `+15.4160%`, `+6.3199%`, `-13.4130%`, `-42.5899%`; funding `1/2/5/10/20bps/day` -> `+29.9911%`, `+29.4486%`, `+27.8349%`, `+25.1897%`, `+20.0619%`.

Paper-equivalent parity evidence: `$10,000` equity, `$100` price, `0.15`, `7x` -> replay expected notional `$10,500`, live notional `$10,500`, diff `0.0`, risk check `Passed`. Preflight result: `paper_run_allowed`, `ready_for_paper=true`, `ready_for_real=false`; no real-money execution was attempted.

Artifacts are under `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/live_notional_risk_aligned_alpha_zoo_20260518/`:

- `live_notional_risk_aligned_alpha_zoo_latest.json`
- `live_notional_risk_aligned_alpha_zoo_latest.md`
- `live_alpha_zoo_notional_risk_aligned_decision_latest.json`
- `live_readiness_preflight_notional_risk_aligned_latest.json`
- `alpha_zoo_validation_march_high_leverage_candidates_latest.csv`
- `local_verification_live_notional_risk_alignment_20260518T113100Z.log`

Strict zero-liquidation lane remains separate: strict `6x` / `10%` allocation, locked-OOS `+16.7783%`, MDD `6.5951%`, liquidation `0`, min buffer `9150.924760`.

Fresh local verification passed: required live/Alpha Zoo suite `32 passed`; required moonshot validation suite `74 passed`; full pytest `1340 passed`; ruff, compileall, `git diff --check`, and `git diff --cached --check` passed. Next handoff risk: real exchange/testnet order placement, exchange-side isolated leverage/margin confirmation, and production credentials remain intentionally untested and require a separate explicit authorization.

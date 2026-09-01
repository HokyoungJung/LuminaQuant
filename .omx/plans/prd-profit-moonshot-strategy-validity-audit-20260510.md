# PRD: Profit Moonshot Strategy Validity Audit and Retune

## Objective
Prevent theoretically defective, non-live-robust strategy rules from being promoted, then re-rank/retune profit-moonshot candidates under live-deployable constraints.

## User problem
The current selected direct 5x candidate passed performance/liquidation/integer gates but relies on a fixed calendar month + fixed asset rule (`TRX` long in Mar-May, `ETH` short in Jan-Feb). This is not a defensible live strategy without robust independent seasonality evidence and may reorder all candidate rankings once invalidated.

## Requirements
1. Audit all final, candidate, retune, hybrid, and current-base strategy rows used by the latest selection pipeline for strategy-validity flaws.
2. Reject or quarantine alpha rules that are fixed month + fixed asset calendar/seasonality rules, or otherwise hard-code future-relevant calendar behavior without robust thesis/evidence.
3. Add strategy-validity gates to final selection so theoretical defects block `deployable_candidate` even if performance is good.
4. Retune/reselect from allowed dynamic, state-based strategies only; preserve train/validation-only selection and locked-OOS gate/report-only policy.
5. Preserve integer leverage, liquidation, margin buffer, OOS MDD, performance, and memory gates.
6. Produce reports showing invalidated candidates, new ranking, whether any strategy remains deployable, and live recommendation.
7. Persist outputs under `.omx/notepad.md`, `.omx/plans`, `docs/session_handoff_*`, and `var/reports/.../strategy_validity_*` / final selection artifacts.
8. Run targeted tests, full pytest, ruff, compileall, git diff check; commit/push to private/main; verify CI/private-ci green.

## Acceptance criteria
- Calendar-fixed TRX/ETH direct winner is explicitly rejected by a strategy-validity gate.
- Non-integer leverage and liquidation gates continue to function.
- Locked-OOS remains report/gate-only.
- Tests cover at least: fixed calendar rule blocks promotion; dynamic non-calendar rule is allowed; final selection cannot promote rows with strategy-validity failures; audit report includes all relevant rows.
- Final report clearly says deploy / no-deploy with evidence.


## Strategy-validity audit closure requirements
- Audit artifact must include a `closure_manifest` array. Each item must include `path`, `artifact_kind`, `row_count`, and `source_role`.
- Final selection must be fail-closed: every row must include `strategy_validity.pass`, `primary_signal_type`, `primary_signal_evidence`, `audited_sleeves`, and `audit_sources`. Missing metadata blocks promotion.
- Calendar-primary alpha (`calendar_rotation`/`calendar_spread` where month/window is the primary signal) is invalid by default regardless of fixed or dynamic asset target unless separately justified by robust external/mechanistic evidence plus out-of-family robustness tests.
- Dynamic state-based primary signals may pass even with secondary time/session filters. Allowed primary signal families include residual, funding/OI, flow, adaptive/trend, cross-sectional rank, compression/volatility, and pair residual/momentum spreads.
- This handoff must not start a new alpha search. If existing valid dynamic candidates do not pass live gates, final recommendation is `no_live_promotion`.

### Closure manifest parity requirement
`closure_manifest` must include entries for all applicable roles: `final_selection_json`, `final_selection_md`, `liquidation_validation`, `candidate_portfolio`, `candidate_hybrid`, `merged_candidate_csv`, `current_base`, `passing_artifacts`, plus per-row/per-sleeve traced source coverage for every final row kind. Every manifest entry requires `path`, `artifact_kind`, `row_count`, and `source_role`; optional/missing artifacts must be listed in `missing_optional_sources` with reason rather than silently omitted.

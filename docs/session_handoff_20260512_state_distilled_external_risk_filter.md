# Session handoff — state-distilled external-risk teacher pass — 2026-05-12

## Scope

Continue profit-moonshot state-distilled work without calendar/month/day/hour entry rules. The calendar/current-base tuple was used only as a teacher/reference for market-state hypotheses, never as a live selector.

## Where the calendar-teacher strategy went

The earlier calendar-teacher conversion is `state_distilled_leadership_unwind`. It did produce valid, zero-liquidation performance, but it did not recover the invalid calendar tuple's economics:

- invalid current-base/calendar reference: locked-OOS `+6.4281%`, return/MDD `6.9169`, Sharpe `5.2024`; strategy-validity failed, so `hypothesis_reference_only`.
- best strict valid state-distilled row before this pass: `fresh_state_distilled_both_lb168_fast72_z075_ret180_h168_ls590_ss100_tp600` at 4x, locked-OOS `+2.4722%`, MDD `2.5328%`, Sharpe `1.5131`, liquidation `0`; not deployable vs reference.

## New code

- `scripts/research/fetch_profit_moonshot_external_state.py`
  - Fetches daily FRED CSV data for `DTWEXBGS`, `VIXCLS`, `DGS2`, `DGS10`, `DCOILWTICO`.
  - Builds lagged `external_*` state features so hourly replay uses prior-observation data only.
- `scripts/research/replay_profit_moonshot_fresh_start.py`
  - Adds `--external-state-csv` join path.
  - Adds non-calendar families:
    - `calendar_teacher_state_similarity`
    - `calendar_teacher_state_fade`
    - `state_distilled_external_risk_filter`
- `scripts/research/run_profit_moonshot_liquidation_aware_validation.py`
  - Adds the same `--external-state-csv` join path for liquidation-aware replay.

## Artifacts

- External state: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/external_market_state_20260512/`
- Teacher similarity replay: `.../calendar_teacher_state_similarity_20260512/`
- Teacher fade replay: `.../calendar_teacher_state_fade_20260512/`
- External-risk state-distilled replay: `.../state_distilled_external_risk_filter_20260512/`
- Liquidation-aware validation: `.../liquidation_aware_state_distilled_external_risk_filter_20260512/`

## Results

### Teacher similarity / fade

- `calendar_teacher_state_similarity`: 972 specs, 0 survivor; all train/validation negative.
- `calendar_teacher_state_fade`: 324 specs, 0 survivor; all train/validation negative.

### State-distilled external risk filter

Replay: 1,728 specs, 565 train/validation-positive, 0 replay survivor under legacy shadow-MDD gate, peak RSS `280.348 MiB`.

Best train/validation-positive OOS diagnostic row after freeze (report-only, not a selector):

`fresh_state_distilled_ext_both_lb336_fast168_z050_ret180_h120_tp750_fl0_xr200`

- train `+3.2248%`
- validation `+1.8146%`
- locked-OOS `+1.4128%`
- OOS MDD `0.5320%`
- OOS Sharpe `3.6699`
- replay liquidations `0`

Liquidation-aware train/validation-selected strict row:

`fresh_state_distilled_ext_both_lb168_fast72_z075_ret180_h168_tp600_fl0_xr125` at 4x

| Split | Return | MDD | Sharpe | Sortino | Calmar | Liq | Min margin buffer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 30.9030% | 10.2437% | 1.8484 | 1.8818 | 3.0172 | 0 | 9740.5206 |
| validation | 12.4704% | 2.5167% | 5.7588 | 6.7743 | 42.5165 | 0 | 9959.0876 |
| locked-OOS | 2.4852% | 2.5328% | 1.5096 | 1.8652 | 5.7103 | 0 | 9875.2252 |

Decision: strategy-validity passes and strict liquidation lane passes, but deployable success is still false because OOS return and return/MDD do not beat the invalid current-base reference. 5x/6x remains diagnostic-only because train/validation liquidations appear.

## Next recommended work

1. Do not chase the raw calendar tuple as a live target; it remains a teacher/hypothesis source only.
2. Convert the external-risk filter into a train/validation-only nested selector that prioritizes validation robustness/turnover/MDD before any OOS report is opened.
3. Try portfolio-level combination of the external-risk filtered sleeve with residual-pair reversion rather than a single sleeve.
4. If higher leverage is needed, add margin-forecast exposure scaling before leverage validation; current 5x/6x diagnostics are blocked by train liquidations.
